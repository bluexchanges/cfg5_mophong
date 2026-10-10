"""Checkpoint migration and boundary switch for the optional transfer pipeline."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import torch
from src.data.common import ROOT
from src.training import pretrain
from src.training.fork_run import fork_run
from tests.test_pretraining import TinyPolicy, tiny_loader

sys.path.insert(0, str(ROOT / "scripts/slurm"))
import switch_to_cuda_prefetch as switch


class PrefetchTransitionTests(unittest.TestCase):
    def test_cancel_is_scoped_and_waits_for_saved_epoch(self):
        running = [("13925", switch.OLD_NAME, "R"), ("13921", "T27", "R")]
        with patch.object(switch, "jobs", side_effect=[running, running, []]), \
             patch.object(switch, "last_epoch", side_effect=[16, 17, 17]), \
             patch.object(switch.time, "sleep"), \
             patch.object(switch.subprocess, "run") as command:
            switch.wait_for_boundary("user08", "13925", "history", 16)
        command.assert_called_once_with(
            ["scancel", "--user", "user08", "--name", switch.OLD_NAME, "13925"], check=True)

    def test_handoff_submits_after_stop_and_existing_job_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            base, bundle = Path(temp) / "base", Path(temp) / "bundle"
            repo = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act"
            parent = repo / "outputs/act_pretrain_bs4_w2"
            parent.mkdir(parents=True)
            (parent / "latest.pt").write_bytes(b"fixture")
            (parent / "history.json").write_text(json.dumps([{"epoch": 16}]))
            cfg = repo / "configs/act/cfg5_all_pretrain_bs4_w2_40epoch.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({"save_every_epochs": 1, "batch_size": 4, "num_workers": 2,
                                       "output_dir": "outputs/act_pretrain_bs4_w2"}))
            template = bundle / "scripts/slurm/act_cuda_prefetch.sbatch"
            template.parent.mkdir(parents=True)
            template.write_text("#!/bin/bash\nBUNDLE=__BUNDLE__\n")
            events = []
            with patch.object(switch, "validate_bundle"), \
                 patch.object(switch, "jobs", side_effect=[[("13925", switch.OLD_NAME, "R")], []]), \
                 patch.object(switch, "wait_for_boundary", side_effect=lambda *a: events.append("saved-and-stopped")), \
                 patch.object(switch.subprocess, "run", side_effect=lambda *a, **k:
                              (events.append("submit") or CompletedProcess([], 0, stdout="14000\n"))):
                switch.transition(bundle, base, repo, "user08")
            self.assertEqual(events, ["saved-and-stopped", "submit"])
            self.assertEqual((base / "act_cuda_prefetch.jobid").read_text(), "14000\n")
        with patch.object(switch, "validate_bundle"), \
             patch.object(switch, "jobs", return_value=[("14000", switch.NEW_NAME, "PD")]), \
             patch.object(switch.subprocess, "run") as command:
            switch.transition(Path("bundle"), Path("base"), Path("repo"), "user08")
        command.assert_not_called()

    def test_fork_prefetch_preserves_weights_optimizer_and_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            run = {"output_dir": str(Path(temp) / "parent"), "processed_root": temp,
                   "seed": 7, "epochs": 1, "batch_size": 4, "num_workers": 2,
                   "save_every_epochs": 1, "log_every_batches": 0, "grad_clip_norm": 1.0}
            data, contract, stats = {"dataset_root": str(Path(temp) / "raw")}, {}, {"smoke_only": False}
            with patch.object(pretrain, "RH20TACTPolicy", TinyPolicy), \
                 patch.object(pretrain, "make_loader", tiny_loader):
                pretrain.train(run, {}, data, {}, stats, contract, device=torch.device("cpu"))
            source = Path(run["output_dir"]) / "latest.pt"
            before_bytes = source.read_bytes()
            target = dict(run, epochs=40, cuda_prefetch=True, output_dir=str(Path(temp) / "child"))
            child = fork_run(target, {}, data, {}, stats, contract, source)
            a = torch.load(source, weights_only=True)
            b = torch.load(child, weights_only=True)
            self.assertEqual(source.read_bytes(), before_bytes)
            self.assertEqual((a["epoch"], a["global_step"]), (b["epoch"], b["global_step"]))
            self.assertTrue(torch.equal(a["model"]["weight"], b["model"]["weight"]))
            self.assertTrue(torch.equal(a["optimizer"]["state"][0]["exp_avg"],
                                        b["optimizer"]["state"][0]["exp_avg"]))
            self.assertEqual(b["run_config_hash"], pretrain.resume_signature(target))
            self.assertNotEqual(a["run_config_hash"], b["run_config_hash"])


if __name__ == "__main__":
    unittest.main()
