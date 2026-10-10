"""One-GPU fallback: old workers, safe pending-job replacement and resume."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.common import ROOT
from src.training import pretrain
from src.training.fork_run import fork_run
from tests.test_pretraining import TinyPolicy, tiny_loader

sys.path.insert(0, str(ROOT / "scripts/slurm"))
import continue_bs8_to80 as runner
import submit_bs8_to80 as submit


class BS8TransitionTests(unittest.TestCase):
    def test_target_keeps_exact_previous_worker_and_optional_signature(self):
        for workers in (0, 2, 4):
            old = {"epochs": 40, "batch_size": 16, "num_workers": workers,
                   "save_every_epochs": 1, "output_dir": "parent", "seed": 42}
            cfg = runner.target_config(old)
            self.assertEqual((cfg["epochs"], cfg["batch_size"], cfg["num_workers"]), (80, 8, workers))
            self.assertNotIn("gpu_count", cfg)
            self.assertEqual(old["output_dir"], "parent")
            self.assertEqual(cfg["seed"], 42)
        with self.assertRaises(ValueError):
            runner.target_config(dict(old, gpu_count=2))
        with self.assertRaises(ValueError):
            runner.target_config(dict(old, num_workers=8))

    def test_cancel_only_exact_pending_job_and_leave_other_job(self):
        with patch.object(submit, "queue", side_effect=[
                [("13988", "cfg5-act-bs16-g2-to80", "PD"), ("13936", "T27", "R")],
                [("13936", "T27", "R")]]), patch.object(submit.subprocess, "run") as command:
            submit.replace_pending("13988", "user08")
        self.assertEqual(command.call_args.args[0], ["scancel", "--user", "user08",
                         "--name=cfg5-act-bs16-g2-to80", "--state=PENDING", "13988"])

    def test_running_wrong_name_or_other_act_is_never_cancelled(self):
        for entries in ([('13988', 'cfg5-act-bs16-g2-to80', 'R')],
                        [('13988', 'T27', 'PD')],
                        [('13988', 'cfg5-act-bs16-g2-to80', 'PD'), ('14000', 'cfg5-act-other', 'R')]):
            with patch.object(submit, "queue", return_value=entries), \
                    patch.object(submit.subprocess, "run") as command:
                with self.assertRaises(RuntimeError):
                    submit.replace_pending("13988", "user08")
                command.assert_not_called()

    def test_pending_to_running_race_stops_without_second_job(self):
        with patch.object(submit, "queue", side_effect=[
                [('13988', 'cfg5-act-bs16-g2-to80', 'PD')],
                [('13988', 'cfg5-act-bs16-g2-to80', 'R')]]), \
                patch.object(submit.subprocess, "run") as command:
            with self.assertRaisesRegex(RuntimeError, "chuyển R"):
                submit.replace_pending("13988", "user08")
        self.assertEqual(command.call_count, 1)
        self.assertIn("--state=PENDING", command.call_args.args[0])

    def test_bs8_fork_resume_preserves_completed_epoch_and_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            parent, child = temp / "parent", temp / "child"
            old = {"output_dir": str(parent), "processed_root": str(temp), "seed": 7,
                   "epochs": 1, "batch_size": 1, "num_workers": 2,
                   "save_every_epochs": 1, "log_every_batches": 0, "grad_clip_norm": 1.0}
            data = {"dataset_root": str(temp / "raw")}
            stats = {"smoke_only": False}
            contract = {"split_hash": "fixed"}
            with patch.object(pretrain, "RH20TACTPolicy", TinyPolicy), \
                    patch.object(pretrain, "make_loader", tiny_loader):
                history = pretrain.train(old, {}, data, {}, stats, contract, device=torch.device("cpu"))
                saved = torch.load(parent / "latest.pt", weights_only=True)
                # Tiny fixture represents a completed 40th epoch, without full training.
                saved["epoch"] = history[-1]["epoch"] = 39
                torch.save(saved, parent / "latest.pt")
                pretrain.write_json(parent / "history.json", history)
                target = dict(runner.target_config(old), output_dir=str(child))
                latest = fork_run(target, {}, data, {}, stats, contract, parent / "latest.pt")
                forked = torch.load(latest, weights_only=True)
                self.assertEqual(forked["epoch"], 39)
                torch.testing.assert_close(forked["optimizer"]["state"][0]["exp_avg"],
                                           saved["optimizer"]["state"][0]["exp_avg"])
                result = pretrain.train(dict(target, epochs=41), {}, data, {}, stats, contract,
                                        device=torch.device("cpu"), resume=latest)
                self.assertEqual(result[-1]["epoch"], 40)
                self.assertEqual(result[-1]["batch_size"], 8)
                self.assertEqual(result[-1]["num_workers"], 2)
                self.assertEqual(result[-1]["global_step"], saved["global_step"] + 1)
                self.assertEqual(torch.load(parent / "latest.pt", weights_only=True)["epoch"], 39)


if __name__ == "__main__":
    unittest.main()
