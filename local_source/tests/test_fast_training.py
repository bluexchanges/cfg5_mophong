"""Throughput selection, process scope and saved-epoch handoff without a GPU."""
import hashlib
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

from src.data.common import ROOT

sys.path.insert(0, str(ROOT / "scripts/slurm"))
import fast_train as tuner
import switch_to_fast as switch


def row(batch, workers, rate, memory=40):
    return {"status": "ok", "batch_size": batch, "num_workers": workers,
            "samples_per_s": rate, "peak_reserved_bytes": memory, "total_vram_bytes": 80}


class FastTrainingTests(unittest.TestCase):
    def test_choose_speed_with_vram_headroom_and_tie_tolerance(self):
        rows = [row(4, 2, 16), row(8, 2, 25), row(8, 4, 25.4),
                row(16, 4, 26), row(32, 4, 40, 75), {"status": "oom"}]
        self.assertEqual(tuner.choose_trial(rows)["batch_size"], 8)
        self.assertEqual(tuner.choose_trial(rows)["num_workers"], 4)
        self.assertEqual(tuner.choose_trial([row(4, 2, 16), row(8, 4, 15)])["batch_size"], 4)

    def test_invalid_results_never_start_training(self):
        for rows in ([{"status": "oom"}], [row(8, 4, float("nan"))], [row(8, 4, 30, 79)]):
            with self.assertRaises(RuntimeError):
                tuner.choose_trial(rows)

    def test_trial_config_does_not_mutate_source_or_change_learning_rate(self):
        old = {"batch_size": 4, "num_workers": 2, "epochs": 40,
               "output_dir": "old", "lr": 0.00001, "seed": 42}
        cfg = tuner.trial_config(old, 16, 4)
        self.assertEqual(old["output_dir"], "old")
        self.assertEqual((cfg["lr"], cfg["seed"]), (old["lr"], old["seed"]))
        self.assertEqual((cfg["batch_size"], cfg["num_workers"]), (16, 4))

    def test_trials_are_isolated_and_timeout_does_not_modify_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "latest.pt"
            source.write_bytes(b"untouched checkpoint")
            commands = []
            def execute(cmd, **kwargs):
                commands.append(cmd)
                batch = int(cmd[cmd.index("--batch") + 1])
                workers = int(cmd[cmd.index("--workers") + 1])
                if batch == 32:
                    raise TimeoutExpired(cmd, 300)
                result = Path(cmd[cmd.index("--result") + 1])
                tuner.write_json(result, row(batch, workers, 16 + batch))
                return CompletedProcess(cmd, 0)
            with patch.object(tuner.subprocess, "run", side_effect=execute):
                chosen = tuner.run_trials(source, temp / "reports")
            self.assertEqual(len(commands), len(tuner.CANDIDATES))
            self.assertEqual(chosen["batch_size"], 16)
            self.assertEqual(source.read_bytes(), b"untouched checkpoint")
            self.assertTrue(all("--trial" in command for command in commands))

    def test_watcher_identity_matches_only_own_actual_script(self):
        with tempfile.TemporaryDirectory() as temp:
            base, proc = Path(temp) / "base", Path(temp) / "proc"
            p = proc / "123"
            p.mkdir(parents=True)
            allowed = base / "act_cuda_prefetch_bundle/scripts/slurm/switch_to_cuda_prefetch.py"
            cmd = p / "cmdline"
            cmd.write_bytes(b"python3\x00-u\x00" + str(allowed).encode() + b"\x00")
            uid = p.stat().st_uid
            self.assertIsNotNone(switch.watcher_identity(123, base, proc_root=proc, uid=uid))
            self.assertIsNone(switch.watcher_identity(123, base, proc_root=proc, uid=uid + 1))
            cmd.write_bytes(b"python3\x00scripts/train.py\x00--flag\x00" + str(allowed).encode() + b"\x00")
            self.assertIsNone(switch.watcher_identity(123, base, proc_root=proc, uid=uid))
            cmd.write_bytes(b"grep\x00" + str(allowed).encode() + b"\x00")
            self.assertIsNone(switch.watcher_identity(123, base, proc_root=proc, uid=uid))

    def test_running_job_cancelled_only_after_checkpoint_and_by_exact_name(self):
        running = [("13925", "cfg5-act-bs4-w2", "R"), ("13921", "T27", "R")]
        with patch.object(switch, "last_epoch", side_effect=[16, 16, 17, 17]), \
             patch.object(switch, "jobs", side_effect=[running, running]), \
             patch.object(switch, "wait_gone"), patch.object(switch.time, "sleep"), \
             patch.object(switch.subprocess, "run") as command:
            switch.wait_boundary("user08", running[0], Path("history"))
        command.assert_called_once_with(
            ["scancel", "--user", "user08", "--name", "cfg5-act-bs4-w2", "13925"], check=True)

    def test_takeover_waits_for_slurm_child_and_never_targets_trainer(self):
        events = []
        def identity(pid, _base):
            return ("python3", "known-watcher") if pid == 123 else None
        child_states = iter([["456"], []])
        def children(_folder):
            state = next(child_states)
            events.append("child-alive" if state else "child-finished")
            return state
        with patch.object(Path, "iterdir", return_value=[Path("/proc/123"), Path("/proc/987")]), \
             patch.object(switch.signal, "SIGSTOP", 19, create=True), \
             patch.object(switch.signal, "SIGCONT", 18, create=True), \
             patch.object(switch, "watcher_identity", side_effect=identity), \
             patch.object(switch, "live_children", side_effect=children), \
             patch.object(switch.time, "sleep"), \
             patch.object(switch.os, "kill", side_effect=lambda pid, sig: events.append((pid, sig))):
            switch.stop_known_watchers(Path("base"))
        self.assertEqual(events, [(123, 19), "child-alive", "child-finished",
                                  (123, signal.SIGTERM), (123, 18)])

    def test_pending_job_cancel_is_filtered_and_unknown_jobs_rejected(self):
        with patch.object(switch, "wait_gone"), patch.object(switch.subprocess, "run") as command:
            switch.wait_boundary("user08", ("999", "cfg5-act-prefetch", "PD"), Path("folder"))
        self.assertIn("--state=PENDING", command.call_args.args[0])
        with patch.object(switch, "jobs", return_value=[("1", "cfg5-act-unknown", "R")]), \
             patch.object(switch.subprocess, "run") as command:
            with self.assertRaises(RuntimeError):
                switch.transition(Path("bundle"), Path("base"), Path("repo"), "user08")
        command.assert_not_called()

    def test_latest_source_uses_highest_saved_epoch_and_child_on_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            for name, epoch in (("act_pretrain_bs4_w2", 17), ("act_pretrain_prefetch", 17),
                                ("act_pretrain_fast", 16)):
                p = repo / "outputs" / name
                p.mkdir(parents=True)
                (p / "latest.pt").write_bytes(b"checkpoint fixture")
                (p / "history.json").write_text(json.dumps([{"epoch": epoch}]))
            source, epoch = switch.latest_source(repo)
            self.assertEqual(source.parent.name, "act_pretrain_prefetch")
            self.assertEqual(epoch, 17)

    def test_submit_after_boundary_and_do_not_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            bundle = base / "bundle"
            repo = base / "repo"
            cfg = repo / "configs/act/cfg5_all_pretrain_bs4_w2_40epoch.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({"save_every_epochs": 1, "output_dir": "outputs/act_pretrain_bs4_w2"}))
            template = bundle / "scripts/slurm/act_fast40.sbatch"
            template.parent.mkdir(parents=True)
            template.write_text("BUNDLE=__BUNDLE__\nSOURCE=__SOURCE__\n")
            events = []
            with patch.object(switch, "jobs", side_effect=[[("13925", "cfg5-act-bs4-w2", "R")], []]), \
                 patch.object(switch, "wait_boundary", side_effect=lambda *a: events.append("saved-and-stopped")), \
                 patch.object(switch, "latest_source", return_value=(repo / "latest.pt", 17)), \
                 patch.object(switch.subprocess, "run", side_effect=lambda *a, **k:
                              (events.append("submitted") or CompletedProcess([], 0, stdout="14001\n"))):
                switch.transition(bundle, base, repo, "user08")
            self.assertEqual(events, ["saved-and-stopped", "submitted"])
            self.assertNotIn("__SOURCE__", (base / "act_fast40.sbatch").read_text())
        with patch.object(switch, "jobs", return_value=[("14001", switch.NEW_NAME, "PD")]), \
             patch.object(switch.subprocess, "run") as command:
            switch.transition(Path("bundle"), Path("base"), Path("repo"), "user08")
        command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
