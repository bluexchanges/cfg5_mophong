"""Safe epoch-boundary switch and source integrity without a Slurm cluster."""

import hashlib
import json
import sys
import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from src.data.common import ROOT

sys.path.insert(0, str(ROOT / "scripts/slurm"))
import install_worker_bundle as installer
import switch_to_workers as switch


class WorkerTransitionTests(unittest.TestCase):
    def test_cancel_only_after_history_boundary_and_only_matching_job(self):
        running = [("123", switch.OLD_NAME, "R"), ("999", "someone_else", "R")]
        with patch.object(switch, "jobs", side_effect=[running, running, []]), \
             patch.object(switch, "last_epoch", side_effect=[15, 16, 16]), \
             patch.object(switch.time, "sleep"), \
             patch.object(switch.subprocess, "run") as command:
            switch.wait_for_boundary("fixture-user", "123", "history", 15, poll_seconds=0)
        command.assert_called_once_with(
            ["scancel", "--user", "fixture-user", "--name", switch.OLD_NAME, "123"], check=True)

    def test_finished_job_needs_no_cancel(self):
        with patch.object(switch, "jobs", return_value=[]), \
             patch.object(switch.subprocess, "run") as command:
            switch.wait_for_boundary("fixture-user", "123", "history", 15)
        command.assert_not_called()

    def test_unexpected_state_is_not_cancelled(self):
        with patch.object(switch, "jobs", return_value=[("123", switch.OLD_NAME, "PD")]), \
             patch.object(switch.subprocess, "run") as command:
            with self.assertRaises(RuntimeError):
                switch.wait_for_boundary("fixture-user", "123", "history", 15)
        command.assert_not_called()

    def test_installer_rejects_corrupt_source_before_editing_repo(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            repo = Path(temp) / "base/edabk/04-Src/software/cfg5_act"
            manifest = {}
            for name in installer.FILES:
                path = bundle / name
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = b"value = 1\n"
                path.write_bytes(payload)
                manifest[name] = hashlib.sha256(payload).hexdigest()
            (bundle / "bundle_hashes.json").write_text(json.dumps(manifest))
            target = repo / installer.FILES[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"old_code = True\n")
            (bundle / installer.FILES[0]).write_bytes(b"value = 1\x00\n")
            with self.assertRaisesRegex(ValueError, "sai checksum"):
                installer.install(bundle, repo)
            self.assertEqual(target.read_bytes(), b"old_code = True\n")
            (bundle / installer.FILES[0]).write_bytes(b"value = 1\n")
            installer.install(bundle, repo)
            self.assertEqual(target.read_bytes(), b"value = 1\n")
            backup = next((Path(temp) / "base").glob("workers2-code-backup-*"))
            self.assertEqual((backup / installer.FILES[0]).read_bytes(), b"old_code = True\n")

    def test_transition_submits_only_after_old_job_leaves_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            base, bundle = Path(temp) / "base", Path(temp) / "bundle"
            repo = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act"
            source = repo / "outputs/act_pretrain_bs4"
            source.mkdir(parents=True)
            (source / "latest.pt").write_bytes(b"fixture; not loaded by head-node watcher")
            (source / "history.json").write_text(json.dumps([{"epoch": 15}]))
            config = repo / "configs/act/cfg5_all_pretrain_bs4_40epoch.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"save_every_epochs": 1,
                                           "output_dir": "outputs/act_pretrain_bs4"}))
            template = bundle / "scripts/slurm/act_bs4_workers2.sbatch"
            template.parent.mkdir(parents=True)
            template.write_text("#!/bin/bash\nBUNDLE=__BUNDLE__\n")
            result = CompletedProcess([], 0, stdout="456\n", stderr="")
            events = []
            with patch.object(switch, "validate_bundle"), \
                 patch.object(switch, "jobs", side_effect=[[("123", switch.OLD_NAME, "R")], []]), \
                 patch.object(switch, "wait_for_boundary", side_effect=lambda *a: events.append("boundary")), \
                 patch.object(switch.subprocess, "run", side_effect=lambda *a, **k: (events.append("submit") or result)):
                switch.transition(bundle, base, repo, "fixture-user")
            self.assertEqual(events, ["boundary", "submit"])
            self.assertEqual((base / "act_bs4_workers2.jobid").read_text(), "456\n")
            self.assertNotIn("__BUNDLE__", (base / "act_bs4_workers2.sbatch").read_text())

    def test_existing_worker_job_is_not_submitted_again(self):
        with patch.object(switch, "validate_bundle"), \
             patch.object(switch, "jobs", return_value=[("456", switch.NEW_NAME, "PD")]), \
             patch.object(switch.subprocess, "run") as command:
            switch.transition(Path("bundle"), Path("base"), Path("repo"), "fixture-user")
        command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
