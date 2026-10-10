"""Verify held-out evaluation has no optimizer updates or target leakage into inference."""

import importlib.util
import tempfile
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.common import ROOT, load_config, read_json
from src.models.rh20t_act import RH20TACTPolicy

spec = importlib.util.spec_from_file_location("heldout_test", ROOT / "scripts/evaluate_pretrain_test.py")
heldout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heldout)
submit_spec = importlib.util.spec_from_file_location("submit_test", ROOT / "scripts/slurm/submit_pretrain_test.py")
submit = importlib.util.module_from_spec(submit_spec)
submit_spec.loader.exec_module(submit)


class SmallDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, i):
        mask = torch.ones(20, dtype=torch.bool)
        mask[:i+1] = False
        target = torch.ones(20, 8)
        target[mask] = 999  # Padding must never contribute to either metric.
        return {"state": torch.zeros(8), "images": torch.zeros(8, 3, 32, 32),
                "actions": target, "is_pad": mask, "episode": f"test-{i}"}


class SpyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def configure_optimizer(self):
        raise AssertionError("Evaluation must not construct an optimizer")

    def forward(self, state, images, actions=None, is_pad=None):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append(actions is not None)
        prediction = self.weight.expand(len(state), 20, 8)
        if actions is None:
            return prediction
        count = int((~is_pad).sum()) * 8
        l1 = (prediction[~is_pad] - actions[~is_pad]).abs().mean()
        return {"l1": l1, "kl": torch.tensor(0.0), "valid_action_elements": torch.tensor(count)}


class HeldoutTests(unittest.TestCase):
    def test_no_weight_update_and_no_targets_in_inference(self):
        policy = SpyPolicy()
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        stats = {"action": {"mean": [0]*8, "std": [1]*8}}
        inference, reconstruction, _ = heldout.evaluate_batches(
            policy, DataLoader(SmallDataset(), batch_size=1), stats, 10, "cpu", 42)
        self.assertEqual(policy.calls, [False, True, False, True])
        for k, value in before.items():
            self.assertTrue(torch.equal(value, policy.state_dict()[k]))
        self.assertIsNone(policy.weight.grad)
        self.assertEqual(inference.summary()["valid_action_steps"], 3)
        self.assertEqual(inference.summary()["normalized_l1"], 1)
        self.assertEqual(reconstruction["valid_action_elements"], 24)
        self.assertEqual(reconstruction["loss"], 1)

    def test_actual_tiny_act_does_not_change_parameters_or_buffers(self):
        model = read_json(ROOT / "configs/act/rh20t_all8.json")
        model.update(hidden_dim=32, dim_feedforward=64, enc_layers=1,
                     dec_layers=1, nheads=4, dropout=0.0)
        data = load_config(ROOT / model["data_config"])
        data.update(image_height=32, image_width=32)
        policy = RH20TACTPolicy(model, data)
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        heldout.evaluate_batches(policy, DataLoader(SmallDataset(), batch_size=2),
                                 {"action": {"mean": [0]*8, "std": [1]*8}}, 10, "cpu", 42)
        self.assertTrue(all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in policy.parameters()))

    def test_physical_units_quaternion_sign_and_mask(self):
        stats = {"action": {"mean": [0,0,0,1,0,0,0,0], "std": [1]*8}}
        metric = heldout.InferenceMetrics(stats)
        prediction, target = np.zeros((1,2,8)), np.zeros((1,2,8))
        prediction[0,0,0] = 0.003
        prediction[0,0,7] = 0.002
        prediction[0,0,3] = -2  # q=-1 and q=+1 are the same orientation.
        target[0,1] = 1000
        metric.add(prediction, target, [[False,True]], ["test-a"])
        report = metric.summary()
        self.assertAlmostEqual(report["position_mean_distance_mm"], 3)
        self.assertAlmostEqual(report["gripper_mae_mm"], 2)
        self.assertAlmostEqual(report["orientation_mean_angle_deg"], 0)
        self.assertEqual(report["valid_action_steps"], 1)

    def test_freeze_refuses_unfinished_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.training.pretrain.load_run", return_value=(
                    {"output_dir": "fixture"}, {}, {"dataset_root": str(root/"raw")}, {}, {}, {})), \
                 patch("src.data.common.read_json", return_value=[{"epoch": 75}]):
                with self.assertRaisesRegex(ValueError, "80 epoch"):
                    heldout.freeze(ROOT, root / "evaluation")
            self.assertFalse((root / "evaluation").exists())

    def test_submission_waits_for_training_and_does_not_cancel_jobs(self):
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            base = home / "cfg5-pretrain"
            base.mkdir()
            (base / "act_bs8_to80.jobid").write_text("123\n")
            with patch.dict(sys.modules, {"fcntl": fake_fcntl}), \
                 patch.object(Path, "home", return_value=home), \
                 patch.object(sys, "argv", ["submit_pretrain_test.py"]), \
                 patch.object(submit.subprocess, "check_output", side_effect=[
                     "123|cfg5-act-bs8-to80|R\n55|T27|R\n", "124\n"]) as command:
                submit.main()
            self.assertIn("--dependency=afterok:123", command.call_args_list[1].args[0])
            self.assertEqual(command.call_args_list[1].args[0][0], "sbatch")
            self.assertEqual((base / "act_pretrain_test.jobid").read_text(), "124\n")

    def test_submission_rejects_unfinished_training_missing_from_queue(self):
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            base = home / "cfg5-pretrain"
            history = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act/outputs/act_pretrain_bs8_to80/history.json"
            history.parent.mkdir(parents=True)
            history.write_text(json.dumps([{"epoch": 75}]))
            (base / "act_bs8_to80.jobid").write_text("123\n")
            with patch.dict(sys.modules, {"fcntl": fake_fcntl}), \
                 patch.object(Path, "home", return_value=home), \
                 patch.object(sys, "argv", ["submit_pretrain_test.py"]), \
                 patch.object(submit.subprocess, "check_output", return_value="") as command:
                with self.assertRaisesRegex(ValueError, "80 epoch"):
                    submit.main()
            self.assertEqual(command.call_count, 1)

    def test_submission_avoids_duplicate_test_job(self):
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "cfg5-pretrain").mkdir()
            with patch.dict(sys.modules, {"fcntl": fake_fcntl}), \
                 patch.object(Path, "home", return_value=home), \
                 patch.object(sys, "argv", ["submit_pretrain_test.py"]), \
                 patch.object(submit.subprocess, "check_output", return_value="124|cfg5-act-test|PD\n") as command:
                submit.main()
            self.assertEqual(command.call_count, 1)

    def test_early_selection_uses_best_saved_epoch_not_latest(self):
        history = [{"epoch": 57, "val": {"loss": .197}}, {"epoch": 75, "val": {"loss": .206}}]
        saved = {"epoch": 57, "val": {"loss": .197}}
        winner, completed = heldout.selected_record(history, saved, True)
        self.assertEqual(winner["epoch"] + 1, 58)
        self.assertEqual(completed, 76)
        with self.assertRaisesRegex(ValueError, "80 epoch"):
            heldout.selected_record(history, saved)

    def test_early_selection_handles_atomic_best_before_history(self):
        history = [{"epoch": 75, "val": {"loss": .197}}]
        winner, completed = heldout.selected_record(history, {"epoch": 76, "val": {"loss": .195}}, True)
        self.assertEqual(completed, 77)
        self.assertEqual(winner["val"]["loss"], .195)
        with self.assertRaises(ValueError):
            heldout.selected_record(history, {"epoch": 79, "val": {"loss": .195}}, True)

    def test_stop_cancels_only_parent_and_its_pending_test(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "outputs/act_pretrain_bs8_to80"
            output.mkdir(parents=True)
            (output / "history.json").write_text(json.dumps([{"epoch": 75}]))
            (output / "best.pt").write_bytes(b"fixture")
            jobs = [["123", "cfg5-act-bs8-to80", "R", "(null)"],
                    ["124", "cfg5-act-test", "PD", "afterok:123(unfulfilled)"],
                    ["55", "T27", "R", "(null)"]]
            with patch.object(submit.subprocess, "run") as cancel, \
                 patch.object(submit, "wait_gone") as wait:
                submit.stop_source(repo, repo, "123", jobs)
            calls = [call.args[0] for call in cancel.call_args_list]
            self.assertEqual([call[-1] for call in calls], ["124", "123"])
            self.assertIn("--state=PENDING", calls[0])
            self.assertIn("--name=cfg5-act-bs8-to80", calls[1])
            self.assertEqual(wait.call_count, 2)
            with patch.object(submit.subprocess, "run") as cancel:
                with self.assertRaises(ValueError):
                    submit.stop_source(repo, repo, "55", jobs)
                cancel.assert_not_called()

    def test_stop_refuses_unrelated_pending_test_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "outputs/act_pretrain_bs8_to80"
            output.mkdir(parents=True)
            (output / "history.json").write_text(json.dumps([{"epoch": 75}]))
            (output / "best.pt").write_bytes(b"fixture")
            jobs = [["123", "cfg5-act-bs8-to80", "R", "(null)"],
                    ["124", "cfg5-act-test", "PD", "afterok:999(unfulfilled)"]]
            with patch.object(submit.subprocess, "run") as cancel:
                with self.assertRaises(ValueError):
                    submit.stop_source(repo, repo, "123", jobs)
                cancel.assert_not_called()

    def test_stop_now_submits_without_afterok_and_sets_evaluation_mode(self):
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            base = home / "cfg5-pretrain"
            output = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act/outputs/act_pretrain_bs8_to80"
            output.mkdir(parents=True)
            (output / "history.json").write_text(json.dumps([{"epoch": 75}]))
            (output / "best.pt").write_bytes(b"fixture")
            (base / "act_bs8_to80.jobid").write_text("123\n")
            with patch.dict(sys.modules, {"fcntl": fake_fcntl}), \
                 patch.object(Path, "home", return_value=home), \
                 patch.object(sys, "argv", ["submit_pretrain_test.py", "--stop-now"]), \
                 patch.object(submit.subprocess, "run") as cancel, \
                 patch.object(submit.subprocess, "check_output", side_effect=[
                     "123|cfg5-act-bs8-to80|R|(null)\n", "", "124\n"]) as command:
                submit.main()
            self.assertEqual(cancel.call_args.args[0][-1], "123")
            submitted = command.call_args
            self.assertEqual(submitted.args[0][0], "sbatch")
            self.assertFalse(any("--dependency" in x for x in submitted.args[0]))
            self.assertEqual(submitted.kwargs["env"]["CFG5_TEST_STOPPED_EARLY"], "1")


if __name__ == "__main__":
    unittest.main()
