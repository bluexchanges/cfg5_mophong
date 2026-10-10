"""Padding-weighted gradients, checkpoint migration and real two-CUDA gate."""

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from src.training import pretrain
from src.training.multi_gpu import LossShard, ParallelACT, reduce_shards
from src.training.fork_run import fork_run
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/slurm"))
from continue_bs16_gpu2 import target_config
from submit_bs16_gpu2 import active_act
from status_to80 import durations


class DeterministicPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.config = {"kl_weight": 2.0}

    def forward(self, state, images, actions, is_pad):
        l1 = (self.weight * state[:, 0, None, None] - actions)[~is_pad].abs().mean()
        kl = (state[:, 0] * self.weight).square().mean()
        return {"l1": l1, "kl": kl, "loss": l1 + 2 * kl,
                "valid_action_elements": (~is_pad).sum() * 8}

    def configure_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


def batch(device="cpu"):
    state = torch.arange(1, 4, dtype=torch.float32, device=device)[:, None].expand(3, 8)
    images = torch.zeros(3, 8, 3, 2, 2, device=device)
    actions = torch.ones(3, 4, 8, device=device)
    mask = torch.tensor([[False, True, True, True], [False, False, True, True],
                         [False, False, False, False]], device=device)
    return state, images, actions, mask


class MultiGPUCPUChecks(unittest.TestCase):
    def test_queue_guard_preserves_other_jobs_and_checks_pending_act(self):
        from subprocess import CompletedProcess
        with patch("submit_bs16_gpu2.subprocess.run", return_value=CompletedProcess(
                [], 0, stdout="13936 T27\n13999 cfg5-act-to80\n")):
            self.assertEqual(active_act("user08"), ["13999 cfg5-act-to80"])

    def test_epoch_duration_skips_legacy_records(self):
        rows = [{"epoch": 39, "train": {}, "val": {}},
                {"epoch": 40, "train": {"timing": {"epoch_seconds": 60}},
                 "val": {"timing": {"epoch_seconds": 10}}}]
        self.assertEqual(durations(rows), [(41, 60, 10)])

    def test_uneven_shards_preserve_full_loss_and_gradient(self):
        one, two = DeterministicPolicy(), DeterministicPolicy()
        inputs = batch()
        expected = one(*inputs)
        shards = [LossShard(two)(*[tensor[a:b] for tensor in inputs])
                  for a, b in ((0, 2), (2, 3))]
        reduced = reduce_shards({key: torch.cat([part[key] for part in shards])
                                 for key in shards[0]}, 2.0)
        torch.testing.assert_close(reduced["loss"], expected["loss"])
        self.assertEqual(int(reduced["valid_action_elements"]), 56)
        expected["loss"].backward()
        reduced["loss"].backward()
        torch.testing.assert_close(one.weight.grad, two.weight.grad)

    def test_config_preserves_learning_contract_and_source(self):
        old = {"epochs": 40, "batch_size": 4, "num_workers": 2,
               "output_dir": "parent", "save_every_epochs": 1, "seed": 42,
               "model_config": "fixed", "stats": "train_only"}
        result = target_config(old)
        self.assertEqual(old["batch_size"], 4)
        self.assertEqual((result["epochs"], result["batch_size"], result["num_workers"],
                          result["gpu_count"]), (80, 16, 8, 2))
        for key in ("seed", "model_config", "stats"):
            self.assertEqual(old[key], result[key])

    def test_fork_preserves_optimizer_weights_epoch_and_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            parent, child = temp / "parent", temp / "child"
            parent.mkdir()
            old = {"epochs": 40, "batch_size": 4, "num_workers": 2,
                   "output_dir": str(parent), "save_every_epochs": 1}
            model = DeterministicPolicy()
            optimizer = model.configure_optimizer()
            model(*batch())["loss"].backward()
            optimizer.step()
            contract = {"split_hash": "fixture"}
            saved = {"kind": "train_state", "smoke": False, "epoch": 39,
                     "global_step": 1234, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "contract": contract,
                     "run_config_hash": pretrain.resume_signature(old), "best_val_loss": 0.2,
                     "metadata": {"run_config": old}, "rng": pretrain.capture_rng(),
                     "loader_generator_state": torch.Generator().get_state()}
            torch.save(saved, parent / "latest.pt")
            torch.save({"kind": "best_model", "smoke": False, "epoch": 35,
                        "val": {"loss": 0.2}, "contract": contract,
                        "model": model.state_dict()}, parent / "best.pt")
            (parent / "history.json").write_text(json.dumps([{"epoch": 39, "global_step": 1234}]))
            original = (parent / "latest.pt").read_bytes()
            run = target_config(old)
            run["output_dir"] = str(child)
            source = fork_run(run, {}, {"dataset_root": str(temp / "raw")}, {}, {},
                              contract, parent / "latest.pt")
            result = torch.load(source, weights_only=True)
            self.assertEqual((result["epoch"], result["global_step"]), (39, 1234))
            torch.testing.assert_close(result["model"]["weight"], saved["model"]["weight"])
            torch.testing.assert_close(result["optimizer"]["state"][0]["exp_avg"],
                                       saved["optimizer"]["state"][0]["exp_avg"])
            self.assertEqual((parent / "latest.pt").read_bytes(), original)
            self.assertEqual(result["run_config_hash"], pretrain.resume_signature(run))
            self.assertEqual(result["metadata"]["run_config"]["gpu_count"], 2)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "Cần 2 GPU CUDA")
class TwoGPUChecks(unittest.TestCase):
    def test_backward_update_and_checkpoint_keys_match_single_gpu(self):
        torch.cuda.set_device(0)
        single = DeterministicPolicy().cuda()
        parallel = ParallelACT(deepcopy(single), 2)
        inputs = batch("cuda:0")
        optimizers = [single.configure_optimizer(), parallel.configure_optimizer()]
        for policy, optimizer in zip((single, parallel), optimizers):
            result = policy(*inputs)
            result["loss"].backward()
            optimizer.step()
        self.assertEqual(list(single.state_dict()), list(parallel.state_dict()))
        torch.testing.assert_close(single.weight, parallel.base_policy.weight)
        torch.testing.assert_close(single.weight.grad, parallel.base_policy.weight.grad)
        clone = ParallelACT(DeterministicPolicy().cuda(), 2)
        clone.load_state_dict(parallel.state_dict())
        clone.configure_optimizer().load_state_dict(optimizers[1].state_dict())
        torch.testing.assert_close(clone.base_policy.weight, single.weight)
        # A final batch of one uses a single replica without duplicate samples.
        tail = clone(*[tensor[-1:] for tensor in inputs])
        self.assertEqual(int(tail["valid_action_elements"]), 32)

    def test_reduced_real_act_on_two_gpus(self):
        from src.data.common import ROOT, read_json, load_config
        from src.models.rh20t_act import RH20TACTPolicy
        model = deepcopy(read_json(ROOT / "configs/act/rh20t_all8.json"))
        model.update(hidden_dim=32, dim_feedforward=64, enc_layers=1, dec_layers=1,
                     nheads=4, dropout=0.0)
        data = load_config(ROOT / model["data_config"])
        data.update(image_height=32, image_width=32)
        policy = ParallelACT(RH20TACTPolicy(model, data).cuda(), 2)
        inputs = (torch.zeros(3, 8, device="cuda"), torch.zeros(3, 8, 3, 32, 32, device="cuda"),
                  torch.zeros(3, 20, 8, device="cuda"), torch.zeros(3, 20, dtype=torch.bool, device="cuda"))
        result = policy(*inputs)
        self.assertTrue(torch.isfinite(result["loss"]).item())
        result["loss"].backward()
        policy.configure_optimizer().step()
        self.assertEqual(int(result["valid_action_elements"]), 480)
        for index in range(2):
            torch.cuda.synchronize(index)
            self.assertGreater(torch.cuda.max_memory_allocated(index), 0)


if __name__ == "__main__":
    if os.environ.get("CFG5_REQUIRE_TWO_GPU") == "1" and torch.cuda.device_count() != 2:
        raise RuntimeError("GPU gate cần đúng 2 GPU, không được bỏ qua")
    unittest.main()
