"""Tiny correctness gate; no dataset/model assets or checkpoint writes.

CFG5_PREFETCH_SOURCE may select a bundled pretrain.py before installation.
"""
import importlib.util
import os
import unittest
from contextlib import closing
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.training import pretrain

if os.environ.get("CFG5_PREFETCH_SOURCE"):
    spec = importlib.util.spec_from_file_location("prefetch_candidate", os.environ["CFG5_PREFETCH_SOURCE"])
    pretrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretrain)


class Samples(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, i):
        return {"state": torch.full((8,), float(i)),
                "images": torch.full((8, 3, 16, 16), float(i)),
                "actions": torch.full((20, 8), i / 10.),
                "is_pad": torch.arange(20) >= i + 1,
                "episode": str(i)}


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.config = {"kl_weight": 0.0}

    def forward(self, state, images, actions, is_pad):
        prediction = self.weight * (state[:, :1] + images.mean((1, 2, 3, 4))[:, None])
        error = (prediction[:, None, :] - actions).square()
        loss = error[~is_pad].mean()
        return {"loss": loss, "l1": loss, "kl": loss * 0,
                "valid_action_elements": (~is_pad).sum() * 8}


class SerialTests(unittest.TestCase):
    def test_cpu_rejects_cuda_prefetch(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            next(pretrain.device_batches([], "cpu", cuda_prefetch=True))

    def test_serial_all_samples_and_partial_batch(self):
        p = Policy()
        result = pretrain.run_epoch(p, DataLoader(Samples(), batch_size=4), "cpu")
        self.assertEqual((result["samples"], result["batches"], result["episodes"]), (11, 3, 11))


@unittest.skipUnless(torch.cuda.is_available(), "GPU CUDA required")
class CudaTests(unittest.TestCase):
    def loader(self):
        return DataLoader(Samples(), batch_size=4, pin_memory=True)

    def test_gpu_batches_preserve_values_and_last_batch(self):
        seen = []
        with closing(pretrain.device_batches(self.loader(), "cuda", cuda_prefetch=True)) as batches:
            for cpu, gpu in batches:
                # Churn allocator while copy/compute streams have outstanding work.
                scratch = torch.empty((1024, 1024), device="cuda")
                for key in gpu:
                    self.assertTrue(torch.equal(cpu[key], gpu[key].cpu()))
                seen.extend(cpu["episode"])
                del scratch
        self.assertEqual(seen, [str(i) for i in range(11)])

    def test_training_matches_serial_and_limits_do_not_overfetch(self):
        for limit in (0, 1, 2):
            a = Policy().cuda()
            b = deepcopy(a)
            oa = torch.optim.AdamW(a.parameters(), lr=0.01)
            ob = torch.optim.AdamW(b.parameters(), lr=0.01)
            ra = pretrain.run_epoch(a, self.loader(), "cuda", optimizer=oa, max_batches=limit)
            rb = pretrain.run_epoch(b, self.loader(), "cuda", optimizer=ob, max_batches=limit,
                                    cuda_prefetch=True)
            for key in ("samples", "batches", "episodes", "valid_action_elements", "loss"):
                self.assertAlmostEqual(ra[key], rb[key], places=6)
            self.assertTrue(torch.equal(a.weight, b.weight))
            self.assertTrue(torch.equal(oa.state[a.weight]["exp_avg"], ob.state[b.weight]["exp_avg"]))

    def test_early_close_and_unpinned_rejection(self):
        with closing(pretrain.device_batches(self.loader(), "cuda", cuda_prefetch=True)) as batches:
            next(batches)
        torch.cuda.synchronize()
        with self.assertRaisesRegex(ValueError, "pin_memory"):
            next(pretrain.device_batches(DataLoader(Samples(), batch_size=4), "cuda", cuda_prefetch=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
