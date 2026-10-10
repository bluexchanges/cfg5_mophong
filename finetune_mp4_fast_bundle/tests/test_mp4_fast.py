"""Raw uint8 transfer matches existing normalization; prefetch cannot block on next CPU batch."""
from contextlib import closing, nullcontext
from copy import deepcopy
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.common import ROOT, read_json
from src.training import pretrain
from tests import test_mp4_dataset as fixtures
from tests.test_cuda_prefetch import Samples


class RawSamples(Samples):
    def __getitem__(self, i):
        row = super().__getitem__(i)
        row["images"] = torch.full((8, 3, 16, 16), i, dtype=torch.uint8)
        return row


class Uint8TransferTests(unittest.TestCase):
    def test_numpy_float32_and_torch_uint8_normalization_match(self):
        cfg = read_json(ROOT / "configs/data/cfg5_all8.json")
        rgb = np.random.default_rng(42).integers(0, 256, (2, 8, 5, 7, 3), dtype=np.uint8)
        old = rgb.astype(np.float32) / 255.0
        old = (old - np.asarray(cfg["image_normalization_mean"], np.float32)) / np.asarray(cfg["image_normalization_std"], np.float32)
        raw = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 1, 4, 2, 3)))
        batch = {"images": raw, "state": torch.zeros(2, 8), "actions": torch.zeros(2, 20, 8),
                 "is_pad": torch.zeros(2, 20, dtype=torch.bool)}
        norm = {"mean": cfg["image_normalization_mean"], "std": cfg["image_normalization_std"]}
        actual = pretrain.move_batch(batch, "cpu", image_normalization=norm)
        np.testing.assert_allclose(actual["images"].numpy(), old.transpose(0, 1, 4, 2, 3), atol=1e-6, rtol=1e-6)
        self.assertEqual(actual["images"].dtype, torch.float32)
        self.assertEqual(raw.dtype, torch.uint8)
        self.assertEqual(raw.numel() * raw.element_size() * 4, actual["images"].numel()*actual["images"].element_size())
        with self.assertRaises(ValueError):
            pretrain.move_batch(batch, "cpu")

    def test_mp4_dataset_uint8_sample_matches_original_targets_and_images(self):
        fixture = fixtures.MP4DatasetTests()
        fixture.setUp()
        try:
            old, raw = fixture.base(), fixture.base(images_uint8=True)
            a, b = old[0], raw[0]
            for key in ("state", "actions", "is_pad"):
                np.testing.assert_array_equal(a[key], b[key])
            batch = {key: torch.from_numpy(np.asarray(b[key])).unsqueeze(0)
                     for key in ("images", "state", "actions", "is_pad")}
            cfg = fixture.cfg
            actual = pretrain.move_batch(batch, "cpu", image_normalization={
                "mean": cfg["image_normalization_mean"], "std": cfg["image_normalization_std"]})
            np.testing.assert_allclose(actual["images"][0].numpy(), a["images"], atol=1e-6, rtol=1e-6)
        finally:
            old._cache = raw._cache = None
            del a, b, batch, actual, old, raw
            fixture.tearDown()

    def test_threaded_prefetch_rejects_cpu(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            next(pretrain.device_batches([], "cpu", cuda_prefetch=True, threaded=True))

    def test_slow_next_batch_does_not_delay_current_batch(self):
        release = threading.Event()
        blocked = threading.Event()
        class Tensor:
            def is_pinned(self): return True
            def record_stream(self, stream): pass
        class Event:
            def record(self, stream): pass
            def query(self): return True
            def synchronize(self): pass
        class Stream:
            def synchronize(self): pass
            def wait_event(self, event): pass
        first = {key: Tensor() for key in ("state", "images", "actions", "is_pad")}
        def source():
            yield first
            blocked.set()
            if not release.wait(2):
                raise RuntimeError("Current batch did not return before second CPU fetch")
        fake_stream = Stream()
        with patch.object(torch.cuda, "Stream", return_value=fake_stream), \
             patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
             patch.object(torch.cuda, "current_stream", return_value=fake_stream), \
             patch.object(torch.cuda, "Event", side_effect=Event), \
             patch.object(pretrain, "move_batch", side_effect=lambda batch, *a, **kw: batch):
            iterator = pretrain.threaded_device_batches(source(), "cuda")
            try:
                batch, _ = next(iterator)
                self.assertIs(batch, first)
                self.assertTrue(blocked.wait(1))
            finally:
                release.set()
                iterator.close()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU gate required on server")
class FastCudaTests(unittest.TestCase):
    def test_threaded_uint8_matches_serial_optimizer_and_early_close(self):
        from torch.utils.data import DataLoader
        from tests.test_cuda_prefetch import Policy
        norm = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}
        for limit in (1, 2, 0):
            a = Policy().cuda()
            b = deepcopy(a)
            oa = torch.optim.AdamW(a.parameters(), lr=.01)
            ob = torch.optim.AdamW(b.parameters(), lr=.01)
            kwargs = {"max_batches": limit, "image_normalization": norm}
            ra = pretrain.run_epoch(a, DataLoader(RawSamples(), batch_size=4, pin_memory=True),
                                    "cuda", optimizer=oa, **kwargs)
            rb = pretrain.run_epoch(b, DataLoader(RawSamples(), batch_size=4, pin_memory=True),
                                    "cuda", optimizer=ob, cuda_prefetch=True,
                                    cuda_prefetch_threaded=True, **kwargs)
            for key in ("samples", "batches", "episodes", "valid_action_elements", "loss"):
                self.assertAlmostEqual(ra[key], rb[key], places=6)
            torch.testing.assert_close(a.weight, b.weight)
        with closing(pretrain.device_batches(DataLoader(RawSamples(), batch_size=4, pin_memory=True),
                                              "cuda", cuda_prefetch=True, threaded=True,
                                              image_normalization=norm)) as stream:
            next(stream)
        torch.cuda.synchronize()

    def test_threaded_prefetch_with_two_spawn_workers(self):
        from torch.utils.data import DataLoader
        norm = {"mean": [.485, .456, .406], "std": [.229, .224, .225]}
        loader = DataLoader(RawSamples(), batch_size=4, num_workers=2,
                            multiprocessing_context="spawn", pin_memory=True)
        seen = []
        with closing(pretrain.device_batches(loader, "cuda", cuda_prefetch=True,
                                              threaded=True, image_normalization=norm)) as batches:
            for cpu, gpu in batches:
                expected = pretrain.move_batch(cpu, "cuda", image_normalization=norm)
                torch.testing.assert_close(gpu["images"], expected["images"])
                seen.extend(cpu["episode"])
        self.assertEqual(seen, [str(i) for i in range(11)])


if __name__ == "__main__":
    unittest.main()
