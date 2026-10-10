"""MP4/NPY equality, bounded cache, exact coverage, sharding and resume order."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
    os.environ[_name] = "1"

from copy import deepcopy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.act_dataset import ACTDataset
from src.data.common import digest, read_json, write_json
from src.data.mp4_dataset import (MP4ACTDataset, EpisodeBatchStream, make_mp4_loader,
                                 prepare_light, verify_light)
from src.data.sim_demos import inspect_episode, preprocess_episode, split_records
from tests import test_sim_finetune_data as fixtures


class MP4DatasetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SimFineTuneTests()
        self.fixture.setUp()
        self.cfg = self.fixture.cfg
        self.records = [inspect_episode(self.fixture.episode(i, gaps=True), self.cfg) for i in range(5)]
        self.split = split_records(self.records, self.cfg, "fixture")
        self.stats = self.fixture.stats(self.split)
        self.light = self.fixture.root / "light"
        self.npy = self.fixture.root / "npy"
        for record in self.records:
            prepare_light(record, self.cfg, self.light, self.split["split_hash"])
            preprocess_episode(record, self.cfg, self.npy, self.split["split_hash"])

    def tearDown(self):
        self.fixture.tearDown()

    def base(self, **kwargs):
        return MP4ACTDataset(self.light, self.split, self.stats, self.cfg, **kwargs)

    def test_exact_images_targets_padding_and_cache_pickle(self):
        old = ACTDataset(self.npy, self.split, self.stats, self.cfg)
        new = self.base()
        for i in range(len(old)):
            a, b = old[i], new[i]
            for key in ("images", "state", "actions", "is_pad"):
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual((a["episode"], a["timestamp_ms"]), (b["episode"], b["timestamp_ms"]))
        self.assertEqual(new._cache[1]["images"].dtype, np.uint8)
        self.assertIsNone(pickle.loads(pickle.dumps(new))._cache)
        self.assertFalse(any(self.light.rglob("images.npy")))
        # Gap padding and final sample remain exact through inherited target logic.
        self.assertEqual([int((~new[i]["is_pad"]).sum()) for i in range(4)], [2, 1, 2, 1])

    def test_limits_changed_source_numeric_corruption_and_partial(self):
        with self.assertRaises(MemoryError):
            self.base(cache_gib=1e-8)
        record = self.records[0]
        folder = self.light / record["episode"]
        numeric = folder / "action.npy"
        data = np.load(numeric)
        data[0, 0] += .1
        np.save(numeric, data, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "Numeric file"):
            verify_light(record, self.cfg, self.light, self.split["split_hash"])
        other = self.records[1]
        raw = self.fixture.raw / other["episode"]
        metadata = read_json(raw / "metadata.json")
        metadata["changed"] = True
        write_json(raw / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "Raw thay đổi"):
            verify_light(other, self.cfg, self.light, self.split["split_hash"])
        last = self.records[-1]
        (self.light / (last["episode"] + ".partial")).mkdir()
        with self.assertRaises(FileExistsError):
            verify_light(last, self.cfg, self.light, self.split["split_hash"])

    def test_episode_ownership_no_loss_no_duplication(self):
        stream = EpisodeBatchStream(self.base(), batch_size=3, shuffle=True)
        stream.epoch_seed.fill_(42)
        owners, samples, batches = {}, [], 0
        for worker_id in range(2):
            info = type("Info", (), {"id": worker_id, "num_workers": 2})()
            with patch("src.data.mp4_dataset.get_worker_info", return_value=info):
                for batch in stream:
                    batches += 1
                    for ep, timestamp in zip(batch["episode"], batch["timestamp_ms"].tolist()):
                        self.assertEqual(owners.setdefault(ep, worker_id), worker_id)
                        samples.append((ep, timestamp))
            self.assertIsNone(stream.base._cache)
        expected = [(ep, i * 100 if i < 2 else (i+2)*100) for ep, i, _ in stream.base.index]
        self.assertEqual(sorted(samples), sorted(expected))
        self.assertEqual(len(set(samples)), len(samples))
        self.assertEqual(batches, len(stream))

    def test_spawn_persistent_workers_epoch_order_and_resume(self):
        def make(generator):
            return make_mp4_loader(self.light, self.split, self.stats, self.cfg, "train", 3, 2,
                                   generator=generator, prefetch_factor=1)
        def consume(loader):
            return [(ep, t) for batch in loader for ep, t in
                    zip(batch["episode"], batch["timestamp_ms"].tolist())]
        generator = torch.Generator().manual_seed(123)
        original = make(generator)
        resumed = None
        try:
            first = consume(original)
            saved_rng = generator.get_state()
            second = consume(original)
            third = consume(original)
            resumed_generator = torch.Generator()
            resumed_generator.set_state(saved_rng)
            resumed = make(resumed_generator)
            self.assertEqual(second, consume(resumed))
            self.assertEqual(third, consume(resumed))
            self.assertEqual(len(first), len(original.dataset.base))
            self.assertEqual(len(set(first)), len(first))
            self.assertNotEqual(first, second)
            self.assertTrue(torch.equal(generator.get_state(), resumed_generator.get_state()))
        finally:
            for loader in (original, resumed):
                if loader is not None and loader._iterator is not None:
                    loader._iterator._shutdown_workers()

    def test_real_reduced_act_backward_and_validation_via_dispatch(self):
        from src.models.rh20t_act import RH20TACTPolicy
        from src.training.pretrain import make_loader, run_epoch
        torch.set_num_threads(2)
        model = deepcopy(self.fixture.model)
        model.update(hidden_dim=32, dim_feedforward=64, enc_layers=1,
                     dec_layers=1, nheads=4, dropout=0.0)
        policy = RH20TACTPolicy(model, self.cfg)
        optimizer = policy.configure_optimizer()
        loader = make_loader(self.light, self.split, self.stats, self.cfg, "train", 3, 0,
                             sample_backend="mp4_episode_cache_v1",
                             generator=torch.Generator().manual_seed(1), prefetch_factor=1)
        before = policy.model.action_head.weight.detach().clone()
        train_result = run_epoch(policy, loader, torch.device("cpu"), optimizer=optimizer, max_batches=1)
        self.assertEqual(train_result["samples"], 3)
        self.assertFalse(torch.equal(before, policy.model.action_head.weight))
        val_loader = make_loader(self.light, self.split, self.stats, self.cfg, "val", 3, 0,
                                 sample_backend="mp4_episode_cache_v1", prefetch_factor=1)
        val_result = run_epoch(policy, val_loader, torch.device("cpu"))
        self.assertEqual(val_result["samples"], 4)
        self.assertEqual(val_result["valid_action_steps"], 6)

    def test_decode_failure_does_not_return_partial_sample(self):
        class BrokenVideo:
            def isOpened(self):
                return True
            def read(self):
                return False, None
            def release(self):
                pass
        dataset = self.base()
        with patch("src.data.mp4_dataset.cv2.VideoCapture", return_value=BrokenVideo()):
            with self.assertRaisesRegex(ValueError, "Video lỗi"):
                dataset[0]
        self.assertIsNone(dataset._cache)


if __name__ == "__main__":
    unittest.main()
