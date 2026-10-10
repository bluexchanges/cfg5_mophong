"""Small correctness tests: no training dataset scan and no long rollout."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

from src.data.common import ROOT, digest, read_json, write_json
from src.data.feature_cache import (cache_contract, cache_episode, check_episode,
                                   FeatureACTDataset, quantize_features)
from src.data.mp4_dataset import prepare_light
from src.data.sim_demos import inspect_episode, split_records, preprocess_episode
from src.models.mcr_act import MCRACTPolicy, PREPROCESS, make_mcr_backbone, preprocess_mcr
from src.evaluation.transfer_probe import bounded_action, select_episodes, recorded_batches
from tests import test_sim_finetune_data as fixtures


class FakeEncoder:
    def extract_features(self, rgb):
        # Tests cache alignment with image-dependent, signed, nonuniform features.
        means = rgb.float().mean((2, 3, 4))[:, :, None, None, None] / 255
        pattern = torch.linspace(-1, 1, 2048 * 7 * 7).reshape(1, 1, 2048, 7, 7)
        return means * pattern


class TransferMCRTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_mcr_forward_gradients_and_frozen_bn(self):
        data = read_json(ROOT / "configs/data/cfg5_all8.json")
        cfg = read_json(ROOT / "configs/act/rh20t_all8.json")
        cfg.update(backbone="mcr_resnet50", feature_preprocess=PREPROCESS, hidden_dim=32,
                   dim_feedforward=64, enc_layers=1, dec_layers=1, nheads=4, dropout=0., lr_backbone=0.)
        policy = MCRACTPolicy(cfg, data)
        policy.train()
        self.assertFalse(policy.model.backbones.training)
        self.assertTrue(all(not p.requires_grad for p in policy.model.backbones.parameters()))
        state = torch.zeros((1, 8))
        feature = torch.randn((1, 8, 2048, 7, 7))
        action = torch.randn((1, 20, 8))
        pad = torch.zeros((1, 20), dtype=torch.bool)
        loss = policy.forward_features(state, feature, action, pad)
        loss["loss"].backward()
        self.assertIsNotNone(policy.model.input_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in policy.model.backbones.parameters()))
        self.assertEqual(policy.forward_features(state, feature).shape, (1, 20, 8))
        # Cached INT8 path must agree with the live frozen visual path in eval mode.
        policy.config["cache_quantization"] = "symmetric_per_frame_camera_channel_maxabs_v1"
        policy.eval()
        rgb = torch.full((1, 8, 3, 32, 32), 120, dtype=torch.uint8)
        with torch.inference_mode():
            q, scales, _ = quantize_features(policy.extract_features(rgb).numpy())
            cached = torch.from_numpy(q.astype(np.float32) * scales.astype(np.float32))
            torch.testing.assert_close(policy(state, rgb), policy.forward_features(state, cached), atol=2e-6, rtol=2e-5)
        cfg["lr_backbone"] = 1e-6
        full = MCRACTPolicy(cfg, data)
        full.train()
        self.assertTrue(full.model.backbones.training)
        full(torch.zeros((1, 8)), torch.zeros((1, 8, 3, 224, 224)), action, pad)["loss"].backward()
        self.assertIsNotNone(full.model.backbones[0][0].body.conv1.weight.grad)

    def test_real_official_checkpoint_and_preprocess(self):
        path = ROOT / "outputs/backbones/mcr_resnet50.pth"
        if not path.exists():
            self.skipTest("Official checkpoint asset absent; no implicit download")
        net = make_mcr_backbone(path).eval()
        with torch.inference_mode():
            output = net(preprocess_mcr(torch.full((1, 3, 36, 64), 128, dtype=torch.uint8)))
        self.assertEqual(output.shape, (1, 2048))
        self.assertTrue(torch.isfinite(output).all())

    def test_quantization_zero_and_signed_features(self):
        features = np.random.default_rng(3).normal(size=(1, 8, 2048, 7, 7)).astype(np.float32)
        q, scale, error = quantize_features(features)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(scale.dtype, np.float16)
        self.assertLess(error, .03)
        zero, scales, error = quantize_features(np.zeros_like(features))
        self.assertTrue((scales > 0).all())
        self.assertEqual(error, 0)
        self.assertFalse(zero.any())

    def test_cache_targets_jpeg_read_and_reuse(self):
        fixture = fixtures.SimFineTuneTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        cfg = fixture.cfg
        records = [inspect_episode(fixture.episode(i, gaps=True), cfg) for i in range(3)]
        split = split_records(records, cfg, "fixture")
        stats = fixture.stats(split)
        source, output = fixture.root / "light", fixture.root / "features"
        contract = cache_contract(cfg, split, stats, 90)
        for r in records:
            prepare_light(r, cfg, source, split["split_hash"])
            cache_episode(r["episode"], source, output, cfg, split, contract, FakeEncoder(), "cpu", batch_size=2)
            check_episode(output / r["episode"], cfg, split, contract, full=True)
        data = FeatureACTDataset(output, split, stats, cfg, "train", feature_contract=contract)
        jpg = FeatureACTDataset(output, split, stats, cfg, "train", feature_contract=contract, stage="full")
        self.assertEqual(data[0]["images"].shape, (8, 2048, 7, 7))
        self.assertEqual(jpg[0]["images"].shape, (8, 3, 224, 224))
        for i in range(len(data)):
            np.testing.assert_array_equal(data[i]["actions"], jpg[i]["actions"])
            np.testing.assert_array_equal(data[i]["is_pad"], jpg[i]["is_pad"])
        self.assertEqual([int((~data[i]["is_pad"]).sum()) for i in range(4)], [2, 1, 2, 1])
        ep = split["splits"]["train"][0]
        self.assertEqual(cache_episode(ep, source, output, cfg, split, contract, FakeEncoder(), "cpu", resume=True)[1], "SKIP_VERIFIED")
        with self.assertRaises(ValueError):
            FeatureACTDataset(output, split, stats, cfg, "train", feature_contract={**contract, "jpeg_quality": 80})
        meta = read_json(output / ep / "episode.json")
        self.assertIn("quantization_max_absolute_error", meta)
        # Exercise the actual shared train/validation loops with cached samples.
        from torch.utils.data import DataLoader
        from src.training.pretrain import run_epoch, validate_with_fixed_seed
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from scripts.train_mcr_act import CachedPolicy
        model = deepcopy(fixture.model)
        model.update(backbone="mcr_resnet50", feature_preprocess=PREPROCESS, hidden_dim=32,
                     dim_feedforward=64, enc_layers=1, dec_layers=1, nheads=4, dropout=0., lr_backbone=0.)
        policy = MCRACTPolicy(model, cfg)
        wrapped = CachedPolicy(policy)
        before = policy.model.input_proj.weight.detach().clone()
        bn_before = policy.model.backbones[0][0].body.bn1.running_mean.clone()
        result = run_epoch(wrapped, DataLoader(data, batch_size=1), "cpu", optimizer=policy.configure_optimizer(), max_batches=1)
        self.assertEqual(result["samples"], 1)
        self.assertFalse(torch.equal(before, policy.model.input_proj.weight))
        torch.testing.assert_close(bn_before, policy.model.backbones[0][0].body.bn1.running_mean, rtol=0, atol=0)
        validation = validate_with_fixed_seed(wrapped, DataLoader(data, batch_size=1), "cpu", 42, max_batches=1)
        self.assertEqual(validation["samples"], 1)

    def test_selection_locked_val_and_tamper_rejection(self):
        fixture = fixtures.SimFineTuneTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        records = [inspect_episode(fixture.episode(i), fixture.cfg) for i in range(3)]
        manifest = {"episodes": records, "config_hash": digest(fixture.cfg)}
        manifest["manifest_hash"] = digest(manifest)
        split = split_records(records, fixture.cfg, manifest["manifest_hash"])
        selected = select_episodes(split, manifest, "val", 1, 42)
        self.assertEqual([r["episode"] for r in selected], split["splits"]["val"])
        self.assertEqual(selected, select_episodes(split, manifest, "val", 1, 42))
        with self.assertRaises(ValueError):
            select_episodes(split, manifest, "train", 1)
        with self.assertRaises(ValueError):
            select_episodes(split, {**manifest, "episodes": []}, "val", 1)

    def test_recorded_inference_never_uses_target_as_input(self):
        fixture = fixtures.SimFineTuneTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        record = inspect_episode(fixture.episode(1, gaps=True), fixture.cfg)
        stats = fixture.stats({"split_hash": "fixture"})
        batches = list(recorded_batches(record, fixture.cfg, stats, batch_size=3))
        self.assertEqual([len(b["state"]) for b in batches], [3, 1])
        self.assertEqual(int((~batches[0]["is_pad"][0]).sum()), 2)
        self.assertFalse(np.array_equal(batches[0]["state"][0], batches[0]["actions"][0, 0]))

    def test_npy_and_mp4_produce_identical_feature_cache(self):
        fixture = fixtures.SimFineTuneTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        record = inspect_episode(fixture.episode(12), fixture.cfg)
        split = {"split_hash": "fixture"}
        stats = fixture.stats(split)
        contract = cache_contract(fixture.cfg, split, stats, 90)
        light, decoded = fixture.root / "light", fixture.root / "decoded"
        prepare_light(record, fixture.cfg, light, "fixture")
        preprocess_episode(record, fixture.cfg, decoded, "fixture")
        outputs = []
        for source, backend, name in [(light, "mp4_episode_cache_v1", "mp4"), (decoded, "npy", "npy")]:
            target = fixture.root / name
            cache_episode(record["episode"], source, target, fixture.cfg, split, contract, FakeEncoder(), "cpu", backend=backend, batch_size=2)
            check_episode(target / record["episode"], fixture.cfg, split, contract, full=True)
            outputs.append(target / record["episode"])
        for key in ("features", "scales", "state", "action"):
            np.testing.assert_array_equal(np.load(outputs[0] / f"{key}.npy"), np.load(outputs[1] / f"{key}.npy"))

    def test_explicit_clipping_and_quaternion_failure(self):
        cfg = {"workspace_min_m": [.2, -.3, .01], "workspace_max_m": [.7, .3, .6], "gripper_width_range_m": [0, .08]}
        raw = np.array([1., 0, .2, 2, 0, 0, 0, .12])
        action, clipped = bounded_action(raw, cfg)
        self.assertTrue(clipped)
        self.assertEqual(action[0], .7)
        self.assertEqual(action[7], .08)
        self.assertEqual(action[3], 1.)
        with self.assertRaises(ValueError):
            bounded_action(np.r_[raw[:3], [0, 0, 0, 0], raw[7]], cfg)

    def test_real_scene_rollout_video_two_steps(self):
        from src.evaluation.transfer_probe import rollout
        class StayPolicy:
            def __call__(self, state, images):
                return state[:, None, :].repeat(1, 20, 1)
        data = read_json(ROOT / "configs/data/cfg5_all8.json")
        stats = {"state": {"mean": [0.] * 8, "std": [1.] * 8},
                 "action": {"mean": [0.] * 8, "std": [1.] * 8}}
        raw = ROOT.parent / "pick_place_000042"
        if not raw.exists():
            self.skipTest("Local real episode sample absent")
        with tempfile.TemporaryDirectory() as name:
            result = rollout({"path": str(raw)}, StayPolicy(), stats, data, Path(name), "cpu", max_steps=2)
            self.assertEqual(result["steps"], 2)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertTrue((Path(name) / "rollout.mp4").stat().st_size > 0)
            cap = __import__("cv2").VideoCapture(str(Path(name) / "rollout.mp4"))
            self.assertTrue(cap.read()[0])
            cap.release()


if __name__ == "__main__":
    unittest.main()
