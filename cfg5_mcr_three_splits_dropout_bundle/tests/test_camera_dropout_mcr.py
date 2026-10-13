from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

from src.data.common import ROOT, read_json, write_json, digest
from src.data.feature_cache import cache_episode, require_ready, FeatureACTDataset
from src.models.mcr_act import MCRACTPolicy, PREPROCESS, drop_camera_views
from tests.test_staged_mcr import StagedMCRTests
from tests.test_transfer_mcr import FakeEncoder


class CameraDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_only_real_views_drop_keep_one_no_mutation_reproducible(self):
        valid = torch.tensor([[True, False, False, False, False, False, False, False],
                              [False, False, True, True, False, False, False, False]])
        original = valid.clone()
        torch.manual_seed(9)
        first = drop_camera_views(valid, .99)
        torch.manual_seed(9)
        second = drop_camera_views(valid, .99)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(valid, original))
        self.assertTrue(first.any(1).all())
        self.assertFalse((first & ~valid).any())
        self.assertTrue(first[0, 0])
        with self.assertRaises(ValueError):
            drop_camera_views(torch.zeros_like(valid), .1)

    def test_empirical_ten_percent_drop_when_eight_real_views(self):
        torch.manual_seed(11)
        keep = drop_camera_views(torch.ones((10000, 8), dtype=torch.bool), .1)
        dropped = (~keep).float().mean().item()
        self.assertGreater(dropped, .095)
        self.assertLess(dropped, .105)
        self.assertTrue(keep.any(1).all())

    def test_real_act_gradient_and_eval_disable_both_dropouts(self):
        cfg = read_json(ROOT / "configs/data/cfg5_all8.json")
        model = read_json(ROOT / "configs/act/rh20t_all8.json")
        model.update(backbone="mcr_resnet50", feature_preprocess=PREPROCESS,
                     hidden_dim=32, dim_feedforward=64, enc_layers=1, dec_layers=1,
                     nheads=4, dropout=.1, camera_dropout=.1, lr_backbone=0.)
        policy = MCRACTPolicy(model, cfg)
        policy.train()
        state = torch.zeros((1, 8))
        features = torch.randn((1, 8, 2048, 7, 7), requires_grad=True)
        valid = torch.tensor([[False, True, True, False, False, False, False, False]])
        keep = torch.tensor([[False, True, False, False, False, False, False, False]])
        with patch("src.models.mcr_act.drop_camera_views", return_value=keep) as dropout:
            result = policy.forward_features(state, features, torch.zeros((1, 20, 8)),
                                             torch.zeros((1, 20), dtype=torch.bool), camera_valid=valid)
            result["loss"].backward()
            dropout.assert_called_once()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(torch.count_nonzero(features.grad[:, 2]).item(), 0)
        self.assertGreater(torch.count_nonzero(features.grad[:, 1]).item(), 0)
        self.assertIsNotNone(policy.model.input_proj.weight.grad)
        self.assertTrue(all(p.grad is None for p in policy.model.backbones.parameters()))
        self.assertFalse(policy.model.backbones.training)
        policy.eval()
        with patch("src.models.mcr_act.drop_camera_views") as dropout, torch.no_grad():
            first = policy.forward_features(state, features, camera_valid=valid)
            second = policy.forward_features(state, features, camera_valid=valid)
            dropout.assert_not_called()
        torch.testing.assert_close(first, second, rtol=0, atol=0)


class FrozenTestCacheTests(unittest.TestCase):
    def test_test_readiness_separate_and_training_does_not_sample_test(self):
        fixture = StagedMCRTests()
        fixture.setUp()
        self.addCleanup(fixture.fixture.tearDown)
        with self.assertRaises(ValueError):
            require_ready(fixture.root, fixture.ctx, "frozen", split_name="test")
        for name in fixture.split["splits"]["test"]:
            cache_episode(name, fixture.source, fixture.root, fixture.cfg, fixture.split,
                          fixture.contract, FakeEncoder(), "cpu", batch_size=2, save_jpeg=False)
        write_json(fixture.root / "READY_TEST_FROZEN.json", {
            "contract_hash": digest(fixture.contract), "episodes": fixture.split["splits"]["test"],
            "contains_test": True})
        require_ready(fixture.root, fixture.ctx, "frozen", split_name="test")
        train = FeatureACTDataset(fixture.root, fixture.split, fixture.stats, fixture.cfg, "train",
                                  feature_contract=fixture.contract)
        test = FeatureACTDataset(fixture.root, fixture.split, fixture.stats, fixture.cfg, "test",
                                 feature_contract=fixture.contract)
        self.assertTrue(set(train.episodes).isdisjoint(test.episodes))
        self.assertEqual(test.episodes, fixture.split["splits"]["test"])
        self.assertEqual(test[0]["images"].shape, (8, 2048, 7, 7))
        for name in fixture.split["splits"]["test"]:
            self.assertFalse((fixture.root / name / "rgb.sqlite").exists())

    def test_build_cli_includes_test_but_keeps_train_readiness_separate(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import prepare_mcr_cache as cli
        fixture = StagedMCRTests()
        fixture.setUp()
        self.addCleanup(fixture.fixture.tearDown)
        fake = FakeEncoder()
        fake.to = lambda *_: fake
        fake.eval = lambda: fake
        fake.requires_grad_ = lambda *_: fake
        run = {"processed_root": str(fixture.source), "cache_source_backend": "mp4_episode_cache_v1"}
        argv = ["cache", "build", "--cache-root", str(fixture.root), "--features-only", "--include-test",
                "--device", "cpu", "--home-used-gib", "0", "--batch-size", "2", "--decoder-workers", "2"]
        locks = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None)
        with patch.dict(sys.modules, {"fcntl": locks}), patch.object(sys, "argv", argv), \
             patch.object(cli, "context", return_value=(run, {}, fixture.cfg, fixture.split, fixture.stats)), \
             patch.object(cli, "MCRACTPolicy", return_value=fake):
            cli.main()
        train_ready = read_json(fixture.root / "READY_FROZEN.json")
        test_ready = read_json(fixture.root / "READY_TEST_FROZEN.json")
        self.assertEqual(train_ready["episodes"], fixture.names)
        self.assertFalse(train_ready["contains_test"])
        self.assertEqual(test_ready["episodes"], fixture.split["splits"]["test"])
        self.assertTrue(test_ready["contains_test"])
        require_ready(fixture.root, fixture.ctx, "frozen", split_name="test")


if __name__ == "__main__":
    unittest.main()
