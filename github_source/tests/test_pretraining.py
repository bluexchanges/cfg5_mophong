"""Trainer contract, weighted validation and epoch-boundary resume without CFG5 payload."""

import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.data.common import ROOT, load_config, read_json

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from src.models.rh20t_act import RH20TACTPolicy
    from src.training import pretrain
    from src.training.fork_run import fork_run
else:
    Dataset = object
    nn = type("NoTorch", (), {"Module": object})


class TinyDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.index = [(f"{split}-{i}", 0, 2) for i in range(len(self))]

    def __len__(self):
        return 3 if self.split == "train" else 2

    def __getitem__(self, index):
        mask = torch.ones(20, dtype=torch.bool)
        mask[:index + 1] = False
        return {"state": torch.ones(8), "images": torch.zeros(8, 3, 2, 2),
                "actions": torch.full((20, 8), 0.25), "is_pad": mask,
                "episode": f"{self.split}-{index}"}


class TinyPolicy(nn.Module):
    def __init__(self, _model_config, _data_config):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.config = {"kl_weight": 0.0}

    def configure_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)

    def forward(self, state, images, actions, is_pad):
        valid = ~is_pad
        l1 = (self.weight - actions[valid]).square().mean()
        kl = l1 * 0
        return {"loss": l1, "l1": l1, "kl": kl,
                "valid_action_elements": valid.sum() * 8}


def tiny_loader(_root, _splits, _stats, _data_cfg, split_name, batch_size,
                _num_workers, *, generator=None, allow_partial_smoke=False):
    return DataLoader(TinyDataset(split_name), batch_size=batch_size,
                      shuffle=split_name == "train", generator=generator)


@unittest.skipUnless(TORCH, "Cần requirements-act.txt")
class PretrainingTests(unittest.TestCase):
    def test_reduced_real_act_runs_optimizer_and_validation(self):
        torch.set_num_threads(2)
        model_cfg = deepcopy(read_json(ROOT / "configs/act/rh20t_all8.json"))
        model_cfg.update(hidden_dim=32, dim_feedforward=64, enc_layers=1,
                         dec_layers=1, nheads=4, dropout=0.0)
        data_cfg = load_config(ROOT / model_cfg["data_config"])
        data_cfg.update(image_height=32, image_width=32)
        policy = RH20TACTPolicy(model_cfg, data_cfg)
        before = policy.model.action_head.weight.detach().clone()
        class OneSample(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                mask = torch.ones(20, dtype=torch.bool)
                mask[:3] = False
                return {"state": torch.zeros(8), "images": torch.zeros(8, 3, 32, 32),
                        "actions": torch.zeros(20, 8), "is_pad": mask,
                        "episode": "tiny-act"}
        loader = DataLoader(OneSample(), batch_size=1)
        trained = pretrain.run_epoch(policy, loader, torch.device("cpu"),
                                     optimizer=policy.configure_optimizer())
        validated = pretrain.validate_with_fixed_seed(policy, loader, torch.device("cpu"), 42)
        self.assertEqual(trained["valid_action_elements"], 24)
        self.assertEqual(validated["samples"], 1)
        self.assertFalse(torch.equal(before, policy.model.action_head.weight))

    def test_episode_sampler_covers_each_sample_once_with_locality(self):
        dataset = type("IndexOnly", (), {"index": [("a", 0, 3), ("a", 1, 3),
                                                  ("b", 0, 2), ("b", 1, 2)]})()
        sampler = pretrain.EpisodeShuffleSampler(dataset, torch.Generator().manual_seed(3))
        order = list(sampler)
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertIn(order[:2], ([0, 1], [1, 0], [2, 3], [3, 2]))

    def test_workers_prefetch_preserves_samples_across_epochs(self):
        with patch.object(pretrain, "ACTDataset", side_effect=lambda *a: TinyDataset(a[4])):
            serial = pretrain.make_loader("", {}, {}, {}, "train", 2, 0,
                                         generator=torch.Generator().manual_seed(3))
            self.assertFalse(serial.persistent_workers)
            parallel = pretrain.make_loader("", {}, {}, {}, "train", 2, 2,
                                           generator=torch.Generator().manual_seed(3))
        for _ in range(2):
            episodes = [episode for batch in parallel for episode in batch["episode"]]
            self.assertEqual(sorted(episodes), ["train-0", "train-1", "train-2"])
        self.assertTrue(parallel.persistent_workers)
        self.assertEqual(parallel.prefetch_factor, 2)
        self.assertEqual(parallel.multiprocessing_context.get_start_method(), "spawn")

    def test_real_full_config_points_to_all_cfg5_and_locked_artifacts(self):
        run, model, data, splits, stats, contract = pretrain.load_run(
            ROOT / "configs/act/cfg5_all_pretrain.json")
        self.assertEqual(model["data_config"], "configs/data/cfg5_all8.json")
        self.assertEqual(len(splits["splits"]["train"]), 531)
        self.assertEqual(len(splits["splits"]["val"]), 66)
        self.assertEqual(run["locked_membership"], "data/splits/cfg5_all_splits.json")
        self.assertEqual(stats["source_split"], "train")
        self.assertEqual(contract["split_hash"], splits["split_hash"])

    def test_validation_counts_valid_elements_not_average_of_batch_means(self):
        metrics = pretrain.Metrics()
        metrics.add({"l1": torch.tensor(1.0), "kl": torch.tensor(2.0),
                     "valid_action_elements": torch.tensor(8)},
                    {"state": torch.zeros(1, 8), "episode": ["a"]})
        metrics.add({"l1": torch.tensor(3.0), "kl": torch.tensor(4.0),
                     "valid_action_elements": torch.tensor(24)},
                    {"state": torch.zeros(3, 8), "episode": ["b", "c", "d"]})
        result = metrics.summary(10.0)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["valid_action_steps"], 4)
        self.assertAlmostEqual(result["l1"], 2.5)
        self.assertAlmostEqual(result["kl"], 3.5)
        self.assertAlmostEqual(result["loss"], 37.5)

    def test_train_checkpoint_resume_and_contract_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            run = {"output_dir": temp, "processed_root": temp, "seed": 7,
                   "epochs": 1, "batch_size": 1, "num_workers": 0,
                   "save_every_epochs": 1, "log_every_batches": 100,
                   "grad_clip_norm": 1.0}
            contract = {"split_hash": "fixture", "camera_serials": ["fixture"]}
            data_cfg = {"dataset_root": str(Path(temp) / "raw")}
            with patch.object(pretrain, "RH20TACTPolicy", TinyPolicy), \
                 patch.object(pretrain, "make_loader", tiny_loader):
                first = pretrain.train(run, {}, data_cfg, {}, {"smoke_only": False},
                                       contract, device=torch.device("cpu"))
                latest = Path(temp) / "latest.pt"
                best = Path(temp) / "best.pt"
                self.assertTrue(latest.exists())
                self.assertTrue(best.exists())
                self.assertEqual(first[0]["train"]["batches"], 3)
                self.assertEqual(first[0]["val"]["samples"], 2)
                saved = torch.load(latest, weights_only=True)
                self.assertEqual(saved["global_step"], 3)
                self.assertEqual(saved["epoch"], 0)
                self.assertIn("normalization", saved["metadata"])
                self.assertIn("versions", saved["metadata"])
                run["epochs"] = 2
                second = pretrain.train(run, {}, data_cfg, {}, {"smoke_only": False},
                                        contract, device=torch.device("cpu"), resume=latest)
                self.assertEqual(second[-1]["epoch"], 1)
                self.assertEqual(torch.load(latest, weights_only=True)["global_step"], 6)
                self.assertEqual(len(pretrain.read_json(Path(temp) / "history.json")), 2)
                with self.assertRaisesRegex(ValueError, "Checkpoint không khớp"):
                    pretrain.check_contract(saved, {"split_hash": "different"})

    def test_batch_four_fork_preserves_state_and_strict_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            parent, child = Path(temp) / "parent", Path(temp) / "child"
            run = {"output_dir": str(parent), "processed_root": temp, "seed": 7,
                   "epochs": 1, "batch_size": 1, "num_workers": 0,
                   "save_every_epochs": 1, "log_every_batches": 100,
                   "grad_clip_norm": 1.0}
            contract = {"split_hash": "fixture", "camera_serials": ["fixture"]}
            data = {"dataset_root": str(Path(temp) / "raw")}
            stats = {"smoke_only": False}
            with patch.object(pretrain, "RH20TACTPolicy", TinyPolicy), \
                 patch.object(pretrain, "make_loader", tiny_loader):
                pretrain.train(run, {}, data, {}, stats, contract, device=torch.device("cpu"))
                source = parent / "latest.pt"
                before = torch.load(source, weights_only=True)
                target = dict(run, output_dir=str(child), batch_size=4, num_workers=2, epochs=2)
                with self.assertRaisesRegex(ValueError, "Run config đã đổi"):
                    pretrain.train(target, {}, data, {}, stats, contract,
                                   device=torch.device("cpu"), resume=source)
                # The failed resume created only an empty output directory.
                with self.assertRaisesRegex(ValueError, "Fork chỉ cho đổi"):
                    fork_run(dict(target, seed=8), {}, data, {}, stats, contract, source)
                with self.assertRaisesRegex(ValueError, "output_dir mới"):
                    fork_run(dict(target, output_dir=str(parent)), {}, data, {}, stats,
                             contract, source)
                latest = fork_run(target, {}, data, {}, stats, contract, source)
                forked = torch.load(latest, weights_only=True)
                self.assertEqual(forked["epoch"], before["epoch"])
                self.assertEqual(forked["global_step"], before["global_step"])
                self.assertTrue(torch.equal(forked["model"]["weight"], before["model"]["weight"]))
                self.assertTrue(torch.equal(forked["rng"]["torch"], before["rng"]["torch"]))
                self.assertTrue(torch.equal(forked["loader_generator_state"],
                                            before["loader_generator_state"]))
                self.assertEqual(forked["optimizer"]["param_groups"], before["optimizer"]["param_groups"])
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(forked["optimizer"]["state"][0][key],
                                                before["optimizer"]["state"][0][key]))
                self.assertEqual((child / "best.pt").read_bytes(), (parent / "best.pt").read_bytes())
                with self.assertRaises(FileExistsError):
                    fork_run(target, {}, data, {}, stats, contract, source)
                history = pretrain.train(target, {}, data, {}, stats, contract,
                                         device=torch.device("cpu"), resume=latest)
                self.assertEqual([r["batch_size"] for r in history], [1, 4])
                self.assertEqual([r["num_workers"] for r in history], [0, 2])
                self.assertEqual(history[-1]["global_step"], 4)  # 3 old + 1 new batch
                self.assertEqual(history[-1]["train"]["samples"], 3)
                final = torch.load(latest, weights_only=True)
                self.assertEqual(final["metadata"]["fork"], forked["metadata"]["fork"])
                self.assertEqual(final["optimizer"]["state"][0]["step"].item(), 4)
                self.assertEqual(torch.load(source, weights_only=True)["global_step"], 3)
                target["epochs"] = 3
                resumed = pretrain.train(target, {}, data, {}, stats, contract,
                                         device=torch.device("cpu"), resume=latest)
                self.assertEqual(resumed[-1]["global_step"], 5)


if __name__ == "__main__":
    unittest.main()
