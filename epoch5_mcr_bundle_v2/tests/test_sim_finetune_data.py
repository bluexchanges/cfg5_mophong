"""Command alignment, source collisions, split leakage and pretrain initialization."""
from copy import deepcopy
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from src.data.act_dataset import ACTDataset
from src.data.common import ROOT, digest, read_json, write_json
from src.data.sim_demos import (SAMPLING, REPRESENTATION, discover_sources, file_hash,
                               inherited_normalization, inspect_episode, load_trajectory,
                               preprocess_episode, read_camera_times, split_records, verify_processed)
from src.training.pretrain import initialize_finetune
from src.training import pretrain


class SimFineTuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.cfg = deepcopy(read_json(ROOT / "configs/data/cfg5_all8.json"))
        self.cfg.update(dataset_root=str(self.raw), raw_sources=[str(self.raw)], image_width=32,
                        image_height=32, source_type="mujoco_scripted_controller",
                        representation=REPRESENTATION, action_target_offset=0, gripper_scale_to_m=1.0)
        self.model = read_json(ROOT / "configs/act/rh20t_all8.json")

    def tearDown(self):
        self.temp.cleanup()

    def episode(self, seed, profile="rich", gaps=False):
        p = self.raw / f"pick_place_{seed:06d}"
        p.mkdir()
        times = np.array([0, 100, 400, 500] if gaps else [0, 100, 200, 300], dtype=np.int64)
        state = np.tile([.4 + seed * .0001, 0, .15, 0, 1, 0, 0, .04], (4, 1)).astype(np.float32)
        action = state.copy()
        action[:, 0] += np.arange(4) * .001 + .01
        action[:, 7] = .001
        np.savez(p / "robot_trajectory.npz", timestamps_ms=times,
                 measured_tcp_gripper=state, commanded_tcp_gripper=action)
        write_json(p / "metadata.json", {"schema_version": 1, "episode": p.name, "seed": seed,
                   "source_type": "mujoco_scripted_controller", "representation": REPRESENTATION,
                   "timestamp_unit": "millisecond", "control_hz": 10, "sampling": SAMPLING,
                   "camera_order": ["cam_" + cam for cam in self.cfg["camera_serials"]],
                   "camera_rig_hash": "rig", "frame_count_each_camera": 4, "success": True,
                   "failure": None, "variation": {"varied": True, "profile": profile}})
        write_json(p / "scene_config.json", {"task": "fixture"})
        for cam in self.cfg["camera_serials"]:
            folder = p / f"cam_{cam}"
            folder.mkdir()
            np.save(folder / "timestamps.npy", {"color": times.tolist()})
            video = cv2.VideoWriter(str(folder / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
            self.assertTrue(video.isOpened())
            for _ in times:
                video.write(np.full((32, 32, 3), [20, 40, 180], np.uint8))
            video.release()
        return p

    def stats(self, split):
        return {"config_hash": digest(self.cfg), "split_hash": split["split_hash"], "source_split": "train",
                "episodes": [], "smoke_only": False,
                "state": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10},
                "action": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10}}

    def test_same_tick_command_padding_last_sample_and_resume(self):
        records = [inspect_episode(self.episode(i), self.cfg) for i in range(3)]
        split = split_records(records, self.cfg, "fixture")
        selected = next(row for row in records if row["episode"] == split["splits"]["train"][0])
        output = self.root / "processed"
        preprocess_episode(selected, self.cfg, output, split["split_hash"])
        verify_processed(selected, self.cfg, output, split["split_hash"])
        dataset = ACTDataset(output, split, self.stats(split), self.cfg)
        self.assertEqual(len(dataset), 4)
        with np.load(Path(selected["path"]) / "robot_trajectory.npz") as trajectory:
            expected = trajectory["commanded_tcp_gripper"]
            np.testing.assert_allclose(dataset[0]["actions"][:4], expected)
            np.testing.assert_allclose(dataset[3]["actions"][0], expected[3])
            self.assertFalse(np.allclose(dataset[0]["actions"][0], trajectory["measured_tcp_gripper"][0]))
        self.assertEqual(int((~dataset[3]["is_pad"]).sum()), 1)
        self.assertEqual(dataset[0]["images"].shape, (8, 3, 32, 32))
        with self.assertRaises(FileExistsError):
            preprocess_episode(selected, self.cfg, output, split["split_hash"])

    def test_chunks_never_cross_gap(self):
        rows = [inspect_episode(self.episode(i, gaps=True), self.cfg) for i in range(3)]
        split = split_records(rows, self.cfg, "fixture")
        row = next(r for r in rows if r["episode"] in split["splits"]["train"])
        out = self.root / "processed"
        preprocess_episode(row, self.cfg, out, split["split_hash"])
        data = ACTDataset(out, split, self.stats(split), self.cfg)
        self.assertEqual([int((~data[i]["is_pad"]).sum()) for i in range(4)], [2, 1, 2, 1])

    def test_split_profiles_seed_groups_and_determinism(self):
        rows = [{"episode": f"demo_{i}", "seed": i // 2, "trajectory_hash": f"traj{i}",
                 "profile": "fixed" if i < 20 else "rich"} for i in range(40)]
        first = split_records(rows, self.cfg, "fixture")
        self.assertEqual(first, split_records(list(reversed(rows)), self.cfg, "fixture"))
        self.assertEqual(first["counts"], {"train": 32, "val": 4, "test": 4})
        membership = {e: name for name, ids in first["splits"].items() for e in ids}
        for i in range(0, 40, 2):
            self.assertEqual(membership[f"demo_{i}"], membership[f"demo_{i+1}"])
        for name in ("train", "val", "test"):
            self.assertEqual(set(first["profiles"][name]), {"fixed", "rich"})

    def test_duplicate_content_skipped_but_conflict_rejected(self):
        p = self.episode(1)
        second = self.root / "second"
        second.mkdir()
        duplicate = second / p.name
        shutil.copytree(p, duplicate)
        accepted, rejected, duplicates = discover_sources([self.raw, second], self.cfg)
        self.assertEqual((len(accepted), len(rejected), len(duplicates)), (1, 0, 1))
        metadata = read_json(duplicate / "metadata.json")
        metadata["extra"] = "changed"
        write_json(duplicate / "metadata.json", metadata)
        with self.assertRaises(RuntimeError):
            discover_sources([self.raw, second], self.cfg)

    def test_one_fixed_group_kept_wholly_in_train_with_913_episodes(self):
        fixed = [{"episode": f"fixed_{i:04d}", "seed": i, "trajectory_hash": "same_fixed_trajectory",
                  "profile": "fixed"} for i in range(200)]
        rich = [{"episode": f"rich_{i:04d}", "seed": i + 1000, "trajectory_hash": f"rich_traj{i}",
                 "profile": "rich"} for i in range(713)]
        result = split_records(fixed + rich, self.cfg, "fixture")
        self.assertEqual(result["counts"], {"train": 731, "val": 91, "test": 91})
        self.assertTrue({row["episode"] for row in fixed}.issubset(result["splits"]["train"]))
        self.assertEqual(result["train_only_profiles"], ["fixed"])
        self.assertEqual(result["group_summary"]["fixed"]["independent_groups"], 1)
        self.assertEqual(set(result["profiles"]["val"]), {"rich"})
        self.assertEqual(set(result["profiles"]["test"]), {"rich"})
        self.assertEqual(result, split_records(list(reversed(fixed + rich)), self.cfg, "fixture"))

    def test_same_seed_fixed_group_and_inexact_remaining_counts(self):
        fixed = [{"episode": f"fixed_{i}", "seed": 42, "trajectory_hash": f"fixed_traj{i}",
                  "profile": "fixed"} for i in range(2)]
        rich = [{"episode": f"rich_{i}", "seed": 100 + i // 2, "trajectory_hash": f"rich_traj{i}",
                 "profile": "rich"} for i in range(10)]
        result = split_records(fixed + rich, self.cfg, "fixture")
        self.assertEqual(sum(result["counts"].values()), 12)
        self.assertTrue(all(result["counts"][name] > 0 for name in ("train", "val", "test")))
        self.assertTrue({r["episode"] for r in fixed}.issubset(result["splits"]["train"]))
        self.assertTrue(any("approximate counts" in note for note in result["assignment_notes"]))
        membership = {ep: name for name, ids in result["splits"].items() for ep in ids}
        for i in range(0, 10, 2):
            self.assertEqual(membership[f"rich_{i}"], membership[f"rich_{i+1}"])

    def test_camera_timestamp_shift_rejected(self):
        p = self.episode(1)
        np.save(p / f"cam_{self.cfg['camera_serials'][0]}/timestamps.npy", {"color": [100, 200, 300, 400]})
        with self.assertRaises(ValueError):
            inspect_episode(p, self.cfg)

    def test_pickle_global_blocked(self):
        class Bad:
            def __reduce__(self):
                return (eval, ("1+2",))
        f = self.root / "bad.npy"
        np.save(f, {"color": Bad()})
        with self.assertRaises(pickle.UnpicklingError):
            read_camera_times(f)

    def test_raw_write_guard_and_changed_source(self):
        p = self.episode(1)
        record = inspect_episode(p, self.cfg)
        with self.assertRaises(ValueError):
            preprocess_episode(record, self.cfg, self.raw / "processed", "split")
        write_json(p / "scene_config.json", {"changed": True})
        with self.assertRaises(ValueError):
            preprocess_episode(record, self.cfg, self.root / "processed", "split")

    def test_init_weights_keep_normalization_reject_changes(self):
        source_cfg = deepcopy(self.cfg)
        source_cfg.update(representation="rh20t_aligned_tcp_base_wxyz_gripper_m", action_target_offset=1)
        source_stats = {"source_split": "train", "smoke_only": False, "episodes": ["source_train"],
                        "split_hash": "source", "state": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10},
                        "action": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10}}
        policy = nn.Linear(2, 2)
        saved = {"kind": "best_model", "smoke": False, "epoch": 57, "contract": {"source": "fixture"},
                 "model": {key: torch.ones_like(value) for key, value in policy.state_dict().items()},
                 "metadata": {"model_config": self.model, "data_config": source_cfg, "normalization": source_stats}}
        stats = inherited_normalization(saved, "hash", self.cfg, {"split_hash": "new"})
        result = initialize_finetune(policy, saved, self.model, self.cfg, stats, "hash")
        self.assertEqual(result["checkpoint_epoch"], 58)
        self.assertTrue(torch.all(policy.weight == 1))
        broken = deepcopy(stats)
        broken["action"]["mean"][0] = .5
        with self.assertRaises(ValueError):
            initialize_finetune(policy, saved, self.model, self.cfg, broken, "hash")
        with self.assertRaises(ValueError):
            initialize_finetune(policy, saved, self.model, self.cfg, stats, "wrong-hash")

    def test_new_run_fresh_optimizer_then_own_checkpoint_resume(self):
        from tests.test_pretraining import TinyPolicy, tiny_loader
        source_cfg = deepcopy(self.cfg)
        source_cfg.update(representation="rh20t_aligned_tcp_base_wxyz_gripper_m", action_target_offset=1)
        source_stats = {"source_split": "train", "smoke_only": False, "episodes": ["source_train"],
                        "split_hash": "old", "state": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10},
                        "action": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10}}
        saved = {"kind": "best_model", "smoke": False, "epoch": 57, "model": TinyPolicy({}, {}).state_dict(),
                 "contract": {"source": "fixture"}, "metadata": {"model_config": {},
                 "data_config": source_cfg, "normalization": source_stats}}
        source = self.root / "source_best.pt"
        torch.save(saved, source)
        checksum = file_hash(source)
        stats = inherited_normalization(saved, checksum, self.cfg, {"split_hash": "new"})
        run = {"output_dir": str(self.root / "finetune"), "processed_root": "unused", "seed": 42,
               "batch_size": 1, "num_workers": 0, "epochs": 1, "save_every_epochs": 1,
               "log_every_batches": 100, "grad_clip_norm": 1.0}
        with patch.object(pretrain, "RH20TACTPolicy", TinyPolicy), patch.object(pretrain, "make_loader", tiny_loader):
            history = pretrain.train(run, {}, self.cfg, {}, stats, {"target": "fixture"},
                                     device="cpu", init_checkpoint=source)
            self.assertEqual(history[0]["epoch"], 0)
            self.assertEqual(history[0]["global_step"], 3)
            latest = Path(run["output_dir"]) / "latest.pt"
            loaded = torch.load(latest, weights_only=True)
            self.assertEqual(loaded["metadata"]["initialization"]["checkpoint_epoch"], 58)
            self.assertEqual(int(next(iter(loaded["optimizer"]["state"].values()))["step"]), 3)
            run["epochs"] = 2
            resumed = pretrain.train(run, {}, self.cfg, {}, stats, {"target": "fixture"},
                                     device="cpu", resume=latest)
            self.assertEqual(resumed[-1]["global_step"], 6)
        self.assertEqual(file_hash(source), checksum)

    def test_cli_prepare_parallel_preprocess_check_and_resume(self):
        repo = self.root / "repo"
        names = ["scripts/prepare_finetune_data.py", "scripts/_bootstrap.py", "src/data/common.py",
                 "src/data/geometry.py", "src/data/splits.py", "src/data/sim_demos.py", "src/data/act_dataset.py"]
        for name in names:
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for i in range(8):
            self.episode(i, profile="fixed" if i < 4 else "rich")
        other = self.root / "other"
        other.mkdir()
        shutil.copytree(self.raw / "pick_place_000000", other / "pick_place_000000")
        source_stats = {"source_split": "train", "smoke_only": False, "episodes": ["source_train"],
                        "split_hash": "old", "state": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10},
                        "action": {"mean": [0.] * 8, "std": [1.] * 8, "count": 10}}
        checkpoint = self.root / "fixture_best.pt"
        torch.save({"kind": "best_model", "smoke": False, "epoch": 57,
                    "metadata": {"model_config": self.model, "data_config": self.cfg,
                                 "normalization": source_stats}}, checkpoint)
        def cli(*args):
            result = subprocess.run([sys.executable, "-X", "utf8", "scripts/prepare_finetune_data.py", *args],
                                    cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout
        output = cli("prepare", "--sources", str(self.raw), str(other), "--init-checkpoint", str(checkpoint),
                     "--expected-sha", file_hash(checkpoint))
        self.assertIn("FINETUNE_SPLIT_LOCKED", output)
        split = read_json(repo / "data/splits/pick_place_finetune_splits.json")
        self.assertEqual(sum(split["counts"].values()), 8)
        output = cli("preprocess", "--workers", "2", "--resume")
        self.assertIn("FINETUNE_PREPROCESS_OK", output)
        output = cli("check")
        self.assertIn("FINETUNE_DATASET_OK", output)
        output = cli("preprocess", "--workers", "2", "--resume")
        self.assertIn("SKIP_VERIFIED", output)
        with self.assertRaises(FileNotFoundError):
            (repo / "data/processed/pick_place_finetune" / split["splits"]["test"][0]).stat()


if __name__ == "__main__":
    unittest.main()
