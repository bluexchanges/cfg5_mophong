import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.data.common import ROOT, digest, load_config, read_json, write_json
from src.data.cfg5 import make_timeline, past_indices, preprocess_episode, source_fingerprint, verify_processed_episode
from src.data.act_dataset import ACTDataset, fit_normalization
from src.data.splits import create_splits, validate_splits
from src.data.geometry import continuous_quaternions, pose_matrix
from src.simulation.camera_rig import camera_world_poses


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_config(ROOT / "configs/data/cfg5_base8.json")
        self.cfg.update(dataset_root=str(self.root / "raw"), image_width=16, image_height=12, chunk_size=4, min_segment_steps=2)
        self.ep = self.root / "raw/task_0008_user_0001_scene_0001_cfg_0005"
        (self.ep / "transformed").mkdir(parents=True)
        self.times = np.arange(1000, 1600, 100)
        write_json(self.ep / "metadata.json", {"finish_time": 1500, "rating": 9, "calib": 1})
        tcp, grip = {}, {}
        for serial in self.cfg["camera_serials"]:
            folder = self.ep / f"cam_{serial}"
            folder.mkdir()
            np.save(folder / "timestamps.npy", {"color": self.times.tolist(), "depth": []})
            writer = cv2.VideoWriter(str(folder / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (16, 12))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                frame = np.zeros((12, 16, 3), dtype=np.uint8)
                frame[:, :, 2] = 160+i*10  # BGR đỏ
                writer.write(frame)
            writer.release()
            tcp[serial] = [{"timestamp": int(t), "tcp": np.array([.4+i*.001, 0, .2, 1, 0, 0, 0])} for i, t in enumerate(self.times)]
            grip[serial] = {int(t): {"gripper_info": [30., 0, 0], "gripper_command": [40., 0, 0]} for t in self.times}
        np.save(self.ep / "transformed/tcp_base.npy", tcp)
        np.save(self.ep / "transformed/gripper.npy", grip)

    def test_causal_sampling_no_future(self):
        indices, valid, age = past_indices(np.array([100, 200, 300]), np.array([90, 150, 290, 500]), 100)
        np.testing.assert_array_equal(indices, [0, 0, 1, 2])
        np.testing.assert_array_equal(valid, [False, True, True, False])
        np.testing.assert_array_equal(age, [-10, 50, 90, 200])

    def test_gap_starts_new_segment(self):
        path = self.ep / f"cam_{self.cfg['camera_serials'][0]}/timestamps.npy"
        np.save(path, {"color": [1000, 1100, 1200, 1400, 1500]})
        self.cfg["max_observation_age_ms"] = 50
        timeline = make_timeline(self.ep, self.cfg)
        np.testing.assert_array_equal(timeline["timestamps_ms"], [1000, 1100, 1200, 1400, 1500])
        np.testing.assert_array_equal(timeline["segment_ids"], [0, 0, 0, 1, 1])

    def test_video_count_mismatch_rejected(self):
        path = self.ep / f"cam_{self.cfg['camera_serials'][0]}/timestamps.npy"
        np.save(path, {"color": [1000, 1100, 1200]})
        with self.assertRaisesRegex(ValueError, "count"):
            make_timeline(self.ep, self.cfg, check_video=True)

    def test_quaternion_sign_and_zero(self):
        result = continuous_quaternions(np.array([[1, 0, 0, 0], [-1, 0, 0, 0]]))
        np.testing.assert_allclose(result[0], result[1])
        with self.assertRaises(ValueError):
            pose_matrix([0, 0, 0, 0, 0, 0, 0])

    def make_split(self):
        episodes = [{"episode": self.ep.name, "task": 8, "user": 1, "calib": "1"},
                    {"episode": "heldout_val", "task": 8, "user": 2, "calib": "2"},
                    {"episode": "heldout_test", "task": 8, "user": 3, "calib": "3"}]
        manifest = {"config_hash": digest(self.cfg), "manifest_hash": "test", "episodes": episodes}
        split = create_splits(manifest, self.cfg)
        # Fixture cần đúng episode đã tạo ở train; đây là split synthetic riêng.
        names = ["train", "val", "test"]
        containing = next(n for n in names if self.ep.name in split["splits"][n])
        for field in ("splits", "groups", "counts"):
            split[field]["train"], split[field][containing] = split[field][containing], split[field]["train"]
        split["split_hash"] = digest({k: v for k, v in split.items() if k != "split_hash"})
        return split

    def test_preprocess_future_mask_rgb_and_train_stats(self):
        split = self.make_split()
        out = self.root / "processed"
        record = {"episode": self.ep.name, "source_fingerprint": source_fingerprint(self.ep, self.cfg)}
        meta = preprocess_episode(record, self.cfg, out, split["split_hash"])
        self.assertFalse(meta["smoke_only"])
        stats = fit_normalization(out, split, self.cfg)
        self.assertEqual(stats["source_split"], "train")
        self.assertEqual(stats["episodes"], [self.ep.name])
        self.assertAlmostEqual(stats["state"]["mean"][0], .4025, places=6)
        dataset = ACTDataset(out, split, stats, self.cfg)
        self.assertEqual(len(dataset), 5)
        first, last = dataset[0], dataset[-1]
        physical = first["actions"][0] * stats["action"]["std"] + stats["action"]["mean"]
        self.assertAlmostEqual(physical[0], .401, places=6)
        self.assertAlmostEqual(physical[7], .04, places=6)
        self.assertGreater(first["images"][0, 0].mean(), first["images"][0, 2].mean())
        np.testing.assert_array_equal(last["is_pad"], [False, True, True, True])
        self.assertEqual(first["images"].shape, (8, 3, 12, 16))
        bad_stats = copy.deepcopy(stats)
        bad_stats["episodes"].append("heldout_val")
        with self.assertRaisesRegex(ValueError, "train"):
            ACTDataset(out, split, bad_stats, self.cfg)
        with self.assertRaises(FileExistsError):
            preprocess_episode(record, self.cfg, out, split["split_hash"])
        record["quality"] = meta["quality"]
        self.assertEqual(verify_processed_episode(record, self.cfg, out, split["split_hash"])["steps"], 6)
        saved = read_json(out / self.ep.name / "episode.json")
        saved["split_hash"] = "wrong"
        write_json(out / self.ep.name / "episode.json", saved)
        with self.assertRaisesRegex(ValueError, "contract khác"):
            verify_processed_episode(record, self.cfg, out, split["split_hash"])

    def test_chunks_never_cross_gap(self):
        split = self.make_split()
        out = self.root / "processed"
        record = {"episode": self.ep.name, "source_fingerprint": source_fingerprint(self.ep, self.cfg)}
        preprocess_episode(record, self.cfg, out, split["split_hash"])
        # Tạo hai đoạn tách biệt trong fixture processed, mỗi đoạn ba bước.
        np.save(out / self.ep.name / "segment_ids.npy", np.array([0, 0, 0, 1, 1, 1]))
        stats = fit_normalization(out, split, self.cfg)
        dataset = ACTDataset(out, split, stats, self.cfg)
        self.assertEqual(len(dataset), 4)
        np.testing.assert_array_equal(dataset[0]["is_pad"], [False, False, True, True])
        np.testing.assert_array_equal(dataset[1]["is_pad"], [False, True, True, True])
        self.assertEqual(dataset[2]["timestamp_ms"], 1300)
        self.assertEqual(stats["action"]["count"], 4)

    def test_split_determinism_leakage_and_stats_contract(self):
        split = self.make_split()
        self.assertEqual(split, self.make_split())
        bad = copy.deepcopy(split)
        bad["splits"]["val"].append(self.ep.name)
        with self.assertRaises(ValueError):
            validate_splits(bad)
        bad = copy.deepcopy(split)
        bad["counts"]["train"] += 1
        bad["split_hash"] = digest({k: v for k, v in bad.items() if k != "split_hash"})
        with self.assertRaisesRegex(ValueError, "count"):
            validate_splits(bad)
        with self.assertRaises(ValueError):
            fit_normalization(self.root / "not_processed", split, self.cfg)

    def test_seeded_random_split_has_exact_episode_counts_and_whole_groups(self):
        # 10 episodes, nhóm có kích thước khác nhau; 80/10/10 phải là 8/1/1.
        sizes = [4, 3, 1, 1, 1]
        episodes = [
            {"episode": f"group{group}_scene{scene}", "task": 8, "user": group, "calib": "1"}
            for group, size in enumerate(sizes) for scene in range(size)
        ]
        manifest = {"config_hash": digest(self.cfg), "manifest_hash": "synthetic_exact",
                    "episodes": episodes}
        result = create_splits(manifest, self.cfg)
        self.assertEqual(result["counts"], {"train": 8, "val": 1, "test": 1})
        self.assertEqual(result, create_splits(manifest, self.cfg))
        owner = {episode: split for split, ids in result["splits"].items() for episode in ids}
        for group, size in enumerate(sizes):
            self.assertEqual(len({owner[f"group{group}_scene{scene}"] for scene in range(size)}), 1)

        impossible = [
            {"episode": f"g{group}_{scene}", "task": 8, "user": group, "calib": "1"}
            for group in range(3) for scene in range(4)
        ]
        manifest["episodes"] = impossible
        with self.assertRaisesRegex(ValueError, "Không thể đạt đúng"):
            create_splits(manifest, self.cfg)

    def test_two_wrist_cameras_move_six_fixed_do_not(self):
        rig = {"cameras": [{"name": str(i), "parent_frame": "robot_base" if i < 6 else "rh20t_aligned_tcp",
                            "T_parent_from_camera_mujoco": np.eye(4).tolist()} for i in range(8)]}
        tcp = np.eye(4)
        a = camera_world_poses(rig, np.eye(4), tcp)
        tcp[0, 3] = .1
        b = camera_world_poses(rig, np.eye(4), tcp)
        for i in range(8):
            self.assertAlmostEqual(b[str(i)][0, 3]-a[str(i)][0, 3], 0 if i < 6 else .1)


if __name__ == "__main__":
    unittest.main()
