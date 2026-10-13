"""Missing views must never act as fabricated observations or affect ACT predictions."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import torch

from src.data.common import ROOT, digest, read_json, write_json
from src.data.cfg5_masked import AlignedFrames, BACKEND, KEYS, fingerprint, timeline, restore_missing_joints
from src.data.cfg5_masked import APPROVED_MISSING_JOINT_EPISODES, can_exclude_missing_joint
from src.data.feature_cache import JPEGFrames, cache_contract, cache_episode, check_episode
from src.data.sim_demos import file_hash
from src.models.mcr_act import MCRACTPolicy, PREPROCESS


class MaskedMCRTests(unittest.TestCase):
    def test_only_approved_absent_joints_can_be_excluded_with_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {"representation": "panda_joint_positions_rad_gripper_width_m",
                   "joint_repair_root": str(root / "repairs")}
            for name in APPROVED_MISSING_JOINT_EPISODES:
                ep = root / "raw" / name
                ep.mkdir(parents=True)
                self.assertFalse(can_exclude_missing_joint(ep, cfg))
                self.assertTrue(can_exclude_missing_joint(ep, cfg, True))
                self.assertFalse(can_exclude_missing_joint(ep, {**cfg, "representation": "tcp_base_xyz_wxyz_gripper_width_m"}, True))
                repaired = root / "repairs" / name / "joint.npy"
                repaired.parent.mkdir(parents=True)
                repaired.write_bytes(b"existing supplemental joint; must be audited, not excluded")
                self.assertFalse(can_exclude_missing_joint(ep, cfg, True))
                repaired.unlink()
                raw = ep / "transformed/joint.npy"
                raw.parent.mkdir()
                raw.write_bytes(b"existing raw joint; must be audited even when malformed")
                before = raw.read_bytes()
                self.assertFalse(can_exclude_missing_joint(ep, cfg, True))
                self.assertEqual(raw.read_bytes(), before)
            unknown = root / "raw/task_9999_user_0007_scene_0005_cfg_0005"
            unknown.mkdir()
            self.assertFalse(can_exclude_missing_joint(unknown, cfg, True))

    def test_joint_targets_and_cache_keep_rating_zero_missing_seven_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = read_json(ROOT / "configs/data/cfg5_all8.json")
            cfg.update(dataset_root=str(root / "raw"), episode_subdirs=["training_data"], source_type=BACKEND,
                       representation="panda_joint_positions_rad_gripper_width_m", require_all_cameras=False,
                       image_width=32, image_height=24, min_rating=None)
            name = "task_0001_user_0001_scene_0001_cfg_0005"
            ep = root / "raw/training_data/task_0001" / name
            (ep / "transformed").mkdir(parents=True)
            write_json(ep / "metadata.json", {"rating": 0, "finish_time": -1, "calib": "fixture"})
            serial = cfg["state_camera_serial"]
            times = np.arange(1000, 1400, 100, dtype=np.int64)
            np.save(ep / "transformed/tcp_base.npy", {serial: [{"timestamp": int(t), "tcp": None} for t in times]})
            joints = {int(t): np.r_[np.arange(7)*.1 + i*.01, np.zeros(7)] for i, t in enumerate(times)}
            np.save(ep / "transformed/joint.npy", {serial: joints})
            np.save(ep / "transformed/gripper.npy", {serial: {int(t): {"gripper_info": [30.], "gripper_command": [40.]} for t in times}})
            cam = ep / f"cam_{serial}"; cam.mkdir()
            np.save(cam / "timestamps.npy", {"color": times})
            writer = cv2.VideoWriter(str(cam / "color.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            for i in range(4): writer.write(np.full((24, 32, 3), 80+i*20, np.uint8))
            writer.release()
            arrays, quality = timeline(ep, cfg, None)
            self.assertEqual(quality["rating"], 0)
            self.assertFalse(quality["rating_filter_applied"])
            self.assertEqual(quality["finish_time_source"], "recorded_state_end")
            self.assertEqual(arrays["camera_valid"].sum(1).tolist(), [1]*4)
            np.testing.assert_allclose(arrays["state"][0, :7], joints[1000][:7])
            self.assertAlmostEqual(arrays["action"][0, 7], .04)
            source, cache = root / "light", root / "cache"
            folder = source / name; folder.mkdir(parents=True)
            for key in KEYS: np.save(folder / f"{key}.npy", arrays[key], allow_pickle=False)
            split = {"split_hash": "fixture"}
            meta = {"episode": name, "steps": 4, "config_hash": digest(cfg), "split_hash": "fixture", "smoke_only": False,
                    "camera_serials": cfg["camera_serials"], "raw_path": str(ep.resolve()), "quality": quality,
                    "source_fingerprint": fingerprint(ep, cfg), "image_storage": BACKEND,
                    "numeric_sha256": {k: file_hash(folder / f"{k}.npy") for k in KEYS}}
            write_json(folder / "episode.json", meta)
            class Encoder:
                def extract_features(self, rgb, camera_valid):
                    mean = rgb.float().mean((2, 3, 4)) / 255
                    mean = mean.masked_fill(~camera_valid, 0)
                    return mean[..., None, None, None].expand(-1, -1, 2048, 7, 7)
            contract = cache_contract(cfg, split, {}, 90)
            cache_episode(name, source, cache, cfg, split, contract, Encoder(), "cpu", backend=BACKEND, batch_size=2)
            check_episode(cache / name, cfg, split, contract, full=True)
            with closing(sqlite3.connect(cache / name / "rgb.sqlite")) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM rgb").fetchone()[0], 4)
            features = np.load(cache / name / "features.npy")
            self.assertFalse(features[~arrays["camera_valid"]].any())
            # Missing raw joint file is restored from the original archive into a
            # separate derived folder. Raw remains untouched; targets stay identical.
            joint_path = ep / "transformed/joint.npy"
            data = joint_path.read_bytes()
            archive = root / "simulation.zip"
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr(f"zip/RH20T_cfg5/training_data/task_0001/{name}/transformed/joint.npy", data)
            joint_path.unlink()
            cfg["joint_repair_root"] = str(root / "repairs")
            repaired = restore_missing_joints([ep], cfg, archive)
            self.assertEqual(repaired["restored"], [name])
            self.assertFalse(joint_path.exists())
            restored, _ = timeline(ep, cfg, None)
            np.testing.assert_array_equal(restored['state'], arrays['state'])
            np.testing.assert_array_equal(restored['action'], arrays['action'])

    def test_causal_decode_repeated_skipped_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "color.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            for i in range(7):
                writer.write(np.full((24, 32, 3), i*30, np.uint8))
            writer.release()
            frames = AlignedFrames(path, np.array([0, 0, -1, 4, 6]), {"image_width": 32, "image_height": 24})
            try:
                rows = [frames.read()[1] for _ in range(5)]
            finally:
                frames.release()
            np.testing.assert_array_equal(rows[0], rows[1])
            self.assertIsNone(rows[2])
            self.assertAlmostEqual(float(rows[3].mean()), 120, delta=5)
            self.assertAlmostEqual(float(rows[4].mean()), 180, delta=5)

    def test_sparse_jpeg_database_uses_actual_camera_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.sqlite"
            valid = np.zeros((1, 8), bool); valid[0, [2, 7]] = True
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE rgb(tick INTEGER,camera INTEGER,jpeg BLOB,PRIMARY KEY(tick,camera))")
                for camera in (2, 7):
                    ok, blob = cv2.imencode('.jpg', np.full((24, 32, 3), camera*20, np.uint8))
                    self.assertTrue(ok)
                    db.execute("INSERT INTO rgb VALUES(?,?,?)", (0, camera, blob.tobytes()))
                db.commit()
            reader = JPEGFrames(path, valid, (24, 32, 3))
            rgb = reader[0]
            self.assertEqual(rgb.shape, (8, 24, 32, 3))
            self.assertEqual(int(rgb[2].mean()), 40)
            self.assertEqual(int(rgb[7].mean()), 140)
            self.assertFalse(rgb[~valid[0]].any())
            reader.db.close()

    def test_missing_camera_features_cannot_change_prediction_or_receive_gradient(self):
        torch.set_num_threads(2)
        cfg = read_json(ROOT / "configs/act/rh20t_all8.json")
        data = read_json(ROOT / "configs/data/cfg5_all8.json")
        cfg.update(backbone="mcr_resnet50", feature_preprocess=PREPROCESS, hidden_dim=32,
                   dim_feedforward=64, enc_layers=1, dec_layers=1, nheads=4, dropout=0., lr_backbone=0.)
        policy = MCRACTPolicy(cfg, data).eval()
        valid = torch.tensor([[True, False, True, False, False, False, False, False],
                              [False, False, False, False, False, False, False, True]])
        state = torch.zeros((2, 8))
        features = torch.randn((2, 8, 2048, 7, 7), requires_grad=True)
        prediction = policy.forward_features(state, features, camera_valid=valid)
        changed = features.detach().clone()
        changed[~valid] = torch.randn_like(changed[~valid]) * 100
        alternate = policy.forward_features(state, changed, camera_valid=valid)
        torch.testing.assert_close(prediction, alternate, atol=1e-6, rtol=1e-5)
        prediction.square().mean().backward()
        self.assertEqual(float(features.grad[~valid].abs().max()), 0.)
        self.assertGreater(float(features.grad[valid].abs().max()), 0.)
        calls = []
        handle = policy.model.backbones[0][0].register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
        try:
            with torch.inference_mode():
                extracted = policy.extract_features(torch.full((2, 8, 3, 24, 32), 100, dtype=torch.uint8), camera_valid=valid)
        finally:
            handle.remove()
        self.assertEqual(calls, [3])
        self.assertFalse(extracted[~valid].any())


if __name__ == "__main__":
    unittest.main()
