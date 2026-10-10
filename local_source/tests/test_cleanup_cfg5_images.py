import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("cleanup_cfg5", Path(__file__).parents[1] / "scripts/cleanup_cfg5_images.py")
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, counts=(1, 1, 1)):
    source = root / "data/processed/cfg5_all"
    source.mkdir(parents=True)
    cfg = {"dataset_root": "RH20T_cfg5", "representation": "rh20t_aligned_tcp_base_wxyz_gripper_m",
           "camera_serials": [str(i) for i in range(8)], "image_height": 1, "image_width": 2}
    names = [f"task_{i:04d}_user_0001_scene_0001_cfg_0005" for i in range(sum(counts))]
    a, b, c = counts
    split = {"split_hash": "locked", "config_hash": cleanup.digest(cfg),
             "splits": {"train": names[:a], "val": names[a:a+b], "test": names[a+b:a+b+c]}}
    for name in names:
        folder = source / name
        folder.mkdir()
        write_json(folder / "episode.json", {"split_hash": "locked", "config_hash": cleanup.digest(cfg),
                   "camera_serials": cfg["camera_serials"], "smoke_only": False, "steps": 1})
        np.save(folder / "images.npy", np.zeros((1, 8, 1, 2, 3), dtype=np.uint8), allow_pickle=False)
        for key in ("state", "action", "timestamps_ms", "segment_ids"):
            np.save(folder / f"{key}.npy", np.zeros((1,), dtype=np.float32), allow_pickle=False)
    return source, cfg, split


class CleanupTests(unittest.TestCase):
    def test_delete_only_train_val_images_and_keep_test_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, cfg, split = fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            items, missing = cleanup.collect_images(source, split, cfg, expected=(1, 1, 1))
            self.assertEqual(len(items), 2)
            self.assertEqual(missing, [])
            removed = {i["path"] for i in items}
            freed = cleanup.delete_images(items, source, root / "deleted.jsonl")
            self.assertEqual(freed, sum(i["bytes"] for i in items))
            for name, data in before.items():
                if name in removed:
                    self.assertFalse(Path(name).exists())
                else:
                    self.assertEqual(Path(name).read_bytes(), data)

    def test_overlap_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cfg, split = fixture(Path(tmp).resolve())
            split["splits"]["test"] = split["splits"]["train"][:]
            with self.assertRaises(ValueError):
                cleanup.collect_images(source, split, cfg, expected=(1, 1, 1))
            self.assertEqual(len(list(source.rglob("images.npy"))), 3)

    def test_changed_file_aborts_before_any_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, cfg, split = fixture(root)
            items, _ = cleanup.collect_images(source, split, cfg, expected=(1, 1, 1))
            with Path(items[-1]["path"]).open("ab") as f:
                f.write(b"changed")
            with self.assertRaises(ValueError):
                cleanup.delete_images(items, source, root / "deleted.jsonl")
            self.assertEqual(len(list(source.rglob("images.npy"))), 3)

    def test_outside_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, cfg, split = fixture(root)
            other = root / "new_images.npy"
            other.write_bytes(b"new data")
            with self.assertRaises(ValueError):
                cleanup.delete_images([{"path": str(other), **cleanup.signature(other)}], source, root / "journal")
            self.assertEqual(other.read_bytes(), b"new data")

    def test_other_job_blocks_even_pending_or_completing(self):
        for state in ("PENDING", "COMPLETING", "RUNNING"):
            response = type("Response", (), {"stdout": f"14127|{state}|cfg5-mcr-cache\n14130|RUNNING|cfg5-mcr-cache\n"})()
            with patch.object(cleanup.subprocess, "run", return_value=response):
                with self.assertRaises(RuntimeError):
                    cleanup.check_jobs(["14130"])

    def test_real_cli_audit_then_apply_on_tiny_597_episode_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, cfg, split = fixture(root, (531, 66, 66))
            old = {"processed_root": "data/processed/cfg5_all", "model_config": "configs/model.json",
                   "splits": "data/splits/locked.json", "locked_membership": "data/splits/locked.json"}
            new = {"processed_root": "data/processed/pick_place_finetune_mp4", "sample_backend": "mp4_episode_cache_v1"}
            new_source = root / new["processed_root"]
            new_source.mkdir()
            (new_source / "do_not_delete").write_bytes(b"new numeric")
            write_json(root / "configs/act/cfg5_all_pretrain_bs8_80epoch.json", old)
            write_json(root / "configs/act/pick_place_finetune_mp4_w16.json", new)
            write_json(root / "configs/model.json", {"data_config": "configs/data.json"})
            write_json(root / "configs/data.json", cfg)
            write_json(root / "data/splits/locked.json", split)
            write_json(root / "data/features/pick_place_mcr/context.json", {
                "source_root": str(new_source), "source_config": "configs/act/pick_place_finetune_mp4_w16.json",
                "data_config": {"dataset_root": "downloads/new"}})
            checkpoint = root / "outputs/act_pretrain/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"epoch5 checkpoint preserved")
            base = ["cleanup", "--repo", str(root), "--allow-job", "14130"]
            with patch.object(cleanup, "check_jobs", return_value=[]), patch.object(sys, "argv", base):
                cleanup.main()
            self.assertEqual(len(list(source.rglob("images.npy"))), 663)
            with patch.object(cleanup, "check_jobs", return_value=[]), patch.object(sys, "argv", base + ["--apply"]):
                cleanup.main()
            self.assertEqual(len(list(source.rglob("images.npy"))), 66)
            self.assertEqual(len(list(source.rglob("state.npy"))), 663)
            self.assertEqual(checkpoint.read_bytes(), b"epoch5 checkpoint preserved")
            self.assertEqual((new_source / "do_not_delete").read_bytes(), b"new numeric")
            self.assertEqual(len(list((root / "outputs/data_cleanup").glob("*.deleted.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
