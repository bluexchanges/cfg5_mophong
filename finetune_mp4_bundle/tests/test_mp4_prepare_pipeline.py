"""Exercise the actual locked-input preparation/check path with two spawned workers."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from src.data.common import ROOT, digest, read_json, write_json
from src.data.sim_demos import file_hash, inspect_episode, split_records
from tests import test_sim_finetune_data as fixtures

sys.path.insert(0, str(ROOT / "scripts"))
from scripts import prepare_mp4_finetune as entry


class MP4PreparePipelineTests(unittest.TestCase):
    def test_locked_split_prepare_check_and_resume_without_images_or_test(self):
        fixture = fixtures.SimFineTuneTests()
        fixture.setUp()
        try:
            root, cfg = fixture.root, fixture.cfg
            records = [inspect_episode(fixture.episode(i), cfg) for i in range(5)]
            manifest = {"schema_version": 1, "config_hash": digest(cfg), "config": cfg,
                        "episodes": records, "rejected": [], "duplicates": []}
            manifest["manifest_hash"] = digest(manifest)
            split = split_records(records, cfg, manifest["manifest_hash"])
            stats = fixture.stats(split)
            checkpoint = root / "outputs/act_pretrain_bs8_to80/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture checkpoint; preparation only")
            checksum = file_hash(checkpoint)
            stats["inherited_from"] = {"checkpoint_sha256": checksum}
            data_path = root / "configs/data/fixture.json"
            model_path = root / "configs/act/fixture_model.json"
            split_path = root / "data/splits/pick_place_finetune_splits.json"
            stats_path = root / "data/splits/pick_place_finetune_normalization.json"
            model = deepcopy(fixture.model)
            model["data_config"] = str(data_path)
            for path, value in ((data_path, cfg), (model_path, model), (split_path, split), (stats_path, stats),
                                (root / "data/splits/pick_place_finetune_manifest.json", manifest)):
                write_json(path, value)
            run = read_json(ROOT / "configs/act/cfg5_all_pretrain.json")
            run.update(model_config=str(model_path), splits=str(split_path), locked_membership=str(split_path),
                       stats=str(stats_path), epochs=20, output_dir="outputs/fixture", processed_root="unused")
            base = root / entry.BASE
            write_json(base, run)
            locked_bytes = {p: p.read_bytes() for p in (data_path, model_path, split_path, stats_path, checkpoint)}
            args = argparse.Namespace(base_config=str(base), workers=2, resume=True)
            with patch.object(entry, "ROOT", root), patch.object(entry, "BEST_SHA", checksum):
                entry.prepare(args)
                entry.check(argparse.Namespace(config=str(root / entry.RUN)))
                entry.prepare(args)
            runtime = read_json(root / entry.RUN)
            self.assertEqual((runtime["batch_size"], runtime["num_workers"], runtime["val_num_workers"]), (8, 16, 2))
            light = root / runtime["processed_root"]
            self.assertFalse(any(light.rglob("images.npy")))
            self.assertEqual({p.name for p in light.iterdir()},
                             set(split["splits"]["train"] + split["splits"]["val"]))
            for path, content in locked_bytes.items():
                self.assertEqual(path.read_bytes(), content)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
