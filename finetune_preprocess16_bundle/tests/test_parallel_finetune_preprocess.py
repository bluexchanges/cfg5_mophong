"""Small real video fixtures: preflight, spawn processing, resume, corruption."""
import importlib
from pathlib import Path
import sys
import unittest

from src.data.common import ROOT
from src.data.sim_demos import inspect_episode, split_records
from tests.test_sim_finetune_data import SimFineTuneTests

sys.path.insert(0, str(ROOT / "scripts"))
parallel = importlib.import_module("preprocess_finetune_parallel")


class ParallelPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SimFineTuneTests()
        self.fixture.setUp()
        self.records = [inspect_episode(self.fixture.episode(i), self.fixture.cfg) for i in range(3)]
        self.split = split_records(self.records, self.fixture.cfg, "fixture")
        self.output = self.fixture.root / "processed16"
        self.payloads = [(row, self.fixture.cfg, str(self.output), self.split["split_hash"])
                         for row in self.records[:2]]

    def tearDown(self):
        self.fixture.tearDown()

    def test_spawn_preflight_process_and_verified_resume(self):
        checked = list(parallel.bounded_results(parallel.check_episode, self.payloads, 2))
        self.assertEqual(len(checked), 2)
        self.assertTrue(all(not row["existing"] for row in checked))
        processed = list(parallel.bounded_results(parallel.process_episode, self.payloads, 2))
        self.assertEqual(sum(row["steps"] for row in processed), 8)
        checked = list(parallel.bounded_results(parallel.check_episode, self.payloads, 2))
        self.assertTrue(all(row["existing"] for row in checked))
        # Header corruption must refuse reuse, rather than silently skip an existing folder.
        image_file = self.output / self.records[0]["episode"] / "images.npy"
        image_file.write_bytes(b"broken")
        with self.assertRaises((ValueError, OSError, EOFError)):
            parallel.check_episode(self.payloads[0])

    def test_changed_raw_and_partial_refused_before_decode(self):
        raw = Path(self.records[0]["path"]) / "scene_config.json"
        raw.write_text('{"task":"changed"}')
        with self.assertRaisesRegex(ValueError, "Raw đổi"):
            parallel.check_episode(self.payloads[0])
        partial = self.output / (self.records[1]["episode"] + ".partial")
        partial.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "Episode dở"):
            parallel.check_episode(self.payloads[1])


if __name__ == "__main__":
    unittest.main()
