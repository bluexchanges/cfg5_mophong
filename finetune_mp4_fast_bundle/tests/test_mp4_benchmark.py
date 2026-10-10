"""Selection must reject failed/OOM/headroom cases, never edit data contract."""
import unittest
from copy import deepcopy
import sys
from src.data.common import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_mp4_fast import select_case


class SelectionTests(unittest.TestCase):
    def test_speed_and_vram_headroom(self):
        base = {"ok": True, "samples_per_s": 7., "peak_reserved_GiB": 20., "device_GiB": 80., "num_workers": 16}
        fast = dict(base, samples_per_s=12., peak_reserved_GiB=50., num_workers=24)
        oom_risk = dict(base, samples_per_s=14., peak_reserved_GiB=70.)
        failed = {"ok": False}
        rows = [base, fast, oom_risk, failed]
        before = deepcopy(rows)
        self.assertIs(select_case(rows), fast)
        self.assertEqual(before, rows)
        with self.assertRaises(RuntimeError):
            select_case([oom_risk, failed])


if __name__ == "__main__":
    unittest.main()
