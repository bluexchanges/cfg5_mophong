from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_mcr_cache import check_storage


class StoragePolicyTests(unittest.TestCase):
    def test_default_budget_still_blocks_shared_home(self):
        with self.assertRaisesRegex(OSError, "ngân sách"):
            check_storage(1805 * 2**30, 493 * 2**30, int(1.8e12), 49911 * 2**30)

    def test_explicit_override_allows_sufficient_filesystem(self):
        check_storage(1805 * 2**30, 493 * 2**30, int(1.8e12), 49911 * 2**30,
                      skip_home_budget=True)

    def test_override_keeps_disk_reserve_before_episode(self):
        with self.assertRaisesRegex(OSError, "Ổ đĩa thiếu"):
            check_storage(1805 * 2**30, 2 * 2**30, int(1.8e12), 11 * 2**30,
                          skip_home_budget=True)


if __name__ == "__main__":
    unittest.main()
