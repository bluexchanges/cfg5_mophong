"""Bundle checks before writes, backups and idempotent installation on small files."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.data.common import ROOT

spec = importlib.util.spec_from_file_location("install_finetune", ROOT / "scripts/install_finetune_bundle.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo, self.bundle = self.root / "repo", self.root / "bundle"
        self.repo.mkdir()
        self.bundle.mkdir()
        self.rows = []
        for name in ("src/data/common.py", "scripts/train.py", "src/data/sim_demos.py"):
            path = self.bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"new file\n")
            row = {"path": name, "sha256": installer.checksum(path), "normalized_sha256": installer.checksum(path, True)}
            if name != "src/data/sim_demos.py":
                target = self.repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"old file\r\n")
                row["previous_normalized_sha256"] = installer.checksum(target, True)
            self.rows.append(row)
        self.manifest = {"files": self.rows}
        (self.bundle / "bundle_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        shutil.copy2(ROOT / "scripts/install_finetune_bundle.py", self.bundle / "install_bundle.py")

    def tearDown(self):
        self.temp.cleanup()

    def test_install_backups_and_idempotency(self):
        self.assertEqual(len(installer.plan_install(self.bundle, self.repo, self.manifest)), 3)
        result = subprocess.run([sys.executable, "-X", "utf8", str(self.bundle / "install_bundle.py"),
                                 "--repo", str(self.repo)], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FINETUNE_BUNDLE_INSTALLED", result.stdout)
        self.assertEqual(installer.plan_install(self.bundle, self.repo, self.manifest), [])
        self.assertEqual(len(list((self.repo / "outputs/install_backups").rglob("*.py"))), 2)

    def test_custom_version_and_corrupt_bundle_refused_before_any_write(self):
        before = {p: p.read_bytes() for p in self.repo.rglob("*.py")}
        target = self.repo / "scripts/train.py"
        target.write_bytes(b"custom server edit\n")
        with self.assertRaises(ValueError):
            installer.plan_install(self.bundle, self.repo, self.manifest)
        self.assertFalse((self.repo / "src/data/sim_demos.py").exists())
        self.assertEqual((self.repo / "src/data/common.py").read_bytes(), before[self.repo / "src/data/common.py"])
        target.write_bytes(before[target])
        (self.bundle / "scripts/train.py").write_bytes(b"corrupted source\n")
        with self.assertRaises(ValueError):
            installer.plan_install(self.bundle, self.repo, self.manifest)
        self.assertFalse((self.repo / "outputs/install_backups").exists())

    def test_path_escape_refused(self):
        with self.assertRaises(ValueError):
            installer.plan_install(self.bundle, self.repo, {"files": [{"path": "../outside.py"}]})


if __name__ == "__main__":
    unittest.main()
