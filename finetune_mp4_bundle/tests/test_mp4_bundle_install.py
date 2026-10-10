"""Installer must reject changed versions/path escapes before writing any file."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.install_mp4_bundle import checksum, plan_install


class MP4InstallerTests(unittest.TestCase):
    def test_plan_baseline_idempotent_and_custom_edit_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo, bundle = root / "repo", root / "bundle"
            repo.mkdir(); bundle.mkdir()
            (repo / "required.py").write_text("required\n", encoding="utf-8")
            target, source = repo / "dataset.py", bundle / "dataset.py"
            target.write_text("old\n", encoding="utf-8")
            previous = checksum(target, True)
            source.write_text("new\n", encoding="utf-8")
            manifest = {"prerequisites": {"required.py": checksum(repo / "required.py", True)},
                        "files": [{"path": "dataset.py", "sha256": checksum(source),
                                   "normalized_sha256": checksum(source, True),
                                   "previous_normalized_sha256": previous}]}
            self.assertEqual(len(plan_install(bundle, repo, manifest)), 1)
            target.write_bytes(source.read_bytes())
            self.assertEqual(plan_install(bundle, repo, manifest), [])
            target.write_text("custom\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "custom code"):
                plan_install(bundle, repo, manifest)
            self.assertEqual(target.read_text(), "custom\n")
            manifest["files"][0]["path"] = "../escape.py"
            with self.assertRaisesRegex(ValueError, "path"):
                plan_install(bundle, repo, manifest)

    def test_prerequisite_and_bundle_checksum_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo, bundle = root / "repo", root / "bundle"
            repo.mkdir(); bundle.mkdir()
            source = bundle / "add.py"
            source.write_text("add\n", encoding="utf-8")
            manifest = {"prerequisites": {}, "files": [{"path": "add.py", "sha256": checksum(source),
                                                          "normalized_sha256": checksum(source, True)}]}
            self.assertEqual(len(plan_install(bundle, repo, manifest)), 1)
            source.write_text("corrupt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                plan_install(bundle, repo, manifest)
            manifest["prerequisites"] = {"missing.py": "0" * 64}
            with self.assertRaisesRegex(ValueError, "Prerequisite"):
                plan_install(bundle, repo, manifest)
            self.assertFalse((repo / "add.py").exists())


if __name__ == "__main__":
    unittest.main()
