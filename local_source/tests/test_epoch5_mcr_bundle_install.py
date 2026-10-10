"""Installer integrity and idempotency against the actual built manifest."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from src.data.common import ROOT


spec = importlib.util.spec_from_file_location("epoch5_installer", ROOT / "scripts/install_epoch5_mcr_bundle.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class BundleTests(unittest.TestCase):
    def test_real_bundle_plan_idempotency_custom_refusal_and_escape(self):
        bundle = ROOT.parent / "epoch5_mcr_bundle_v2"
        if not bundle.exists():
            self.skipTest("Build bundle before installer integration test")
        manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(installer.plan_install(bundle, ROOT, manifest), [])
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name)
            for relative in manifest["prerequisites"]:
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            plan = installer.plan_install(bundle, repo, manifest)
            self.assertEqual(len(plan), len(manifest["files"]))
            for source, target in plan:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            self.assertEqual(installer.plan_install(bundle, repo, manifest), [])
            # Old test ran its evaluator from a separate bundle, not inside repo.
            self.assertFalse((repo / "scripts/evaluate_pretrain_test.py").exists())
            self.assertTrue((repo / "src/evaluation/transfer_metrics.py").exists())
            result = subprocess.run([sys.executable, "-I", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); from src.evaluation.transfer_metrics import InferenceMetrics; print('METRIC_IMPORT_OK')",
                str(repo)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("METRIC_IMPORT_OK", result.stdout)
            (repo / "tests/test_sim_finetune_data.py").write_text("older server test fixture", encoding="utf-8")
            self.assertEqual(installer.plan_install(bundle, repo, manifest), [])
            protected = repo / "scripts/probe_epoch5.py"
            protected.write_text("custom server code", encoding="utf-8")
            with self.assertRaises(ValueError):
                installer.plan_install(bundle, repo, manifest)
            self.assertEqual(protected.read_text(), "custom server code")
            with self.assertRaises(ValueError):
                installer.safe_path(repo, "../outside.py")

    def test_missing_runtime_lists_every_path_without_writing(self):
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name)
            manifest = {"prerequisites": ["src/data/common.py", "src/data/mp4_dataset.py"], "files": []}
            with self.assertRaisesRegex(FileNotFoundError, "common.py.*mp4_dataset.py"):
                installer.plan_install(repo, repo, manifest)
            self.assertEqual(list(repo.iterdir()), [])
