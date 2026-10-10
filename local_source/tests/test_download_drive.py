"""Fixtures nhỏ kiểm wrapper; không đăng nhập/tải Google Drive thật."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_drive.py"
spec = importlib.util.spec_from_file_location("download_drive", SCRIPT)
drive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drive)


class FakeProcess:
    def __init__(self, code):
        self.returncode = code
        self.pid = 12345

    def poll(self):
        return self.returncode


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = dict(remote="mydrive:", folder_id=drive.DEFAULT_FOLDER,
                        dest=str(self.root / "data có dấu"), workers=4,
                        retry_seconds=1, max_attempts=3)

    def tearDown(self):
        self.temp.cleanup()

    def run_codes(self, codes, verify=False):
        calls = []
        def spawn(command, **kwargs):
            calls.append(command)
            return FakeProcess(next(codes))
        with patch.object(drive, "rclone_executable", return_value=["rclone"]), \
             patch.object(drive.subprocess, "Popen", side_effect=spawn), \
             patch.object(drive.threading.Event, "wait", return_value=False):
            rc = drive.run_worker(self.cfg, self.root, "fixture", verify_only=verify)
        return rc, calls, json.loads((self.root / "state.json").read_text())

    def test_retry_network_keeps_existing_data_and_checks_before_complete(self):
        data = Path(self.cfg["dest"])
        data.mkdir()
        original = data / "already_complete.bin"
        original.write_bytes(b"fixture complete file")
        rc, calls, state = self.run_codes(iter([5, 0, 0]))
        self.assertEqual(rc, 0)
        self.assertEqual([c[1] for c in calls], ["copy", "copy", "check"])
        self.assertEqual(original.read_bytes(), b"fixture complete file")
        self.assertEqual(state["status"], "complete")
        self.assertIn("verified_at", state)
        self.assertIn("--checksum", calls[0])
        self.assertEqual(calls[0][calls[0].index("--transfers") + 1], "4")
        self.assertNotIn("--ignore-existing", calls[0])

    def test_check_failure_repairs_with_copy_then_rechecks(self):
        rc, calls, state = self.run_codes(iter([0, 1, 0, 0]))
        self.assertEqual([c[1] for c in calls], ["copy", "check", "copy", "check"])
        self.assertEqual(state["attempt"], 2)
        self.assertEqual(rc, 0)

    def test_fatal_auth_error_never_marks_complete(self):
        rc, calls, state = self.run_codes(iter([7]))
        self.assertEqual(rc, 7)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("verified_at", state)

    def test_verify_never_copies_or_repairs_files(self):
        rc, calls, state = self.run_codes(iter([1]), verify=True)
        self.assertEqual(rc, 1)
        self.assertEqual([c[1] for c in calls], ["check"])
        self.assertIn("--one-way", calls[0])
        self.assertEqual(state["status"], "failed")

    def test_attempt_limit_ends_unknown_errors(self):
        rc, calls, state = self.run_codes(iter([1, 1, 1]))
        self.assertEqual(rc, 1)
        self.assertEqual(state["attempt"], 3)
        self.assertEqual(state["status"], "failed")

    def test_url_id_and_dataset_binding(self):
        url = "https://drive.google.com/drive/folders/" + drive.DEFAULT_FOLDER + "?fbclid=test"
        self.assertEqual(drive.folder_id(url), drive.DEFAULT_FOLDER)
        with self.assertRaises(ValueError):
            drive.folder_id("https://example.com/drive/folders/test")
        with self.assertRaises(ValueError):
            drive.folder_id("https://drive.google.com/file/d/test/view")
        drive.bind_config(self.root, self.cfg)
        with self.assertRaises(RuntimeError):
            drive.bind_config(self.root, dict(self.cfg, folder_id="different_folder"))

    def test_stop_refuses_reused_pid_and_other_host(self):
        state = dict(pid=123, hostname=drive.socket.gethostname(), token="job_token")
        with patch.object(drive.Path, "stat") as stat, \
             patch.object(drive.Path, "read_bytes", return_value=b"python\0scripts/train.py\0"):
            stat.return_value.st_uid = 1000
            with patch.object(drive.os, "getuid", return_value=1000, create=True):
                self.assertFalse(drive.process_matches(state))
        self.assertFalse(drive.process_matches(dict(state, hostname="another_headnode")))


@unittest.skipUnless(sys.platform == "linux", "Lifecycle/lock integration cần Linux")
class LinuxLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / "provider"
        self.dest = self.root / "destination"
        self.src.mkdir()
        self.dest.mkdir()
        for i in range(24):
            (self.src / f"{i:03d}.bin").write_bytes(bytes([i]) * 1024)
        # Một file tải xong từ lượt trước: restart không được tải/ghi lại.
        self.finished = self.dest / "000.bin"
        self.finished.write_bytes((self.src / "000.bin").read_bytes())
        self.finished_mtime = self.finished.stat().st_mtime_ns
        self.bin = self.root / "bin"
        self.bin.mkdir()
        stub = self.bin / "rclone"
        stub.write_text('''#!/usr/bin/env python3
import hashlib, os, pathlib, sys, time
a = sys.argv[1:]
if a[0] == 'listremotes':
    print('mydrive:')
    sys.exit(0)
src = pathlib.Path(os.environ['FAKE_DRIVE_SOURCE'])
dest = pathlib.Path(a[2])
if a[0] == 'copy':
    time.sleep(0.2)
    for p in sorted(src.iterdir()):
        target = dest / p.name
        if target.exists() and target.read_bytes() == p.read_bytes():
            continue
        time.sleep(0.035)
        temp = target.with_suffix('.partial')
        temp.write_bytes(p.read_bytes())
        temp.replace(target)
    sys.exit(0)
if a[0] == 'check':
    report = pathlib.Path(a[a.index('--combined') + 1])
    rows = []
    ok = True
    for p in sorted(src.iterdir()):
        target = dest / p.name
        same = target.exists() and target.read_bytes() == p.read_bytes()
        rows.append(('= ' if same else '* ') + p.name)
        ok &= same
    report.write_text('\\n'.join(rows))
    sys.exit(0 if ok else 1)
sys.exit(2)
''', encoding="utf-8")
        stub.chmod(0o700)
        self.env = dict(os.environ, HOME=str(self.root), FAKE_DRIVE_SOURCE=str(self.src),
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        self.cfg = dict(remote="mydrive:", folder_id=drive.DEFAULT_FOLDER, dest=str(self.dest),
                        workers=4, retry_seconds=1, max_attempts=3)
        key = drive.hashlib.sha256(str(self.dest).encode()).hexdigest()[:20]
        self.state_root = self.root / "cfg5-pretrain/logs/drive_download" / key

    def command(self, action):
        return subprocess.run([sys.executable, str(SCRIPT), action, "--dest", str(self.dest),
                               "--retry-seconds", "1", "--max-attempts", "3"],
                              env=self.env, capture_output=True, text=True, timeout=15)

    def wait_status(self, choices):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = drive.read_json(self.state_root / "state.json")
            if state.get("status") in choices:
                return state
            time.sleep(0.025)
        self.fail("Worker không đạt trạng thái " + str(choices))

    def tearDown(self):
        state = drive.read_json(self.state_root / "state.json")
        if drive.process_matches(state):
            self.command("stop")
            self.wait_status({"stopped", "complete", "failed"})
        # Đợi lock được worker nhả sau state cuối, không xóa workspace/source thật.
        time.sleep(0.1)
        self.temp.cleanup()

    def test_detached_single_instance_and_verified_completion(self):
        first = self.command("start")
        self.assertEqual(first.returncode, 0, first.stderr)
        state = self.wait_status({"copying"})
        self.assertEqual(os.getsid(state["pid"]), state["pid"])
        duplicate = self.command("start")
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("Đã có tiến trình", duplicate.stderr)
        state = self.wait_status({"complete", "failed"})
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(list(self.dest.glob("*.bin"))), 24)
        self.assertEqual(self.finished.stat().st_mtime_ns, self.finished_mtime)
        self.assertIn("DRIVE_DOWNLOAD_OK", (self.state_root / "download.log").read_text())

    def test_stop_then_restart_keeps_completed_files(self):
        self.assertEqual(self.command("start").returncode, 0)
        self.wait_status({"copying"})
        # Chờ có một file mới hoàn chỉnh rồi dừng giữa copy.
        deadline = time.monotonic() + 5
        while len(list(self.dest.glob("*.bin"))) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        kept = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.dest.glob("*.bin")}
        self.assertEqual(self.command("stop").returncode, 0)
        self.wait_status({"stopped"})
        time.sleep(0.1)
        self.assertEqual(self.command("start").returncode, 0)
        self.assertEqual(self.wait_status({"complete", "failed"})["status"], "complete")
        for name, (payload, mtime) in kept.items():
            p = self.dest / name
            self.assertEqual(p.read_bytes(), payload)
            self.assertEqual(p.stat().st_mtime_ns, mtime)

    def test_unmanaged_rclone_same_destination_is_not_killed(self):
        # argv0 mô phỏng binary rclone, dùng Python làm process fixture nhỏ.
        old = subprocess.Popen(["rclone", "-c", "import time; time.sleep(10)", "copy", str(self.dest)],
                               executable=sys.executable)
        try:
            result = self.command("start")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rclone cũ cùng đích", result.stderr)
            self.assertIsNone(old.poll())
        finally:
            old.terminate()
            old.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
