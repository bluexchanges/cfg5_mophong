#!/usr/bin/env python3
"""Tải thư mục Google Drive bằng rclone; Linux, Python stdlib, resume theo file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

DEFAULT_FOLDER = "1vzCuuAVDapO1bBW1ijk4YYlFp4uKoItI"
FATAL_CODES = {2, 3, 4, 7, 8, 9, 10}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def folder_id(value):
    if value.startswith("https://"):
        url = urlparse(value)
        if url.hostname != "drive.google.com":
            raise ValueError("Link phải là thư mục drive.google.com/drive/folders/ID")
        match = re.fullmatch(r"/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]+)/?", url.path)
        if not match:
            raise ValueError("Cần link thư mục Drive, không phải link file")
        value = match.group(1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Folder ID không hợp lệ")
    return value


def config(args):
    remote = args.remote.rstrip(":")
    if not re.fullmatch(r"[A-Za-z0-9_. -]+", remote) or remote.startswith("-"):
        raise ValueError("Remote phải là tên đã tạo bằng rclone config")
    dest = Path(args.dest).expanduser().resolve()
    if dest == Path(dest.anchor) or dest == Path.home().resolve():
        raise ValueError("Chọn thư mục dữ liệu riêng, không dùng / hoặc toàn bộ HOME")
    if args.workers < 1 or args.retry_seconds < 1 or args.max_attempts < 0:
        raise ValueError("workers/retry-seconds phải dương; max-attempts >= 0")
    return dict(remote=remote + ":", folder_id=folder_id(args.folder), dest=str(dest),
                workers=args.workers, retry_seconds=args.retry_seconds,
                max_attempts=args.max_attempts)


def state_dir(cfg):
    key = hashlib.sha256(cfg["dest"].encode()).hexdigest()[:20]
    return Path.home() / "cfg5-pretrain/logs/drive_download" / key


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def acquire_lock(root):
    if sys.platform != "linux":
        raise RuntimeError("Lệnh chạy nền dùng trên server Linux; không chạy tải trên Windows")
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(root / "download.lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RuntimeError("Đã có tiến trình tải/kiểm tra đích này. Dùng status, không start lần hai")
    return fd


def bind_config(root, cfg):
    old = read_json(root / "config.json")
    if old and any(old[k] != cfg[k] for k in ("remote", "folder_id", "dest")):
        raise RuntimeError("Đích này đã gắn với nguồn Drive khác; chọn --dest mới")
    atomic_json(root / "config.json", cfg)


def rclone_executable():
    exe = shutil.which("rclone")
    if not exe:
        raise RuntimeError("Chưa có rclone trong PATH")
    return [exe]


def preflight(cfg):
    result = subprocess.run(rclone_executable() + ["listremotes"], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30)
    if result.returncode or cfg["remote"] not in result.stdout.splitlines():
        raise RuntimeError("Không đọc được remote " + cfg["remote"] + "; chạy rclone listremotes/config trên server")


def unmanaged_downloads(cfg):
    """Chặn rclone cũ chạy ngoài script cùng đích; không tự hủy tiến trình."""
    found = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return found
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if not argv or Path(argv[0]).name != "rclone" or "copy" not in argv:
                continue
            # Cùng thư mục đích, kể cả nguồn Drive khác, cũng không cho ghi song song.
            absolute = [str(Path(a).resolve()) for a in argv[1:] if a.startswith("/")]
            if cfg["dest"] in absolute:
                found.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return found


def command(cfg, mode, root):
    common = ["--drive-root-folder-id", cfg["folder_id"], "--checkers", "8",
              "--contimeout", "30s", "--timeout", "5m", "--retries", "3",
              "--low-level-retries", "10", "--retries-sleep", "10s",
              "--stats", "30s", "--stats-log-level", "INFO", "--log-level", "INFO"]
    base = rclone_executable() + [mode, cfg["remote"], cfg["dest"]]
    if mode == "copy":
        extra = ["--transfers", str(cfg["workers"]), "--checksum",
                 "--multi-thread-streams", "0", "--create-empty-src-dirs"]
    else:
        extra = ["--one-way", "--combined", str(root / "check_files.txt")]
    return base + common + extra


def run_worker(cfg, root, token, verify_only=False):
    stop = threading.Event()
    child = None
    state = {"pid": os.getpid(), "hostname": socket.gethostname(), "token": token,
             "config": cfg, "started_at": now()}

    def update(status, **values):
        state.update(status=status, updated_at=now(), **values)
        atomic_json(root / "state.json", state)

    def interrupted(signum, frame):
        stop.set()
        if child is not None and child.poll() is None:
            child.terminate()

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, interrupted)
    if hasattr(signal, "SIGHUP"):
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def call(mode):
        nonlocal child
        update("verifying" if mode == "check" else "copying")
        child = subprocess.Popen(command(cfg, mode, root), stdin=subprocess.DEVNULL)
        update(state["status"], rclone_pid=child.pid)
        while child.poll() is None:
            if stop.wait(0.2):
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        return child.returncode

    try:
        update("starting", attempt=0)
        attempt = 0
        while not stop.is_set():
            attempt += 1
            update("starting", attempt=attempt)
            rc = 0 if verify_only else call("copy")
            if rc == 0 and not stop.is_set():
                rc = call("check")
                if rc == 0 and not stop.is_set():
                    update("complete", exit_code=0, verified_at=now())
                    print("DRIVE_DOWNLOAD_OK: copy/check đều thành công", flush=True)
                    return 0
            if stop.is_set():
                break
            if verify_only or rc in FATAL_CODES or (cfg["max_attempts"] and attempt >= cfg["max_attempts"]):
                update("failed", exit_code=rc)
                print("DRIVE_DOWNLOAD_FAILED: xem log, sửa lỗi rồi start lại", flush=True)
                return rc or 1
            delay = min(cfg["retry_seconds"] * (2 ** min(attempt - 1, 4)), 600)
            update("retry_wait", exit_code=rc, retry_in_seconds=delay)
            print(f"RETRY: exit={rc}; thử lại sau {delay}s; giữ file đã hoàn tất", flush=True)
            stop.wait(delay)
        update("stopped", exit_code=130)
        return 130
    except Exception as exc:
        update("failed", error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def process_matches(state):
    if state.get("hostname") != socket.gethostname() or not state.get("token"):
        return False
    try:
        proc = Path("/proc") / str(int(state["pid"]))
        if proc.stat().st_uid != os.getuid():
            return False
        argv = (proc / "cmdline").read_bytes().decode().split("\0")
        return "_worker" in argv and "--token" in argv and state["token"] in argv
    except (OSError, ValueError, KeyError):
        return False


def show_status(root):
    state = read_json(root / "state.json")
    if not state:
        print("Chưa có lượt tải do script quản lý")
    else:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        if state.get("status") in {"starting", "copying", "verifying", "retry_wait"}:
            print("Tiến trình còn chạy trên máy này:", process_matches(state))
    print("Log:", root / "download.log")
    log = root / "download.log"
    if log.exists():
        from collections import deque
        with log.open(encoding="utf-8", errors="replace") as handle:
            print("".join(deque(handle, maxlen=22)), end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop", "verify", "_worker"])
    parser.add_argument("--remote", default="mydrive", help="Tên remote đã xác thực")
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="Drive folder ID hoặc link thư mục")
    parser.add_argument("--dest", default=str(Path.home() / "cfg5-pretrain/downloads/finetune_data"))
    parser.add_argument("--workers", type=int, default=4, help="Số file tải đồng thời")
    parser.add_argument("--retry-seconds", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=0, help="0: retry không giới hạn với lỗi có thể retry")
    parser.add_argument("--token", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cfg = config(args)
    root = state_dir(cfg)
    if args.action == "status":
        show_status(root)
        return 0
    if args.action == "stop":
        state = read_json(root / "state.json")
        if not process_matches(state):
            raise RuntimeError("Không có tiến trình script hợp lệ trên máy này; không gửi signal")
        os.kill(state["pid"], signal.SIGTERM)
        print("Đã yêu cầu dừng lượt tải; file hoàn tất vẫn giữ nguyên. Dùng status kiểm tra")
        return 0
    if args.action == "_worker":
        if args.lock_fd is None or not args.token:
            raise RuntimeError("Không gọi _worker bằng tay; dùng start")
        import fcntl
        fd = args.lock_fd
        expected = (root / "download.lock").stat()
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("Lock FD không khớp thư mục trạng thái")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run_worker(cfg, root, args.token)
        finally:
            os.close(fd)

    fd = acquire_lock(root)
    try:
        old = unmanaged_downloads(cfg)
        if old:
            raise RuntimeError(f"Có rclone cũ cùng đích, PID={old}. Không khởi động lượt tải trùng")
        preflight(cfg)
        bind_config(root, cfg)
        Path(cfg["dest"]).mkdir(parents=True, exist_ok=True)
        token = os.urandom(16).hex()
        if args.action == "verify":
            return run_worker(cfg, root, token, verify_only=True)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker",
               "--remote", cfg["remote"], "--folder", cfg["folder_id"], "--dest", cfg["dest"],
               "--workers", str(cfg["workers"]), "--retry-seconds", str(cfg["retry_seconds"]),
               "--max-attempts", str(cfg["max_attempts"]), "--token", token, "--lock-fd", str(fd)]
        with (root / "download.log").open("a", encoding="utf-8") as log:
            log.write("\n=== START " + now() + " ===\n")
            log.flush()
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, pass_fds=(fd,))
        # Đợi worker ghi state mới, tránh đọc nhầm kết quả complete của lượt trước.
        for _ in range(100):
            state = read_json(root / "state.json")
            if state.get("token") == token:
                break
            if child.poll() is not None:
                raise RuntimeError("Worker khởi động lỗi; xem " + str(root / "download.log"))
            time.sleep(0.05)
        else:
            raise RuntimeError("Worker chưa ghi trạng thái; kiểm tra log, không start trùng")
        print(f"Đã chạy nền: PID={child.pid}; tải đồng thời {cfg['workers']} file")
        print("Đích:", cfg["dest"])
        print("Log:", root / "download.log")
        print("Đóng SSH/tắt laptop không dừng tiến trình tải trên server")
        return 0
    finally:
        # Không LOCK_UN: worker kế thừa cùng open-file description đang giữ lock.
        os.close(fd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print("LỖI:", exc, file=sys.stderr)
        sys.exit(1)
