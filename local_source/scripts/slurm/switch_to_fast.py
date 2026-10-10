"""Take over known watchers, wait for a saved epoch, then submit one fast40 job."""
import argparse
import getpass
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

NEW_NAME = "cfg5-act-fast40"
SOURCES = {
    "cfg5-act-bs4-w2": ("act_pretrain_bs4_w2", "cfg5_all_pretrain_bs4_w2_40epoch.json"),
    "cfg5-act-prefetch": ("act_pretrain_prefetch", "cfg5_all_pretrain_prefetch_40epoch.json"),
}


def validate_bundle(bundle):
    manifest = json.loads((bundle / "bundle_hashes.json").read_text(encoding="utf-8"))
    required = ("src/training/pretrain.py", "src/training/fork_run.py",
                "scripts/fork_training_run.py", "scripts/slurm/install_worker_bundle.py",
                "scripts/slurm/fast_train.py", "scripts/slurm/act_fast40.sbatch",
                "scripts/slurm/switch_to_fast.py", "tests/test_cuda_prefetch.py")
    for name in required:
        payload = (bundle / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest[name]:
            raise ValueError(f"Bundle sai checksum: {name}")
        if name.endswith(".py"):
            compile(payload, name, "exec")


def jobs(user):
    result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j %t"],
                            check=True, capture_output=True, text=True)
    return [tuple(line.split()) for line in result.stdout.splitlines() if line.strip()]


def last_epoch(folder):
    records = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"History trống: {folder}")
    return records[-1]["epoch"]


def watcher_identity(pid, base, *, proc_root=Path("/proc"), uid=None):
    """Match our own exact watcher script arguments, never a trainer or grep."""
    if uid is None:
        uid = os.getuid()
    allowed = {
        str(base / "act_workers2_to40_bundle/scripts/slurm/switch_to_workers.py"),
        str(base / "act_cuda_prefetch_bundle/scripts/slurm/switch_to_cuda_prefetch.py"),
    }
    try:
        folder = proc_root / str(pid)
        if folder.stat().st_uid != uid:
            return None
        args = (folder / "cmdline").read_bytes().rstrip(b"\x00").split(b"\x00")
        words = [a.decode("utf-8") for a in args]
        if not words or not Path(words[0]).name.startswith("python"):
            return None
        script_index = 1
        while script_index < len(words) and words[script_index] in ("-u", "-B", "-E", "-s", "-S", "-I"):
            script_index += 1
        if script_index >= len(words) or words[script_index] not in allowed:
            return None
        return tuple(words)
    except (OSError, UnicodeError):
        return None


def live_children(folder, proc_root=Path("/proc")):
    pid = folder.name
    try:
        child_ids = (folder / "task" / pid / "children").read_text().split()
    except FileNotFoundError:
        return []
    live = []
    for child_id in child_ids:
        try:
            state = (proc_root / child_id / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state != "Z":
                live.append(child_id)
        except FileNotFoundError:
            pass
    return live


def stop_known_watchers(base):
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        pid = int(folder.name)
        identity = watcher_identity(pid, base)
        if identity and watcher_identity(pid, base) == identity:
            stopped = False
            try:
                # Freeze the controller, let any in-flight sbatch/scancel
                # child finish, then terminate it before taking its locks.
                # This avoids a late submission from an orphaned sbatch.
                os.kill(pid, signal.SIGSTOP)
                stopped = True
                for _ in range(40):
                    if not live_children(folder):
                        break
                    time.sleep(0.25)
                else:
                    raise RuntimeError(f"Watcher {pid} còn lệnh Slurm đang chạy; chưa tiếp quản")
                os.kill(pid, signal.SIGTERM)
                print(f"Tiếp quản từ watcher PID={pid}: {identity}", flush=True)
            except ProcessLookupError:
                pass
            finally:
                if stopped:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass


def wait_gone(user, job_id):
    for _ in range(60):
        if not any(row[0] == job_id for row in jobs(user)):
            return
        time.sleep(5)
    raise RuntimeError("Job cũ chưa rời queue sau 5 phút; chưa gửi fast40")


def wait_boundary(user, row, folder):
    job_id, name, state = row
    if name not in SOURCES:
        raise RuntimeError(f"Tên job ngoài phạm vi: {name}")
    if state == "PD":
        subprocess.run(["scancel", "--user", user, "--name", name, "--state=PENDING", job_id], check=True)
        print(f"Hủy lượt chờ {job_id} ({name}) để thay bằng fast40", flush=True)
        wait_gone(user, job_id)
        return
    if state == "CG":
        wait_gone(user, job_id)
        return
    if state != "R":
        raise RuntimeError(f"Không tự hủy trạng thái {state}")
    target = last_epoch(folder) + 1
    print(f"Chờ job {job_id} lưu xong epoch {target + 1}, rồi chuyển fast40", flush=True)
    while True:
        current = next((r for r in jobs(user) if r[0] == job_id), None)
        if current is None:
            return
        if current[1] != name:
            raise RuntimeError("Danh tính job thay đổi; không hủy")
        if current[2] == "CG":
            wait_gone(user, job_id)
            return
        if current[2] != "R":
            raise RuntimeError(f"Job không còn R: {current}; không hủy")
        if last_epoch(folder) >= target:
            # history is committed after the complete epoch checkpoint.
            subprocess.run(["scancel", "--user", user, "--name", name, job_id], check=True)
            print(f"Checkpoint epoch {last_epoch(folder)+1} đã lưu; dừng job {job_id}", flush=True)
            wait_gone(user, job_id)
            return
        time.sleep(5)


def latest_source(repo):
    folders = ["act_pretrain_bs4_w2", "act_pretrain_prefetch", "act_pretrain_fast"]
    found = []
    for priority, name in enumerate(folders):
        folder = repo / "outputs" / name
        if (folder / "latest.pt").exists() and (folder / "history.json").exists():
            found.append((last_epoch(folder), priority, folder / "latest.pt"))
    if not found:
        raise FileNotFoundError("Chưa có checkpoint worker2/prefetch/fast để tiếp tục")
    epoch, _, source = max(found)
    return source, epoch


def transition(bundle, base, repo, user):
    rows = [r for r in jobs(user) if r[1].startswith("cfg5-act")]
    if any(r[1] == NEW_NAME for r in rows):
        print("fast40 đã chạy/chờ; không gửi trùng", flush=True)
        return
    if len(rows) > 1 or any(r[1] not in SOURCES for r in rows):
        raise RuntimeError(f"Có job ACT ngoài phạm vi hoặc nhiều job; không tự dừng: {rows}")
    if rows:
        row = rows[0]
        folder_name, config_name = SOURCES[row[1]]
        folder = repo / "outputs" / folder_name
        if row[2] == "R":
            cfg = json.loads((repo / "configs/act" / config_name).read_text(encoding="utf-8"))
            if cfg["save_every_epochs"] != 1 or cfg["output_dir"] != "outputs/" + folder_name:
                raise ValueError("Nguồn cần lưu mỗi epoch và đúng output")
        wait_boundary(user, row, folder)
    if any(r[1].startswith("cfg5-act") for r in jobs(user)):
        raise RuntimeError("Có job ACT mới xuất hiện; chưa gửi fast40")
    source, epoch = latest_source(repo)
    if epoch + 1 >= 40:
        print("Đã xong 40 epoch; chưa gửi job thêm", flush=True)
        return
    script = base / "act_fast40.sbatch"
    template = (bundle / "scripts/slurm/act_fast40.sbatch").read_text(encoding="utf-8")
    script.write_text(template.replace("__BUNDLE__", shlex.quote(str(bundle))).replace(
        "__SOURCE__", shlex.quote(str(source))), encoding="utf-8")
    result = subprocess.run(["sbatch", "--parsable", str(script)], check=True,
                            capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch trả kết quả lạ: {result.stdout!r}")
    (base / "act_fast40.jobid").write_text(job_id + "\n")
    print(f"Submitted batch job {job_id}; 8 CPU; nguồn đã lưu {epoch+1} epoch", flush=True)
    print(f"Log: {base}/logs/cfg5-act-fast40-{job_id}.out", flush=True)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.home() / "cfg5-pretrain")
    parser.add_argument("--takeover", action="store_true", help="Tiếp quản hai watcher workers2/prefetch đã biết")
    args = parser.parse_args()
    base = args.base.resolve()
    bundle = Path(__file__).resolve().parents[2]
    repo = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act"
    validate_bundle(bundle)
    with ExitStack() as stack:
        own = stack.enter_context((base / "switch_fast40.lock").open("a+"))
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Đã có watcher fast40", flush=True)
            return
        if args.takeover:
            stop_known_watchers(base)
        # Own every earlier transition lock before examining/cancelling jobs.
        for name in ("switch_workers2.lock", "switch_cuda_prefetch.lock"):
            lock = stack.enter_context((base / name).open("a+"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not args.takeover:
                    raise RuntimeError("Watcher cũ đang chạy; dùng --takeover để chuyển quyền")
                for _ in range(20):
                    time.sleep(0.5)
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                else:
                    raise RuntimeError("Chưa lấy được khóa watcher cũ; không đụng job")
        transition(bundle, base, repo, getpass.getuser())


if __name__ == "__main__":
    main()
