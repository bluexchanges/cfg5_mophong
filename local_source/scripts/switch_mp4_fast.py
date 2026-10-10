"""Chờ epoch đã lưu rồi đổi code và đăng ký benchmark/train; chạy nhẹ trên head node."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def call(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Lệnh thất bại: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def saved_epoch(repo):
    source = repo / "outputs/pick_place_finetune_mp4_w16"
    history = source / "history.json"
    if not history.exists() or not (source / "latest.pt").is_file():
        return -1
    rows = json.loads(history.read_text(encoding="utf-8"))
    full = [r for r in rows if r.get("smoke") is False]
    if not full:
        return -1
    row = full[-1]
    if row["train"]["episodes"] != 731 or row["val"]["episodes"] != 91:
        raise ValueError("Counts train/val khác run đã kiểm; không tự chuyển")
    return row["epoch"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=int)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--use-saved-epoch", action="store_true",
                        help="Dừng ở checkpoint hiện có; bỏ phần epoch chưa lưu")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    bundle = Path(__file__).resolve().parent
    if not (bundle / "install_fast.py").is_file():
        raise FileNotFoundError("Chạy switch_mp4_fast.py ở bundle đã unzip")
    lock = (bundle / "switch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((repo / "configs/act/pick_place_finetune_mp4_w16.json").read_text(encoding="utf-8"))
    if config["save_every_epochs"] != 1:
        raise ValueError("Watcher cần latest.pt được lưu mỗi epoch")
    initial = saved_epoch(repo)
    if args.use_saved_epoch and initial < 0:
        raise ValueError("Chưa có checkpoint fine-tune, không dừng giữa epoch")
    job = str(args.job)
    job_info = call("squeue", "-j", job, "-h", "-o", "%u|%j|%T")
    if job_info and (not job_info.startswith(call("id", "-un") + "|")
                     or "cfg5-ft-mp4-w16" not in job_info):
        raise ValueError("Job nguồn không đúng user/name, không cancel")
    target = initial if args.use_saved_epoch else initial + 1
    print(f"Checkpoint hiện có: {initial+1} epoch. Chờ lưu đủ {target+1} epoch; job={job}", flush=True)
    while saved_epoch(repo) < target:
        state = call("squeue", "-j", job, "-h", "-o", "%T")
        if not state:
            raise RuntimeError("Job đã kết thúc trước checkpoint cần chờ; xem sacct/log")
        time.sleep(30)
    if call("squeue", "-j", job, "-h", "-o", "%T"):
        call("scancel", job)
    for _ in range(120):
        if not call("squeue", "-j", job, "-h", "-o", "%T"):
            break
        time.sleep(5)
    else:
        raise RuntimeError("Job chưa giải phóng; chưa cài code")
    print("Epoch đã lưu; job cũ đã dừng. Cài bundle fast.", flush=True)
    print(call(str(repo / ".venv-data/bin/python"), str(bundle / "install_fast.py"), "--repo", str(repo)), flush=True)
    bench = call("sbatch", "--parsable", str(repo / "scripts/slurm/benchmark_pick_place_mp4_fast.sbatch"))
    bench = bench.split(";")[0]
    if not bench.isdigit():
        raise ValueError("Không đọc được job ID benchmark")
    (Path.home() / "cfg5-pretrain/mp4_fast_bench.jobid").write_text(bench + "\n")
    train = call("sbatch", "--parsable", "--dependency=afterok:" + bench,
                 "--kill-on-invalid-dep=yes", str(repo / "scripts/slurm/train_pick_place_mp4_fast.sbatch"))
    train = train.split(";")[0]
    (Path.home() / "cfg5-pretrain/mp4_fast_train.jobid").write_text(train + "\n")
    print(f"BENCH_JOB={bench}; TRAIN_JOB={train}; train chỉ chạy sau benchmark đạt.", flush=True)


if __name__ == "__main__":
    main()
