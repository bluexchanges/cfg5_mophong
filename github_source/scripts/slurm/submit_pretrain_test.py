"""Submit the held-out test once, after the current BS8-to80 job succeeds."""

import argparse
import getpass
import json
import os
import subprocess
import re
import time
from pathlib import Path


def queue_jobs():
    queue = subprocess.check_output(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%j|%t|%E"], text=True)
    return [line.strip().split("|") for line in queue.splitlines() if line.strip()]


def wait_gone(job):
    for _ in range(30):
        if not any(row[0] == job for row in queue_jobs()):
            return
        time.sleep(1)
    raise RuntimeError(f"Job {job} chưa rời queue; không gửi test khi training chưa dừng")


def stop_source(base, repo, parent, jobs):
    run = repo / "outputs/act_pretrain_bs8_to80"
    history = json.loads((run / "history.json").read_text())
    if not history or not (run / "best.pt").is_file() or not (run / "best.pt").stat().st_size:
        raise ValueError("Chưa có history/best.pt hoàn chỉnh để test; không tự dừng")
    source = [row for row in jobs if row[0] == parent]
    if source and (source[0][1] != "cfg5-act-bs8-to80" or source[0][2] not in ("R", "PD", "CG")):
        raise ValueError(f"Job nguồn không đúng lượt BS8: {source}")
    other = [row for row in jobs if row[1].startswith("cfg5-act")
             and row[0] != parent and row[1] != "cfg5-act-test"]
    if other:
        raise ValueError(f"Có ACT job khác; kiểm tra trước khi chốt test: {other}")
    pending = [row for row in jobs if row[1] == "cfg5-act-test"]
    for row in pending:
        dependency = row[3] if len(row) > 3 else ""
        if row[2] != "PD" or not re.fullmatch(r"afterok:" + re.escape(parent) + r"(?:\([^)]*\))?", dependency):
            raise ValueError(f"Không thay job test này tự động: {row}")
    # Remove the old afterok test first, since cancelling its parent invalidates it.
    for row in pending:
        subprocess.run(["scancel", f"--user={getpass.getuser()}", "--name=cfg5-act-test",
                        "--state=PENDING", row[0]], check=True)
        wait_gone(row[0])
    if source:
        print(f"Dừng job {parent}; epoch đã lưu trong history: {history[-1]['epoch']+1}", flush=True)
        subprocess.run(["scancel", f"--user={getpass.getuser()}", "--name=cfg5-act-bs8-to80", parent], check=True)
        wait_gone(parent)
    print("Training đã dừng; sẽ khóa best.pt hiện tại và test không cập nhật trọng số", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-now", action="store_true", help="Dừng đúng job BS8 và test best checkpoint đã lưu")
    args = parser.parse_args()
    import fcntl
    base = Path.home() / "cfg5-pretrain"
    repo = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act"
    bundle = Path(__file__).resolve().parents[2]
    if args.stop_now:
        for name in ("scripts/evaluate_pretrain_test.py", "scripts/slurm/pretrain_test.sbatch"):
            payload = (bundle / name).read_bytes()
            if name.endswith(".py"):
                compile(payload, name, "exec")
            elif not payload.startswith(b"#!/usr/bin/env bash"):
                raise ValueError("File sbatch của test không hợp lệ; không dừng training")
    (base / "logs").mkdir(exist_ok=True)
    with (base / "submit_pretrain_test.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = repo / "outputs/act_pretrain_bs8_to80"
        completed = [output / name / "test_report.json" for name in ("test_final", "test_epoch80")]
        existing = next((path for path in completed if path.exists()), None)
        if existing:
            print(f"Đã có kết quả test: {existing}")
            return
        jobs = queue_jobs()
        previous = [row for row in jobs if row[1] == "cfg5-act-test"]
        if previous and (not args.stop_now or any(row[2] != "PD" for row in previous)):
            print(f"Đã có job test: {previous}; không gửi thêm")
            return
        parent = (base / "act_bs8_to80.jobid").read_text().strip()
        if not parent.isdigit():
            raise ValueError("Job ID training không hợp lệ")
        active = [row for row in jobs if row[0] == parent]
        command = ["sbatch", "--parsable"]
        if args.stop_now:
            stop_source(base, repo, parent, jobs)
            active = []
        elif active:
            if active[0][1] != "cfg5-act-bs8-to80" or active[0][2] not in ("R", "PD", "CG"):
                raise ValueError(f"Job nguồn không đúng lượt BS8: {active}")
            command += [f"--dependency=afterok:{parent}", "--kill-on-invalid-dep=yes"]
        else:
            history = json.loads((repo / "outputs/act_pretrain_bs8_to80/history.json").read_text())
            if not history or history[-1]["epoch"] + 1 != 80:
                raise ValueError("Training chưa lưu đủ 80 epoch và không còn trong queue; kiểm tra job trước")
        env = dict(os.environ, CFG5_TEST_BUNDLE=str(bundle), CFG5_TEST_STOPPED_EARLY="1" if args.stop_now else "0")
        reply = subprocess.check_output(command + [str(bundle / "scripts/slurm/pretrain_test.sbatch")],
                                        text=True, env=env).strip()
        job = reply.split(";", 1)[0]
        if not job.isdigit():
            raise RuntimeError(f"Không đọc được Job ID: {reply}")
        (base / "act_pretrain_test.jobid").write_text(job + "\n")
        print(f"Submitted batch job {job}")
        print(f"Chờ job {parent} COMPLETED rồi test" if active else "Test best checkpoint của lượt đã chốt")


if __name__ == "__main__":
    main()
