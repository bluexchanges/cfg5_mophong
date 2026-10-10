"""Read-only epoch/batch progress and train+val durations; no torch import."""

import json
import argparse
import getpass
import re
import subprocess
from math import ceil
from pathlib import Path


def durations(rows):
    result = []
    for row in rows:
        try:
            train = row["train"]["timing"]["epoch_seconds"]
            val = row["val"]["timing"]["epoch_seconds"]
        except KeyError:
            continue
        result.append((row["epoch"] + 1, train, val))
    return result


def read_tail(path, size=200000):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - size))
        return handle.read().decode("utf-8", errors="replace").splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("bs8", "bs16"), default="bs16")
    args = parser.parse_args()
    repo = Path.home() / "cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act"
    base = Path.home() / "cfg5-pretrain"
    is_bs8 = args.run == "bs8"
    batch_size = 8 if is_bs8 else 16
    output = repo / ("outputs/act_pretrain_bs8_to80" if is_bs8 else "outputs/act_pretrain_bs16_w8_gpu2")
    history = output / "history.json"
    if not history.exists():
        history = repo / "outputs/act_pretrain_fast/history.json"
    rows = json.loads(history.read_text())
    last = rows[-1]
    completed = last["epoch"] + 1
    print(f"Checkpoint/history đã lưu: epoch {completed}/80; global_step={last['global_step']}")
    if is_bs8:
        config = repo / "configs/act/cfg5_all_pretrain_bs8_80epoch.json"
        workers = json.loads(config.read_text())["num_workers"] if config.exists() else last.get("num_workers", "từ checkpoint cũ")
        print(f"Lượt mới: batch=8; workers={workers}; xin 8 CPU / 1 GPU / 64 GB RAM")
    else:
        print("Lượt mới: batch tổng=16 (khoảng 8/GPU); 8 workers; xin 16 CPU / 2 GPU")
    print(f"Loss cuối: train={last['train']['loss']:.6f}; val={last['val']['loss']:.6f}")
    timed = durations(rows)
    print("Thời gian train + val (không gồm lưu checkpoint):")
    for epoch, train, val in timed[-8:]:
        print(f"  E{epoch}: train={train/60:.2f} phút; val={val/60:.2f} phút; tổng={(train+val)/60:.2f} phút")
    if not timed:
        print("  History cũ chưa ghi timing.")
    job_prefix = "act_bs8_to80" if is_bs8 else "act_bs16_gpu2_to80"
    id_path = base / f"{job_prefix}.jobid"
    if not id_path.exists():
        return
    job = id_path.read_text().strip()
    queue = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %t %M %R"],
                           check=True, text=True, capture_output=True)
    current = [line for line in queue.stdout.splitlines() if line.split()[0] == job]
    if current:
        print("JobID trạng_thái đã_chạy node/lý_do: " + current[0])
    else:
        subprocess.run(["sacct", "-j", job, "-X", "--format=JobID,State,ExitCode,Elapsed"], check=True)
    log_name = "cfg5-act-bs8-to80" if is_bs8 else "cfg5-act-bs16-g2-to80"
    log = base / f"logs/{log_name}-{job}.out"
    if log.exists():
        epoch, progress = completed + 1, None
        for line in read_tail(log):
            match = re.search(r"Đã lưu (\d+) epoch;", line)
            if match:
                epoch = int(match[1]) + 1
            if line.startswith('{"epoch":'):
                epoch = json.loads(line)["epoch"] + 2
                progress = None
            if re.match(r"^(train|val) batch=", line):
                progress = line
        if progress:
            phase = progress.split()[0]
            count = last[phase]["samples"]
            print(f"Log: đang epoch {epoch}; {phase} có khoảng {ceil(count/batch_size)} batch")
            print(progress)
        else:
            print("Chưa có dòng tiến độ batch mới; xem các dòng log cuối:")
            print("\n".join(read_tail(log)[-3:]))


if __name__ == "__main__":
    main()
