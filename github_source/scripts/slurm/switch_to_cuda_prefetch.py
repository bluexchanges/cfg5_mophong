"""Wait for the next saved epoch, stop the matching job and submit CUDA prefetch.

Runs as a small stdlib-only watcher on the head node. No torch/checkpoint load
or dataset read here. Install and GPU training happen inside the new job.
"""

import argparse
import getpass
import json
import shlex
import subprocess
import time
from pathlib import Path

from install_worker_bundle import validate_bundle

OLD_NAME = "cfg5-act-bs4-w2"
NEW_NAME = "cfg5-act-prefetch"


def jobs(user):
    result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j %t"],
                            check=True, capture_output=True, text=True)
    return [tuple(line.split()) for line in result.stdout.splitlines() if line.strip()]


def last_epoch(history):
    records = json.loads(Path(history).read_text(encoding="utf-8"))
    if not records:
        raise ValueError("History nguồn trống")
    return records[-1]["epoch"]


def matching_old(rows, job_id):
    return next((row for row in rows if row[0] == job_id and row[1] == OLD_NAME), None)


def wait_for_boundary(user, job_id, history, initial, *, poll_seconds=10):
    target = initial + 1  # JSON epochs are zero-based.
    print(f"Đang chờ job {job_id} lưu xong epoch {target + 1}; không dừng giữa epoch",
          flush=True)
    while True:
        rows = jobs(user)
        current = matching_old(rows, job_id)
        if current is None:
            print("Job cũ đã kết thúc; sẽ tiếp từ checkpoint epoch cuối đã lưu", flush=True)
            return
        if current[2] != "R":
            if current[2] in ("CG", "CD", "F", "CA", "TO"):
                time.sleep(poll_seconds)
                continue
            raise RuntimeError(f"Job cũ không đang R: {current}; không tự hủy")
        if last_epoch(history) >= target:
            # history is atomically committed after latest.pt; only accept save_every_epochs=1.
            subprocess.run(["scancel", "--user", user, "--name", OLD_NAME, job_id], check=True)
            print(f"Đã lưu epoch {last_epoch(history) + 1}; gửi dừng đúng job {job_id}", flush=True)
            for _ in range(60):
                if matching_old(jobs(user), job_id) is None:
                    return
                time.sleep(5)
            raise RuntimeError("Job cũ chưa rời queue sau 5 phút; chưa gửi lượt mới")
        time.sleep(poll_seconds)


def transition(bundle, base, repo, user):
    validate_bundle(bundle)
    rows = jobs(user)
    if any(row[1] == NEW_NAME for row in rows):
        print("Job CUDA prefetch đã chạy/chờ; không gửi trùng", flush=True)
        return
    act = [row for row in rows if row[1].startswith("cfg5-act")]
    if any(row[1] != OLD_NAME for row in act) or len(act) > 1:
        raise RuntimeError(f"Có job ACT khác; không tự dừng: {act}")
    history = repo / "outputs/act_pretrain_bs4_w2/history.json"
    source = repo / "outputs/act_pretrain_bs4_w2/latest.pt"
    if not source.exists():
        raise FileNotFoundError(f"Chưa có checkpoint batch 4: {source}")
    if act:
        if act[0][2] != "R":
            raise RuntimeError(f"Job cũ chưa R: {act[0]}; không tự hủy")
        config = json.loads((repo / "configs/act/cfg5_all_pretrain_bs4_w2_40epoch.json").read_text())
        if config["save_every_epochs"] != 1 or config["output_dir"] != "outputs/act_pretrain_bs4_w2":
            raise ValueError("Job nguồn cần lưu mỗi epoch và đúng output batch 4")
        if config["batch_size"] != 4 or config["num_workers"] != 2 or config.get("cuda_prefetch", False):
            raise ValueError("Job nguồn phải là batch4/workers2 chưa bật CUDA prefetch")
        wait_for_boundary(user, act[0][0], history, last_epoch(history))
    if any(row[1].startswith("cfg5-act") for row in jobs(user)):
        raise RuntimeError("Có job ACT mới xuất hiện; chưa gửi lượt CUDA prefetch")
    child_history = repo / "outputs/act_pretrain_prefetch/history.json"
    if last_epoch(child_history if child_history.exists() else history) + 1 >= 40:
        print("Đã đạt tổng 40 epoch; không chạy thêm", flush=True)
        return
    template = (bundle / "scripts/slurm/act_cuda_prefetch.sbatch").read_text(encoding="utf-8")
    script = base / "act_cuda_prefetch.sbatch"
    script.write_text(template.replace("__BUNDLE__", shlex.quote(str(bundle))), encoding="utf-8")
    result = subprocess.run(["sbatch", "--parsable", str(script)], check=True,
                            capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch trả kết quả không hợp lệ: {result.stdout!r}")
    (base / "act_cuda_prefetch.jobid").write_text(job_id + "\n")
    print(f"Submitted batch job {job_id}; batch=4, workers=2, CUDA prefetch, tổng 40 epoch", flush=True)
    print(f"Log: {base}/logs/cfg5-act-prefetch-{job_id}.out", flush=True)


def main():
    import fcntl  # Linux head node only; helper functions are portable for fixture tests.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path.home() / "cfg5-pretrain")
    args = parser.parse_args()
    base = args.base.resolve()
    bundle = Path(__file__).resolve().parents[2]
    repo = base / "edabk-physical-ai-robot/04-Src/software/cfg5_act"
    with (base / "switch_cuda_prefetch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Đã có một tiến trình chờ chuyển CUDA prefetch", flush=True)
            return
        transition(bundle, base, repo, getpass.getuser())


if __name__ == "__main__":
    main()
