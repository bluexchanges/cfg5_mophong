"""Replace only the specified pending two-GPU job, then submit one BS8 job."""

import argparse
import getpass
import shlex
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path


def queue(user):
    result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j %t"],
                            check=True, text=True, capture_output=True)
    return [tuple(line.split()) for line in result.stdout.splitlines() if line.strip()]


def replace_pending(job, user):
    entries = queue(user)
    target = [row for row in entries if row[0] == job]
    if not target:
        return
    if target != [(job, "cfg5-act-bs16-g2-to80", "PD")]:
        raise RuntimeError(f"Chỉ thay job BS16 hai GPU đang PD; hiện thấy {target}")
    other = [row for row in entries if row[0] != job and row[1].startswith("cfg5-act")]
    if other:
        raise RuntimeError(f"Có job ACT khác; chưa thay/gửi: {other}")
    subprocess.run(["scancel", "--user", user, "--name=cfg5-act-bs16-g2-to80",
                    "--state=PENDING", job], check=True)
    for _ in range(20):
        remaining = [row for row in queue(user) if row[0] == job]
        if not remaining:
            print(f"Đã hủy job hai GPU đang chờ {job}", flush=True)
            return
        if remaining[0][2] == "R":
            raise RuntimeError("Job cũ đã chuyển R; không dừng giữa epoch/không gửi thêm")
        time.sleep(1)
    raise RuntimeError("Job cũ chưa rời queue; chưa gửi job mới")


def main():
    import fcntl
    from install_worker_bundle import validate_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-pending", help="ID job BS16 hai GPU đang PD được phép thay")
    args = parser.parse_args()
    if args.replace_pending and not args.replace_pending.isdigit():
        raise ValueError("JobID phải là số")
    user = getpass.getuser()
    base = Path.home() / "cfg5-pretrain"
    bundle = Path(__file__).resolve().parents[2]
    validate_bundle(bundle)
    with ExitStack() as stack:
        for name in ("submit_bs16_gpu2.lock", "submit_bs8_to80.lock", "switch_fast40.lock",
                     "switch_workers2.lock", "switch_cuda_prefetch.lock"):
            handle = stack.enter_context((base / name).open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.replace_pending:
            replace_pending(args.replace_pending, user)
        active = [row for row in queue(user) if row[1].startswith("cfg5-act")]
        if active:
            raise RuntimeError(f"Đã có job ACT; không gửi thêm: {active}")
        (base / "logs").mkdir(exist_ok=True)
        script = base / "act_bs8_to80.sbatch"
        template = (bundle / "scripts/slurm/act_bs8_to80.sbatch").read_text()
        script.write_text(template.replace("__BUNDLE__", shlex.quote(str(bundle))), encoding="utf-8")
        result = subprocess.run(["sbatch", "--parsable", str(script)],
                                check=True, text=True, capture_output=True)
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdigit():
            raise RuntimeError(f"sbatch trả kết quả lạ: {result.stdout!r}")
        (base / "act_bs8_to80.jobid").write_text(job + "\n")
        print(f"Submitted batch job {job}; 1 GPU / 8 CPU / 64 GB RAM; batch=8; worker từ checkpoint cũ")
        print(f"Log: {base}/logs/cfg5-act-bs8-to80-{job}.out")


if __name__ == "__main__":
    main()
