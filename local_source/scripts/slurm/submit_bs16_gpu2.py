"""Submit one ACT job; refuse duplicates and active transition controllers."""

import getpass
import shlex
import subprocess
from contextlib import ExitStack
from pathlib import Path


def active_act(user):
    result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i %j"],
                            check=True, text=True, capture_output=True)
    return [line for line in result.stdout.splitlines()
            if len(line.split()) == 2 and line.split()[1].startswith("cfg5-act")]


def main():
    import fcntl
    base = Path.home() / "cfg5-pretrain"
    bundle = Path(__file__).resolve().parents[2]
    with ExitStack() as stack:
        for name in ("submit_bs16_gpu2.lock", "switch_fast40.lock", "switch_workers2.lock",
                     "switch_cuda_prefetch.lock"):
            handle = stack.enter_context((base / name).open("a+"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = active_act(getpass.getuser())
        if active:
            raise RuntimeError(f"Đã có job ACT; không gửi thêm: {active}")
        (base / "logs").mkdir(exist_ok=True)
        script = base / "act_bs16_gpu2_to80.sbatch"
        text = (bundle / "scripts/slurm/act_bs16_gpu2_to80.sbatch").read_text()
        script.write_text(text.replace("__BUNDLE__", shlex.quote(str(bundle))), encoding="utf-8")
        result = subprocess.run(["sbatch", "--parsable", str(script)],
                                check=True, capture_output=True, text=True)
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdigit():
            raise RuntimeError(f"sbatch trả kết quả lạ: {result.stdout!r}")
        (base / "act_bs16_gpu2_to80.jobid").write_text(job + "\n")
        print(f"Submitted batch job {job}; xin 2 GPU / 16 CPU / 128 GB RAM")
        print(f"Log: {base}/logs/cfg5-act-bs16-g2-to80-{job}.out")


if __name__ == "__main__":
    main()
