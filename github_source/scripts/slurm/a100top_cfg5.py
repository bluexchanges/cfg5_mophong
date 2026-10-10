#!/usr/bin/env python3
"""Dashboard CFG5 ACT: stdlib only, read-only; attach to an existing Slurm job.

Run on the head node: python3 a100top_cfg5.py [--job JOBID].
q exits only the monitor. No sbatch, scancel, model load or training here.
"""

import argparse
import csv
import getpass
import io
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=6, check=True)
    return result.stdout


def queue(user):
    text = command(["squeue", "-u", user, "-h", "-o", "%i|%j|%t|%M|%L|%R"])
    return [dict(zip(("id", "name", "state", "elapsed", "left", "node"), line.split("|", 5)))
            for line in text.splitlines() if line.strip()]


def safe_json(path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def tail(path):
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def locate(repo, name, detail, job_id):
    output = re.search(r"\bStdOut=(\S+)", detail)
    stdout = Path(output[1].replace("%j", job_id)) if output else Path("/nonexistent/log")
    command_path = re.search(r"\bCommand=(\S+)", detail)
    script = tail(command_path[1]) if command_path else ""
    config = re.search(r"--config\s+([^\s\\]+)", script)
    candidate = config[1].strip("\"'") if config else ""
    if "$" in candidate:
        assigned = re.search(r"^\s*CONFIG=[\"']?([^\s\"']+)", script, re.M)
        candidate = assigned[1] if assigned else ""
    defaults = {
        "cfg5-act-bs4-w2": "cfg5_all_pretrain_bs4_w2_40epoch.json",
        "cfg5-act-bs4-to40": "cfg5_all_pretrain_bs4_40epoch.json",
        "cfg5-act-to20": "cfg5_all_pretrain_20epoch.json",
        "cfg5-act-to10": "cfg5_all_pretrain_10epoch.json",
    }
    cfg = safe_json(repo / candidate, {}) if candidate else {}
    if not cfg:
        cfg = safe_json(repo / "configs/act" / defaults.get(name, "cfg5_all_pretrain.json"), {})
    out = repo / cfg.get("output_dir", "outputs/act_pretrain")
    if "--smoke" in script:
        out = out / "smoke"
    cpus = re.search(r"\bNumCPUs=(\d+)", detail)
    cfg = dict(cfg, _cpus=int(cpus[1]) if cpus else None)
    if "--smoke" in script:
        cfg["epochs"] = 1
    return stdout, out / "history.json", cfg


def progress(text, history, cfg):
    completed = history[-1]["epoch"] + 1 if history else 0
    train_n = history[-1]["train"]["samples"] if history else None
    val_n = history[-1]["val"]["samples"] if history else None
    result = {"saved": completed, "epoch": completed + 1, "target": cfg.get("epochs"),
              "phase": "chuẩn bị", "batch": 0, "samples": 0, "total": None, "rate": None}
    start = re.search(r"Đã lưu (\d+) epoch; chạy tới tổng (\d+)", text)
    if start:
        result.update(epoch=int(start[1]) + 1, target=int(start[2]))
    batch = cfg.get("batch_size", 1)
    latest = None
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                record = json.loads(line)
                if "epoch" in record and "train" in record and "val" in record:
                    result.update(epoch=record["epoch"] + 1, phase="lưu checkpoint",
                                  samples=record["val"]["samples"], batch=record["val"]["batches"],
                                  total=record["val"]["samples"], rate=None)
                    latest = record
            except ValueError:
                pass
        match = re.search(r"^(train|val) batch=(\d+) samples=(\d+)", line)
        if match:
            if latest:
                result["epoch"] = latest["epoch"] + 2
                latest = None
            rate = re.search(r"samples_per_s=([\d.]+)", line)
            wait = re.search(r"data_wait_ms=([\d.]+)", line)
            step = re.search(r"step_ms=([\d.]+)", line)
            result.update(phase=match[1], batch=int(match[2]), samples=int(match[3]),
                          total=train_n if match[1] == "train" else val_n,
                          rate=float(rate[1]) if rate else None,
                          wait=float(wait[1]) if wait else None,
                          step=float(step[1]) if step else None)
    result["batches"] = math.ceil(result["total"] / batch) if result["total"] else None
    # The log prints every 100 batches, so its progress is a lower bound between updates.
    return result


def process_stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": pid, "state": fields[0], "ticks": int(fields[11]) + int(fields[12]),
                "start": int(fields[19]), "rss": int(fields[21]) * os.sysconf("SC_PAGE_SIZE")}
    except (OSError, ValueError, IndexError):
        return None


class Probe:
    def __init__(self, job_id):
        self.job_id = job_id
        self.pid = None
        self.next_find = 0
        self.previous = {}

    def sample(self):
        now = time.monotonic()
        if now >= self.next_find or not self.pid or not Path(f"/proc/{self.pid}").exists():
            self.next_find = now + 30
            self.pid = None
            for line in command(["scontrol", "listpids", self.job_id]).splitlines():
                part = line.split()
                if part and part[0].isdigit():
                    try:
                        args = Path(f"/proc/{part[0]}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                        if "scripts/train.py" in args:
                            self.pid = int(part[0])
                            break
                    except OSError:
                        pass
        if not self.pid:
            return {"processes": [], "gpu": None, "note": "Chưa thấy train.py trên node"}
        try:
            children = Path(f"/proc/{self.pid}/task/{self.pid}/children").read_text().split()
        except OSError:
            children = []
        processes = []
        for pid in [self.pid] + [int(item) for item in children]:
            stat = process_stat(pid)
            if stat is None:
                continue
            before = self.previous.get(pid)
            stat["cpu"] = None
            if before and before[1]["start"] == stat["start"]:
                stat["cpu"] = max(0, (stat["ticks"] - before[1]["ticks"]) /
                                  os.sysconf("SC_CLK_TCK") / max(now - before[0], 1e-9) * 100)
            self.previous[pid] = (now, stat.copy())
            processes.append(stat)
        apps = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"])
        uuid = next((row[1].strip() for row in csv.reader(io.StringIO(apps))
                     if len(row) > 1 and row[0].strip() == str(self.pid)), None)
        gpu = None
        if uuid:
            fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
            values = command(["nvidia-smi", "-i", uuid, "--query-gpu=" + fields,
                              "--format=csv,noheader,nounits"])
            row = next(csv.reader(io.StringIO(values)), [])
            gpu = [item.strip() for item in row] if len(row) >= 6 else None
        return {"processes": processes, "gpu": gpu,
                "note": "CPU: tiến trình train + con; VRAM/GPU: toàn card được chọn"}


def bar(value, width=32):
    count = round(max(0, min(100, value)) / 100 * width)
    return "[" + "#" * count + "-" * (width - count) + "]"


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60}h{minutes % 60:02d}m"


def render(rows, job_id, p, cfg, history, sample, width=120, show_processes=True):
    lines = [f"a100top CFG5 | {getpass.getuser()} | {datetime.now():%Y-%m-%d %H:%M:%S}",
             "q: thoát monitor   p: hiện/ẩn process   +/-: tốc độ refresh", "",
             f"{'JOB':<9} {'TÊN':<24} {'ST':<4} {'ĐÃ CHẠY':<12} {'HẠN CÒN':<12} NODE / LÝ DO"]
    for row in rows:
        marker = ">" if row["id"] == job_id else " "
        lines.append(f"{marker}{row['id']:<8} {row['name'][:23]:<24} {row['state']:<4} "
                     f"{row['elapsed']:<12} {row['left']:<12} {row['node']}")
    lines += ["", "=" * min(width, 110),
              f"Job {job_id}: epoch hiện tại {p['epoch']}/{p['target'] or '?'} | "
              f"đã lưu {p['saved']} epoch | batch={cfg.get('batch_size', '?')} "
              f"workers={cfg.get('num_workers', '?')}"]
    total = p["total"]
    percent = 100 * p["samples"] / total if total else 0
    phase = p["phase"]
    lines.append(f"{phase:>7} {bar(percent)} {percent:.1f}% | "
                 f"batch {p['batch']}/{p['batches'] or '?'} | mẫu {p['samples']}/{total or '?'}")
    if p.get("rate"):
        remaining = max(0, total - p["samples"]) / p["rate"] if total else 0
        lines.append(f"{p['rate']:.2f} mẫu/s | ETA riêng phase {phase}: ~{duration(remaining)} | "
                     f"data_wait_ms={p.get('wait')} step_ms={p.get('step')}")
    else:
        lines.append("ETA: chưa đủ số đo; tiến độ cập nhật theo log mỗi 100 batch")
    if history:
        last = history[-1]
        best = min(history, key=lambda row: row["val"]["loss"])
        lines.append(f"Epoch {last['epoch'] + 1} đã lưu: train loss {last['train']['loss']:.4f} "
                     f"| val loss {last['val']['loss']:.4f}")
        lines.append(f"Best lịch sử: epoch {best['epoch'] + 1}, val {best['val']['loss']:.4f}")
    lines.append("")
    gpu = sample.get("gpu")
    if gpu:
        try:
            util, used, capacity = map(float, gpu[1:4])
            lines += [f"GPU  {bar(util)} {util:5.1f}% | {gpu[0]}",
                      f"VRAM {bar(used / capacity * 100)} {used / 1024:.1f}/{capacity / 1024:.1f} GiB "
                      f"| {gpu[4]} °C | {gpu[5]} W"]
        except (ValueError, ZeroDivisionError):
            lines.append("GPU: " + ", ".join(gpu))
    else:
        lines.append("GPU: chưa có số đo cho tiến trình train")
    processes = sample.get("processes", [])
    if processes:
        values = [row["cpu"] for row in processes if row["cpu"] is not None]
        cpus = cfg.get("_cpus")
        if values and cpus:
            lines.append(f"CPU  {bar(sum(values) / cpus)} {sum(values):.1f}% trên {cpus} lõi cấp cho job "
                         "(100% ≈ 1 lõi)")
        else:
            lines.append(f"CPU train + con: {sum(values):.1f}% (100% ≈ 1 lõi)" if values else "CPU: đang lấy mẫu đầu")
        lines.append(f"RAM RSS main: {processes[0]['rss'] / 1024**3:.2f} GiB; không phải tổng RAM riêng của job")
        if show_processes:
            lines.append("    PID    ST     CPU%     RSS GiB    VAI TRÒ")
            for index, row in enumerate(processes):
                cpu = f"{row['cpu']:.1f}" if row["cpu"] is not None else "--"
                lines.append(f"{row['pid']:>9}    {row['state']}    {cpu:>7}     {row['rss'] / 1024**3:>7.2f}    "
                             + ("train" if index == 0 else "tiến trình con / worker"))
    lines += ["", sample.get("note", ""),
              "Refresh GPU/CPU theo màn hình; queue mỗi 15s. Monitor không dừng training."]
    return [line[:max(10, width - 1)] for line in lines]


class JobEnded(Exception):
    pass


def view(args):
    import termios
    import tty

    rows = queue(getpass.getuser())
    selected = next((row for row in rows if row["id"] == args.job), None)
    if selected is None or selected["state"] != "R":
        print("Job không còn R; chạy lại monitor để chọn job mới")
        return 2
    detail = command(["scontrol", "show", "job", args.job, "-o"])
    stdout, history_path, cfg = locate(args.repo, selected["name"], detail, args.job)
    probe = Probe(args.job)
    interval, processes, next_queue = args.interval, True, time.monotonic() + 15
    fd = sys.stdin.fileno()
    old_terminal = termios.tcgetattr(fd)
    def stopped(*_args):
        raise JobEnded()
    old_signal = signal.signal(signal.SIGTERM, stopped)
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[?1049h\033[?25l")
        while True:
            started = time.monotonic()
            error = ""
            try:
                if started >= next_queue:
                    rows = queue(getpass.getuser())
                    next_queue = started + 15
                    if not any(row["id"] == args.job and row["state"] == "R" for row in rows):
                        raise JobEnded()
                history = safe_json(history_path, [])
                p = progress(tail(stdout), history, cfg)
                sample = probe.sample()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                error = f"Chưa lấy được số đo mới: {exc}"
                sample = {"processes": [], "gpu": None, "note": error}
                history = safe_json(history_path, [])
                p = progress(tail(stdout), history, cfg)
            size = shutil.get_terminal_size((120, 35))
            lines = render(rows, args.job, p, cfg, history, sample, size.columns, processes)
            colored = []
            for line in lines[1:size.lines - 1]:
                color = ("\033[33m" if line.startswith("GPU ") else
                         "\033[34m" if line.startswith("VRAM") else
                         "\033[32m" if line.startswith("CPU ") else
                         "\033[35m" if line.startswith(">" + args.job) else "")
                colored.append(color + line + ("\033[0m" if color else ""))
            sys.stdout.write("\033[H\033[2J\033[36m" + lines[0] + "\033[0m\n" +
                             "\n".join(colored) + "\033[J")
            sys.stdout.flush()
            deadline = max(started + interval, time.monotonic() + 0.1)
            while time.monotonic() < deadline:
                ready, _, _ = select.select([sys.stdin], [], [], max(0, min(0.2, deadline - time.monotonic())))
                if ready:
                    key = os.read(fd, 1)
                    if key in (b"q", b"Q"):
                        return 0
                    if key == b"p":
                        processes = not processes
                    if key == b"+":
                        interval = max(3, interval - 1)
                    if key == b"-":
                        interval = min(30, interval + 1)
    except JobEnded:
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal)
        signal.signal(signal.SIGTERM, old_signal)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="Job đang R; mặc định tự chọn job cfg5-act")
    parser.add_argument("--repo", type=Path, default=Path.home() /
                        "cfg5-pretrain/edabk-physical-ai-robot/04-Src/software/cfg5_act")
    parser.add_argument("--interval", type=float, default=3)
    parser.add_argument("--on-node", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.interval = max(3, args.interval)
    if not sys.stdin.isatty():
        parser.error("Mở trong terminal SSH tương tác")
    if args.on_node:
        return view(args)
    rows = queue(getpass.getuser())
    selected = next((row for row in rows if row["state"] == "R" and row["name"].startswith("cfg5-act")
                     and (not args.job or row["id"] == args.job)), None)
    if selected is None:
        print("Chưa thấy job ACT đang R. Queue hiện tại:")
        for row in rows:
            print(row["id"], row["name"], row["state"], row["node"])
        return 0
    print(f"Mở dashboard cho job {selected['id']} trên {selected['node']}; q để thoát", flush=True)
    import termios
    original = termios.tcgetattr(sys.stdin.fileno())
    try:
        return subprocess.call(["srun", "--mpi=none", "--jobid=" + selected["id"], "--overlap",
                                "--exact", "-N1", "-n1", "-c1", "--pty", "python3",
                                str(Path(__file__).resolve()), "--on-node", "--job", selected["id"],
                                "--repo", str(args.repo), "--interval", str(args.interval)])
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
        print("Đã đóng monitor. Mở lại để chọn job hiện đang chạy.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
    except (OSError, subprocess.SubprocessError) as exc:
        sys.exit(f"Không mở được monitor: {exc}")
