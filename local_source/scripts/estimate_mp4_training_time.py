"""Đọc log/history để ước tính thời gian còn lại; không load model hay sửa job."""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PHASE = re.compile(r"^(train|val) batch=(\d+) samples=(\d+).*samples_per_s=([\d.]+)")
LOADER = re.compile(r"MP4_LOADER split=(train|val).*samples=(\d+)")
HEADER = re.compile(r"Đã lưu (\d+) epoch; chạy tới tổng (\d+)")


def parse_log(lines, windows=20):
    result = {"completed": 0, "started": False, "phase": None,
              "progress": {"train": 0, "val": 0}, "totals": {},
              "recent": {name: deque(maxlen=windows) for name in ("train", "val")}}
    previous = {"train": 0, "val": 0}
    for line in lines:
        match = LOADER.search(line)
        if match:
            result["totals"][match[1]] = int(match[2])
        match = HEADER.search(line)
        if match:
            result.update(completed=int(match[1]), started=True, phase=None)
            previous = {"train": 0, "val": 0}
            result["progress"] = previous.copy()
        match = PHASE.search(line)
        if match:
            name, _, count, rate = match.groups()
            count, rate = int(count), float(rate)
            delta = count - previous[name]
            if delta > 0 and rate > 0:
                result["recent"][name].append((delta, delta / rate))
            previous[name] = count
            result["progress"][name] = count
            result["phase"] = name
        if line.lstrip().startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in row and row.get("smoke") is False and "train" in row and "val" in row:
                result["completed"] = int(row["epoch"]) + 1
                result["phase"] = None
                previous = {"train": 0, "val": 0}
                result["progress"] = previous.copy()
    return result


def forecast(run, history, log):
    recent_epochs = [r for r in history if r.get("smoke") is False][-3:]
    saved = max((r["epoch"] + 1 for r in history if r.get("smoke") is False), default=0)
    completed = max(saved, log["completed"])
    phase = log["phase"] if log["completed"] >= saved else None
    remaining_epochs = max(0, run["epochs"] - completed)
    rates, full_phase_seconds, provenance = {}, {}, {}
    for name in ("train", "val"):
        rows = [r[name] for r in recent_epochs if r[name].get("timing", {}).get("epoch_seconds", 0) > 0]
        windows = log["recent"][name]
        if windows:
            rates[name] = sum(n for n, _ in windows) / sum(seconds for _, seconds in windows)
        elif rows:
            rates[name] = sum(r["samples"] for r in rows) / sum(r["timing"]["epoch_seconds"] for r in rows)
        else:
            rates[name] = None
        if rows:
            full_phase_seconds[name] = statistics.mean(r["timing"]["epoch_seconds"] for r in rows)
            provenance[name] = f"trung bình {len(rows)} epoch đã xong gần nhất"
        elif rates[name] and name in log["totals"]:
            full_phase_seconds[name] = log["totals"][name] / rates[name]
            provenance[name] = "ngoại suy từ tốc độ log gần đây; chưa có full epoch"
        else:
            full_phase_seconds[name] = None
            provenance[name] = "chưa có số đo"
    parts = {}
    for name in ("train", "val"):
        full = full_phase_seconds[name]
        if remaining_epochs == 0:
            parts[name] = 0.0
        elif full is None:
            parts[name] = None
        elif phase is None:
            parts[name] = remaining_epochs * full
        elif phase == "val" and name == "train":
            parts[name] = (remaining_epochs - 1) * full
        elif phase == name:
            total = log["totals"].get(name)
            if total is not None and rates[name]:
                current = max(0, total - log["progress"][name]) / rates[name]
            else:
                current = full
            parts[name] = current + (remaining_epochs - 1) * full
        else:  # Currently train: a complete validation is still due this epoch.
            parts[name] = remaining_epochs * full
    return {"saved": saved, "completed": completed, "phase": phase, "rates": rates,
            "full_seconds": full_phase_seconds, "provenance": provenance,
            "remaining_seconds": parts,
            "total_seconds": sum(parts.values()) if all(v is not None for v in parts.values()) else None}


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours} giờ {minutes:02d} phút {seconds:02d} giây"


def slurm_seconds(text):
    if text in ("UNLIMITED", "NOT_SET", "N/A", "INVALID"):
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    values = [int(v) for v in text.split(":")]
    if len(values) == 3:
        h, m, s = values
    elif len(values) == 2:
        h, (m, s) = 0, values
    else:
        h, m, s = 0, values[0], 0
    return days * 86400 + h * 3600 + m * 60 + s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--config", default="configs/act/pick_place_finetune_mp4_w16.json")
    parser.add_argument("--job", help="Mặc định đọc ~/cfg5-pretrain/mp4_train.jobid")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--windows", type=int, default=20, help="Số dòng tốc độ gần nhất mỗi phase")
    args = parser.parse_args()
    if args.windows < 1:
        parser.error("--windows phải >=1")
    try:
        run = json.loads((args.repo / args.config).read_text(encoding="utf-8"))
        if run.get("sample_backend") != "mp4_episode_cache_v1":
            raise ValueError("Config không phải run MP4 đã chuẩn bị")
        job = args.job or (Path.home() / "cfg5-pretrain/mp4_train.jobid").read_text().strip()
        if not job.isdigit():
            raise ValueError("Job ID không hợp lệ")
        path = args.log or Path.home() / f"cfg5-pretrain/logs/cfg5-ft-mp4-train-{job}.out"
        with path.open(encoding="utf-8", errors="replace") as stream:
            log = parse_log(stream, args.windows)
        history_path = args.repo / run["output_dir"] / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not log["started"]:
        print("Chưa thấy trainer bắt đầu; job có thể đang chờ/kiểm dữ liệu. Chưa đủ dữ liệu tính ETA.")
        return
    result = forecast(run, history, log)
    print(f"Job {job}; mục tiêu {run['epochs']} epoch; đã lưu {result['saved']} epoch")
    if result["phase"]:
        name = result["phase"]
        print(f"Epoch {result['completed'] + 1}: {name}, "
              f"sample đã log {log['progress'][name]}/{log['totals'].get(name, '?')}")
    else:
        print(f"Đã hoàn thành tính toán {result['completed']} epoch; có thể đang khởi động/ghi checkpoint")
    for name in ("train", "val"):
        rate, full, left = result["rates"][name], result["full_seconds"][name], result["remaining_seconds"][name]
        if rate and full is not None:
            print(f"{name}: {rate:.2f} mẫu/giây; một phase đầy đủ ≈ {duration(full)} "
                  f"({result['provenance'][name]})")
        elif rate:
            print(f"{name}: {rate:.2f} mẫu/giây; chưa biết tổng số mẫu")
        if left is not None:
            print(f"  {name} còn lại ≈ {duration(left)}")
        else:
            print(f"  {name}: chưa đủ số đo; chưa tính được phần này")
    now = datetime.now(timezone(timedelta(hours=7)))
    total = result["total_seconds"]
    if total is None:
        print("Chưa tính được giờ xong toàn run vì thiếu số đo validation/train. Chạy lại khi có log val hoặc xong epoch đầu.")
    else:
        print(f"TỔNG TRAIN + VAL CÒN LẠI ≈ {duration(total)}")
        print("DỰ KIẾN XONG (UTC+7):", (now + timedelta(seconds=total)).strftime("%d/%m/%Y %H:%M"))
    print("ETA là ước tính: chưa cộng thời gian ghi checkpoint/xếp hàng/chạy lại epoch bị ngắt. Log cập nhật mỗi100batch.")
    try:
        status = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T|%L"],
                                capture_output=True, text=True, timeout=5, check=True)
        if status.stdout.strip():
            state, left_text = status.stdout.strip().splitlines()[0].split("|")
            left = slurm_seconds(left_text.strip())
            print(f"Slurm: {state}; walltime còn {duration(left) if left is not None else left_text}")
            if state != "RUNNING":
                print("Job không ở RUNNING; giờ xong phía trên chỉ đúng nếu công việc tiếp tục ngay.")
            elif total is not None and left is not None and total > left:
                print("Ước tính vượt walltime còn lại: cần đăng ký lại job để resume latest.pt khi job kết thúc.")
    except (OSError, ValueError, subprocess.SubprocessError):
        print("Không đọc được squeue; job train đã cấu hình walltime tối đa72giờ, không phải72giờ còn lại.")


if __name__ == "__main__":
    main()
