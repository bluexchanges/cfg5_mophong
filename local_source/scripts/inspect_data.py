"""Khảo sát CFG5 chỉ đọc; mặc định chỉ dùng Python standard library.

Payload NPY dạng object chỉ được mở khi chọn --trusted-npy-samples.
Không giải mã video, không load toàn bộ dataset vào RAM.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import struct
import sys

EPISODE = re.compile(r"task_(\d+)_user_(\d+)_scene_(\d+)_cfg_(\d+)(_human(?:_2)?)?$")


def npy_header(path: Path) -> dict:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Không có magic header NPY")
        version = tuple(stream.read(2))
        if version not in ((1, 0), (2, 0), (3, 0)):
            raise ValueError(f"NPY version không hỗ trợ: {version}")
        size = 2 if version == (1, 0) else 4
        length = struct.unpack("<H" if size == 2 else "<I", stream.read(size))[0]
        if length > 65536:
            raise ValueError("NPY header vượt giới hạn khảo sát 64 KiB")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("NPY header bị cắt ngắn")
        header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1"))
    return {"version": list(version), **header}


def payload_summary(value, np, depth=0):
    if isinstance(value, np.ndarray):
        result = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if value.dtype.kind in "biuf":
            result["first_values"] = [payload_summary(v, np, depth + 1) for v in value.reshape(-1)[:14]]
        return result
    if isinstance(value, dict):
        items = list(value.items())[:3]
        return {"type": "dict", "length": len(value),
                "keys_first": [str(k) for k in list(value)[:10]],
                "examples": {str(k): payload_summary(v, np, depth + 1) for k, v in items} if depth < 4 else {}}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value),
                "first_values": [payload_summary(v, np, depth + 1) for v in value[:3]] if depth < 4 else []}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return str(value)
    return value if value is None or isinstance(value, (str, int, float, bool)) else repr(value)[:100]


def inspect_episode(path: Path, trust: bool) -> dict:
    result = {"episode": path.name, "entries": sorted(p.name for p in path.iterdir()), "npy": []}
    try:
        result["metadata"] = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result["metadata_error"] = str(exc)
    # Chỉ lấy camera đầu tiên, các file transformed và robot_command.
    cameras = sorted(path.glob("cam_*"))
    candidates = list((path / "transformed").glob("*.npy")) + list((path / "robot_command").glob("*.npy"))
    if cameras:
        candidates.extend(cameras[0].glob("*.npy"))
    for file in sorted(candidates):
        entry = {"file": file.relative_to(path).as_posix(), "bytes": file.stat().st_size}
        try:
            entry["header"] = npy_header(file)
            if trust and entry["bytes"] <= 5_000_000:
                import numpy as np
                value = np.load(file, allow_pickle=True)
                if value.shape == ():
                    value = value.item()
                entry["payload_sample"] = payload_summary(value, np)
                if file.name == "timestamps.npy" and isinstance(value, dict):
                    entry["timestamp_stats"] = {}
                    for kind, timestamps in value.items():
                        array = np.asarray(timestamps, dtype=np.float64)
                        if array.ndim != 1:
                            continue
                        delta = np.diff(array)
                        entry["timestamp_stats"][str(kind)] = {
                            "count": len(array),
                            "median_delta_raw_units": float(np.median(delta)) if len(delta) else None,
                            "strictly_increasing": bool(np.all(delta > 0)),
                        }
            elif trust:
                entry["payload_skipped"] = "File lớn hơn giới hạn 5 MB"
        except (OSError, ValueError, EOFError, SyntaxError, struct.error) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        result["npy"].append(entry)
    return result


def inspect(root: Path, samples: int, scan_files: bool, trust: bool) -> dict:
    episodes = []
    unexpected = []
    tasks, users, configs, kinds = Counter(), Counter(), Counter(), Counter()
    flat = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("task_")]
    nested = list((root / "training_data").glob("task_*/task_*"))
    nested += list((root / "missing_data").glob("task_*/*/task_*"))
    for path in sorted(flat + nested):
        match = EPISODE.fullmatch(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        task, user, _, cfg, human = match.groups()
        tasks[task] += 1
        users[user] += 1
        configs[cfg] += 1
        kinds["human" if human else "robot"] += 1
        episodes.append(path)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": str(root),
              "scope": "Directory inventory and bounded samples; not a complete integrity or synchronization audit.",
              "episode_directories": len(episodes), "kinds": dict(kinds), "task_count": len(tasks),
              "users": dict(users), "configs": dict(configs), "unexpected_directories": unexpected,
              "calibration_directory_exists": (root / "calib").is_dir(),
              "trusted_object_payloads": trust, "samples": []}
    robots = [p for p in episodes if not p.name.endswith("_human")]
    report["robot_file_availability"] = {}
    for name in ("joint.npy", "gripper.npy", "tcp.npy", "tcp_base.npy"):
        missing = [p.name for p in robots if not (p / "transformed" / name).is_file()]
        report["robot_file_availability"][name] = {"present": len(robots) - len(missing), "missing_episodes": missing}
    for human in (False, True):
        selected = [p for p in episodes if p.name.endswith("_human") == human][:samples]
        report["samples"].extend(inspect_episode(p, trust) for p in selected)
    if scan_files:
        extensions, names = Counter(), Counter()
        file_count, total_bytes = 0, 0
        for folder, _, files in os.walk(root):
            for name in files:
                file = Path(folder) / name
                file_count += 1
                total_bytes += file.stat().st_size
                extensions[file.suffix.lower() or "<none>"] += 1
                if name in ("color.mp4", "depth.mp4", "joint.npy", "gripper.npy", "tcp.npy", "tcp_base.npy"):
                    names[name] += 1
        report["file_inventory"] = {"files": file_count, "bytes": total_bytes, "extensions": dict(extensions), "selected_names": dict(names)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("RH20T_cfg5"))
    parser.add_argument("--output", type=Path, default=Path("docs/audit/project_inventory.json"))
    parser.add_argument("--samples-per-kind", type=int, default=1)
    parser.add_argument("--scan-files", action="store_true", help="Đếm file và dung lượng bằng metadata; không đọc nội dung video")
    parser.add_argument("--trusted-npy-samples", action="store_true", help="Cho phép NumPy mở object arrays của dữ liệu cục bộ tin cậy; cần NumPy")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f"Không tìm thấy dataset: {root}")
    if output.is_relative_to(root):
        parser.error("Output phải nằm ngoài dataset gốc")
    if not 0 <= args.samples_per_kind <= 10:
        parser.error("--samples-per-kind phải nằm trong 0..10")
    if args.trusted_npy_samples:
        try:
            import numpy  # noqa: F401
        except ImportError:
            parser.error("Chế độ payload cần NumPy; bỏ --trusted-npy-samples để khảo sát bằng Python chuẩn")
    report = inspect(root, args.samples_per_kind, args.scan_files, args.trusted_npy_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("episode_directories", "kinds", "task_count", "users")}, ensure_ascii=False))
    print(f"Report: {output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
