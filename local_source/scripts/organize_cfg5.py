"""Audit toàn CFG5, rồi MOVE episode theo task trên cùng ổ; không copy dữ liệu.

Tách --plan/--apply/--verify/--rollback để có manifest và resume an toàn.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time

import _bootstrap
from src.data.cfg5 import make_timeline, source_fingerprint
from src.data.common import ROOT, data_root, digest, load_config, read_json, write_json

EPISODE = re.compile(r"^task_(\d{4})_user_\d{4}_scene_\d{4}_cfg_0005(?:_human(?:_2)?)?$")
PLAN = ROOT / "docs/audit/cfg5_all_organization_plan.json"
PROGRESS = ROOT / "docs/audit/cfg5_all_organization_progress.json"


def checked_root(cfg):
    root = data_root(cfg)
    if root != (ROOT / "RH20T_cfg5").resolve() or not root.is_dir():
        raise ValueError("Chỉ hỗ trợ dataset root đã chỉ định trong workspace")
    return root


def assert_no_reparse(path):
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError(f"Symlink/junction cần xử lý riêng: {path}")


def footprint(folder):
    """Metadata only; no video/NPY payload read."""
    files = bytes_total = 0
    dirs = [folder]
    while dirs:
        current = dirs.pop()
        assert_no_reparse(current)
        with os.scandir(current) as scan:
            for entry in scan:
                path = Path(entry.path)
                assert_no_reparse(path)
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(path)
                elif entry.is_file(follow_symlinks=False):
                    size = entry.stat(follow_symlinks=False).st_size
                    files += 1; bytes_total += size
                else:
                    raise ValueError(f"File type không được hỗ trợ: {path}")
    return files, bytes_total


def classify(folder, cfg):
    match = EPISODE.fullmatch(folder.name)
    if not match:
        raise ValueError(f"Tên thư mục episode không nhận diện được: {folder.name}")
    task = f"task_{match.group(1)}"
    kind = "human" if "_human" in folder.name else "robot"
    status, reason, quality, fingerprint = "missing_data", None, None, None
    if kind == "human":
        reason = "not_direct_act_input:human_video_has_no_robot_8d_target"
    else:
        try:
            metadata = read_json(folder / "metadata.json")
            if metadata.get("rating", -1) < cfg["min_rating"]:
                raise ValueError("rating_below_minimum")
            missing = []
            for name in ("transformed/tcp_base.npy", "transformed/gripper.npy"):
                if not (folder / name).is_file(): missing.append(name)
            for serial in cfg["camera_serials"]:
                for suffix in ("color.mp4", "timestamps.npy"):
                    item = f"cam_{serial}/{suffix}"
                    if not (folder / item).is_file(): missing.append(item)
            if missing:
                raise ValueError("missing_required_files: " + ",".join(missing))
            timeline = make_timeline(folder, cfg, check_video=True)
            quality = timeline["quality"]
            fingerprint = source_fingerprint(folder, cfg)
            status = "training_data"
        except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
    fs = folder.stat()
    count, size = footprint(folder)
    subgroup = ("human_for_future_one_shot" if kind == "human" else "robot_incomplete")
    destination = (f"{status}/{task}/{folder.name}" if status == "training_data"
                   else f"{status}/{task}/{subgroup}/{folder.name}")
    counterpart = re.sub(r"_human(?:_2)?$", "", folder.name) if kind == "human" else None
    return {"episode": folder.name, "task": task, "kind": kind, "category": status,
            "reason": reason, "quality": quality, "source_fingerprint": fingerprint,
            "source_rel": folder.name, "destination_rel": destination, "robot_counterpart": counterpart,
            "files": count, "bytes": size, "directory_dev": fs.st_dev, "directory_inode": fs.st_ino}


def make_plan(cfg, workers):
    root = checked_root(cfg)
    if PLAN.exists() or PROGRESS.exists():
        raise FileExistsError("Plan/progress đã có; dùng --apply/--verify hoặc kiểm trước khi làm plan mới")
    if (root / "training_data").exists() or (root / "missing_data").exists():
        raise FileExistsError("Dataset đã được chia; không lập plan từ layout pha trộn")
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("task_"))
    if not folders: raise ValueError("Không thấy episode tại root")
    started = time.monotonic()
    print(f"Planning {len(folders)} episode folders, workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = []
        for record in pool.map(lambda p: classify(p, cfg), folders):
            records.append(record)
            if len(records) % 100 == 0:
                print(f"Checked {len(records)}/{len(folders)} in {time.monotonic()-started:.0f}s", flush=True)
    counts = Counter(r["category"] for r in records)
    kinds = Counter(r["kind"] for r in records)
    reasons = Counter(r["reason"] for r in records if r["reason"])
    by_task = defaultdict(lambda: {"training_data": 0, "missing_data": 0})
    for record in records: by_task[record["task"]][record["category"]] += 1
    plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(root), "data_config_hash": digest(cfg), "move_method": "same-volume directory rename; no file copy",
            "criteria": ["all_robot_tasks", "rating>=2", "8 specified RGB+timestamps", "tcp_base+gripper",
                         "video first frame/count matches timestamps", "causal aligned timeline with segment>=3"],
            "human_policy": "human videos retained intact under human_for_future_one_shot; not bad data, but no direct robot TCP+gripper target8D for ACT. RH20T paper proposes human-robot hierarchy, while reported ACT experiment trains on robot trajectories.",
            "counts": dict(counts), "kinds": dict(kinds), "reasons": dict(reasons),
            "tasks": {k: by_task[k] for k in sorted(by_task)},
            "file_count": sum(r["files"] for r in records), "total_bytes": sum(r["bytes"] for r in records),
            "episodes": records}
    plan["plan_hash"] = digest(plan)
    write_json(PLAN, plan)
    print(f"Plan saved: {PLAN}; categories={dict(counts)}; kinds={dict(kinds)}; {time.monotonic()-started:.0f}s", flush=True)


def load_plan(cfg):
    plan = read_json(PLAN)
    plan_hash = plan.pop("plan_hash")
    if digest(plan) != plan_hash or plan["data_config_hash"] != digest(cfg):
        raise ValueError("Plan hash/config không khớp")
    plan["plan_hash"] = plan_hash
    if plan["dataset_root"] != str(checked_root(cfg)):
        raise ValueError("Plan thuộc dataset root khác")
    return plan


def verified_destination(root, record):
    src = root / record["source_rel"]
    dst = root / record["destination_rel"]
    if src.exists() or not dst.is_dir():
        raise ValueError(f"Move chưa hoàn tất hoặc nguồn/đích trùng: {record['episode']}")
    assert_no_reparse(dst)
    info = dst.stat()
    if info.st_dev != record["directory_dev"] or info.st_ino != record["directory_inode"]:
        raise ValueError(f"Directory identity đổi, có thể đã copy/replace: {dst}")
    count, size = footprint(dst)
    if (count, size) != (record["files"], record["bytes"]):
        raise ValueError(f"File count/bytes khác plan: {dst}")


def apply(plan):
    root = Path(plan["dataset_root"])
    if Path(root.anchor).resolve() != Path((root / "training_data").anchor).resolve():
        raise ValueError("Đích không cùng volume")
    for category in ("training_data", "missing_data"):
        (root / category).mkdir(exist_ok=True)
        for task in plan["tasks"]:
            (root / category / task).mkdir(exist_ok=True)
    done = 0
    for record in plan["episodes"]:
        src = root / record["source_rel"]
        dst = root / record["destination_rel"]
        if src.is_dir() and not dst.exists():
            assert_no_reparse(src)
            info = src.stat()
            if info.st_dev != record["directory_dev"] or info.st_ino != record["directory_inode"]:
                raise ValueError(f"Nguồn đổi từ lúc lập plan: {src}")
            count, size = footprint(src)
            if (count, size) != (record["files"], record["bytes"]):
                raise ValueError(f"Nguồn đổi file count/bytes: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            verified_destination(root, record)
        elif not src.exists() and dst.is_dir():
            verified_destination(root, record)  # resume
        else:
            raise ValueError(f"Nguồn/đích không rõ; dừng, không ghi đè: {src} -> {dst}")
        done += 1
        if done % 50 == 0 or done == len(plan["episodes"]):
            write_json(PROGRESS, {"plan_hash": plan["plan_hash"], "verified_moves": done,
                                  "total": len(plan["episodes"]), "updated_utc": datetime.now(timezone.utc).isoformat()})
            print(f"Moved and verified {done}/{len(plan['episodes'])}", flush=True)


def verify(plan):
    root = Path(plan["dataset_root"])
    for i, record in enumerate(plan["episodes"], start=1):
        verified_destination(root, record)
        if i % 250 == 0: print(f"Verified {i}/{len(plan['episodes'])}", flush=True)
    roots = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("task_")]
    if roots: raise ValueError(f"Episode còn tại root: {roots[:10]}")
    report = {"plan_hash": plan["plan_hash"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "episode_count": len(plan["episodes"]), "category_counts": plan["counts"],
              "file_count": plan["file_count"], "total_bytes": plan["total_bytes"],
              "directory_identity_and_file_count_bytes_verified": True, "remaining_root_episodes": 0,
              "copy_operation_used": False, "training_started": False}
    write_json(ROOT / "docs/audit/cfg5_all_organization_verified.json", report)
    print("Verified all moves:", plan["counts"], "files", plan["file_count"], flush=True)


def rollback(plan):
    root = Path(plan["dataset_root"])
    for record in reversed(plan["episodes"]):
        src, dst = root / record["source_rel"], root / record["destination_rel"]
        if src.is_dir() and not dst.exists():
            continue
        if src.exists() or not dst.is_dir():
            raise ValueError(f"Rollback dừng vì nguồn/đích không rõ: {src} <- {dst}")
        verified_destination(root, record)
        dst.rename(src)
        if src.stat().st_ino != record["directory_inode"]:
            raise ValueError(f"Rollback identity sai: {src}")
    print("Returned all episode directories to root; plan remains for audit.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    for key in ("plan", "apply", "verify", "rollback"):
        options.add_argument("--" + key, action="store_true")
    parser.add_argument("--config", default="configs/data/cfg5_all8.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.workers not in range(1, 9): parser.error("--workers phải1..8")
    if args.plan: make_plan(cfg, args.workers)
    elif args.apply: apply(load_plan(cfg))
    elif args.verify: verify(load_plan(cfg))
    elif args.rollback: rollback(load_plan(cfg))
