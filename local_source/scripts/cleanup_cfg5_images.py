"""Audit/delete only locked CFG5 train+val images.npy; never remove directories."""
import argparse
import ast
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import subprocess


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_path(path, root, *, directory=False):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not path.is_relative_to(root):
        raise ValueError(f"Đường dẫn symlink/ngoài phạm vi: {path}")
    info = path.stat()
    if directory and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Không phải thư mục: {path}")
    if not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise ValueError(f"Không phải file thường độc lập: {path}")
    return path


def split_names(split, expected=(531, 66, 66)):
    groups = split["splits"]
    all_names = []
    for label, count in zip(("train", "val", "test"), expected):
        names = groups[label]
        if len(names) != count or any(not isinstance(n, str) or not re.fullmatch(r"task_[0-9]+_user_[0-9]+_scene_[0-9]+_cfg_0005", n) for n in names):
            raise ValueError(f"Membership CFG5 {label} không đúng số lượng/tên episode")
        all_names.extend(names)
    if len(all_names) != len(set(all_names)):
        raise ValueError("Episode trùng hoặc train/val/test chồng nhau")
    return groups["train"] + groups["val"]


def signature(path):
    s = path.stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "device": s.st_dev, "inode": s.st_ino}


def check_image_header(path, steps, cfg):
    # Read only the NPY header, never image pixels or a pickle.
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Không phải NPY: {path}")
        version = stream.read(2)
        if version == b"\x01\x00":
            length = struct.unpack("<H", stream.read(2))[0]
        elif version in (b"\x02\x00", b"\x03\x00"):
            length = struct.unpack("<I", stream.read(4))[0]
        else:
            raise ValueError(f"NPY version không hỗ trợ: {path}")
        if length > 65536:
            raise ValueError("NPY header quá lớn")
        header = ast.literal_eval(stream.read(length).decode("utf-8" if version[0] == 3 else "latin1").strip())
        offset = stream.tell()
    shape = (steps, 8, cfg["image_height"], cfg["image_width"], 3)
    if header != {"descr": "|u1", "fortran_order": False, "shape": shape}:
        raise ValueError(f"Header ảnh không khớp metadata: {path}")
    size = steps * 8 * cfg["image_height"] * cfg["image_width"] * 3
    if path.stat().st_size != offset + size:
        raise ValueError(f"Ảnh thiếu/dư byte: {path}")


def collect_images(source, split, cfg, *, expected=(531, 66, 66)):
    candidates, missing = [], []
    for name in split_names(split, expected):
        folder = checked_path(source / name, source, directory=True)
        meta = read_json(checked_path(folder / "episode.json", source))
        if meta.get("split_hash") != split["split_hash"] or meta.get("config_hash") != digest(cfg) or meta.get("smoke_only") is not False:
            raise ValueError(f"Contract episode không khớp: {folder}")
        if meta.get("camera_serials") != cfg["camera_serials"] or type(meta.get("steps")) is not int or meta["steps"] < 1:
            raise ValueError(f"Camera/steps episode không khớp: {folder}")
        # Numeric labels and metadata stay where they are.
        for filename in ("state.npy", "action.npy", "timestamps_ms.npy", "segment_ids.npy"):
            checked_path(folder / filename, source)
        image = folder / "images.npy"
        if not image.exists() and not image.is_symlink():
            missing.append(name)
            continue
        checked_path(image, source)
        check_image_header(image, meta["steps"], cfg)
        candidates.append({"path": str(image), **signature(image)})
    return candidates, missing


def check_jobs(allowed):
    if set(allowed) & {"14127", "14128"}:
        raise ValueError("Không được cho phép job dùng CFG5 cũ")
    result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%A|%T|%j"],
                            capture_output=True, text=True, check=True, timeout=20)
    active = []
    for line in result.stdout.splitlines():
        job, state, name = (part.strip() for part in line.split("|", 2))
        if job not in allowed or name != "cfg5-mcr-cache":
            raise RuntimeError(f"Chưa dọn: job khác còn trong squeue: {job} {state} {name}")
        active.append({"job": job, "state": state, "name": name})
    return active


def delete_images(items, source, journal):
    # Validate the complete selection before the first deletion.
    for item in items:
        path = checked_path(Path(item["path"]), source)
        if path.name != "images.npy" or signature(path) != {k: item[k] for k in ("bytes", "mtime_ns", "device", "inode")}:
            raise ValueError(f"File đã đổi sau audit: {path}")
    freed = 0
    with journal.open("x", encoding="utf-8") as log:
        for item in items:
            path = checked_path(Path(item["path"]), source)
            if signature(path) != {k: item[k] for k in ("bytes", "mtime_ns", "device", "inode")}:
                raise ValueError(f"File đổi trước xoá: {path}")
            path.unlink()
            freed += item["bytes"]
            log.write(json.dumps({"deleted": str(path), "bytes": item["bytes"]}) + "\n")
            log.flush()
    return freed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--allow-job", action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="Không có cờ này: chỉ audit")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[0-9]+", j) for j in args.allow_job):
        parser.error("Job ID phải là số")
    repo = Path(args.repo).absolute()
    if repo.resolve(strict=True) != repo:
        raise ValueError("Repo phải là đường dẫn thật, không qua symlink")
    active = check_jobs(args.allow_job)
    old_run_path = repo / "configs/act/cfg5_all_pretrain_bs8_80epoch.json"
    new_run_path = repo / "configs/act/pick_place_finetune_mp4_w16.json"
    old, new = read_json(old_run_path), read_json(new_run_path)
    source = checked_path(repo / old["processed_root"], repo, directory=True)
    if source != repo / "data/processed/cfg5_all" or old.get("sample_backend", "npy") != "npy":
        raise ValueError("Root/backend CFG5 cũ không đúng phạm vi đã cho phép")
    new_source = (repo / new["processed_root"]).resolve(strict=True)
    if new.get("sample_backend") != "mp4_episode_cache_v1" or source == new_source or source.is_relative_to(new_source) or new_source.is_relative_to(source):
        raise ValueError("Preprocessing mới trùng/nằm trong root cũ hoặc không phải MP4")
    ctx = read_json(repo / "data/features/pick_place_mcr/context.json")
    if Path(ctx["source_root"]).resolve(strict=True) != new_source or ctx["source_config"] != "configs/act/pick_place_finetune_mp4_w16.json":
        raise ValueError("Cache mới chưa xác nhận dùng đúng preprocessing MP4 mới")
    model = read_json(repo / old["model_config"])
    cfg = read_json(repo / model["data_config"])
    if cfg["representation"] != "rh20t_aligned_tcp_base_wxyz_gripper_m":
        raise ValueError("Không phải representation CFG5 cũ")
    for config in (cfg, ctx["data_config"]):
        roots = [repo / config["dataset_root"], *(Path(p) for p in config.get("raw_sources", []))]
        if any(source.is_relative_to(p.resolve()) for p in roots):
            raise ValueError("Root dọn nằm trong dữ liệu gốc")
    split = read_json(repo / old["splits"])
    if split != read_json(repo / old["locked_membership"]) or split["config_hash"] != digest(cfg):
        raise ValueError("Split/config cũ khác membership đã khóa")
    # The selected initialization checkpoint must still exist; no torch load needed.
    checked_path(repo / "outputs/act_pretrain/best.pt", repo)
    items, missing = collect_images(source, split, cfg)
    selected = set(split["splits"]["train"] + split["splits"]["val"])
    protected = sorted(p.name for p in source.iterdir() if p.name not in selected)
    report = {"source": str(source), "new_source_preserved": str(new_source), "split_hash": split["split_hash"],
              "old_run_sha256": digest(old), "mode": "delete" if args.apply else "audit_only",
              "active_allowed_jobs": active, "files": items, "already_missing_images": missing,
              "protected_root_entries": protected, "bytes_to_free": sum(i["bytes"] for i in items)}
    audit_dir = repo / "outputs/data_cleanup"
    audit_dir.mkdir(parents=True, exist_ok=True)
    checked_path(audit_dir, repo, directory=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = audit_dir / f"cfg5_images_{stamp}.json"
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    total = report["bytes_to_free"]
    print(f"SOURCE={source}")
    print(f"SELECTED=train531+val66; test66 được giữ")
    print(f"FILES={len(items)} ALREADY_MISSING={len(missing)} FREE_GB={total/10**9:.3f} FREE_GiB={total/2**30:.3f}")
    print(f"NEW_SOURCE_PRESERVED={new_source}")
    print(f"AUDIT={report_path}")
    if args.apply:
        check_jobs(args.allow_job)
        freed = delete_images(items, source, audit_dir / f"cfg5_images_{stamp}.deleted.jsonl")
        print(f"CFG5_OLD_IMAGES_DELETED files={len(items)} freed_GiB={freed/2**30:.3f}")
    else:
        print("CFG5_CLEANUP_AUDIT_OK; chưa xoá file nào; thêm --apply để xoá đúng các ảnh đã kiểm")


if __name__ == "__main__":
    main()
