"""Deferred JPEG materialization and verified per-episode feature retirement."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import stat

import numpy as np
import torch

from .common import digest, read_json
from .feature_cache import (BACKEND, FEATURE_ONLY, JPEG_ONLY, FEATURE_SHAPE, NUMERIC,
                            check_episode, open_source_frames, decode_cache_chunk, verify_source_after)
from .sim_demos import file_hash


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".atomic_tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_frozen_checkpoint(path, context):
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    meta = saved.get("metadata", {})
    if (saved.get("smoke") is not False or meta.get("stage") != "frozen"
            or meta.get("model_config", {}).get("backbone") != "mcr_resnet50"
            or meta.get("model_config", {}).get("lr_backbone") != 0
            or not saved.get("model") or saved.get("epoch", -1) < 0):
        raise ValueError("Cần checkpoint MCR+ACT frozen đã học, không dùng smoke/E5/E58/full")
    if (meta.get("feature_contract") != context["contract"]
            or digest(meta.get("data_config")) != digest(context["data_config"])
            or digest(meta.get("normalization")) != digest(context["stats"])):
        raise ValueError("Frozen checkpoint khác cache/data/stats")
    return {"checkpoint": str(path.resolve()), "sha256": file_hash(path), "epoch_completed": saved["epoch"] + 1}


def materialize_jpeg_episode(episode, source_root, cache_root, cfg, split, contract, backend,
                             *, batch_size=8, decoder_workers=8, max_jpeg_bytes=None):
    """Regenerate the exact JPEG input used by frozen extraction; never run ResNet."""
    folder = Path(cache_root) / episode
    meta = read_json(folder / "episode.json")
    final = folder / "rgb.sqlite"
    partial = folder / "rgb.sqlite.partial"
    receipt = folder / "JPEG_PENDING.json"
    if meta.get("image_storage") in (BACKEND, JPEG_ONLY):
        return check_episode(folder, cfg, split, contract, stage="full", full=True), "SKIP_VERIFIED", 0
    check_episode(folder, cfg, split, contract, stage="frozen")
    if partial.exists():
        raise FileExistsError(f"JPEG dở: {partial}; xem JPEG_ERROR.json trước khi xử lý")
    if final.exists():
        # Recover a crash between publishing the database and its metadata.
        pending = read_json(receipt)
        if (pending["source_metadata_hash"] != meta["source_metadata_hash"]
                or pending["contract_hash"] != digest(contract)
                or file_hash(final) != pending["jpeg_sha256"]):
            raise ValueError("JPEG publish receipt differs; features are preserved")
        updated = {**meta, "image_storage": BACKEND, "jpeg_sha256": pending["jpeg_sha256"],
                   "jpeg_file_bytes": pending["jpeg_file_bytes"]}
        atomic_json(folder / "episode.json", updated)
        check_episode(folder, cfg, split, contract, stage="full", full=True)
        receipt.unlink()
        return updated, "RECOVERED_PUBLISH", 0
    source, source_meta, numeric, count, caps, memmap, masks = open_source_frames(episode, source_root, cfg, split, backend)
    try:
        if digest(source_meta) != meta["source_metadata_hash"] or numeric != meta["numeric_sha256"]:
            raise ValueError("Light source changed since frozen extraction")
        if masks is not None and not np.array_equal(masks, np.load(folder / "camera_valid.npy", allow_pickle=False)):
            raise ValueError("Camera mask changed since frozen extraction")
        bytes_encoded = 0
        with closing(sqlite3.connect(partial)) as db, ThreadPoolExecutor(max_workers=decoder_workers) as executor:
            db.execute("CREATE TABLE rgb (tick INTEGER, camera INTEGER, jpeg BLOB NOT NULL, PRIMARY KEY(tick,camera)) WITHOUT ROWID")
            for start in range(0, count, batch_size):
                n = min(batch_size, count-start)
                decoded = decode_cache_chunk(caps, memmap, start, n, cfg, contract["jpeg_quality"], executor)
                rows = [(start+i, cam, decoded[cam][1][i]) for i in range(n) for cam in range(8) if decoded[cam][1][i] is not None]
                bytes_encoded += sum(len(row[2]) for row in rows)
                if max_jpeg_bytes is not None and bytes_encoded * 1.1 + 2**20 > max_jpeg_bytes:
                    raise OSError("JPEG episode vượt ngân sách còn lại; chưa xóa feature")
                db.executemany("INSERT INTO rgb VALUES (?,?,?)", rows)
            if backend == "mp4_episode_cache_v1" and any(cap.read()[0] for cap in caps):
                raise ValueError("Video dư frame")
            db.commit()
        verify_source_after(episode, source_root, cfg, split, backend, source_meta, numeric)
        if max_jpeg_bytes is not None and partial.stat().st_size > max_jpeg_bytes:
            raise OSError("JPEG database vượt ngân sách; chưa xóa feature")
        updated = {**meta, "image_storage": BACKEND, "jpeg_sha256": file_hash(partial),
                   "jpeg_file_bytes": partial.stat().st_size}
        atomic_json(receipt, {"source_metadata_hash": meta["source_metadata_hash"],
                             "contract_hash": digest(contract), "jpeg_sha256": updated["jpeg_sha256"],
                             "jpeg_file_bytes": updated["jpeg_file_bytes"]})
        os.replace(partial, final)
        atomic_json(folder / "episode.json", updated)
        check_episode(folder, cfg, split, contract, stage="full", full=True)
        receipt.unlink()
        return updated, "JPEG_CACHED", updated["jpeg_file_bytes"]
    except Exception as exc:
        atomic_json(folder / "JPEG_ERROR.json", {"error": str(exc), "features_preserved": True})
        raise
    finally:
        for cap in caps:
            cap.release()


def release_feature_episode(episode, cache_root, cfg, split, contract, origin):
    """Call under an exclusive cache lock, after verifying the frozen checkpoint."""
    root = Path(cache_root).resolve(strict=True)
    unresolved = Path(cache_root) / episode
    folder = unresolved.resolve(strict=True)
    if unresolved.is_symlink() or folder.parent != root:
        raise ValueError("Feature retirement path must be a direct episode under cache root")
    if (root / "READY_FROZEN.json").exists() or (root / "READY.json").exists():
        raise ValueError("Revoke frozen readiness before retiring any features")
    meta = check_episode(folder, cfg, split, contract, stage="full", full=True)
    candidates = []
    for filename, key, shape, dtype in (
            ("features.npy", "feature_file_bytes", (meta["steps"], *FEATURE_SHAPE), np.int8),
            ("scales.npy", "scale_file_bytes", (meta["steps"], 8, 2048, 1, 1), np.float16)):
        p = folder / filename
        if not p.exists() and not p.is_symlink():
            continue  # Resume after an interrupted deletion.
        info = p.lstat()
        if (p.is_symlink() or p.resolve() != p.absolute() or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1 or info.st_size != meta[key]):
            raise ValueError(f"Unexpected feature file, preserved: {p}")
        array = np.load(p, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Unexpected feature header, preserved: {p}")
        del array  # Release Windows memmaps before deleting test fixtures.
        candidates.append((p, info.st_size, info.st_mtime_ns, info.st_ino))
    meta = {**meta, "image_storage": JPEG_ONLY, "feature_release_status": "in_progress",
            "frozen_checkpoint": origin}
    atomic_json(folder / "episode.json", meta)
    freed = 0
    for p, size, mtime, inode in candidates:
        current = p.lstat()
        if p.is_symlink() or (current.st_size, current.st_mtime_ns, current.st_ino) != (size, mtime, inode):
            raise ValueError("Feature file changed before unlink; stopped")
        p.unlink()  # Only these two verified files, never recursive directory deletion.
        freed += size
    atomic_json(folder / "episode.json", {**meta, "feature_release_status": "complete"})
    atomic_json(folder / "FEATURE_RELEASE.json", {"checkpoint": origin, "released_filenames": ["features.npy", "scales.npy"],
                "raw_preserved": True, "numeric_preserved": True, "jpeg_verified_sha256": meta["jpeg_sha256"]})
    return freed


def jpeg_storage_projection(root, context):
    cfg, split, contract = context["data_config"], context["splits"], context["contract"]
    names = split["splits"]["train"] + split["splits"]["val"]
    measured, pending, releasable = [], [], 0
    for episode in names:
        folder = Path(root) / episode
        meta = read_json(folder / "episode.json")
        if meta["image_storage"] == FEATURE_ONLY:
            check_episode(folder, cfg, split, contract, stage="frozen")
            pending.append(meta["steps"])
        else:
            checked = check_episode(folder, cfg, split, contract, stage="full")
            measured.append(checked["jpeg_file_bytes"] / checked["steps"])
        for name in ("features.npy", "scales.npy"):
            p = folder / name
            if p.is_file():
                releasable += p.stat().st_size
    bytes_per_tick = max(measured)*1.5 if measured else cfg["image_width"]*cfg["image_height"]*3*8*1.2
    pending_bytes = int(sum(pending)*bytes_per_tick*1.1)
    largest = int(max(pending, default=0)*bytes_per_tick*1.1 + 2**30)
    return {"jpeg_completed_episodes": len(measured), "jpeg_pending_episodes": len(pending),
            "jpeg_pending_bytes": pending_bytes, "feature_releasable_bytes": releasable,
            "largest_episode_extra_bytes": largest,
            "estimated_peak_extra_bytes": max(0, pending_bytes-releasable)+largest,
            "jpeg_estimate_basis": "max measured bytes/tick x1.5" if measured else "uncompressed upper estimate"}
