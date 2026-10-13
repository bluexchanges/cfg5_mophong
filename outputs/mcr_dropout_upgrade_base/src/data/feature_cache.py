"""Spatial MCR cache + compressed RGB for later full fine-tuning, no MP4 at training."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path
import hashlib
import math
import shutil
import sqlite3
import time

import cv2
import numpy as np
import torch

from src.data.act_dataset import ACTDataset
from src.data.common import ROOT, digest, outside_raw, read_json
from src.data.mp4_dataset import verify_light
from src.data import cfg5_masked
from src.data.sim_demos import file_hash, read_camera_times
from src.models.mcr_act import MCR_SHA, PREPROCESS, preprocess_mcr
from src.training.pretrain import load_run, write_json

BACKEND = "mcr_spatial_int8_jpeg_v1"
FEATURE_ONLY = "mcr_spatial_int8_deferred_jpeg_v1"
JPEG_ONLY = "mcr_jpeg_after_frozen_v1"
FEATURE_SHAPE = (8, 2048, 7, 7)
NUMERIC = ("state", "action", "timestamps_ms", "segment_ids")


def context(run_path):
    run, model, cfg, split, stats, _ = load_run(ROOT / run_path)
    return run, model, cfg, split, stats


def cache_contract(cfg, split, stats, quality):
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality phải 1..100")
    contract = {"backend": BACKEND, "config_hash": digest(cfg), "split_hash": split["split_hash"],
            "stats_hash": digest(stats), "mcr_sha256": MCR_SHA, "preprocess": PREPROCESS,
            "feature_shape": list(FEATURE_SHAPE), "feature_dtype": "int8", "scale_dtype": "float16",
            "quantization": "symmetric_per_frame_camera_channel_maxabs_v1",
            "jpeg_quality": quality, "features_input": "JPEG decoded once before feature extraction"}
    if cfg.get("source_type") == cfg5_masked.BACKEND:
        contract["missing_views"] = "per_tick_camera_valid_attention_mask_v1"
    return contract


def require_ready(root, context, stage):
    names = context["splits"]["splits"]["train"] + context["splits"]["splits"]["val"]
    expected = {"contract_hash": digest(context["contract"]), "episodes": names, "contains_test": False}
    marker = Path(root) / ("READY_FROZEN.json" if stage == "frozen" else "READY_FULL.json")
    if not marker.exists() and context.get("storage_mode") != "features_only_v1":
        marker = Path(root) / "READY.json"  # Existing combined caches remain usable.
    if not marker.is_file() or read_json(marker) != expected:
        raise ValueError(f"Cache chưa READY cho stage {stage}: {marker}")
    return expected


def check_episode(folder, cfg, split, contract, *, full=False, stage=None):
    folder = Path(folder)
    meta = read_json(folder / "episode.json")
    if (meta["feature_contract"] != contract or meta["config_hash"] != digest(cfg)
            or meta["split_hash"] != split["split_hash"] or meta["smoke_only"] is not False
            or meta["camera_serials"] != cfg["camera_serials"]):
        raise ValueError(f"Cache contract khác: {folder}")
    storage = meta.get("image_storage")
    if storage not in (BACKEND, FEATURE_ONLY, JPEG_ONLY):
        raise ValueError("Unsupported MCR episode storage")
    if stage not in (None, "frozen", "full"):
        raise ValueError("Stage phải frozen/full")
    want_features = stage == "frozen" or (stage is None and storage != JPEG_ONLY)
    want_jpeg = stage == "full" or (stage is None and storage != FEATURE_ONLY)
    if want_features and storage == JPEG_ONLY:
        raise ValueError("Features đã được dọn; không resume frozen với JPEG-only cache")
    if want_jpeg and storage == FEATURE_ONLY:
        raise ValueError("JPEG chưa được tạo; cache chỉ dùng cho frozen stage")
    for key in NUMERIC:
        if file_hash(folder / f"{key}.npy") != meta["numeric_sha256"][key]:
            raise ValueError(f"Numeric cache đã đổi {folder}/{key}")
    if want_features:
        features = np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
        if features.shape != (meta["steps"], *FEATURE_SHAPE) or features.dtype != np.int8:
            raise ValueError("Feature header sai")
        scales = np.load(folder / "scales.npy", mmap_mode="r", allow_pickle=False)
        if scales.shape != (meta["steps"], 8, 2048, 1, 1) or scales.dtype != np.float16 or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Feature scales sai")
        if (folder / "features.npy").stat().st_size != meta["feature_file_bytes"]:
            raise ValueError("Feature file bị thiếu/dư")
        if (folder / "scales.npy").stat().st_size != meta["scale_file_bytes"]:
            raise ValueError("Scale file bị thiếu/dư")
    expected_jpegs = meta["steps"] * 8
    if "missing_views" in contract:
        mask = np.load(folder / "camera_valid.npy", allow_pickle=False)
        if mask.dtype != bool or mask.shape != (meta["steps"], 8) or not mask.any(1).all():
            raise ValueError("Camera mask invalid")
        if file_hash(folder / "camera_valid.npy") != meta["camera_valid_sha256"]:
            raise ValueError("Camera mask changed")
        expected_jpegs = int(mask.sum())
    if want_jpeg:
        if (folder / "rgb.sqlite").stat().st_size != meta["jpeg_file_bytes"]:
            raise ValueError("JPEG file bị thiếu/dư")
        with closing(sqlite3.connect((folder / "rgb.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("SELECT count(*) FROM rgb").fetchone()[0] != expected_jpegs:
                raise ValueError("JPEG cache thiếu view/frame")
            if "missing_views" in contract:
                actual = np.zeros_like(mask)
                for tick, camera in db.execute("SELECT tick,camera FROM rgb"):
                    if not 0 <= tick < len(mask) or not 0 <= camera < 8:
                        raise ValueError("JPEG key outside mask")
                    actual[tick, camera] = True
                if not np.array_equal(actual, mask):
                    raise ValueError("JPEG rows differ from real camera mask")
            if full and db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("JPEG database lỗi")
    if full:
        if want_features:
            for chunk in meta["feature_chunks"]:
                array = np.ascontiguousarray(features[chunk["start"]:chunk["end"]])
                if hashlib.sha256(array.tobytes()).hexdigest() != chunk["sha256"]:
                    raise ValueError(f"Feature payload sai tại {chunk['start']}")
                if hashlib.sha256(np.ascontiguousarray(scales[chunk["start"]:chunk["end"]]).tobytes()).hexdigest() != chunk["scale_sha256"]:
                    raise ValueError("Scale payload SHA sai")
        if want_jpeg and file_hash(folder / "rgb.sqlite") != meta["jpeg_sha256"]:
            raise ValueError("JPEG payload SHA sai")
    return meta


def quantize_features(features):
    if features.shape[-3:] != (2048, 7, 7) or not np.isfinite(features).all():
        raise ValueError("Spatial features MCR sai")
    scales = np.maximum(np.max(np.abs(features), axis=(-2, -1), keepdims=True) / 127, 1e-6).astype(np.float16)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise FloatingPointError("Feature scale overflow/underflow")
    values = np.clip(np.rint(features / scales.astype(np.float32)), -127, 127).astype(np.int8)
    error = float(np.max(np.abs(features - values.astype(np.float32) * scales.astype(np.float32))))
    return values, scales, error


def jpg_chunk(cap, count, shape, quality):
    blobs, frames = [], []
    for _ in range(count):
        ok, bgr = cap.read()
        if not ok:
            raise ValueError("MP4 thiếu/sai frame")
        if bgr is None:
            blobs.append(None)
            frames.append(np.zeros(shape, dtype=np.uint8))
            continue
        if bgr.shape != shape:
            raise ValueError("MP4 thiếu/sai frame")
        good, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not good:
            raise ValueError("Không encode được JPEG")
        blob = encoded.tobytes()
        decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        blobs.append(blob)
    return np.stack(frames), blobs


def open_source_frames(episode, source_root, cfg, split, backend):
    source = Path(source_root) / episode
    source_meta = read_json(source / "episode.json")
    if source_meta["config_hash"] != digest(cfg) or source_meta["split_hash"] != split["split_hash"] or source_meta["smoke_only"]:
        raise ValueError("Source processed contract sai")
    arrays = {key: np.load(source / f"{key}.npy", allow_pickle=False, mmap_mode="r") for key in NUMERIC}
    numeric_before = {key: file_hash(source / f"{key}.npy") for key in NUMERIC}
    count = len(arrays["state"])
    if count != source_meta["steps"] or arrays["state"].shape != (count, 8):
        raise ValueError("Source numeric shape sai")
    caps, image_memmap, camera_valid = [], None, None
    if backend == cfg5_masked.BACKEND:
        cfg5_masked.verify_light(episode, cfg, source_root, split["split_hash"])
        camera_valid = np.load(source / "camera_valid.npy", allow_pickle=False)
        frame_indices = np.load(source / "frame_indices.npy", allow_pickle=False)
        class MissingFrames:
            def read(self): return True, None
            def release(self): pass
        try:
            for cam, serial in enumerate(cfg["camera_serials"]):
                caps.append(cfg5_masked.AlignedFrames(Path(source_meta["raw_path"]) / f"cam_{serial}/color.mp4",
                            frame_indices[:, cam], cfg) if camera_valid[:, cam].any() else MissingFrames())
        except Exception:
            for cap in caps: cap.release()
            raise
    elif backend == "mp4_episode_cache_v1":
        record = {"episode": episode, "path": source_meta["raw_path"],
                  "source_fingerprint": source_meta["source_fingerprint"], "quality": source_meta["quality"]}
        verify_light(record, cfg, source_root, split["split_hash"])
        for cam in cfg["camera_serials"]:
            raw = Path(source_meta["raw_path"]) / f"cam_{cam}"
            if not np.array_equal(read_camera_times(raw / "timestamps.npy"), arrays["timestamps_ms"]):
                for opened in caps:
                    opened.release()
                raise ValueError("Camera timestamp lệch")
            cap = cv2.VideoCapture(str(raw / "color.mp4"))
            caps.append(cap)
            if not cap.isOpened():
                for opened in caps:
                    opened.release()
                raise ValueError("MP4 không mở được")
    elif backend == "npy":
        image_memmap = np.load(source / "images.npy", mmap_mode="r", allow_pickle=False)
        if image_memmap.shape != (count, 8, cfg["image_height"], cfg["image_width"], 3) or image_memmap.dtype != np.uint8:
            raise ValueError("Source NPY images shape/dtype sai")
    else:
        raise ValueError("Unsupported cache source backend")
    return source, source_meta, numeric_before, count, caps, image_memmap, camera_valid


def decode_cache_chunk(caps, image_memmap, start, count, cfg, quality, executor):
    if caps:
        futures = [executor.submit(jpg_chunk, cap, count, (cfg["image_height"], cfg["image_width"], 3), quality) for cap in caps]
        return [future.result() for future in futures]
    def encode_camera(camera):
        class Frames:
            def __init__(self, data): self.data, self.row = data, 0
            def read(self):
                frame = cv2.cvtColor(self.data[self.row], cv2.COLOR_RGB2BGR)
                self.row += 1
                return True, frame
        return jpg_chunk(Frames(image_memmap[start:start+count, camera]), count,
                         (cfg["image_height"], cfg["image_width"], 3), quality)
    return list(executor.map(encode_camera, range(8)))


def verify_source_after(episode, source_root, cfg, split, backend, source_meta, numeric_before):
    source = Path(source_root) / episode
    if backend == cfg5_masked.BACKEND:
        cfg5_masked.verify_light(episode, cfg, source_root, split["split_hash"])
    elif backend == "mp4_episode_cache_v1":
        record = {"episode": episode, "path": source_meta["raw_path"],
                  "source_fingerprint": source_meta["source_fingerprint"], "quality": source_meta["quality"]}
        verify_light(record, cfg, source_root, split["split_hash"])
    elif digest(read_json(source / "episode.json")) != digest(source_meta):
        raise ValueError("Source metadata đổi trong khi cache")
    if any(file_hash(source / f"{key}.npy") != numeric_before[key] for key in NUMERIC):
        raise ValueError("Source numeric đổi trong khi cache")


def cache_episode(episode, source_root, target_root, cfg, split, contract, policy, device,
                  *, backend="mp4_episode_cache_v1", batch_size=8, decoder_workers=8, resume=False, save_jpeg=True):
    final = Path(target_root) / episode
    partial = final.with_name(final.name + ".partial")
    if final.exists():
        if not resume:
            raise FileExistsError(final)
        return check_episode(final, cfg, split, contract, stage="frozen" if not save_jpeg else None), "SKIP_VERIFIED"
    if partial.exists():
        raise FileExistsError(f"Episode dở: {partial}; kiểm ERROR.json, giữ lại để audit trước khi thử lại")
    source, source_meta, numeric_before, count, caps, image_memmap, camera_valid = open_source_frames(episode, source_root, cfg, split, backend)
    partial.mkdir(parents=True)
    feature_chunks = []
    started = time.perf_counter()
    try:
        for key in NUMERIC:
            shutil.copyfile(source / f"{key}.npy", partial / f"{key}.npy")
        if camera_valid is not None:
            shutil.copyfile(source / "camera_valid.npy", partial / "camera_valid.npy")
        feature_file = np.lib.format.open_memmap(partial / "features.npy", mode="w+", dtype=np.int8,
                                               shape=(count, *FEATURE_SHAPE))
        scale_file = np.lib.format.open_memmap(partial / "scales.npy", mode="w+", dtype=np.float16,
                                             shape=(count, 8, 2048, 1, 1))
        quantization_max_error = 0.0
        database = closing(sqlite3.connect(partial / "rgb.sqlite")) if save_jpeg else nullcontext(None)
        with database as db, ThreadPoolExecutor(max_workers=decoder_workers) as executor:
            if db is not None:
                db.execute("CREATE TABLE rgb (tick INTEGER, camera INTEGER, jpeg BLOB NOT NULL, PRIMARY KEY(tick,camera)) WITHOUT ROWID")
            for start in range(0, count, batch_size):
                n = min(batch_size, count - start)
                decoded = decode_cache_chunk(caps, image_memmap, start, n, cfg, contract["jpeg_quality"], executor)
                rgb = np.stack([row[0] for row in decoded], axis=1)
                if db is not None:
                    db.executemany("INSERT INTO rgb VALUES (?,?,?)", [(start+i, cam, decoded[cam][1][i]) for i in range(n) for cam in range(8) if decoded[cam][1][i] is not None])
                tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 1, 4, 2, 3))).to(device)
                with torch.inference_mode():
                    valid_tensor = (torch.from_numpy(camera_valid[start:start+n]).to(device)
                                    if camera_valid is not None else None)
                    features = (policy.extract_features(tensor, camera_valid=valid_tensor) if valid_tensor is not None
                                else policy.extract_features(tensor)).cpu().numpy()
                if not np.isfinite(features).all():
                    raise FloatingPointError("MCR features overflow/NaN")
                values, scales, error = quantize_features(features)
                quantization_max_error = max(quantization_max_error, error)
                feature_file[start:start+n] = values
                scale_file[start:start+n] = scales
                feature_chunks.append({"start": start, "end": start+n, "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                                       "scale_sha256": hashlib.sha256(scales.tobytes()).hexdigest()})
            if backend == "mp4_episode_cache_v1" and caps and any(cap.read()[0] for cap in caps):
                raise ValueError("Video dư frame")
            if db is not None:
                db.commit()
        feature_file.flush()
        scale_file.flush()
        del feature_file, scale_file
        verify_source_after(episode, source_root, cfg, split, backend, source_meta, numeric_before)
        if any(file_hash(source / f"{key}.npy") != numeric_before[key] or file_hash(partial / f"{key}.npy") != numeric_before[key] for key in NUMERIC):
            raise ValueError("Source numeric đổi trong khi cache")
        meta = {**source_meta, "image_storage": BACKEND if save_jpeg else FEATURE_ONLY, "feature_contract": contract,
                "source_metadata_hash": digest(source_meta), "feature_file_bytes": (partial / "features.npy").stat().st_size,
                "scale_file_bytes": (partial / "scales.npy").stat().st_size,
                "feature_chunks": feature_chunks, "jpeg_sha256": file_hash(partial / "rgb.sqlite") if save_jpeg else None,
                "jpeg_file_bytes": (partial / "rgb.sqlite").stat().st_size if save_jpeg else 0,
                "quantization_max_absolute_error": quantization_max_error,
                "numeric_sha256": {key: file_hash(partial / f"{key}.npy") for key in NUMERIC},
                "cache_seconds": time.perf_counter() - started}
        if camera_valid is not None:
            meta["camera_valid_sha256"] = file_hash(partial / "camera_valid.npy")
        write_json(partial / "episode.json", meta)
        partial.rename(final)
        return meta, "CACHED"
    except Exception as exc:
        write_json(partial / "ERROR.json", {"error": str(exc), "incomplete": True})
        raise
    finally:
        for cap in caps:
            cap.release()


class JPEGFrames:
    def __init__(self, path, masks=None, shape=None):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.masks, self.shape = masks, shape

    def __getitem__(self, tick):
        rows = self.db.execute("SELECT camera,jpeg FROM rgb WHERE tick=? ORDER BY camera", (int(tick),)).fetchall()
        expected = np.flatnonzero(self.masks[int(tick)]).tolist() if self.masks is not None else list(range(8))
        if [r[0] for r in rows] != expected:
            raise ValueError("JPEG cache thiếu camera")
        images = np.zeros((8, *self.shape), np.uint8) if self.masks is not None else []
        for camera, blob in rows:
            frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG decode lỗi")
            if self.masks is not None:
                if frame.shape != self.shape:
                    raise ValueError("JPEG shape differs from data config")
                images[camera] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(images)

    def __del__(self):
        if hasattr(self, "db"):
            self.db.close()


class FeatureACTDataset(ACTDataset):
    def __init__(self, *args, feature_contract, stage="frozen", **kwargs):
        if stage not in ("frozen", "full"):
            raise ValueError("Stage phải frozen/full")
        self.stage = stage
        super().__init__(*args, **kwargs)
        for episode in self.episodes:
            meta = read_json(self.root / episode / "episode.json")
            storage = meta.get("image_storage")
            if meta.get("feature_contract") != feature_contract or storage not in (BACKEND, FEATURE_ONLY, JPEG_ONLY):
                raise ValueError("Cache dùng backbone/transform/split/stats khác")
            if self.stage == "frozen":
                if storage == JPEG_ONLY:
                    raise ValueError("Features đã dọn; không resume frozen")
                shape = np.load(self.root / episode / "features.npy", mmap_mode="r", allow_pickle=False).shape
                if shape != (meta["steps"], *FEATURE_SHAPE):
                    raise ValueError("Cached features shape sai")
            elif storage == FEATURE_ONLY or not (self.root / episode / "rgb.sqlite").is_file():
                raise ValueError("Full stage cần JPEG, không cần features.npy")

    def _load_arrays(self, episode):
        folder = self.root / episode
        arrays = {key: np.load(folder / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                  for key in ("state", "action", "timestamps_ms")}
        if self.cfg.get("source_type") == cfg5_masked.BACKEND:
            arrays["camera_valid"] = np.load(folder / "camera_valid.npy", mmap_mode="r", allow_pickle=False)
        arrays["images"] = (np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
                            if self.stage == "frozen" else JPEGFrames(folder / "rgb.sqlite", arrays.get("camera_valid"),
                              (self.cfg["image_height"], self.cfg["image_width"], 3)))
        if self.stage == "frozen":
            arrays["scales"] = np.load(folder / "scales.npy", mmap_mode="r", allow_pickle=False)
        return arrays

    def _prepare_images(self, images):
        if self.stage == "frozen":
            # ACTDataset calls this for the index currently fetched.
            return np.array(images, dtype=np.int8, copy=True)
        tensor = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
        return preprocess_mcr(tensor).numpy()

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        if self.stage == "frozen":
            _, start, _ = self.index[index]
            batch["images"] = batch["images"].astype(np.float32) * self._cache[1]["scales"][start].astype(np.float32)
        if "camera_valid" in self._cache[1]:
            _, start, _ = self.index[index]
            batch["camera_valid"] = np.array(self._cache[1]["camera_valid"][start], copy=True)
        return batch
