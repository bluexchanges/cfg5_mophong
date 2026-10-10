"""CFG5 RGB → observation 8 camera + TCP base 8D, đọc theo từng episode."""
from collections import Counter
import hashlib
from pathlib import Path
import re

import cv2
import numpy as np

from .common import data_root, episode_path, digest, outside_raw, read_json, write_json
from .geometry import continuous_quaternions

NAME = re.compile(r"task_(\d+)_user_(\d+)_scene_(\d+)_cfg_0005$")


def load_object(path):
    # Chỉ dùng với RH20T cục bộ do người dùng cung cấp; NPY object chứa pickle.
    value = np.load(path, allow_pickle=True)
    if value.shape != () or not isinstance(value.item(), dict):
        raise ValueError(f"Không phải scalar dictionary NPY: {path}")
    return value.item()


def camera_timestamps(folder):
    times = np.asarray(load_object(folder / "timestamps.npy")["color"], dtype=np.int64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Timestamp color rỗng/trùng/không tăng: {folder.name}")
    return times


def video_info(path):
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError(f"Không mở được video: {path}")
        ok, frame = video.read()
        if not ok:
            raise ValueError(f"Không decode được frame đầu: {path}")
        return {"frames": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                "container_fps": float(video.get(cv2.CAP_PROP_FPS)),
                "width": int(frame.shape[1]), "height": int(frame.shape[0])}
    finally:
        video.release()


def past_indices(source, query, max_age):
    """Sample-and-hold causal: không lấy ảnh/state tương lai."""
    indices = np.searchsorted(source, query, side="right") - 1
    safe = np.clip(indices, 0, len(source)-1)
    ages = query - source[safe]
    valid = (indices >= 0) & (ages >= 0) & (ages <= max_age)
    return safe, valid, ages


def source_fingerprint(ep, cfg):
    files = [ep / "metadata.json", ep / "transformed/tcp_base.npy", ep / "transformed/gripper.npy"]
    for camera in cfg["camera_serials"]:
        files.extend([ep / f"cam_{camera}/timestamps.npy", ep / f"cam_{camera}/color.mp4"])
    records = []
    for file in files:
        stat = file.stat()
        record = {"path": file.relative_to(ep).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if file.suffix != ".mp4":
            record["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        records.append(record)
    return digest(records)


def make_timeline(ep, cfg, check_video=False):
    metadata = read_json(ep / "metadata.json")
    camera_times, video_metadata = [], {}
    for camera in cfg["camera_serials"]:
        folder = ep / f"cam_{camera}"
        times = camera_timestamps(folder)
        if check_video:
            info = video_info(folder / "color.mp4")
            if info["frames"] != len(times):
                raise ValueError(f"Video/timestamp count không khớp {camera}: {info['frames']} != {len(times)}")
            info["median_timestamp_dt_ms"] = float(np.median(np.diff(times)))
            video_metadata[camera] = info
        camera_times.append(times)
    serial = cfg["state_camera_serial"]
    tcp = load_object(ep / "transformed/tcp_base.npy")[serial]
    grip = load_object(ep / "transformed/gripper.npy")[serial]
    poses, actions, times = [], [], []
    lo, hi = cfg["gripper_valid_range_m"]
    rejected = 0
    for item in sorted(tcp, key=lambda r: int(r["timestamp"])):
        timestamp = int(item["timestamp"])
        try:
            pose = np.asarray(item["tcp"], dtype=np.float64)
            measured = float(grip[timestamp]["gripper_info"][0]) * cfg["gripper_scale_to_m"]
            commanded = float(grip[timestamp]["gripper_command"][0]) * cfg["gripper_scale_to_m"]
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise ValueError("TCP không hữu hạn")
            if abs(np.linalg.norm(pose[3:]) - 1) > cfg["quaternion_norm_tolerance"]:
                raise ValueError("Quaternion norm sai")
            if not (lo <= measured <= hi and lo <= commanded <= hi):
                raise ValueError("Gripper sai phạm vi")
            if times and timestamp <= times[-1]:
                raise ValueError("Timestamp state trùng")
        except (KeyError, ValueError, TypeError, IndexError):
            rejected += 1
            continue
        times.append(timestamp)
        poses.append(np.r_[pose, measured])
        actions.append(np.r_[pose, commanded])
    if len(times) < 2:
        raise ValueError("Không đủ state hợp lệ")
    times = np.asarray(times, dtype=np.int64)
    poses, actions = np.asarray(poses), np.asarray(actions)
    tick = 1000.0 / cfg["control_hz"]
    if not float(tick).is_integer():
        raise ValueError("Bản này cần chu kỳ control nguyên millisecond")
    tick = int(tick)
    begin = max(times[0], *(t[0] for t in camera_times))
    end = min(times[-1], int(metadata["finish_time"]), *(t[-1] for t in camera_times))
    grid = np.arange(((begin+tick-1)//tick)*tick, end+1, tick, dtype=np.int64)
    if len(grid) < cfg["min_segment_steps"]:
        raise ValueError("Không có khoảng thời gian giao đủ dài cho 8 camera/state")
    state_idx, valid, state_ages = past_indices(times, grid, cfg["max_state_age_ms"])
    camera_indices, camera_ages = [], []
    for timestamps in camera_times:
        indices, ok, ages = past_indices(timestamps, grid, cfg["max_observation_age_ms"])
        valid &= ok
        camera_indices.append(indices)
        camera_ages.append(ages)
    # Không tạo chunk vượt qua đoạn mất đồng bộ; chỉ giữ đoạn đủ dài.
    keep, segments, segment_number = [], [], 0
    raw_indices = np.flatnonzero(valid)
    for indices in np.split(raw_indices, np.flatnonzero(np.diff(raw_indices) != 1)+1):
        if len(indices) >= cfg["min_segment_steps"]:
            keep.extend(indices.tolist())
            segments.extend([segment_number] * len(indices))
            segment_number += 1
    if not keep:
        raise ValueError("Không có segment đủ dài sau lọc độ trễ 8 camera")
    keep = np.asarray(keep)
    state, action = poses[state_idx[keep]].copy(), actions[state_idx[keep]].copy()
    segments = np.asarray(segments, dtype=np.int32)
    for segment in np.unique(segments):
        selected = segments == segment
        state[selected, 3:7] = continuous_quaternions(state[selected, 3:7])
        action[selected, 3:7] = state[selected, 3:7]
    frame_indices = np.stack(camera_indices, axis=1)[keep]
    capture_times = np.stack([ts[idx] for ts, idx in zip(camera_times, camera_indices)], axis=1)[keep]
    return {
        "timestamps_ms": grid[keep], "state": state.astype(np.float32), "action": action.astype(np.float32),
        "segment_ids": segments, "frame_indices": frame_indices, "camera_timestamps_ms": capture_times,
        "state_timestamps_ms": times[state_idx[keep]],
        "quality": {"steps": len(keep), "segments": segment_number, "grid_candidates": len(grid),
                    "rejected_state_records": rejected, "max_camera_age_ms": int(np.max(grid[keep, None]-capture_times)),
                    "max_state_age_ms": int(np.max(state_ages[keep])),
                    "segment_lengths": [int(np.sum(segments == s)) for s in range(segment_number)],
                    "videos": video_metadata},
    }


def build_manifest(cfg):
    accepted, rejected, skipped = [], [], Counter()
    root = data_root(cfg)
    episodes = list(root.glob("task_*"))
    episodes += list((root / "training_data").glob("task_*/task_*"))
    for ep in sorted(episodes):
        match = NAME.fullmatch(ep.name)
        if not match:
            skipped["human_or_alternate_name"] += 1
            continue
        task, user, scene = map(int, match.groups())
        if cfg["task_ids"] and task not in cfg["task_ids"]:
            skipped["task_not_selected"] += 1
            continue
        try:
            meta = read_json(ep / "metadata.json")
            if meta.get("rating", -1) < cfg["min_rating"]:
                raise ValueError("rating dưới ngưỡng")
            for name in cfg["camera_serials"]:
                if not (ep / f"cam_{name}/color.mp4").is_file():
                    raise ValueError(f"Thiếu RGB camera {name}")
            timeline = make_timeline(ep, cfg, check_video=True)
            accepted.append({"episode": ep.name, "task": task, "user": user, "scene": scene,
                             "calib": str(meta["calib"]), "rating": meta["rating"],
                             "source_fingerprint": source_fingerprint(ep, cfg), "quality": timeline["quality"]})
        except (OSError, ValueError, KeyError, EOFError, TypeError) as exc:
            rejected.append({"episode": ep.name, "reason": str(exc)})
    content = {"schema_version": 1, "config_hash": digest(cfg), "config": cfg,
               "episodes": accepted, "rejected": rejected, "skipped_counts": dict(skipped)}
    content["manifest_hash"] = digest(content)
    return content


def preprocess_episode(record, cfg, output_root, split_hash, max_steps=None):
    ep = episode_path(cfg, record["episode"])
    if source_fingerprint(ep, cfg) != record["source_fingerprint"]:
        raise ValueError("Raw files đổi từ lúc lập manifest; cần khảo sát lại")
    timeline = make_timeline(ep, cfg, check_video=True)
    count = len(timeline["timestamps_ms"])
    count = min(count, max_steps) if max_steps is not None else count
    if count < 2:
        raise ValueError("Cần ít nhất 2 bước cho observation/future target")
    output_root = outside_raw(output_root, cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / ep.name
    partial = output_root / (ep.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(f"Đã có output/partial: {final}; chọn output-root mới để tránh ghi đè")
    partial.mkdir()
    try:
        for key in ("timestamps_ms", "state", "action", "segment_ids", "frame_indices", "camera_timestamps_ms", "state_timestamps_ms"):
            np.save(partial / f"{key}.npy", timeline[key][:count], allow_pickle=False)
        shape = (count, 8, cfg["image_height"], cfg["image_width"], 3)
        images = np.lib.format.open_memmap(partial / "images.npy", mode="w+", dtype=np.uint8, shape=shape)
        for camera_index, camera in enumerate(cfg["camera_serials"]):
            indices = timeline["frame_indices"][:count, camera_index]
            cap = cv2.VideoCapture(str(ep / f"cam_{camera}/color.mp4"))
            current, last = -1, None
            try:
                for row, wanted in enumerate(indices):
                    if wanted != current:
                        while current < wanted:
                            if not cap.grab():
                                raise ValueError(f"Video kết thúc trước frame {wanted}: {camera}")
                            current += 1
                        ok, frame = cap.retrieve()
                        if not ok:
                            raise ValueError(f"Decode lỗi frame {wanted}: {camera}")
                        last = cv2.cvtColor(cv2.resize(frame, (cfg["image_width"], cfg["image_height"]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    images[row, camera_index] = last
            finally:
                cap.release()
        images.flush()
        del images
        metadata = {"schema_version": 1, "episode": ep.name, "config_hash": digest(cfg),
                    "split_hash": split_hash, "source_fingerprint": record["source_fingerprint"],
                    "representation": cfg["representation"], "camera_serials": cfg["camera_serials"],
                    "steps": count, "smoke_only": max_steps is not None, "quality": timeline["quality"],
                    "action_semantics": "future measured aligned TCP base pose plus future gripper command width; not recorded arm command"}
        write_json(partial / "episode.json", metadata)
        partial.rename(final)
        return metadata
    except Exception as exc:
        write_json(partial / "ERROR.json", {"error": str(exc), "incomplete": True})
        raise


def verify_processed_episode(record, cfg, output_root, split_hash, max_steps=None):
    """Only skip a completed episode whose source, contract and files still match."""
    folder = outside_raw(output_root, cfg) / record["episode"]
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    meta = read_json(folder / "episode.json")
    expected_steps = record["quality"]["steps"]
    if max_steps is not None:
        expected_steps = min(expected_steps, max_steps)
    expected = {"episode": record["episode"], "config_hash": digest(cfg),
                "split_hash": split_hash, "source_fingerprint": record["source_fingerprint"],
                "representation": cfg["representation"], "camera_serials": cfg["camera_serials"],
                "steps": expected_steps, "smoke_only": max_steps is not None}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Output đã có nhưng contract khác: {folder}")
    if meta.get("quality") != record["quality"]:
        raise ValueError(f"Output đã có nhưng timeline quality khác manifest: {folder}")
    if source_fingerprint(episode_path(cfg, record["episode"]), cfg) != record["source_fingerprint"]:
        raise ValueError(f"Raw source đổi từ lúc lập manifest: {record['episode']}")
    shapes = {"images": (expected_steps, 8, cfg["image_height"], cfg["image_width"], 3),
              "state": (expected_steps, 8), "action": (expected_steps, 8),
              "timestamps_ms": (expected_steps,), "segment_ids": (expected_steps,),
              "frame_indices": (expected_steps, 8),
              "camera_timestamps_ms": (expected_steps, 8),
              "state_timestamps_ms": (expected_steps,)}
    dtypes = {"images": np.uint8, "state": np.float32, "action": np.float32,
              "timestamps_ms": np.int64, "segment_ids": np.int32,
              "frame_indices": np.int64, "camera_timestamps_ms": np.int64,
              "state_timestamps_ms": np.int64}
    for name, shape in shapes.items():
        try:
            array = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Không đọc được processed {name}: {folder}") from exc
        if array.shape != shape or array.dtype != dtypes[name]:
            raise ValueError(f"Processed {name} shape/dtype sai: {array.shape}/{array.dtype} != {shape}/{dtypes[name]}")
    return meta
