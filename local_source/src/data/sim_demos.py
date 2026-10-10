"""Verified MuJoCo pre-step exports: split before chunks; measured input, command target."""
from collections import Counter, defaultdict
import hashlib
import importlib
import os
import pickle
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np

from .common import digest, outside_raw, read_json, write_json
from .geometry import continuous_quaternions
from .splits import random_exact_group_assignment, target_counts, validate_splits

SAMPLING = "pre-step: RGB/state/command have one shared simulation tick; command applies during following step"
REPRESENTATION = "tcp_base_xyz_wxyz_gripper_width_m"


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            checksum.update(block)
    return checksum.hexdigest()


class NumpyTimestampUnpickler(pickle.Unpickler):
    """Legacy exporter uses object NPY. Permit NumPy reconstruction, no arbitrary globals."""
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(importlib.import_module("numpy.core.multiarray"), name)
        raise pickle.UnpicklingError(f"Timestamp pickle global bị chặn: {module}.{name}")


def read_camera_times(path):
    path = Path(path)
    if path.stat().st_size > 16 * 2**20:
        raise ValueError("Timestamp NPY quá lớn cho demo")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError("Timestamp NPY version chưa hỗ trợ")
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, _order, dtype = reader(stream)
        if dtype.hasobject:
            if shape != ():
                raise ValueError("Timestamp object phải là scalar dictionary")
            value = NumpyTimestampUnpickler(stream).load()
            if not isinstance(value, np.ndarray) or value.shape != ():
                raise ValueError("Timestamp object payload sai")
            value = value.item()
            if not isinstance(value, dict) or "color" not in value:
                raise ValueError("Timestamp dictionary thiếu color")
            times = np.asarray(value["color"])
        else:
            times = np.load(path, allow_pickle=False)
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("Timestamp color phải integer millisecond, vector 1D")
    return times.astype(np.int64)


def load_trajectory(episode, cfg):
    episode = Path(episode)
    meta = read_json(episode / "metadata.json")
    expected = {"schema_version": 1, "source_type": "mujoco_scripted_controller",
                "representation": REPRESENTATION, "timestamp_unit": "millisecond",
                "sampling": SAMPLING, "control_hz": cfg["control_hz"],
                "camera_order": ["cam_" + s for s in cfg["camera_serials"]]}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Demo metadata khác schema pre-step/8 camera/đơn vị đã kiểm")
    if meta.get("episode") != episode.name or type(meta.get("seed")) is not int:
        raise ValueError("Tên episode hoặc seed không hợp lệ")
    if meta.get("success") is not True or meta.get("failure") is not None:
        raise ValueError("Không dùng demo thất bại cho behavior cloning bản này")
    if cfg.get("camera_rig_hash") and meta.get("camera_rig_hash") != cfg["camera_rig_hash"]:
        raise ValueError("Camera rig khác nguồn đã khóa")
    path = episode / "robot_trajectory.npz"
    if path.stat().st_size > 64 * 2**20:
        raise ValueError("Trajectory NPZ quá lớn; cần adapter streaming riêng")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 64 * 2**20:
            raise ValueError("Trajectory NPZ giải nén quá lớn cho demo")
    with np.load(path, allow_pickle=False) as data:
        times = data["timestamps_ms"].copy()
        state = data["measured_tcp_gripper"].copy()
        action = data["commanded_tcp_gripper"].copy()
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.integer) or len(times) < 3:
        raise ValueError("Trajectory timestamp phải integer 1D, ít nhất 3 bước")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory timestamp trùng hoặc không tăng")
    tick = 1000 / cfg["control_hz"]
    if not tick.is_integer() or np.any(np.diff(times) % int(tick)):
        raise ValueError("Demo timestamp không nằm trên control grid")
    for key, array in (("state", state), ("action", action)):
        if array.shape != (len(times), 8) or not np.isfinite(array).all():
            raise ValueError(f"{key} phải hữu hạn (T,8)")
        norm = np.linalg.norm(array[:, 3:7], axis=1)
        if np.any(np.abs(norm - 1) > cfg["quaternion_norm_tolerance"]):
            raise ValueError(f"Quaternion {key} không có norm hợp lệ")
        lo, hi = cfg["gripper_valid_range_m"]
        if np.any(array[:, 7] < lo - 1e-6) or np.any(array[:, 7] > hi + 1e-6):
            raise ValueError(f"Gripper {key} ngoài phạm vi mét")
    if meta.get("frame_count_each_camera") != len(times):
        raise ValueError("Metadata/trajectory frame count khác nhau")
    # Gaps create separate segments. Never manufacture missing observations/targets.
    segments = np.r_[0, np.cumsum(np.diff(times) != int(tick))].astype(np.int32)
    for seg in np.unique(segments):
        rows = segments == seg
        state[rows, 3:7] = continuous_quaternions(state[rows, 3:7])
        action[rows, 3:7] = continuous_quaternions(action[rows, 3:7])
        # q and -q are equivalent; choose action signs consistent with measured orientation.
        flip = np.sum(state[rows, 3:7] * action[rows, 3:7], axis=1) < 0
        actions_q = action[rows, 3:7].copy()
        actions_q[flip] *= -1
        action[rows, 3:7] = actions_q
    return meta, times.astype(np.int64), state.astype(np.float32), action.astype(np.float32), segments


def fingerprint(episode, cameras, *, hash_videos=False):
    """Hash small source files; video size/mtime guard. Full video hash for duplicate names."""
    episode = Path(episode)
    files = [episode / name for name in ("metadata.json", "scene_config.json", "robot_trajectory.npz")]
    files += [episode / f"cam_{cam}" / name for cam in cameras for name in ("timestamps.npy", "color.mp4")]
    rows = []
    for path in files:
        stat = path.stat()
        row = {"path": path.relative_to(episode).as_posix(), "size": stat.st_size}
        if path.suffix != ".mp4" or hash_videos:
            row["sha256"] = file_hash(path)
        else:
            row["mtime_ns"] = stat.st_mtime_ns
        rows.append(row)
    return digest(rows)


def inspect_episode(episode, cfg):
    episode = Path(episode).resolve()
    meta, times, state, action, segments = load_trajectory(episode, cfg)
    video_infos = {}
    for camera in cfg["camera_serials"]:
        folder = episode / f"cam_{camera}"
        if not np.array_equal(read_camera_times(folder / "timestamps.npy"), times):
            raise ValueError(f"Camera {camera} không có cùng tick với state/command")
        cap = cv2.VideoCapture(str(folder / "color.mp4"))
        try:
            ok, frame = cap.read()
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not ok or frames != len(times):
                raise ValueError(f"Video {camera} lỗi/thiếu frame")
            if frame.shape != (cfg["image_height"], cfg["image_width"], 3):
                raise ValueError(f"Video {camera} resolution khác cfg")
            video_infos[camera] = {"frames": frames, "width": frame.shape[1], "height": frame.shape[0]}
        finally:
            cap.release()
    trajectory_hash = hashlib.sha256()
    for array in (times, state, action):
        trajectory_hash.update(array.tobytes())
    return {"episode": episode.name, "path": str(episode), "seed": meta["seed"],
            "profile": meta.get("variation", {}).get("profile", "varied" if meta.get("variation", {}).get("varied") else "fixed"),
            "camera_rig_hash": meta["camera_rig_hash"],
            "source_fingerprint": fingerprint(episode, cfg["camera_serials"]),
            "trajectory_hash": trajectory_hash.hexdigest(),
            "quality": {"steps": len(times), "segments": int(segments[-1] + 1), "videos": video_infos},
            "processed_image_bytes": len(times) * 8 * cfg["image_width"] * cfg["image_height"] * 3}


def discover_sources(roots, cfg):
    accepted, rejected, duplicates = [], [], []
    seen_names = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        episodes = ([root] if root.name.startswith("pick_place_") and (root / "metadata.json").exists()
                    else sorted(p for p in root.rglob("pick_place_*") if p.is_dir()))
        if not episodes:
            raise ValueError(f"Không có pick_place episode trong {root}")
        for episode in episodes:
            if not episode.resolve().is_relative_to(root):
                raise ValueError("Episode symlink ra ngoài source root")
            try:
                row = inspect_episode(episode, cfg)
                if row["episode"] in seen_names:
                    old = seen_names[row["episode"]]
                    if fingerprint(old["path"], cfg["camera_serials"], hash_videos=True) != fingerprint(episode, cfg["camera_serials"], hash_videos=True):
                        raise RuntimeError(f"Tên episode trùng nhưng nội dung khác: {old['path']} / {episode}")
                    duplicates.append({"episode": row["episode"], "kept": old["path"], "duplicate": str(episode)})
                    continue
                seen_names[row["episode"]] = row
                accepted.append(row)
                print(f"AUDIT {len(accepted)} {row['episode']} steps={row['quality']['steps']} profile={row['profile']}", flush=True)
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
                rejected.append({"path": str(episode), "reason": str(exc)})
    if len({row["camera_rig_hash"] for row in accepted}) != 1:
        raise ValueError("Không có episode hợp lệ hoặc có nhiều camera rig; cần khảo sát riêng")
    return accepted, rejected, duplicates


def assign_group_targets(items, desired, seed):
    """Prefer exact counts; when impossible, preserve groups and report actual counts."""
    try:
        return random_exact_group_assignment(items, desired, seed), True
    except ValueError:
        pass
    rng = random.Random(seed)
    items = sorted(items)
    rng.shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    names = ("train", "val", "test")
    assigned, counts = {name: [] for name in names}, [0, 0, 0]
    for pos, item in enumerate(items):
        empty = [i for i, count in enumerate(counts) if count == 0 and desired[i] > 0]
        choices = empty if len(items) - pos == len(empty) else list(range(3))
        size = len(item[1])
        def score(index):
            hypothetical = [count + (size if i == index else 0) for i, count in enumerate(counts)]
            return sum((count - goal)**2 / max(goal, 1) for count, goal in zip(hypothetical, desired))
        chosen = min(choices, key=lambda i: (score(i), i))
        assigned[names[chosen]].append(item)
        counts[chosen] += size
    return assigned, False


def split_records(records, cfg, manifest_hash):
    # Connected components: same generating seed OR same numeric trajectory always stay together.
    parent = {row["episode"]: row["episode"] for row in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for row in records:
        for key in (("seed", row["seed"]), ("trajectory", row["trajectory_hash"])):
            if key in seen:
                a, b = find(row["episode"]), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = row["episode"]
    groups = defaultdict(list)
    profiles = {}
    for row in records:
        group = find(row["episode"])
        groups[group].append(row["episode"])
        profiles[row["episode"]] = row["profile"]
    if len(groups) < 3:
        raise ValueError("Cần >=3 nhóm độc lập để chia train/val/test; hai mẫu local chỉ dùng kiểm adapter")
    strata = defaultdict(list)
    for group, episodes in sorted(groups.items()):
        strata["|".join(sorted({profiles[e] for e in episodes}))].append((group, sorted(episodes)))
    assigned = {name: [] for name in ("train", "val", "test")}
    rng = random.Random(cfg["seed"])
    scarce = {profile: items for profile, items in strata.items() if len(items) < 3}
    notes = []
    if scarce:
        # Repeated fixed demonstrations cannot become independent held-out samples.
        # Keep scarce strata in train; hold out independent groups from other strata.
        for profile, items in sorted(scarce.items()):
            assigned["train"].extend(items)
            notes.append(f"Profile {profile}: {len(items)} independent group(s), assigned wholly to train; no independent validation/test for this profile.")
        pool = sorted(item for profile, items in strata.items() if profile not in scarce for item in items)
        if len(pool) < 2:
            raise ValueError("Sau khi giữ profile ít nhóm ở train, cần >=2 nhóm khác cho val/test; không tách frame/group để tạo holdout")
        requested = target_counts(len(records), cfg["split_ratios"])
        pinned = sum(len(ids) for _, ids in assigned["train"])
        remaining = [requested[0] - pinned, requested[1], requested[2]]
        if remaining[0] < 0:
            available = sum(len(ids) for _, ids in pool)
            remaining = [0, (available + 1) // 2, available // 2]
            notes.append("Train-only groups exceed the requested train budget; actual ratio differs from 80/10/10.")
        part, exact = assign_group_targets(pool, remaining, rng.randrange(2**31))
        if not exact:
            notes.append("Exact episode counts infeasible; assigned whole groups with approximate counts.")
        for name in assigned:
            assigned[name].extend(part[name])
        method = "scarce_profiles_train_only_remaining_groups_seeded_assignment"
    else:
        for profile, items in sorted(strata.items()):
            desired = target_counts(sum(len(ids) for _, ids in items), cfg["split_ratios"])
            part, exact = assign_group_targets(items, desired, rng.randrange(2**31))
            if not exact:
                notes.append(f"Profile {profile}: exact counts infeasible; assigned whole groups with approximate counts.")
            for name in assigned:
                assigned[name].extend(part[name])
        method = "profile_stratified_group_assignment"
    result = {"schema_version": 1, "config_hash": digest(cfg), "manifest_hash": manifest_hash,
              "seed": cfg["seed"], "group_fields": ["seed_or_same_trajectory"],
              "assignment_method": method, "assignment_notes": notes,
              "requested_episode_counts": dict(zip(("train", "val", "test"), target_counts(len(records), cfg["split_ratios"]))),
              "train_only_profiles": sorted(scarce),
              "group_summary": {profile: {"independent_groups": len(items),
                                          "episodes": sum(len(ids) for _, ids in items)}
                                for profile, items in sorted(strata.items())},
              "groups": {name: sorted(group for group, _ in items) for name, items in assigned.items()},
              "splits": {name: sorted(e for _, ids in items for e in ids) for name, items in assigned.items()}}
    result["counts"] = {name: len(ids) for name, ids in result["splits"].items()}
    result["profiles"] = {name: dict(Counter(profiles[e] for e in ids)) for name, ids in result["splits"].items()}
    result["split_hash"] = digest(result)
    validate_splits(result)
    return result


def inherited_normalization(saved, checkpoint_hash, cfg, splits):
    source = saved["metadata"]["normalization"]
    if source["source_split"] != "train" or source["smoke_only"]:
        raise ValueError("Checkpoint không có train-only normalization full")
    return {"config_hash": digest(cfg), "split_hash": splits["split_hash"],
            "source_split": "train", "episodes": [], "missing_train_episodes": [], "smoke_only": False,
            "normalization_origin": "pretraining_train_checkpoint",
            "inherited_from": {"checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": saved["epoch"] + 1,
                               "stats_hash": digest(source), "source_split_hash": source["split_hash"],
                               "source_train_episodes": len(source["episodes"])},
            "state": source["state"], "action": source["action"],
            "note": "Numeric stats fit on CFG5 train only; no target val/test statistics fitted. Empty episodes refers to no target episodes fitted."}


def verify_processed(record, cfg, output, split_hash, max_steps=None):
    folder = outside_raw(output, cfg) / record["episode"]
    meta = read_json(folder / "episode.json")
    count = min(record["quality"]["steps"], max_steps) if max_steps else record["quality"]["steps"]
    expected = {"config_hash": digest(cfg), "split_hash": split_hash, "steps": count,
                "source_fingerprint": record["source_fingerprint"], "smoke_only": max_steps is not None,
                "action_target_offset": 0, "camera_serials": cfg["camera_serials"]}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Processed contract khác: {folder}")
    if fingerprint(record["path"], cfg["camera_serials"]) != record["source_fingerprint"]:
        raise ValueError("Raw đổi từ lúc tạo manifest")
    for key, shape, dtype in (("images", (count, 8, cfg["image_height"], cfg["image_width"], 3), np.uint8),
                              ("state", (count, 8), np.float32), ("action", (count, 8), np.float32),
                              ("timestamps_ms", (count,), np.int64), ("segment_ids", (count,), np.int32)):
        array = np.load(folder / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Processed shape/dtype sai: {key}")
    return meta


def preprocess_episode(record, cfg, output, split_hash, max_steps=None):
    episode = Path(record["path"])
    if fingerprint(episode, cfg["camera_serials"]) != record["source_fingerprint"]:
        raise ValueError("Raw đổi từ lúc tạo manifest")
    _meta, times, state, action, segments = load_trajectory(episode, cfg)
    count = min(len(times), max_steps) if max_steps else len(times)
    if count < 2:
        raise ValueError("Smoke phải có ít nhất 2 bước")
    output = outside_raw(output, cfg)
    output.mkdir(parents=True, exist_ok=True)
    final, partial = output / record["episode"], output / (record["episode"] + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(f"Đã có processed/partial: {final}")
    partial.mkdir()
    cv2.setNumThreads(1)
    try:
        for key, value in (("timestamps_ms", times), ("state", state), ("action", action), ("segment_ids", segments)):
            np.save(partial / f"{key}.npy", value[:count], allow_pickle=False)
        images = np.lib.format.open_memmap(partial / "images.npy", mode="w+", dtype=np.uint8,
                    shape=(count, 8, cfg["image_height"], cfg["image_width"], 3))
        try:
            for camera_id, camera in enumerate(cfg["camera_serials"]):
                folder = episode / f"cam_{camera}"
                if not np.array_equal(read_camera_times(folder / "timestamps.npy"), times):
                    raise ValueError("Camera timestamp đổi hoặc lệch tick")
                cap = cv2.VideoCapture(str(folder / "color.mp4"))
                try:
                    for row in range(count):
                        ok, frame = cap.read()
                        if not ok or frame.shape != (cfg["image_height"], cfg["image_width"], 3):
                            raise ValueError(f"Lỗi video {camera} frame={row}")
                        images[row, camera_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if max_steps is None and cap.read()[0]:
                        raise ValueError(f"Video có nhiều frame hơn trajectory: {camera}")
                finally:
                    cap.release()
            images.flush()
        finally:
            del images
        metadata = {"schema_version": 1, "episode": record["episode"], "steps": count,
                    "config_hash": digest(cfg), "split_hash": split_hash, "camera_serials": cfg["camera_serials"],
                    "source_fingerprint": record["source_fingerprint"], "representation": REPRESENTATION,
                    "smoke_only": max_steps is not None, "action_target_offset": 0,
                    "action_semantics": "commanded TCP base xyz/wxyz/gripper metres at observation tick; not measured future pose",
                    "quality": record["quality"]}
        write_json(partial / "episode.json", metadata)
        os.rename(partial, final)
        return metadata
    except Exception as exc:
        write_json(partial / "ERROR.json", {"error": str(exc), "incomplete": True})
        raise
