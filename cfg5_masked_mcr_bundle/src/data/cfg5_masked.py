"""CFG5 robot: keep every episode; missing views are attention-masked.

Missing TCP is derived from recorded Panda joints, never from a guessed action.
Preparation stops with a report if any episode cannot supply training samples.
"""
from collections import Counter
import io
from pathlib import Path
import re
import zipfile

import cv2
import mujoco
import numpy as np

from .cfg5 import load_object, past_indices, video_info
from .common import ROOT, digest, outside_raw, read_json, write_json
from .geometry import continuous_quaternions, pose_matrix
from .sim_demos import file_hash

BACKEND = "cfg5_masked_raw_mp4_v1"
NAME = re.compile(r"task_(\d+)_user_(\d+)_scene_(\d+)_cfg_0005$")
KEYS = ("state", "action", "timestamps_ms", "segment_ids", "frame_indices", "camera_valid")


class PandaFK:
    def __init__(self):
        self.path = ROOT / "mujoco/models/franka_emika_panda/panda.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.adr = [self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
        # Panda hand -> RH20T processed TCP. Checked against measured TCP below.
        self.tool = np.diag([-1., -1., 1., 1.])
        self.tool[2, 3] = .1034

    def matrix(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or len(q) < 7 or not np.isfinite(q[:7]).all():
            raise ValueError("Joint record cannot supply Panda FK")
        self.data.qpos[self.adr] = q[:7]
        mujoco.mj_forward(self.model, self.data)
        hand = np.eye(4)
        hand[:3, :3] = self.data.body("hand").xmat.reshape(3, 3)
        hand[:3, 3] = self.data.body("hand").xpos
        return hand @ self.tool

    def pose(self, q):
        matrix = self.matrix(q)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, matrix[:3, :3].ravel())
        return np.r_[matrix[:3, 3], quat]


def discover(cfg):
    root = (ROOT / cfg["dataset_root"]).resolve()
    episodes, seen = [], set()
    for source in cfg["episode_subdirs"]:
        folder = (root / source).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError(f"Raw source missing/outside root: {folder}")
        for ep in sorted(folder.glob("task_*/task_*_cfg_0005")):
            if ep.is_dir() and NAME.fullmatch(ep.name):
                if not ep.resolve().is_relative_to(root) or ep.name in seen:
                    raise ValueError(f"Duplicate/outside raw episode: {ep}")
                seen.add(ep.name)
                episodes.append(ep)
    return sorted(episodes, key=lambda p: p.name)


def fingerprint(ep, cfg):
    files = [ep / "metadata.json"] + list((ep / "transformed").glob("*.npy"))
    for cam in cfg["camera_serials"]:
        files += [ep / f"cam_{cam}/timestamps.npy", ep / f"cam_{cam}/color.mp4"]
    records = []
    for p in sorted(files):
        row = {"path": p.relative_to(ep).as_posix(), "exists": p.is_file()}
        if p.is_file():
            s = p.stat()
            row.update(size=s.st_size, mtime_ns=s.st_mtime_ns)
            if p.suffix != ".mp4":
                row["sha256"] = file_hash(p)
        records.append(row)
    if not (ep / "transformed/joint.npy").is_file() and cfg.get("joint_repair_root"):
        repaired = ROOT / cfg["joint_repair_root"] / ep.name / "joint.npy"
        row = {"path": "supplemental_recorded_joint", "exists": repaired.is_file()}
        if repaired.is_file():
            s = repaired.stat()
            row.update(size=s.st_size, mtime_ns=s.st_mtime_ns, sha256=file_hash(repaired))
        records.append(row)
    return digest(records)


def restore_missing_joints(episodes, cfg, archive):
    """Read only original recorded joints from ZIP into a separate repair directory."""
    wanted = {ep.name for ep in episodes if not (ep / "transformed/joint.npy").is_file()}
    if not wanted:
        return {"needed": [], "restored": [], "unavailable": []}
    if not Path(archive).is_file():
        raise FileNotFoundError(f"Need original ZIP to restore recorded joints: {archive}")
    repair = outside_raw(ROOT / cfg["joint_repair_root"], cfg)
    found = {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[-2:] == ["transformed", "joint.npy"] and parts[-3] in wanted:
                name = parts[-3]
                if name in found and (found[name].CRC, found[name].file_size) != (info.CRC, info.file_size):
                    raise ValueError(f"Conflicting joint members in ZIP: {name}")
                found[name] = info
        restored = []
        for name, info in sorted(found.items()):
            data = z.read(info)  # zipfile validates CRC; no arbitrary archive paths are extracted.
            value = np.load(io.BytesIO(data), allow_pickle=True)
            if value.shape != () or not isinstance(value.item(), dict):
                raise ValueError(f"Recorded joint NPY schema invalid in ZIP: {name}")
            valid = 0
            for rows in value.item().values():
                if not isinstance(rows, dict):
                    continue
                for q in rows.values():
                    try:
                        array = np.asarray(q, dtype=float)
                        valid += bool(array.ndim == 1 and len(array) >= 7 and np.isfinite(array[:7]).all())
                    except (ValueError, TypeError):
                        pass
            if valid < 2:
                raise ValueError(f"ZIP has no usable recorded joints: {name}")
            folder = (repair / name).resolve()
            if not folder.is_relative_to(repair):
                raise ValueError("Repair path outside output")
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "joint.npy"
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError(f"Different supplemental joints already present: {target}")
            if not target.exists():
                temporary = folder / "joint.npy.partial"
                with temporary.open("xb") as handle:
                    handle.write(data)
                temporary.rename(target)
            write_json(folder / "source.json", {"archive": str(Path(archive).resolve()), "member": info.filename,
                       "crc32": info.CRC, "sha256": file_hash(target), "recorded_joint_records": valid})
            restored.append(name)
    return {"needed": sorted(wanted), "restored": restored, "unavailable": sorted(wanted-set(restored))}


def validate_fk(episodes, cfg, fk, min_episodes=20):
    errors, used = [], []
    used_tasks = set()
    serial = cfg["state_camera_serial"]
    for ep in episodes:
        task = NAME.fullmatch(ep.name).group(1)
        if task in used_tasks:
            continue
        tcp = load_object(ep / "transformed/tcp_base.npy").get(serial, [])
        joint = load_object(ep / "transformed/joint.npy").get(serial, {})
        current = []
        for item in tcp[::max(1, len(tcp)//8)]:
            timestamp = int(item["timestamp"])
            if item.get("tcp") is None or timestamp not in joint:
                continue
            try:
                expected = pose_matrix(item["tcp"])
                actual = fk.matrix(joint[timestamp])
                relative = expected[:3, :3].T @ actual[:3, :3]
                angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
                current.append((float(np.linalg.norm(expected[:3, 3]-actual[:3, 3])), float(angle)))
            except (ValueError, TypeError):
                continue
        if current:
            errors.extend(current)
            used.append(ep.name)
            used_tasks.add(task)
        if len(used) >= min_episodes:
            break
    if len(used) < min_episodes:
        raise ValueError("Not enough independent episodes to verify FK")
    values = np.asarray(errors)
    report = {"episodes": used, "points": len(errors), "max_position_error_m": float(values[:, 0].max()),
              "max_angle_error_rad": float(values[:, 1].max()), "tool_in_hand": fk.tool.tolist(),
              "panda_xml_sha256": file_hash(fk.path), "position_tolerance_m": .003,
              "angle_tolerance_rad": .03}
    if report["max_position_error_m"] > .003 or report["max_angle_error_rad"] > .03:
        raise ValueError(f"FK verification failed; cannot recover labels: {report}")
    return report


def timeline(ep, cfg, fk):
    meta = read_json(ep / "metadata.json")
    joint_target = cfg["representation"] == "panda_joint_positions_rad_gripper_width_m"
    tcp = {} if joint_target else load_object(ep / "transformed/tcp_base.npy")
    grips = load_object(ep / "transformed/gripper.npy")
    joint_file = ep / "transformed/joint.npy"
    if not joint_file.is_file() and cfg.get("joint_repair_root"):
        joint_file = ROOT / cfg["joint_repair_root"] / ep.name / "joint.npy"
    joints = load_object(joint_file) if joint_file.is_file() else {}
    preferred = cfg["state_camera_serial"]
    candidates = [preferred] + [s for s in cfg["camera_serials"] if s != preferred]
    state_rows, failures, chosen = [], Counter(), None
    for serial in candidates:
        rows, current = [], Counter()
        source_records = ([{"timestamp": t, "joint": q} for t, q in joints.get(serial, {}).items()]
                          if joint_target else tcp.get(serial, []))
        for item in source_records:
            t = int(item["timestamp"])
            try:
                g = grips[serial][t]
                measured = float(g["gripper_info"][0]) * cfg["gripper_scale_to_m"]
                command = float(g["gripper_command"][0]) * cfg["gripper_scale_to_m"]
                pose = np.asarray(item["joint"][:7], dtype=float) if joint_target else item.get("tcp")
                recovered = pose is None
                pose = fk.pose(joints[serial][t]) if recovered else np.asarray(pose, dtype=float)
                if pose.shape != (7,) or not np.isfinite(pose).all():
                    raise ValueError("nonfinite_pose")
                if not joint_target and abs(np.linalg.norm(pose[3:])-1) > cfg["quaternion_norm_tolerance"]:
                    raise ValueError("invalid_quaternion")
                lo, hi = cfg["gripper_valid_range_m"]
                if not lo <= measured <= hi or not lo <= command <= hi:
                    raise ValueError("invalid_gripper")
                rows.append((t, np.r_[pose, measured], np.r_[pose, command], recovered))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                current[type(exc).__name__ + ":" + str(exc)] += 1
        # No episode/rating exclusion. Unrecoverable state rows remain in report.
        if len(rows) >= 2:
            state_rows, failures, chosen = rows, current, serial
            break
        failures.update(current)
    if not state_rows:
        raise ValueError(f"No recorded {'joint' if joint_target else 'TCP'} / gripper stream: {dict(failures)}")
    state_rows.sort(key=lambda row: row[0])
    times = np.array([r[0] for r in state_rows], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Duplicate state timestamps need repair")
    state = np.array([r[1] for r in state_rows])
    action = np.array([r[2] for r in state_rows])
    camera_times, camera_issues = [], {}
    for cam in cfg["camera_serials"]:
        folder = ep / f"cam_{cam}"
        try:
            ts = np.asarray(load_object(folder / "timestamps.npy")["color"], dtype=np.int64)
            if ts.ndim != 1 or not len(ts) or np.any(np.diff(ts) <= 0):
                raise ValueError("invalid_timestamps")
            info = video_info(folder / "color.mp4")
            if info["frames"] != len(ts):
                raise ValueError("video_timestamp_count_mismatch")
            camera_times.append(ts)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            camera_times.append(None)
            camera_issues[cam] = str(exc)
    period = 1000 / cfg["control_hz"]
    if not period.is_integer():
        raise ValueError("Control period must be integer milliseconds")
    period = int(period)
    # Camera absence/end never shortens other views' usable intervals.
    finish = meta.get("finish_time", -1)
    finish_known = isinstance(finish, (int, float)) and np.isfinite(finish) and finish >= times[0]
    # -1 means annotation unavailable, not an instruction to discard the recording.
    end = min(times[-1], int(finish)) if finish_known else times[-1]
    grid = np.arange(((times[0]+period-1)//period)*period, end+1, period, dtype=np.int64)
    state_idx, good, _ = past_indices(times, grid, cfg["max_state_age_ms"])
    indices = np.full((len(grid), 8), -1, dtype=np.int64)
    valid = np.zeros((len(grid), 8), dtype=bool)
    for cam, ts in enumerate(camera_times):
        if ts is not None:
            idx, ok, _ = past_indices(ts, grid, cfg["max_observation_age_ms"])
            valid[:, cam] = ok
            indices[ok, cam] = idx[ok]
    good &= valid.any(1)
    kept, segment_ids = [], []
    raw = np.flatnonzero(good)
    for part in np.split(raw, np.flatnonzero(np.diff(raw) != 1)+1):
        if len(part) >= 2:
            segment_ids.extend([len(set(segment_ids))] * len(part))
            kept.extend(part.tolist())
    if not kept:
        raise ValueError("No two consecutive ticks with real state/action and at least one real camera")
    kept = np.asarray(kept)
    segment_ids = np.asarray(segment_ids, dtype=np.int32)
    state, action = state[state_idx[kept]].copy(), action[state_idx[kept]].copy()
    if not joint_target:
        for seg in np.unique(segment_ids):
            ix = segment_ids == seg
            state[ix, 3:7] = continuous_quaternions(state[ix, 3:7])
            action[ix, 3:7] = state[ix, 3:7]
    quality = {"steps": len(kept), "state_serial": chosen, "fk_recovered_records": sum(r[3] for r in state_rows),
               "invalid_state_records": dict(failures), "camera_issues": camera_issues,
               "camera_valid_ticks": valid[kept].sum(0).tolist(), "grid_candidates": len(grid),
               "unused_ticks": len(grid)-len(kept), "rating": meta.get("rating"), "rating_filter_applied": False,
               "finish_time_source": "metadata" if finish_known else "recorded_state_end",
               "original_finish_time": finish}
    arrays = {"state": state.astype(np.float32), "action": action.astype(np.float32), "timestamps_ms": grid[kept],
              "segment_ids": segment_ids, "frame_indices": indices[kept], "camera_valid": valid[kept]}
    return arrays, quality


def verify_light(episode, cfg, source, split_hash):
    folder = Path(source) / episode
    meta = read_json(folder / "episode.json")
    raw = Path(meta["raw_path"]).resolve()
    root = (ROOT / cfg["dataset_root"]).resolve()
    allowed = [(root / s).resolve() for s in cfg["episode_subdirs"]]
    if raw.name != episode or not any(raw.is_relative_to(r) for r in allowed):
        raise ValueError("Raw path outside configured robot sources")
    if (meta["config_hash"] != digest(cfg) or meta["split_hash"] != split_hash
            or meta["image_storage"] != BACKEND or meta["source_fingerprint"] != fingerprint(raw, cfg)):
        raise ValueError("Light source/config/split changed")
    for key in KEYS:
        if file_hash(folder / f"{key}.npy") != meta["numeric_sha256"][key]:
            raise ValueError(f"Light numeric changed: {key}")
    return meta


class AlignedFrames:
    """One sequential decoder per camera, selecting only causal aligned frame IDs."""
    def __init__(self, path, indices, cfg):
        self.cap = cv2.VideoCapture(str(path))
        self.indices, self.cfg, self.row, self.current, self.last = indices, cfg, 0, -1, None
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Cannot open {path}")

    def read(self):
        wanted = int(self.indices[self.row]); self.row += 1
        if wanted < 0:
            return True, None
        if wanted != self.current:
            while self.current < wanted:
                if not self.cap.grab():
                    raise ValueError(f"Video ends before aligned frame {wanted}")
                self.current += 1
            ok, frame = self.cap.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode frame {wanted}")
            self.last = cv2.resize(frame, (self.cfg["image_width"], self.cfg["image_height"]), interpolation=cv2.INTER_AREA)
        return True, self.last

    def release(self):
        self.cap.release()
