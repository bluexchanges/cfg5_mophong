"""Rig 8 camera từ calibration CFG5, frame conventions được ghi trong JSON."""
import hashlib
from pathlib import Path
import numpy as np

from src.data.cfg5 import load_object
from src.data.common import data_root, digest, read_json, ROOT
from src.data.geometry import check_transform, rotvec_matrix


def build_rig(cfg, calibration_id):
    folder = data_root(cfg) / "calib" / str(calibration_id)
    source_path = ROOT / "references/upstream/RH20T__rh20t_api/configs/configs.json"
    robot = next(r for r in read_json(source_path) if r["conf_num"] == 5)
    intrinsics = load_object(folder / "intrinsics.npy")
    extrinsics = {key: check_transform(np.asarray(value)[0])
                  for key, value in load_object(folder / "extrinsics.npy").items()}
    # Franka preprocessing: raw rotation-vector → R * Rx(pi).
    preprocessed_tcp = rotvec_matrix(np.load(folder / "tcp.npy", allow_pickle=False)) @ np.diag([1, -1, -1, 1])
    marker_from_base = np.linalg.inv(extrinsics[robot["in_hand"][0]]) @ np.asarray(robot["tc_mat"]) @ np.linalg.inv(preprocessed_tcp)
    aligned_tcp = np.linalg.inv(np.asarray(robot["align_mat_base"])) @ preprocessed_tcp @ np.asarray(robot["align_mat_tcp"])
    cv_from_mujoco = np.diag([1, -1, -1, 1])
    cameras = []
    for serial in cfg["camera_serials"]:
        camera_from_base = extrinsics[serial] @ marker_from_base @ np.asarray(robot["align_mat_base"])
        base_from_camera = check_transform(np.linalg.inv(camera_from_base))
        wrist = serial in cfg["in_hand_serials"]
        parent_from_camera = np.linalg.inv(aligned_tcp) @ base_from_camera if wrist else base_from_camera
        # K có kích thước 3x4 trong file; RH20T API ghi ảnh calibration 1280x720.
        k = np.asarray(intrinsics[serial], dtype=np.float64)[:, :3].copy()
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise ValueError("Intrinsics không hợp lệ")
        k[0] *= cfg["image_width"] / 1280
        k[1] *= cfg["image_height"] / 720
        cameras.append({"name": f"cam_{serial}", "serial": serial,
                        "parent_frame": "rh20t_aligned_tcp" if wrist else "robot_base",
                        "K_pixels": k.tolist(), "resolution_wh": [cfg["image_width"], cfg["image_height"]],
                        "T_camera_cv_from_base_at_calibration": camera_from_base.tolist(),
                        "T_base_from_camera_cv_at_calibration": base_from_camera.tolist(),
                        "T_parent_from_camera_mujoco": check_transform(parent_from_camera @ cv_from_mujoco).tolist()})
    files = [source_path, *(folder/name for name in ("tcp.npy", "intrinsics.npy", "extrinsics.npy"))]
    return {"schema_version": 1, "config_hash": digest(cfg), "calibration_id": str(calibration_id),
            "camera_count": len(cameras), "camera_order": cfg["camera_serials"],
            "source_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "source_intrinsics_resolution_wh": [1280, 720],
            "T_base_from_aligned_tcp_at_calibration": aligned_tcp.tolist(),
            "convention": "T_A_from_B maps coordinates in B to A; CV +Z forward,+Y down; MuJoCo -Z forward,+Y up",
            "status": "exported calibration transforms; mathematical and rendering verification are recorded separately in docs/audit",
            "cameras": cameras}


def camera_world_poses(rig, world_from_base, base_from_aligned_tcp):
    world_from_base = check_transform(world_from_base)
    base_from_aligned_tcp = check_transform(base_from_aligned_tcp)
    result = {}
    for camera in rig["cameras"]:
        mount = check_transform(camera["T_parent_from_camera_mujoco"])
        if camera["parent_frame"] not in ("robot_base", "rh20t_aligned_tcp"):
            raise ValueError("Camera parent frame không được hỗ trợ")
        parent = np.eye(4) if camera["parent_frame"] == "robot_base" else base_from_aligned_tcp
        result[camera["name"]] = world_from_base @ parent @ mount
    return result
