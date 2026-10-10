"""Kiểm tra transform và đối chiếu TCP ở camera cố định; không render MuJoCo."""
import argparse
import _bootstrap
import numpy as np
from src.data.common import load_config, read_json, write_json, episode_path, outside_raw, require_contract
from src.data.cfg5 import load_object
from src.data.geometry import pose_matrix, check_transform
from src.simulation.camera_rig import build_rig, camera_world_poses


def audit(cfg, rig, split):
    require_contract(rig, cfg)
    require_contract(split, cfg)
    if rig["camera_order"] != cfg["camera_serials"]:
        raise ValueError("Sai camera order")
    tcp = check_transform(rig["T_base_from_aligned_tcp_at_calibration"])
    poses = camera_world_poses(rig, np.eye(4), tcp)
    moved = tcp.copy()
    moved[:3, 3] += [.1, -.02, .03]
    changed = camera_world_poses(rig, np.eye(4), moved)
    records = []
    for camera in rig["cameras"]:
        name = camera["name"]
        expected = np.asarray(camera["T_base_from_camera_cv_at_calibration"]) @ np.diag([1, -1, -1, 1])
        error = float(np.max(np.abs(poses[name]-expected)))
        np.testing.assert_allclose(poses[name], expected, atol=1e-8)
        k = np.asarray(camera["K_pixels"])
        assert k[0, 0] > 0 and k[1, 1] > 0
        wrist = camera["parent_frame"] == "rh20t_aligned_tcp"
        delta = changed[name][:3, 3]-poses[name][:3, 3]
        np.testing.assert_allclose(delta, [.1, -.02, .03] if wrist else [0, 0, 0], atol=1e-8)
        records.append({"serial": camera["serial"], "parent_frame": camera["parent_frame"],
                        "calibration_roundtrip_max_abs_error": error,
                        "base_position_at_calibration_m": poses[name][:3, 3].tolist()})
    # Dữ liệu TCP theo camera của RH20T cho phép kiểm tra chéo frame camera cố định.
    # Với camera wrist đang chuyển động, không áp extrinsic tại calibration cho cả trajectory.
    compared = []
    checked_calibrations = set()
    for episode in split["splits"]["train"]:
        folder = episode_path(cfg, episode)
        calibration = str(read_json(folder / "metadata.json")["calib"])
        if calibration in checked_calibrations:
            continue
        episode_rig = build_rig(cfg, calibration)
        base = load_object(folder / "transformed/tcp_base.npy")
        camera_tcp = load_object(folder / "transformed/tcp.npy")
        for camera in episode_rig["cameras"]:
            if camera["parent_frame"] != "robot_base":
                continue
            serial = camera["serial"]
            left = {int(r["timestamp"]): r["tcp"] for r in base[serial]}
            right = {int(r["timestamp"]): r["tcp"] for r in camera_tcp[serial]}
            shared = sorted(left.keys() & right.keys())
            if not shared:
                raise ValueError("Thiếu timestamp chung để kiểm tra camera")
            errors = []
            for index in np.unique(np.linspace(0, len(shared)-1, min(5, len(shared)), dtype=int)):
                t = shared[index]
                predicted = np.asarray(camera["T_camera_cv_from_base_at_calibration"]) @ pose_matrix(left[t])
                errors.append(float(np.max(np.abs(predicted-pose_matrix(right[t])))))
            if max(errors) > 1e-4:
                raise ValueError(f"Transform không khớp TCP: {episode} {serial}: {max(errors)}")
            compared.append({"episode": episode, "calibration": calibration, "serial": serial,
                             "samples": len(errors), "max_abs_matrix_error": max(errors)})
        checked_calibrations.add(calibration)
    return {"calibration_id": rig["calibration_id"], "camera_count": len(records),
            "cameras": records, "fixed_camera_tcp_crosschecks_train_only": compared,
            "mujoco_rendered": False, "wrist_dynamic_dataset_extrinsics_verified": False,
            "scope": "8 calibration roundtrips, 6 fixed/2 moving mount tests; fixed-camera TCP crosschecks on train only"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--rig", default="configs/mujoco/cfg5_camera_rig.json")
    p.add_argument("--splits", default="data/splits/cfg5_splits.json")
    p.add_argument("--output", default="docs/audit/camera_rig_check.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    result = audit(cfg, read_json(args.rig), read_json(args.splits))
    write_json(outside_raw(args.output, cfg), result)
    print("Camera checks:", result["camera_count"], "fixed-camera TCP checks:", len(result["fixed_camera_tcp_crosschecks_train_only"]))
