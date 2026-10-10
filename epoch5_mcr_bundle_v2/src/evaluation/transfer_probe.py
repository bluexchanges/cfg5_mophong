"""Read-only transfer diagnostic: recorded inference and closed-loop ACT rollout."""
from copy import deepcopy
import html
import math
from pathlib import Path
import random
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

from src.data.common import ROOT, digest, outside_raw, read_json
from src.data.sim_demos import file_hash, load_trajectory, read_camera_times, fingerprint
from src.data.splits import validate_splits
from src.models.rh20t_act import RH20TACTPolicy
from src.simulation.panda_observation import to_act_observation, denormalize_action
from src.training.pretrain import write_json


def load_frozen_checkpoint(path, expected_epoch=5, device="cpu"):
    path = Path(path)
    checksum = file_hash(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("smoke") is not False or saved.get("epoch") != expected_epoch - 1:
        raise ValueError(f"Cần checkpoint full epoch {expected_epoch}, nhận {saved.get('epoch')}")
    meta = saved["metadata"]
    cfg, data, stats = deepcopy(meta["model_config"]), meta["data_config"], meta["normalization"]
    if stats.get("smoke_only") or stats.get("source_split") != "train":
        raise ValueError("Checkpoint thiếu normalization train đầy đủ")
    for key in ("state", "action"):
        if np.asarray(stats[key]["mean"]).shape != (8,) or np.any(np.asarray(stats[key]["std"]) <= 0):
            raise ValueError("Normalization 8D không hợp lệ")
    contract = saved["contract"]
    if any(contract[key] != value for key, value in {
        "model_config_hash": digest(meta["model_config"]), "data_config_hash": digest(data),
        "stats_hash": digest(stats), "camera_serials": data["camera_serials"],
        "upstream_commit": cfg["upstream_commit"], "representation": data["representation"]}.items()):
        raise ValueError("Checkpoint metadata khác contract")
    cfg["backbone_weights_path"] = None  # full checkpoint already contains the backbone
    policy = RH20TACTPolicy(cfg, data)
    policy.load_state_dict(saved["model"], strict=True)
    policy.to(device).eval().requires_grad_(False)
    del saved
    return policy, stats, data, checksum


def select_episodes(splits, manifest, split="val", count=10, seed=42):
    validate_splits(splits)
    if split not in ("val", "test") or not 1 <= count <= len(splits["splits"][split]):
        raise ValueError("Diagnostic chỉ chọn val/test; count phải trong membership")
    if manifest["manifest_hash"] != splits["manifest_hash"]:
        raise ValueError("Split/manifest khác nhau")
    if digest({k: v for k, v in splits.items() if k != "split_hash"}) != splits["split_hash"]:
        raise ValueError("Split hash không hợp lệ")
    if digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("Manifest hash không hợp lệ")
    by_name = {r["episode"]: r for r in manifest["episodes"]}
    names = random.Random(seed).sample(sorted(splits["splits"][split]), count)
    return [by_name[name] for name in names]


def recorded_batches(record, cfg, stats, batch_size=8):
    """Decode every view sequentially once; hold only one batch of RGB in RAM."""
    raw = Path(record["path"])
    if fingerprint(raw, cfg["camera_serials"]) != record["source_fingerprint"]:
        raise ValueError(f"Raw đã đổi: {raw}")
    _, times, state, action, segments = load_trajectory(raw, cfg)
    caps = []
    try:
        for cam in cfg["camera_serials"]:
            folder = raw / f"cam_{cam}"
            if not np.array_equal(read_camera_times(folder / "timestamps.npy"), times):
                raise ValueError("Timestamp camera không khớp robot")
            cap = cv2.VideoCapture(str(folder / "color.mp4"))
            caps.append(cap)
            if not cap.isOpened():
                raise ValueError(f"Không mở video {folder}")
        for start in range(0, len(times), batch_size):
            count = min(batch_size, len(times) - start)
            rgb = np.empty((count, 8, cfg["image_height"], cfg["image_width"], 3), np.uint8)
            targets = np.zeros((count, cfg["chunk_size"], 8), np.float32)
            pad = np.ones(targets.shape[:2], bool)
            for row in range(count):
                i = start + row
                for cam_index, cap in enumerate(caps):
                    ok, frame = cap.read()
                    if not ok or frame.shape != rgb.shape[2:]:
                        raise ValueError(f"Video thiếu/sai frame {raw.name}/{i}")
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb[row, cam_index])
                end = i
                while end < min(len(times), i + cfg["chunk_size"]) and segments[end] == segments[i]:
                    end += 1
                n = end - i
                targets[row, :n] = (action[i:end] - stats["action"]["mean"]) / stats["action"]["std"]
                pad[row, :n] = False
            image = rgb.astype(np.float32) / 255
            image = (image - np.asarray(cfg["image_normalization_mean"], np.float32)) / np.asarray(cfg["image_normalization_std"], np.float32)
            yield {"images": np.ascontiguousarray(image.transpose(0, 1, 4, 2, 3)),
                   "state": ((state[start:start+count] - stats["state"]["mean"]) / stats["state"]["std"]).astype(np.float32),
                   "actions": targets, "is_pad": pad, "episode": [raw.name] * count,
                   "timestamps_ms": times[start:start+count]}
        if any(cap.read()[0] for cap in caps):
            raise ValueError("Video có frame dư")
        if fingerprint(raw, cfg["camera_serials"]) != record["source_fingerprint"]:
            raise ValueError("Raw đổi trong khi đọc")
    finally:
        for cap in caps:
            cap.release()


def bounded_action(raw_action, cfg):
    """Explicit simulation guard; log both raw and executed commands."""
    action = np.asarray(raw_action, dtype=np.float64).copy()
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("Prediction không hữu hạn/không phải 8D")
    norm = np.linalg.norm(action[3:7])
    if norm < 1e-8:
        raise ValueError("Prediction quaternion gần bằng 0")
    action[:3] = np.clip(action[:3], cfg["workspace_min_m"], cfg["workspace_max_m"])
    action[7] = np.clip(action[7], *cfg["gripper_width_range_m"])
    clipped = not np.array_equal(action[np.r_[0:3, 7]], np.asarray(raw_action)[np.r_[0:3, 7]])
    action[3:7] /= norm
    return action, clipped


def rollout(record, policy, stats, data_cfg, out, device, *, max_steps=600, viewer=False):
    from src.simulation.panda_env import PandaEnv
    from src.simulation.scene_builder import load_sim_config, build_scene
    raw = Path(record["path"])
    meta = read_json(raw / "metadata.json")
    cfg = read_json(raw / "scene_config.json")
    if digest(cfg) != meta["scene_config_hash"]:
        raise ValueError("Scene config hash khác demo")
    if (file_hash(ROOT / cfg["camera_rig"]) != meta["camera_rig_hash"]
            and digest(read_json(ROOT / cfg["camera_rig"])) != meta["camera_rig_hash"]):
        raise ValueError("Camera rig hiện tại khác demo")
    if cfg["control_hz"] != data_cfg["control_hz"]:
        raise ValueError("Control Hz khác checkpoint")
    original_cfg = deepcopy(cfg)
    cfg["scene_xml"] = str((out / "scene.xml").resolve())
    config_path = out / "scene_runtime.json"
    write_json(config_path, cfg)
    load_sim_config(config_path)  # validates before constructing MuJoCo
    # Builder's normal output sits in mujoco/scenes. A probe is stored elsewhere;
    # resolve meshdir in this generated derivative, leaving source assets intact.
    generated = build_scene(cfg)
    tree = ET.parse(generated)
    tree.getroot().find("compiler").set("meshdir", str((ROOT / cfg["robot_xml"]).parent / "assets"))
    tree.write(generated, encoding="utf-8", xml_declaration=True)
    build_record = read_json(generated.with_suffix(".manifest.json"))
    build_record["generated_xml_sha256"] = file_hash(generated)
    write_json(generated.with_suffix(".manifest.json"), build_record)
    source_stats = stats
    path = out / "rollout.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), cfg["control_hz"], (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Không tạo được rollout.mp4")
    trace, clipped_count, failure = [], 0, None
    started = time.perf_counter()
    try:
        with PandaEnv(config_path, render_observations=True) as env:
            obs, info = env.reset(seed=meta["seed"])
            with np.load(raw / "robot_trajectory.npz", allow_pickle=False) as source:
                initial = source["measured_tcp_gripper"][0]
            if not np.allclose(obs["state"], initial, atol=2e-4, rtol=0):
                raise ValueError("Reset MuJoCo khác trạng thái đầu demo; kiểm version/assets/scene")
            terminated = truncated = False
            for tick in range(max_steps):
                frame = cv2.cvtColor(env.render("overview", 640, 360), cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"ACT transfer | {raw.name} | step {tick} | clipped {clipped_count}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                writer.write(frame)
                if viewer:
                    if not env.show_viewer().is_running():
                        failure = "viewer_closed"
                        break
                inputs = to_act_observation(obs, source_stats, data_cfg)
                with torch.inference_mode():
                    prediction = policy(torch.from_numpy(inputs["state"]).unsqueeze(0).to(device),
                                        torch.from_numpy(inputs["images"]).unsqueeze(0).to(device))[0, 0].cpu().numpy()
                raw_action = denormalize_action(prediction, source_stats)
                try:
                    action, clipped = bounded_action(raw_action, cfg)
                    clipped_count += int(clipped)
                    before = obs["state"].copy()
                    obs, _, terminated, truncated, info = env.step(action)
                    trace.append({"timestamp_s": tick / cfg["control_hz"], "state": before.tolist(),
                                  "raw_action": raw_action.tolist(), "executed_action": action.tolist(),
                                  "clipped": clipped, "controller": info["controller"]})
                except (ValueError, FloatingPointError) as exc:
                    failure = str(exc)
                    break
                if terminated or truncated:
                    break
                if (tick + 1) % 100 == 0:
                    print(f"ROLLOUT {raw.name} step={tick+1}/{max_steps}", flush=True)
            writer.write(cv2.cvtColor(env.render("overview", 640, 360), cv2.COLOR_RGB2BGR))
            result = {"episode": raw.name, "seed": meta["seed"], "scene_config_hash": digest(original_cfg),
                      "task_success": bool(info["metrics"]["task_success"]), "metrics": info["metrics"],
                      "steps": len(trace), "sim_time_s": info["sim_time_s"], "failure": failure,
                      "horizon_reached": not terminated and not truncated and len(trace) == max_steps,
                      "terminated": terminated, "truncated": truncated, "clipped_commands": clipped_count,
                      "elapsed_seconds": time.perf_counter() - started,
                      "policy": "images/measured_state only; z=0; first predicted action each 10Hz tick",
                      "guard": "workspace/gripper clipping + quaternion normalization; raw actions retained",
                      "optimizer_steps": 0, "video": "rollout.mp4"}
    finally:
        writer.release()
    write_json(out / "rollout_trace.json", trace)
    write_json(out / "rollout_report.json", result)
    return result


def evaluate(args):
    from src.evaluation.transfer_metrics import InferenceMetrics
    policy, stats, source_data, checksum = load_frozen_checkpoint(args.checkpoint, args.expected_epoch, args.device)
    cfg = read_json(ROOT / args.data_config)
    for key in ("camera_serials", "image_width", "image_height", "control_hz", "chunk_size",
                "image_normalization_mean", "image_normalization_std"):
        if cfg[key] != source_data[key]:
            raise ValueError(f"Observation khác checkpoint: {key}")
    splits, manifest = read_json(ROOT / args.splits), read_json(ROOT / args.manifest)
    if manifest["config_hash"] != digest(cfg) or splits["config_hash"] != digest(cfg):
        raise ValueError("Config không khớp manifest/split đã khóa")
    records = select_episodes(splits, manifest, args.split, args.episodes, args.seed)
    out = outside_raw(ROOT / args.output, cfg)
    if out.exists():
        raise FileExistsError(f"Output đã có; dùng --output mới: {out}")
    out.mkdir(parents=True)
    shutil.copyfile(args.checkpoint, out / "frozen_checkpoint.pt")
    if file_hash(out / "frozen_checkpoint.pt") != checksum:
        raise ValueError("Checkpoint nguồn đổi trong lúc tạo bản snapshot")
    selection = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": checksum,
                 "checkpoint_epoch": args.expected_epoch, "split": args.split, "seed": args.seed,
                 "episodes": [r["episode"] for r in records], "split_hash": splits["split_hash"],
                 "normalization": "original checkpoint training stats, unchanged", "optimizer_steps": 0,
                 "scope": "small transfer diagnostic; val is development data; not official full test"}
    write_json(out / "selection.json", selection)
    all_metrics, rows = InferenceMetrics(stats), []
    for record in records:
        folder = out / record["episode"]
        folder.mkdir()
        metrics = InferenceMetrics(stats)
        times, predictions, targets = [], [], []
        with torch.inference_mode():
            for batch in recorded_batches(record, cfg, stats, args.batch_size):
                pred = policy(torch.from_numpy(batch["state"]).to(args.device),
                              torch.from_numpy(batch["images"]).to(args.device)).cpu().numpy()
                for aggregator in (metrics, all_metrics):
                    aggregator.add(pred, batch["actions"], batch["is_pad"], batch["episode"])
                times.extend(batch["timestamps_ms"])
                predictions.extend(pred[:, 0] * stats["action"]["std"] + stats["action"]["mean"])
                targets.extend(batch["actions"][:, 0] * stats["action"]["std"] + stats["action"]["mean"])
        np.savez_compressed(folder / "first_action_predictions.npz", timestamps_ms=times,
                            prediction=predictions, target=targets)
        row = {"episode": record["episode"], "offline": metrics.summary()}
        write_json(folder / "offline_report.json", row)
        if args.mode == "both":
            try:
                row["rollout"] = rollout(record, policy, stats, source_data, folder, args.device,
                                         max_steps=args.max_steps, viewer=args.viewer)
            except Exception as exc:
                row["rollout_error"] = f"{type(exc).__name__}: {exc}"
                write_json(folder / "rollout_error.json", {"error": row["rollout_error"]})
        rows.append(row)
        write_json(out / "progress.json", rows)
        print(f"PROBE {len(rows)}/{len(records)} {record['episode']} offline_l1={row['offline']['normalized_l1']:.6f}", flush=True)
    if file_hash(args.checkpoint) != checksum:
        raise ValueError("Checkpoint nguồn bị thay trong khi đánh giá")
    completed = [r["rollout"] for r in rows if "rollout" in r]
    report = {**selection, "offline": all_metrics.summary(), "results": rows,
              "requested_rollouts": len(rows) if args.mode == "both" else 0,
              "completed_rollouts": len(completed), "rollout_errors": sum("rollout_error" in r for r in rows),
              "successful_rollouts": sum(r["task_success"] for r in completed),
              "success_rate_completed": sum(r["task_success"] for r in completed) / len(completed) if completed else None,
              "success_rate_all_requested": sum(r["task_success"] for r in completed) / len(rows) if args.mode == "both" else None,
              "warning": "0/10 is a diagnostic, not proof of general failure; offline error != success rate"}
    write_json(out / "report.json", report)
    body = []
    for row in rows:
        ep = html.escape(row["episode"])
        outcome = row.get("rollout_error", "SUCCESS" if row.get("rollout", {}).get("task_success") else "FAIL/unfinished")
        body.append(f'<section><h2>{ep}: {html.escape(outcome)}</h2><p>L1={row["offline"]["normalized_l1"]:.5f}; position={row["offline"]["position_mean_distance_mm"]:.2f} mm</p><video controls width="640" src="{ep}/rollout.mp4"></video><p><a href="{ep}/offline_report.json">Offline metrics</a> | <a href="{ep}/rollout_report.json">Rollout metrics</a> | <a href="{ep}/rollout_trace.json">Actions</a></p></section>')
    (out / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>ACT transfer probe</title><h1>ACT E5: transfer diagnostic, optimizer_steps=0</h1><p>Val subset; checkpoint unchanged; clipped actions recorded.</p>' + ''.join(body), encoding="utf-8")
    print(f"TRANSFER_PROBE_OK output={out} rollouts={len(completed)}/{report['requested_rollouts']}", flush=True)
    if args.mode == "both" and len(completed) != len(rows):
        raise RuntimeError("Có rollout không khởi tạo/chạy được; đọc report.json và rollout_error.json")
    return report
