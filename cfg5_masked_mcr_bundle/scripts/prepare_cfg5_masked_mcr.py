"""Prepare every CFG5 robot episode for masked MCR caching; no rating filter."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from collections import Counter
from pathlib import Path
import _bootstrap
import cv2
import numpy as np
from src.data.common import ROOT, digest, outside_raw, read_json, write_json
from src.data.cfg5_masked import BACKEND, KEYS, NAME, PandaFK, discover, fingerprint, timeline, validate_fk, verify_light, restore_missing_joints
from src.data.sim_demos import file_hash
from src.data.splits import create_splits
from src.data.act_dataset import fit_normalization


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", default=["training_data", "robot_missing_data"])
    p.add_argument("--prefix", default="cfg5_robot_all_masked")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--target", choices=("tcp", "joint"), default="tcp")
    p.add_argument("--repair-archive", help="Original ZIP; restore only missing joint.npy into separate derived directory")
    args = p.parse_args()
    if not args.prefix.replace("_", "").isalnum():
        p.error("prefix must use letters/digits/underscore")
    prefix = args.prefix
    cfg = read_json(ROOT / "configs/data/cfg5_all8.json")
    cfg.update(source_type=BACKEND, episode_subdirs=args.sources, require_all_cameras=False,
               rating_filter_applied=False, min_rating=None, action_target_offset=1,
               min_segment_steps=2, missing_views="per_tick_camera_valid_attention_mask_v1",
               missing_tcp="panda_fk_verified_against_measured_tcp_v1")
    cv2.setNumThreads(1)
    episodes = discover(cfg)
    if not episodes:
        raise ValueError("No robot episode in configured sources")
    print(f"INPUT episodes={len(episodes)} sources={args.sources}; no episode/rating/camera-count filter", flush=True)
    fk = None
    if args.target == "joint":
        cfg.update(representation="panda_joint_positions_rad_gripper_width_m", missing_tcp="unused_joint_target",
                   joint_repair_root="data/repairs/cfg5_recorded_joints")
        if args.repair_archive:
            print("JOINT_REPAIR", restore_missing_joints(episodes, cfg, args.repair_archive), flush=True)
        fk_report = {"used": False, "reason": "joint target uses recorded joints, no TCP reconstruction"}
    else:
        fk = PandaFK()
        try:
            fk_report = validate_fk(episodes, cfg, fk)
        except ValueError as exc:
            path = ROOT / f"outputs/data_audit/{prefix}_fk_blocked.json"
            write_json(path, {"error": str(exc), "no_episode_exclusions": True, "input_episodes": len(episodes)})
            raise ValueError(f"TCP recovery not verified; no labels generated. See {path}") from exc
        cfg["fk_validation_hash"] = digest(fk_report)
        print("FK_VERIFIED", fk_report, flush=True)
    accepted, blocked = [], []
    report_path = outside_raw(ROOT / f"outputs/data_audit/{prefix}.json", cfg)
    for i, ep in enumerate(episodes, 1):
        try:
            arrays, quality = timeline(ep, cfg, fk)
            task, user, scene = map(int, NAME.fullmatch(ep.name).groups())
            meta = read_json(ep / "metadata.json")
            accepted.append({"episode": ep.name, "task": task, "user": user, "scene": scene,
                             "calib": str(meta["calib"]), "rating": meta.get("rating"), "path": str(ep.resolve()),
                             "source_fingerprint": fingerprint(ep, cfg), "quality": quality})
        except (OSError, ValueError, KeyError, TypeError, EOFError, IndexError) as exc:
            blocked.append({"episode": ep.name, "path": str(ep.resolve()), "reason": str(exc)})
            print(f"BLOCKED {ep.name}: {exc}", flush=True)
        if i % 25 == 0 or i == len(episodes):
            print(f"AUDIT {i}/{len(episodes)} usable={len(accepted)} blocked={len(blocked)}", flush=True)
    report = {"input_episodes": len(episodes), "usable_episodes": len(accepted), "blocked_episodes": blocked,
              "no_episode_exclusions": True, "rating_filter_applied": False, "fk_validation": fk_report,
              "config": cfg, "episodes": accepted}
    write_json(report_path, report)
    print(f"AUDIT_REPORT {report_path}", flush=True)
    if blocked:
        raise ValueError(f"{len(blocked)} episode cần sửa dữ liệu; dừng, không tự loại khỏi train: {report_path}")
    if args.audit_only:
        print("CFG5_ALL_ROBOT_AUDIT_OK", flush=True)
        return
    manifest = {"schema_version": 1, "config_hash": digest(cfg), "config": cfg,
                "episodes": accepted, "rejected": [], "skipped_counts": {}, "fk_validation": fk_report}
    manifest["manifest_hash"] = digest(manifest)
    split = create_splits(manifest, cfg)
    base = ROOT / f"data/splits/{prefix}"
    files = {"data_config": f"configs/data/{prefix}.json", "model_config": f"configs/act/{prefix}_model.json",
             "run_config": f"configs/act/{prefix}.json", "manifest": f"{base.relative_to(ROOT).as_posix()}_manifest.json",
             "splits": f"{base.relative_to(ROOT).as_posix()}_splits.json", "stats": f"{base.relative_to(ROOT).as_posix()}_normalization.json"}
    source = outside_raw(ROOT / f"data/processed/{prefix}_light", cfg)
    records = {r["episode"]: r for r in accepted}
    # Numeric preparation includes test; image feature cache remains train/val only.
    for i, record in enumerate(accepted, 1):
        name = record["episode"]
        final = source / name
        partial = final.with_name(name + ".partial")
        if partial.exists():
            raise FileExistsError(f"Incomplete numeric episode, inspect ERROR.json: {partial}")
        if final.exists():
            verify_light(name, cfg, source, split["split_hash"])
        else:
            raw = Path(record["path"])
            if fingerprint(raw, cfg) != record["source_fingerprint"]:
                raise ValueError(f"Raw changed since audit: {name}")
            arrays, quality = timeline(raw, cfg, fk)
            if quality != record["quality"]:
                raise ValueError(f"Timeline changed since audit: {name}")
            partial.mkdir(parents=True)
            try:
                for key in KEYS:
                    np.save(partial / f"{key}.npy", arrays[key], allow_pickle=False)
                meta = {"schema_version": 1, "episode": name, "steps": quality["steps"], "smoke_only": False,
                        "config_hash": digest(cfg), "split_hash": split["split_hash"], "raw_path": record["path"],
                        "camera_serials": cfg["camera_serials"], "source_fingerprint": record["source_fingerprint"],
                        "representation": cfg["representation"], "image_storage": BACKEND, "quality": quality,
                        "action_target_offset": 1, "numeric_sha256": {k: file_hash(partial / f"{k}.npy") for k in KEYS},
                        "action_semantics": ("future recorded joint positions plus future recorded gripper command"
                            if args.target == "joint" else "future measured TCP or verified joint-derived FK TCP plus recorded future gripper command")}
                if fingerprint(raw, cfg) != record["source_fingerprint"]:
                    raise ValueError(f"Raw changed during numeric preparation: {name}")
                write_json(partial / "episode.json", meta)
                partial.rename(final)
            except Exception as exc:
                write_json(partial / "ERROR.json", {"error": str(exc), "incomplete": True})
                raise
        if i % 25 == 0 or i == len(accepted):
            print(f"LIGHT {i}/{len(accepted)}", flush=True)
    stats = fit_normalization(source, split, cfg)
    model = read_json(ROOT / "configs/act/rh20t_all8.json")
    model["data_config"] = files["data_config"]
    model["backbone_weights_path"] = None
    run = read_json(ROOT / "configs/act/cfg5_all_pretrain.json")
    run.update(model_config=files["model_config"], splits=files["splits"], locked_membership=files["splits"],
               stats=files["stats"], processed_root=source.relative_to(ROOT).as_posix(),
               output_dir=f"outputs/{prefix}", cache_source_backend=BACKEND, sample_backend="npy")
    for key, content in (("data_config", cfg), ("model_config", model), ("run_config", run),
                         ("manifest", manifest), ("splits", split), ("stats", stats)):
        path = outside_raw(ROOT / files[key], cfg)
        if path.exists() and read_json(path) != content:
            raise FileExistsError(f"Artifacts changed, choose a new --prefix: {path}")
        write_json(path, content)
    print({"episodes": len(accepted), "split_counts": split["counts"], "artifacts": files,
           "light_root": str(source), "test_images_cached": False}, flush=True)
    print("CFG5_ALL_ROBOT_LIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
