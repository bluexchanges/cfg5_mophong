"""Pilot -> resumable cache; train/val by default, separate frozen test readiness on request."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from pathlib import Path
import random
import shutil
import subprocess
import time
import _bootstrap
import torch
import cv2
from src.data.common import ROOT, digest, outside_raw, read_json
from src.data.feature_cache import (context, cache_contract, cache_episode, check_episode, FEATURE_SHAPE)
from src.models.mcr_act import MCRACTPolicy, PREPROCESS
from src.training.pretrain import write_json


def home_usage():
    result = subprocess.run(["du", "-s", "-B1", str(Path.home())], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("Không đo được dung lượng home; dùng --home-used-gib với số đo thật")
    return int(result.stdout.split()[0])


def check_storage(used, additional, quota, free, *, skip_home_budget=False):
    reserve = 10 * 2**30
    if free < additional + reserve:
        raise OSError(f"Ổ đĩa thiếu chỗ: free={free/2**30:.1f}GiB, cần~{(additional+reserve)/2**30:.1f}GiB")
    if not skip_home_budget and used + additional + reserve > quota:
        raise OSError(f"Cache vượt ngân sách: home_used={used/2**30:.1f}GiB, thêm~{additional/2**30:.1f}GiB, quota={quota/2**30:.1f}GiB")


def estimate(root, source, names, cfg, split, contract, *, features_only=False):
    completed, pending_steps, total_steps = [], 0, 0
    for ep in names:
        steps = read_json(source / ep / "episode.json")["steps"]
        total_steps += steps
        if (root / ep).exists():
            completed.append(check_episode(root / ep, cfg, split, contract, stage="frozen" if features_only else None))
        else:
            pending_steps += steps
    per_step = 8 * 2048 * 7 * 7 + 8 * 2048 * 2
    jpeg_per_step = 0 if features_only else (max(m["jpeg_file_bytes"] / m["steps"] for m in completed) * 1.5
                     if completed else cfg["image_width"] * cfg["image_height"] * 3 * 8 * 1.2)
    pending_bytes = int(pending_steps * (per_step + jpeg_per_step + 256) * 1.1 + 2**30)
    return {"total_steps": total_steps, "pending_steps": pending_steps, "completed_episodes": len(completed),
            "feature_payload_GiB": total_steps * per_step / 2**30,
            "estimated_pending_GiB": pending_bytes / 2**30, "pending_bytes": pending_bytes,
            "jpeg_estimate_basis": "not stored in frozen stage" if features_only else ("max completed bytes/tick x1.5" if completed else "uncompressed upper estimate"),
            "quantization_max_absolute_error": max((m["quantization_max_absolute_error"] for m in completed), default=None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("pilot", "build", "check", "estimate"))
    p.add_argument("--base-config", default="configs/act/pick_place_finetune_mp4_w16.json")
    p.add_argument("--cache-root", default="data/features/pick_place_mcr")
    p.add_argument("--mcr-checkpoint", default="outputs/backbones/mcr_resnet50.pth")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--decoder-workers", type=int, default=8)
    p.add_argument("--pilot-episodes", type=int, default=3)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--home-quota-tb", type=float, default=1.8, help="Decimal TB; cluster soft quota")
    p.add_argument("--home-used-gib", type=float)
    p.add_argument("--skip-home-budget", action="store_true",
                   help="Explicitly disable the script home budget; still check filesystem free space. Does not change server quota.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--full-check", action="store_true")
    p.add_argument("--features-only", action="store_true", help="Save features/scales only; regenerate JPEG after frozen training")
    p.add_argument("--include-test", action="store_true", help="Also preprocess frozen test features; never sample test during training")
    args = p.parse_args()
    if min(args.batch_size, args.decoder_workers, args.pilot_episodes) < 1 or args.decoder_workers > 16:
        p.error("batch/pilot phải dương; decoder-workers 1..16")
    if args.home_quota_tb <= 0 or (args.home_used_gib is not None and args.home_used_gib < 0):
        p.error("Quota/usage không hợp lệ")
    if args.include_test and not args.features_only:
        p.error("--include-test currently requires --features-only")
    run, model, cfg, splits, stats = context(args.base_config)
    root = outside_raw(ROOT / args.cache_root, cfg)
    source = ROOT / run["processed_root"]
    contract = cache_contract(cfg, splits, stats, args.jpeg_quality)
    training_names = splits["splits"]["train"] + splits["splits"]["val"]
    names = training_names + (splits["splits"]["test"] if args.include_test else [])
    if len(names) != len(set(names)):
        raise ValueError("Train/val/test overlap")
    report = estimate(root, source, names, cfg, splits, contract, features_only=args.features_only)
    print(report, flush=True)
    if args.command == "estimate":
        return
    if args.command == "check":
        for ep in names:
            check_episode(root / ep, cfg, splits, contract, full=args.full_check, stage="frozen" if args.features_only else None)
        print("MCR_CACHE_CHECK_OK", flush=True)
        return
    if args.command == "build" and report["completed_episodes"] < min(3, len(splits["splits"]["train"])):
        raise ValueError("Chạy pilot tối thiểu 3 episode trước để đo JPEG/storage; không tạo cache full mù")
    selected = (random.Random(42).sample(sorted(splits["splits"]["train"]), min(args.pilot_episodes, len(splits["splits"]["train"])))
                if args.command == "pilot" else names)
    selected_report = estimate(root, source, selected, cfg, splits, contract, features_only=args.features_only)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Không thấy CUDA; chạy bằng sbatch GPU")
    used = int(args.home_used_gib * 2**30) if args.home_used_gib is not None else home_usage()
    quota = int(args.home_quota_tb * 10**12)
    print("MCR_STORAGE_POLICY", {"script_home_budget_enabled": not args.skip_home_budget,
          "home_used_GiB": used / 2**30, "script_budget_GiB": quota / 2**30,
          "filesystem_reserve_GiB": 10, "server_quota_changed": False}, flush=True)
    root.mkdir(parents=True, exist_ok=True)
    additional = selected_report["pending_bytes"]
    free = shutil.disk_usage(root).free
    check_storage(used, additional, quota, free, skip_home_budget=args.skip_home_budget)
    lock = root / ".cache.lock"
    import fcntl
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = {"source_config": args.base_config, "source_root": str(source.resolve()),
                   "contract": contract, "splits": splits, "stats": stats,
                   "data_config": cfg, "source_model_config": model}
        if args.features_only:
            runtime["storage_mode"] = "features_only_v1"
        context_path = root / "context.json"
        if context_path.exists() and read_json(context_path) != runtime:
            raise ValueError("Cache context đã có và khác; dùng cache-root mới")
        write_json(context_path, runtime)
        model = {**model, "backbone": "mcr_resnet50", "backbone_weights_path": None,
                 "lr_backbone": 0., "feature_preprocess": PREPROCESS}
        policy = MCRACTPolicy(model, cfg, mcr_checkpoint=ROOT / args.mcr_checkpoint).to(args.device).eval().requires_grad_(False)
        cv2.setNumThreads(1)
        torch.set_num_threads(min(args.decoder_workers, 8))
        started, added = time.perf_counter(), 0
        for i, ep in enumerate(selected, 1):
            next_report = estimate(root, source, [ep], cfg, splits, contract, features_only=args.features_only)
            if next_report["pending_steps"]:
                check_storage(used + added, next_report["pending_bytes"], quota,
                              shutil.disk_usage(root).free, skip_home_budget=args.skip_home_budget)
            meta, status = cache_episode(ep, source, root, cfg, splits, contract, policy, args.device,
                                         backend=run.get("cache_source_backend", run.get("sample_backend", "npy")), batch_size=args.batch_size,
                                         decoder_workers=args.decoder_workers, resume=True, save_jpeg=not args.features_only)
            if status == "CACHED":
                added += sum(p.stat().st_size for p in (root / ep).iterdir() if p.is_file())
            check_storage(used + added, 0, quota, shutil.disk_usage(root).free,
                          skip_home_budget=args.skip_home_budget)
            print(f"MCR_CACHE {i}/{len(selected)} {ep} {status} steps={meta['steps']} elapsed_s={time.perf_counter()-started:.1f} max_quant_error={meta['quantization_max_absolute_error']:.6f}", flush=True)
        report = estimate(root, source, names, cfg, splits, contract, features_only=args.features_only)
        write_json(root / "storage_report.json", {**report, "home_used_before_GiB": used / 2**30,
                                                 "bytes_added": added, "quota_bytes": quota,
                                                 "script_home_budget_enabled": not args.skip_home_budget})
        if args.command == "build":
            marker = "READY_FROZEN.json" if args.features_only else "READY.json"
            write_json(root / marker, {"contract_hash": digest(contract), "episodes": training_names,
                                            "contains_test": False})
            if args.include_test:
                write_json(root / "READY_TEST_FROZEN.json", {"contract_hash": digest(contract),
                           "episodes": splits["splits"]["test"], "contains_test": True})
            print("MCR_CACHE_SPLITS", {k: len(splits["splits"][k]) for k in
                  (("train", "val", "test") if args.include_test else ("train", "val"))}, flush=True)
        print("MCR_PILOT_OK" if args.command == "pilot" else "MCR_CACHE_READY", report, flush=True)


if __name__ == "__main__":
    main()
