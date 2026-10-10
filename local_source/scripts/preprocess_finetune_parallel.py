"""Kiểm tra và preprocess demo đã khóa split, không đổi representation/tần số."""
import os

# Must be set before importing OpenCV/NumPy, including in spawned workers.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM", "OPENCV_FFMPEG_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import _bootstrap
import cv2
from prepare_finetune_data import paths
from src.data.common import ROOT, digest, load_config, outside_raw, read_json, write_json
from src.data.sim_demos import file_hash, inspect_episode, preprocess_episode, verify_processed
from src.data.splits import validate_splits


def bounded_results(function, payloads, workers):
    """Keep at most workers tasks running/queued; do not launch all episodes on failure."""
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    iterator = iter(payloads)
    running = set()
    try:
        for _ in range(workers):
            payload = next(iterator, None)
            if payload is not None:
                running.add(pool.submit(function, payload))
        while running:
            ready, running = wait(running, return_when=FIRST_COMPLETED)
            # Resolve all completed results before dispatching the next group of tasks.
            results = [future.result() for future in ready]
            yield from results
            for _ in ready:
                payload = next(iterator, None)
                if payload is not None:
                    running.add(pool.submit(function, payload))
    finally:
        for future in running:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def check_episode(payload):
    record, cfg, output, split_hash = payload
    cv2.setNumThreads(1)
    folder = Path(output) / record["episode"]
    if folder.with_name(folder.name + ".partial").exists():
        raise FileExistsError(f"Episode dở: {folder}.partial; đọc ERROR.json, không tự xóa")
    if folder.exists():
        verify_processed(record, cfg, output, split_hash)
        return {"episode": record["episode"], "existing": True,
                "steps": record["quality"]["steps"]}
    actual = inspect_episode(record["path"], cfg)
    for key in ("episode", "seed", "trajectory_hash", "camera_rig_hash", "source_fingerprint"):
        if actual[key] != record[key]:
            raise ValueError(f"Raw đổi từ lúc khóa manifest: {record['episode']} {key}")
    if actual["quality"]["steps"] != record["quality"]["steps"]:
        raise ValueError("Số thời điểm đổi so với manifest")
    return {"episode": record["episode"], "existing": False,
            "steps": record["quality"]["steps"]}


def process_episode(payload):
    record, cfg, output, split_hash = payload
    started = time.monotonic()
    meta = preprocess_episode(record, cfg, output, split_hash)
    # Confirm all final arrays and metadata before marking the episode completed.
    verify_processed(record, cfg, output, split_hash)
    return {"episode": record["episode"], "steps": meta["steps"],
            "seconds": time.monotonic() - started}


def load_locked(name):
    locations = paths(name)
    cfg = load_config(locations["data_config"])
    manifest = read_json(locations["manifest"])
    splits = read_json(locations["splits"])
    stats = read_json(locations["stats"])
    run = read_json(locations["run_config"])
    validate_splits(splits)
    cfg_hash = digest(cfg)
    if (manifest["config_hash"] != cfg_hash or splits["config_hash"] != cfg_hash
            or splits["manifest_hash"] != manifest["manifest_hash"]
            or digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]):
        raise ValueError("Manifest/config/split hash không khớp")
    if (stats["config_hash"] != cfg_hash or stats["split_hash"] != splits["split_hash"]
            or stats["source_split"] != "train" or stats["smoke_only"]):
        raise ValueError("Normalization không thuộc train của split đã khóa")
    if (cfg.get("source_type") != "mujoco_scripted_controller"
            or cfg.get("action_target_offset") != 0 or cfg["control_hz"] != 10):
        raise ValueError("Bản chạy này yêu cầu demo same-tick đã audit ở 10 Hz")
    if Path(run["processed_root"]).resolve() != (ROOT / f"data/processed/{name}").resolve():
        raise ValueError("Run config dùng processed_root khác bản mặc định; kiểm đường dẫn trước")
    checkpoint = ROOT / "outputs/act_pretrain_bs8_to80/best.pt"
    if file_hash(checkpoint) != run["initial_checkpoint_sha256"]:
        raise ValueError("Checkpoint pretrain khác SHA đã khóa")
    return cfg, manifest, splits, run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pick_place_finetune")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or len(set(args.splits)) != len(args.splits):
        raise ValueError("workers phải >=1, không lặp split")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else allocated
    if args.workers > min(allocated, affinity):
        raise ValueError(f"Xin {allocated} CPU, affinity {affinity}; workers={args.workers} vượt CPU được cấp")
    started = time.monotonic()
    cfg, manifest, splits, run = load_locked(args.name)
    output = outside_raw(ROOT / run["processed_root"], cfg)
    by_id = {record["episode"]: record for record in manifest["episodes"]}
    if len(by_id) != len(manifest["episodes"]) or set(by_id) != {
            episode for names in splits["splits"].values() for episode in names}:
        raise ValueError("Manifest/split thiếu hoặc lặp episode")
    ids = [episode for split in args.splits for episode in splits["splits"][split]]
    records = [by_id[episode] for episode in ids]
    print(f"PREFLIGHT_BEGIN episodes={len(records)} workers={args.workers} cpu_allocated={allocated} "
          f"cpu_affinity={affinity} opencv={cv2.__version__} decoder_threads=1 control_hz=10 gpu_used=False", flush=True)
    payloads = [(record, cfg, str(output), splits["split_hash"]) for record in records]
    checks = {}
    for result in bounded_results(check_episode, payloads, args.workers):
        checks[result["episode"]] = result
        if len(checks) % 25 == 0 or len(checks) == len(records):
            print(f"PREFLIGHT {len(checks)}/{len(records)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    pending = [record for record in records if not checks[record["episode"]]["existing"]]
    bytes_per_tick = len(cfg["camera_serials"]) * cfg["image_height"] * cfg["image_width"] * 3
    estimate = sum(row["quality"]["steps"] * bytes_per_tick for row in pending)
    existing = output
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    required = int(estimate * 1.1) + (2**30 if pending else 0)
    summary = {"split_hash": splits["split_hash"], "manifest_hash": manifest["manifest_hash"],
               "splits": args.splits, "output": str(output), "workers": args.workers,
               "episodes": len(records), "verified_existing": len(records)-len(pending),
               "pending": len(pending), "pending_images_GiB": estimate/2**30,
               "required_free_GiB": required/2**30, "free_GiB": free/2**30,
               "quota_note": "disk free does not check user/group/project storage quota",
               "preflight_seconds": time.monotonic()-started}
    report = outside_raw(ROOT / f"outputs/data_audit/{args.name}_preflight16.json", cfg)
    write_json(report, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if free < required:
        raise OSError("Thiếu ổ đĩa (+10% và 1 GiB); chưa ghi ảnh. Kiểm cả quota tài khoản.")
    print("FINETUNE_PREFLIGHT_OK", flush=True)
    if args.check_only:
        return
    phase_start = time.monotonic()
    steps_done = 0
    total_steps = sum(row["quality"]["steps"] for row in pending)
    for number, result in enumerate(bounded_results(process_episode,
            [(row, cfg, str(output), splits["split_hash"]) for row in pending], args.workers), 1):
        steps_done += result["steps"]
        elapsed = time.monotonic()-phase_start
        speed = steps_done / max(elapsed, 1e-9)
        eta = (total_steps-steps_done)/max(speed, 1e-9)
        print(f"PREPROCESS {number}/{len(pending)} episode={result['episode']} "
              f"episode_s={result['seconds']:.1f} ticks_per_s={speed:.2f} "
              f"images_per_s={speed*len(cfg['camera_serials']):.2f} "
              f"write_MiB_per_s={speed*bytes_per_tick/2**20:.2f} "
              f"elapsed_min={elapsed/60:.1f} eta_min={eta/60:.1f}", flush=True)
    # CLI check creates ACT samples for first/middle/last of each selected split.
    import subprocess
    import sys
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts/prepare_finetune_data.py"),
                    "check", "--name", args.name, "--splits", *args.splits], check=True)
    summary.update(completed=True, preprocess_seconds=time.monotonic()-phase_start,
                   total_seconds=time.monotonic()-started)
    write_json(outside_raw(ROOT / f"outputs/data_audit/{args.name}_preprocess16_report.json", cfg), summary)
    print("FINETUNE_PREPROCESS16_OK", flush=True)


if __name__ == "__main__":
    main()
