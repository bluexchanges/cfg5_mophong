"""Dùng split đã khóa: metadata nhỏ, kiểm nguồn, cấu hình train MP4; không ghi ảnh."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "OPENCV_FFMPEG_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time

import _bootstrap
from src.data.common import ROOT, digest, load_config, outside_raw, read_json, require_contract
from src.data.mp4_dataset import BACKEND, MP4ACTDataset, make_mp4_loader, prepare_light, verify_light
from src.data.sim_demos import file_hash
from src.data.splits import validate_splits
from src.training.pretrain import load_run, write_json

BEST_SHA = "7567df2b01291fbd8f096cce5693bde542fbfc9709b9d5b2d7f18d722863ac9e"
BASE = "configs/act/pick_place_finetune.json"
RUN = "configs/act/pick_place_finetune_mp4_w16.json"
SMOKE_RUN = "configs/act/pick_place_finetune_mp4_smoke.json"


def locked_inputs(base_path=BASE):
    run, model, cfg, splits, stats, contract = load_run(ROOT / base_path)
    if cfg.get("source_type") != "mujoco_scripted_controller" or cfg.get("action_target_offset") != 0:
        raise ValueError("Cần locked split demo pre-step")
    validate_splits(splits)
    require_contract(splits, cfg)
    require_contract(stats, cfg)
    if (stats.get("smoke_only") is not False or stats.get("source_split") != "train"
            or stats["split_hash"] != splits["split_hash"]
            or not set(stats["episodes"]).issubset(splits["splits"]["train"])):
        raise ValueError("Normalization không đúng train/locked split")
    manifest = read_json(ROOT / "data/splits/pick_place_finetune_manifest.json")
    if (manifest["config_hash"] != digest(cfg) or manifest["manifest_hash"] != splits["manifest_hash"]
            or digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]):
        raise ValueError("Manifest hash/config/split khác nhau")
    records = {row["episode"]: row for row in manifest["episodes"]}
    ids = [e for name in ("train", "val", "test") for e in splits["splits"][name]]
    if len(records) != len(manifest["episodes"]) or set(ids) != set(records):
        raise ValueError("Manifest membership khác locked splits")
    return run, model, cfg, splits, stats, contract, records


def write_locked_runtime(path, value):
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Runtime config đã tồn tại và khác: {path}; không ghi đè")
    else:
        write_json(path, value)


def work(payload):
    record, cfg, root, split_hash, resume = payload
    if (Path(root) / record["episode"]).exists() and resume:
        verify_light(record, cfg, root, split_hash)
        return record["episode"], "SKIP_VERIFIED"
    prepare_light(record, cfg, root, split_hash)
    return record["episode"], "LIGHT_READY"


def prepare(args):
    run, _, cfg, splits, stats, _, records = locked_inputs(args.base_config)
    checkpoint = ROOT / "outputs/act_pretrain_bs8_to80/best.pt"
    if file_hash(checkpoint) != BEST_SHA or stats.get("inherited_from", {}).get("checkpoint_sha256") != BEST_SHA:
        raise ValueError("Best checkpoint/stats khác bản pretrain epoch58 đã test")
    if args.workers < 1 or args.workers > 16:
        raise ValueError("Preparation workers phải 1..16")
    output = outside_raw(ROOT / "data/processed/pick_place_finetune_mp4", cfg)
    selected = [records[e] for name in ("train", "val") for e in splits["splits"][name]]
    max_bytes = max(r["quality"]["steps"] * 8 * cfg["image_height"] * cfg["image_width"] * 3 for r in selected)
    if max_bytes > 3.5 * 2**30:
        raise MemoryError(f"Episode lớn nhất {max_bytes / 2**30:.2f} GiB vượt cache 3.5 GiB; cần cấu hình cache riêng")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Numeric state/action/times/segments plus metadata; no decoded image estimate.
    estimate = sum(r["quality"]["steps"] * (8*4*2 + 8 + 4) + 32768 for r in selected)
    if shutil.disk_usage(output.parent).free < estimate * 1.1 + 5 * 2**30:
        raise OSError("Thiếu disk cho metadata và dự phòng checkpoint 5 GiB")
    print(f"LIGHT_PREPARE episodes={len(selected)} workers={args.workers} "
          f"estimated_MiB={estimate / 2**20:.1f} max_episode_cache_GiB={max_bytes / 2**30:.3f} "
          f"split_hash={splits['split_hash']}; không tạo images.npy", flush=True)
    payloads = [(r, cfg, str(output), splits["split_hash"], args.resume) for r in selected]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, (episode, status) in enumerate(pool.map(work, payloads, chunksize=1), 1):
            print(f"{status} {i}/{len(selected)} {episode}", flush=True)
    runtime = dict(run, sample_backend=BACKEND, num_workers=16, val_num_workers=2,
                   prefetch_factor=1, video_cache_gib=3.5, gpu_count=1, batch_size=8,
                   cuda_prefetch=True, processed_root=output.relative_to(ROOT).as_posix(),
                   output_dir="outputs/pick_place_finetune_mp4_w16")
    write_locked_runtime(ROOT / RUN, runtime)
    smoke = dict(runtime, num_workers=1, val_num_workers=1,
                 output_dir="outputs/pick_place_finetune_mp4_w16/smoke_check")
    write_locked_runtime(ROOT / SMOKE_RUN, smoke)
    write_json(ROOT / "outputs/data_audit/pick_place_mp4_light.json",
               {"backend": BACKEND, "split_hash": splits["split_hash"], "stats_hash": digest(stats),
                "counts": {name: len(splits["splits"][name]) for name in ("train", "val", "test")},
                "numeric_estimated_MiB": estimate / 2**20,
                "max_episode_cache_GiB": max_bytes / 2**30,
                "test_prepared": False, "run_config": RUN, "epochs": runtime["epochs"]})
    print("MP4_LIGHT_PREPARE_OK", flush=True)


def check(args):
    run, _, cfg, splits, stats, contract, records = locked_inputs(args.config)
    if run.get("sample_backend") != BACKEND:
        raise ValueError("Config chưa chọn backend MP4")
    root = ROOT / run["processed_root"]
    for name in ("train", "val"):
        for episode in splits["splits"][name]:
            verify_light(records[episode], cfg, root, splits["split_hash"])
        dataset = MP4ACTDataset(root, splits, stats, cfg, name, cache_gib=run["video_cache_gib"])
        print(f"CHECK split={name} episodes={len(dataset.episodes)} samples={len(dataset)} "
              f"max_cache_GiB={dataset.max_episode_bytes / 2**30:.3f}", flush=True)
    if file_hash(ROOT / "outputs/act_pretrain_bs8_to80/best.pt") != BEST_SHA:
        raise ValueError("Best checkpoint SHA đã đổi")
    print(json.dumps({"batch_size": run["batch_size"], "train_workers": run["num_workers"],
                      "val_workers": run["val_num_workers"], "epochs": run["epochs"],
                      "backend": BACKEND, "contract": contract}, ensure_ascii=False), flush=True)
    print("MP4_DATA_CHECK_OK; chưa đọc ảnh test", flush=True)


def benchmark(args):
    run, _, cfg, splits, stats, _, _ = locked_inputs(args.config)
    workers = run["num_workers"] if args.workers is None else args.workers
    if workers not in (0, 1, 2, 4, 8, 16) or args.batches < 1:
        raise ValueError("workers: 0/1/2/4/8/16; batches>=1")
    import torch
    loader = make_mp4_loader(ROOT / run["processed_root"], splits, stats, cfg, "train",
                             run["batch_size"], workers, generator=torch.Generator().manual_seed(run["seed"]),
                             prefetch_factor=1, cache_gib=run["video_cache_gib"])
    started = time.perf_counter()
    iterator = iter(loader)
    samples = 0
    first_seconds = None
    waits = []
    for i in range(min(args.batches, len(loader))):
        before = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - before)
        samples += len(batch["state"])
        if first_seconds is None:
            first_seconds = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    print(json.dumps({"workers": workers, "batches": len(waits), "samples": samples,
                      "first_batch_seconds": first_seconds, "elapsed_seconds": elapsed,
                      "loader_samples_per_s_including_startup": samples / elapsed,
                      "mean_fetch_ms_after_first": 1000 * sum(waits[1:]) / max(1, len(waits)-1),
                      "note": "Loader-only CPU benchmark; không phải throughput train/GPU"}), flush=True)
    # This command exits; persistent worker caches are released before subsequent training.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--base-config", default=BASE)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=prepare)
    p = commands.add_parser("check")
    p.add_argument("--config", default=RUN)
    p.set_defaults(func=check)
    p = commands.add_parser("benchmark")
    p.add_argument("--config", default=RUN)
    p.add_argument("--workers", type=int)
    p.add_argument("--batches", type=int, default=100)
    p.set_defaults(func=benchmark)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
