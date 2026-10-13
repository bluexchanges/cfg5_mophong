"""After frozen training: pilot JPEG, then convert and release features episode by episode."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from pathlib import Path
import random
import shutil
import _bootstrap
from src.data.common import ROOT, digest, outside_raw, read_json
from src.data.feature_cache import check_episode, require_ready, FEATURE_ONLY
from src.data.staged_mcr import (atomic_json, jpeg_storage_projection, materialize_jpeg_episode,
                                release_feature_episode, validate_frozen_checkpoint)
from scripts.prepare_mcr_cache import home_usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("pilot", "build", "check"))
    p.add_argument("--cache-root", required=True)
    p.add_argument("--frozen-checkpoint", required=True)
    p.add_argument("--release-features", action="store_true",
                   help="Build only: revoke frozen readiness, verify each JPEG episode, then unlink its features/scales")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--decoder-workers", type=int, default=8)
    p.add_argument("--pilot-episodes", type=int, default=3)
    p.add_argument("--home-quota-tb", type=float, default=1.8)
    p.add_argument("--home-used-gib", type=float)
    p.add_argument("--full-check", action="store_true")
    args = p.parse_args()
    if (min(args.batch_size, args.decoder_workers, args.pilot_episodes) < 1 or args.decoder_workers > 16
            or args.home_quota_tb <= 0 or (args.home_used_gib is not None and args.home_used_gib < 0)):
        p.error("Invalid batch/workers/pilot/storage budget")
    if (args.command == "build") != args.release_features:
        p.error("build requires --release-features; pilot/check never release features")
    ctx = read_json(ROOT / args.cache_root / "context.json")
    if ctx.get("storage_mode") != "features_only_v1":
        raise ValueError("Chỉ chuyển cache features-only staged; không sửa cache combined cũ")
    cfg, split, contract = ctx["data_config"], ctx["splits"], ctx["contract"]
    root = outside_raw(ROOT / args.cache_root, cfg)
    origin = validate_frozen_checkpoint(ROOT / args.frozen_checkpoint, ctx)
    source_run = read_json(ROOT / ctx["source_config"])
    source = Path(ctx["source_root"])
    backend = source_run.get("cache_source_backend", source_run.get("sample_backend", "npy"))
    if (source != (ROOT / source_run["processed_root"]).resolve()
            or digest(read_json(ROOT / source_run["stats"])) != digest(ctx["stats"])
            or read_json(ROOT / source_run["splits"]) != split):
        raise ValueError("Source run/light/statistics changed")
    names = split["splits"]["train"] + split["splits"]["val"]
    expected = {"contract_hash": digest(contract), "episodes": names, "contains_test": False}
    import fcntl
    with (root / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "check":
            require_ready(root, ctx, "full")
            for name in names:
                check_episode(root / name, cfg, split, contract, stage="full", full=args.full_check)
            print("MCR_JPEG_CHECK_OK", flush=True)
            return
        transition_path = root / "JPEG_CONVERSION.json"
        transition = {"cache_context_hash": digest(ctx), "checkpoint": origin, "episodes": names}
        if transition_path.exists():
            if read_json(transition_path) != transition:
                raise ValueError("JPEG conversion thuộc frozen checkpoint/context khác")
            if args.command == "pilot":
                raise ValueError("Conversion đã bắt đầu; resume build, không quay lại pilot")
        else:
            require_ready(root, ctx, "frozen")
        report = jpeg_storage_projection(root, ctx)
        print("JPEG_STORAGE", report, flush=True)
        if args.command == "build" and report["jpeg_completed_episodes"] < min(3, len(split["splits"]["train"])):
            raise ValueError("Chạy JPEG pilot tối thiểu 3 episode sau frozen trước khi convert full")
        selected = (random.Random(42).sample(sorted(split["splits"]["train"]), min(args.pilot_episodes, len(split["splits"]["train"])))
                    if args.command == "pilot" else names)
        used = int(args.home_used_gib * 2**30) if args.home_used_gib is not None else home_usage()
        quota, reserve = int(args.home_quota_tb * 10**12), 10 * 2**30
        if args.command == "pilot":
            pending = [read_json(root / name / "episode.json")["steps"] for name in selected
                       if read_json(root / name / "episode.json")["image_storage"] == FEATURE_ONLY]
            extra = int(sum(pending) * cfg["image_width"] * cfg["image_height"] * 3 * 8 * 1.2 * 1.1 + 2**30)
        else:
            extra = report["estimated_peak_extra_bytes"]
        if used + extra + reserve > quota or shutil.disk_usage(root).free < extra + reserve:
            raise OSError(f"JPEG conversion vượt ngân sách: used={used/2**30:.1f}GiB, peak thêm~{extra/2**30:.1f}GiB; features chưa xóa")
        if args.command == "build":
            atomic_json(transition_path, transition)
            (root / "READY_FULL.json").unlink(missing_ok=True)
            (root / "READY_FROZEN.json").unlink(missing_ok=True)
            # Already-created pilot JPEG episodes release space first.
            selected = sorted(selected, key=lambda name: (read_json(root / name / "episode.json")["image_storage"] == FEATURE_ONLY, name))
        added = released = 0
        for i, name in enumerate(selected, 1):
            remaining = quota - used - added + released - reserve
            if remaining <= 0:
                raise OSError("Home không còn ngân sách để tạo JPEG episode kế tiếp")
            _, status, new_bytes = materialize_jpeg_episode(name, source, root, cfg, split, contract, backend,
                batch_size=args.batch_size, decoder_workers=args.decoder_workers, max_jpeg_bytes=remaining)
            added += new_bytes
            freed = release_feature_episode(name, root, cfg, split, contract, origin) if args.release_features else 0
            released += freed
            atomic_json(root / "jpeg_progress.json", {"checkpoint": origin, "last_episode": name,
                        "episodes_processed_this_job": i, "bytes_added_this_job": added,
                        "feature_bytes_released_this_job": released, "raw_preserved": True})
            print(f"MCR_JPEG {i}/{len(selected)} {name} {status} released_GiB={freed/2**30:.3f}", flush=True)
        report = jpeg_storage_projection(root, ctx)
        atomic_json(root / "jpeg_storage_report.json", report)
        if args.command == "build":
            for name in names:
                check_episode(root / name, cfg, split, contract, stage="full")
                if any((root / name / f).exists() for f in ("features.npy", "scales.npy")):
                    raise ValueError("Feature retirement incomplete; no full readiness marker")
            atomic_json(root / "READY_FULL.json", expected)
        print("MCR_JPEG_PILOT_OK" if args.command == "pilot" else "MCR_JPEG_FULL_READY", report, flush=True)


if __name__ == "__main__":
    main()
