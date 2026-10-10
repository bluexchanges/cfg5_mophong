"""Audit/split/preprocess two MuJoCo demo sources; create a separate ACT fine-tune run."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re
import shutil

import _bootstrap
from src.data.common import ROOT, digest, load_config, outside_raw, read_json, write_json
from src.data.sim_demos import (REPRESENTATION, discover_sources, file_hash, inherited_normalization,
                               inspect_episode, preprocess_episode, split_records, verify_processed)
from src.data.splits import validate_splits

BEST_SHA = "7567df2b01291fbd8f096cce5693bde542fbfc9709b9d5b2d7f18d722863ac9e"


def paths(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", name):
        raise ValueError("name phải lowercase chữ/số/underscore")
    return {"data_config": ROOT / f"configs/data/{name}8.json",
            "model_config": ROOT / f"configs/act/{name}_model.json",
            "run_config": ROOT / f"configs/act/{name}.json",
            "manifest": ROOT / f"data/splits/{name}_manifest.json",
            "splits": ROOT / f"data/splits/{name}_splits.json",
            "stats": ROOT / f"data/splits/{name}_normalization.json"}


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def simulation_config(base, roots):
    cfg = deepcopy(base)
    cfg.update(dataset_root=str(Path(roots[0]).resolve()), raw_sources=[str(Path(r).resolve()) for r in roots],
               source_type="mujoco_scripted_controller", representation=REPRESENTATION,
               action_target_offset=0, gripper_scale_to_m=1.0, gripper_valid_range_m=[0.0, 0.085],
               split_group_fields=["seed_or_same_trajectory"], split_ratios=[0.8, 0.1, 0.1], seed=42)
    return cfg


def inspect(args):
    cfg = simulation_config(read_json(ROOT / "configs/data/cfg5_all8.json"), args.episodes)
    records = [inspect_episode(ep, cfg) for ep in args.episodes]
    result = {"records": records, "notes": "Samples inspected only; no full-server split created, no raw modified."}
    if args.report:
        write_json(outside_raw(args.report, cfg), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def prepare(args):
    import torch
    destinations = paths(args.name)
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(f"Đã khóa artifact {path}; dùng preprocess --resume, không chia lại")
    checkpoint = Path(args.init_checkpoint).resolve()
    checksum = file_hash(checkpoint)
    if args.expected_sha and checksum != args.expected_sha:
        raise ValueError("Checksum best.pt khác checkpoint epoch58 đã test; kiểm lại nguồn")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("kind") != "best_model" or saved.get("smoke") is not False:
        raise ValueError("Cần best_model full, không dùng latest/partial/smoke")
    cfg = simulation_config(saved["metadata"]["data_config"], args.sources)
    for path in destinations.values():
        outside_raw(path, cfg)
    records, rejected, duplicates = discover_sources(args.sources, cfg)
    cfg["camera_rig_hash"] = records[0]["camera_rig_hash"]
    audit = {"accepted": len(records), "rejected": rejected, "duplicates": duplicates,
             "sources": cfg["raw_sources"], "profiles": dict(Counter(row["profile"] for row in records)),
             "records": records, "checkpoint_sha256": checksum, "checkpoint_epoch": saved["epoch"] + 1}
    report = outside_raw(ROOT / f"outputs/data_audit/{args.name}.json", cfg)
    write_json(report, audit)
    if rejected and not args.exclude_invalid:
        raise ValueError(f"Có {len(rejected)} demo lỗi; xem {report}. Chỉ thêm --exclude-invalid sau khi đọc lý do")
    manifest = {"schema_version": 1, "config_hash": digest(cfg), "config": cfg,
                "episodes": records, "rejected": rejected, "duplicates": duplicates}
    manifest["manifest_hash"] = digest(manifest)
    splits = split_records(records, cfg, manifest["manifest_hash"])
    stats = inherited_normalization(saved, checksum, cfg, splits)
    model = deepcopy(saved["metadata"]["model_config"])
    model["data_config"] = relative(destinations["data_config"])
    model["backbone_weights_path"] = None
    model["note"] = "Fine-tune recorded same-tick command, initialized from validation-selected CFG5 best; numeric CFG5 train stats retained."
    run = {"model_config": relative(destinations["model_config"]), "splits": relative(destinations["splits"]),
           "locked_membership": relative(destinations["splits"]), "stats": relative(destinations["stats"]),
           "processed_root": f"data/processed/{args.name}", "output_dir": f"outputs/{args.name}",
           "seed": 42, "epochs": args.epochs, "batch_size": 8, "num_workers": 4,
           "gpu_count": 1, "cuda_prefetch": True, "save_every_epochs": 1,
           "log_every_batches": 100, "grad_clip_norm": 1.0, "require_cuda_for_full_run": True,
           "initial_checkpoint_sha256": checksum}
    if args.epochs < 1:
        raise ValueError("epochs phải >=1")
    payloads = {"data_config": cfg, "model_config": model, "run_config": run,
                "manifest": manifest, "splits": splits, "stats": stats}
    for key, value in payloads.items():
        write_json(destinations[key], value)
    by_id = {row["episode"]: row for row in records}
    sizes = {name: sum(by_id[e]["processed_image_bytes"] for e in ids) / 2**30
             for name, ids in splits["splits"].items()}
    print(json.dumps({"counts": splits["counts"], "profiles": splits["profiles"], "duplicates_skipped": len(duplicates),
                      "estimated_processed_images_GiB": sizes, "report": str(report),
                      "artifacts": {key: relative(path) for key, path in destinations.items()}},
                     ensure_ascii=False, indent=2))
    print("FINETUNE_SPLIT_LOCKED", flush=True)


def work_episode(payload):
    record, cfg, output, split_hash, max_steps = payload
    meta = preprocess_episode(record, cfg, output, split_hash, max_steps)
    return {"episode": meta["episode"], "steps": meta["steps"]}


def preprocess(args):
    ps = paths(args.name)
    cfg, manifest, splits = load_config(ps["data_config"]), read_json(ps["manifest"]), read_json(ps["splits"])
    validate_splits(splits)
    if (manifest["config_hash"] != digest(cfg) or splits["config_hash"] != digest(cfg)
            or splits["manifest_hash"] != manifest["manifest_hash"]
            or digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]):
        raise ValueError("Manifest/config/split hash không khớp")
    output = outside_raw(args.output_root or ROOT / f"data/processed/{args.name}", cfg)
    if args.max_steps and args.max_steps < 2:
        raise ValueError("max-steps phải >=2")
    if args.workers < 1 or args.limit_episodes is not None and args.limit_episodes < 1:
        raise ValueError("workers/limit-episodes phải >=1")
    by_id = {record["episode"]: record for record in manifest["episodes"]}
    selected = [by_id[e] for name in args.splits for e in
                (splits["splits"][name][:args.limit_episodes] if args.limit_episodes else splits["splits"][name])]
    if len({record["episode"] for record in selected}) != len(selected):
        raise ValueError("Không lặp split trong cùng lệnh")
    pending = []
    for record in selected:
        folder = output / record["episode"]
        if folder.with_name(folder.name + ".partial").exists():
            raise FileExistsError(f"Có episode đang dở: {folder}.partial; đọc ERROR.json trước khi xử lý")
        if folder.exists():
            if not args.resume:
                raise FileExistsError(f"Đã có {folder}; dùng --resume")
            verify_processed(record, cfg, output, splits["split_hash"], args.max_steps)
            print(f"SKIP_VERIFIED {record['episode']}", flush=True)
        else:
            pending.append(record)
    estimate = sum(min(row["quality"]["steps"], args.max_steps or row["quality"]["steps"]) *
                   8 * cfg["image_height"] * cfg["image_width"] * 3 for row in pending)
    existing = output
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    print(f"PREPROCESS pending={len(pending)} workers={args.workers} images_GiB={estimate/2**30:.2f} free_GiB={free/2**30:.2f}", flush=True)
    if free < estimate * 1.1 + 2**30:
        raise OSError("Thiếu dung lượng trống (+10% và 1 GiB dự phòng); không decode data")
    payloads = [(row, cfg, str(output), splits["split_hash"], args.max_steps) for row in pending]
    if args.workers == 1:
        results = map(work_episode, payloads)
        for i, result in enumerate(results, 1):
            print(f"PREPROCESS {i}/{len(pending)} {result['episode']} steps={result['steps']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, result in enumerate(pool.map(work_episode, payloads), 1):
                print(f"PREPROCESS {i}/{len(pending)} {result['episode']} steps={result['steps']}", flush=True)
    print("FINETUNE_PREPROCESS_OK", flush=True)


def check(args):
    from src.data.act_dataset import ACTDataset
    ps = paths(args.name)
    cfg, splits, stats = load_config(ps["data_config"]), read_json(ps["splits"]), read_json(ps["stats"])
    root = args.processed_root or ROOT / f"data/processed/{args.name}"
    summary = {}
    for name in args.splits:
        dataset = ACTDataset(root, splits, stats, cfg, name, allow_partial_smoke=args.smoke)
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            sample = dataset[index]
            print(f"CHECK {name} sample={index} episode={sample['episode']} t={sample['timestamp_ms']} actions={sample['actions'].shape} valid={int((~sample['is_pad']).sum())}")
        summary[name] = {"episodes": len(dataset.episodes), "samples": len(dataset),
                         "batches_bs8": (len(dataset) + 7) // 8}
    print(json.dumps(summary, indent=2))
    print("FINETUNE_DATASET_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("inspect", help="Khảo sát episode mẫu; không tạo split")
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--report")
    p.set_defaults(func=inspect)
    p = commands.add_parser("prepare", help="Audit tất cả nguồn; khóa manifest/split và config fine-tune")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--expected-sha", default=BEST_SHA)
    p.add_argument("--exclude-invalid", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.set_defaults(func=prepare)
    p = commands.add_parser("preprocess", help="Giải mã memmap từng episode; có multiprocessing")
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    p.add_argument("--output-root")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit-episodes", type=int)
    p.add_argument("--max-steps", type=int)
    p.set_defaults(func=preprocess)
    p = commands.add_parser("check", help="Kiểm samples train/val, không optimizer")
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    p.add_argument("--processed-root")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(func=check)
    for name, sub in commands.choices.items():
        if name != "inspect":
            sub.add_argument("--name", default="pick_place_finetune")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
