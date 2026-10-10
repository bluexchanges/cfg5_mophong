"""Bounded batch/worker trials, then resume the selected run to 40 epochs.

Trials run in isolated processes and never save learned trial weights.
Only the untouched source checkpoint is used to fork the actual training.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

CANDIDATES = ((4, 2), (8, 2), (8, 4), (16, 2), (16, 4), (32, 4))


def choose_trial(rows, *, vram_limit=0.85):
    valid = [r for r in rows if r.get("status") == "ok"
             and math.isfinite(r["samples_per_s"]) and r["samples_per_s"] > 0
             and r["total_vram_bytes"] > 0
             and 0 <= r["peak_reserved_bytes"] / r["total_vram_bytes"] <= vram_limit]
    if not valid:
        raise RuntimeError("Không có cấu hình đạt giới hạn VRAM; chưa train dài")
    fastest = max(r["samples_per_s"] for r in valid)
    # A short probe cannot reliably distinguish differences below 3%.
    near = [r for r in valid if r["samples_per_s"] >= fastest * 0.97]
    return min(near, key=lambda r: (r["batch_size"], r["num_workers"]))


def trial_config(old, batch_size, num_workers):
    return dict(old, epochs=40, batch_size=batch_size, num_workers=num_workers,
                cuda_prefetch=True, output_dir="outputs/act_pretrain_fast")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def trial(source, batch_size, workers, result_path, warmup=3, measured=20):
    import torch
    from src.training import pretrain
    from src.data.common import ROOT
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "CUDA required"
    saved = torch.load(source, map_location="cpu", weights_only=True)
    assert saved["kind"] == "train_state" and saved["smoke"] is False
    cfg = trial_config(saved["metadata"]["run_config"], batch_size, workers)
    config_path = result_path.with_suffix(".config.json")
    write_json(config_path, cfg)
    run, model_cfg, data_cfg, splits, stats, contract = pretrain.load_run(config_path)
    pretrain.check_contract(saved, contract)
    generator = torch.Generator()
    generator.set_state(saved["loader_generator_state"].cpu())
    loader = pretrain.make_loader(ROOT / run["processed_root"], splits, stats, data_cfg,
                                 "train", batch_size, workers, generator=generator)
    policy = pretrain.RH20TACTPolicy(model_cfg, data_cfg).cuda()
    policy.load_state_dict(saved["model"], strict=True)
    optimizer = policy.configure_optimizer()
    optimizer.load_state_dict(saved["optimizer"])
    pretrain.restore_rng(saved["rng"])
    del saved
    pretrain.run_epoch(policy, loader, "cuda", optimizer=optimizer, max_batches=warmup,
                       grad_clip_norm=run["grad_clip_norm"], cuda_prefetch=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = pretrain.run_epoch(policy, loader, "cuda", optimizer=optimizer,
                               max_batches=measured, grad_clip_norm=run["grad_clip_norm"],
                               cuda_prefetch=True)
    torch.cuda.synchronize()
    if result["batches"] != measured:
        raise RuntimeError("Probe không đủ batch thực tế")
    record = {"status": "ok", "batch_size": batch_size, "num_workers": workers,
              "samples_per_s": result["timing"]["samples_per_s"],
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
              "warmup_batches": warmup, "measured_batches": measured,
              "source_checkpoint": str(source), "train_probe": result,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0)}
    write_json(result_path, record)
    print(json.dumps(record), flush=True)


def run_trials(source, report_dir, *, measured=20, warmup=3, timeout=300):
    rows = []
    for batch, workers in CANDIDATES:
        result_path = report_dir / f"bs{batch}-w{workers}.json"
        print(f"BENCHMARK batch={batch}, workers={workers}: {warmup}+{measured} batch", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--trial",
                   "--source", str(source), "--batch", str(batch), "--workers", str(workers),
                   "--result", str(result_path), "--measured", str(measured), "--warmup", str(warmup)]
        try:
            subprocess.run(command, check=True, timeout=timeout)
            row = json.loads(result_path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            # Parent run remains untouched; each trial has an independent CUDA context.
            row = {"status": "timeout", "batch_size": batch, "num_workers": workers}
            write_json(result_path, row)
        rows.append(row)
        write_json(report_dir / "trials.json", rows)
        if row["status"] == "ok":
            print(f"RESULT batch={batch}, workers={workers}: {row['samples_per_s']:.2f} mẫu/s; "
                  f"VRAM reserved peak={row['peak_reserved_bytes']/1024**3:.2f} GiB", flush=True)
        else:
            print(f"RESULT batch={batch}, workers={workers}: {row['status']}", flush=True)
    chosen = choose_trial(rows)
    write_json(report_dir / "selected.json", chosen)
    return chosen


def continue_training(source, report_dir):
    import torch
    from src.data.common import ROOT
    from src.training import pretrain
    from src.training.fork_run import fork_run
    child = ROOT / "outputs/act_pretrain_fast/latest.pt"
    config_path = ROOT / "configs/act/cfg5_all_pretrain_fast40.json"
    if child.exists():
        saved = torch.load(child, map_location="cpu", weights_only=True)
        if source.resolve() != child.resolve():
            parent = torch.load(source, map_location="cpu", weights_only=True)
            if saved["epoch"] < parent["epoch"]:
                raise RuntimeError("Nhánh fast cũ hơn nguồn; cần output mới để tránh quay lùi")
            del parent
        cfg = dict(saved["metadata"]["run_config"], epochs=40)
        if cfg["output_dir"] != "outputs/act_pretrain_fast" or not cfg.get("cuda_prefetch"):
            raise ValueError("Checkpoint fast không đúng cấu hình")
        del saved
    else:
        chosen = run_trials(source, report_dir)
        saved = torch.load(source, map_location="cpu", weights_only=True)
        cfg = trial_config(saved["metadata"]["run_config"], chosen["batch_size"], chosen["num_workers"])
        del saved
    write_json(config_path, cfg)
    run, model, data, splits, stats, contract = pretrain.load_run(config_path)
    if not child.exists():
        fork_run(run, model, data, splits, stats, contract, source)
    print(f"SELECTED batch_size={run['batch_size']}; num_workers={run['num_workers']}; "
          "CPU allocation=8; cuda_prefetch=True; tổng 40 epoch", flush=True)
    pretrain.train(run, model, data, splits, stats, contract, device=torch.device("cuda"), resume=child)


def main():
    # The script can be executed from a bundle; cwd is the installed project.
    sys.path.insert(0, str(Path.cwd()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=20)
    args = parser.parse_args()
    if args.trial:
        if args.batch not in (4, 8, 16, 32) or args.workers not in (2, 4) or args.result is None:
            parser.error("Trial cần batch/worker/result hợp lệ")
        if not 1 <= args.warmup <= 10 or not 1 <= args.measured <= 100:
            parser.error("Số batch benchmark vượt giới hạn")
        import torch
        try:
            trial(args.source, args.batch, args.workers, args.result, args.warmup, args.measured)
        except torch.cuda.OutOfMemoryError:
            write_json(args.result, {"status": "oom", "batch_size": args.batch,
                                     "num_workers": args.workers})
    else:
        if args.report_dir is None:
            parser.error("Cần --report-dir")
        args.report_dir.mkdir(parents=True, exist_ok=True)
        continue_training(args.source, args.report_dir)


if __name__ == "__main__":
    main()
