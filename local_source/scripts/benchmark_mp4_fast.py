"""Đo batch/worker trên train; không ghi weights từ benchmark, không đọc test."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import _bootstrap
from src.data.common import ROOT, read_json


def select_case(rows):
    safe = [r for r in rows if r.get("ok") and r["samples_per_s"] > 0
            and r["peak_reserved_GiB"] <= .80 * r["device_GiB"]]
    if not safe:
        raise RuntimeError("Không có case đạt và đủ VRAM dự phòng; chưa tạo config train")
    return max(safe, key=lambda r: r["samples_per_s"])


def child(args):
    import hashlib
    import torch
    from itertools import islice
    from src.training import pretrain as p
    from src.models.rh20t_act import RH20TACTPolicy
    torch.set_num_threads(1)
    run, model, data, splits, stats, contract = p.load_run(ROOT / args.config)
    p.seed_everything(run["seed"])
    device = p.choose_device("cuda", require_cuda=True)
    policy = RH20TACTPolicy(model, data).to(device)
    source = Path(args.checkpoint)
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("kind") == "train_state":
        p.check_contract(saved, contract)
        if saved.get("smoke") is not False or saved["run_config_hash"] != p.resume_signature(run):
            raise ValueError("Benchmark cần latest.pt full đúng run nguồn")
        policy.load_state_dict(saved["model"], strict=True)
    else:
        checksum = hashlib.sha256()
        with source.open("rb") as f:
            for block in iter(lambda: f.read(2**20), b""):
                checksum.update(block)
        p.initialize_finetune(policy, saved, model, data, stats, checksum.hexdigest())
    del saved
    optimizer = policy.configure_optimizer()  # temporary; never saved or reused by real run
    norm = {"mean": data["image_normalization_mean"], "std": data["image_normalization_std"]}
    generator = torch.Generator().manual_seed(run["seed"])
    loader = p.make_loader(ROOT / run["processed_root"], splits, stats, data, "train",
                           args.batch, args.workers, generator=generator,
                           sample_backend=run["sample_backend"], prefetch_factor=1,
                           video_cache_gib=run.get("video_cache_gib", 3.5), images_uint8=args.fast)
    # Cover a first episode per worker plus a second decode boundary, not just
    # consume the queues filled during startup. This remains a finite subset.
    import math
    largest = max(len(indices) for _, indices in loader.dataset.groups)
    target_batches = max(args.batches, args.workers * math.ceil(largest / args.batch) + 80)
    policy.train()

    def step(tensors):
        result = policy(tensors["state"], tensors["images"], tensors["actions"], tensors["is_pad"])
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), run["grad_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        # Match trainer scalar synchronization; not a kernel-only timer.
        for key in ("loss", "l1", "kl", "valid_action_elements"):
            float(result[key].detach())
        return float(result["loss"].detach())

    startup = perf_counter()
    samples, measured, wait_seconds, step_seconds = 0, 0, 0., 0.
    first_batch_s = None
    held = None
    try:
        source_batches = islice(loader, args.warmup + target_batches)
        with closing(p.device_batches(source_batches, device, cuda_prefetch=True,
                                      threaded=args.fast, image_normalization=norm)) as batches:
            start = perf_counter()
            for i, (batch, tensors) in enumerate(batches):
                fetched = perf_counter()
                if first_batch_s is None:
                    first_batch_s = fetched - startup
                loss = step(tensors)
                torch.cuda.synchronize()
                ended = perf_counter()
                if i >= args.warmup:
                    samples += len(batch["state"])
                    measured += 1
                    wait_seconds += fetched - start
                    step_seconds += ended - fetched
                    if measured % 40 == 0:
                        print(f"BENCH batch={args.batch} workers={args.workers} fast={args.fast} "
                              f"measured={measured}/{target_batches} samples_per_s="
                              f"{samples / (wait_seconds+step_seconds):.2f}", flush=True)
                else:
                    torch.cuda.reset_peak_memory_stats()
                if held is None or len(tensors["state"]) == args.batch:
                    held = tensors
                start = perf_counter()
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    if measured != target_batches or held is None:
        raise RuntimeError("Không đủ batch train để benchmark")
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Workers stopped; time the same optimizer step with inputs already on CUDA.
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(10):
        step(held)
    torch.cuda.synchronize()
    resident_ms = (perf_counter() - start) * 100
    row = {"ok": True, "batch_size": args.batch, "num_workers": args.workers,
           "fast": args.fast, "samples": samples, "batches": measured,
           "samples_per_s": samples / (wait_seconds + step_seconds),
           "data_wait_ms": wait_seconds * 1000 / measured,
           "step_ms": step_seconds * 1000 / measured,
           "resident_step_ms": resident_ms, "resident_batch_size": len(held["state"]),
           "first_batch_s": first_batch_s, "peak_allocated_GiB": peak_allocated,
           "peak_reserved_GiB": peak_reserved,
           "device_GiB": torch.cuda.get_device_properties(device).total_memory / 2**30,
           "last_loss": loss, "checkpoint": str(Path(args.checkpoint).resolve()),
           "notes": "Temporary optimizer discarded; train only; finite short run, warm filesystem may affect comparisons."}
    p.write_json(args.result, row)
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/act/pick_place_finetune_mp4_w16.json")
    parser.add_argument("--checkpoint", default="outputs/pick_place_finetune_mp4_w16/latest.pt")
    parser.add_argument("--report-dir", default="outputs/mp4_fast_benchmark/latest")
    parser.add_argument("--batches", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.batches < 80 or args.warmup < 1:
        parser.error("Cần ít nhất 80 measured batches + warmup")
    if args.child:
        child(args)
        return
    from src.training import pretrain as p
    run, model, data, splits, stats, contract = p.load_run(ROOT / args.config)
    destination = ROOT / "outputs/pick_place_finetune_mp4_fast"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Output fast đã tồn tại; dùng train job --resume, không benchmark lại lên output cũ")
    report = ROOT / args.report_dir
    report.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError("Chưa có latest.pt fine-tune; chờ lưu epoch trước khi chuyển run")
    # baseline; then uint8+background H2D, higher batch and more workers.
    cases = [(8, 16, False), (8, 16, True), (16, 8, True), (16, 16, True), (16, 24, True),
             (32, 8, True), (32, 16, True), (32, 24, True)]
    rows = []
    for batch, workers, fast in cases:
        name = f"b{batch}_w{workers}_{'fast' if fast else 'baseline'}"
        result = report / (name + ".json")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
                   "--config", args.config, "--checkpoint", str(checkpoint),
                   "--batch", str(batch), "--workers", str(workers), "--result", str(result),
                   "--batches", str(args.batches), "--warmup", str(args.warmup)]
        if fast:
            command.append("--fast")
        print(f"BENCH_START {name}; tiến độ chi tiết: {report / (name + '.log')}", flush=True)
        with (report / (name + ".log")).open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                # Whole Slurm step timeout is the final guard; child workers must not survive.
                raise RuntimeError(f"Case {name} quá 60 phút; dừng job, xem log")
        row = read_json(result) if code == 0 and result.exists() else {
            "ok": False, "batch_size": batch, "num_workers": workers, "fast": fast, "exit_code": code}
        rows.append(row)
        print("BENCH_RESULT " + json.dumps(row), flush=True)
        p.write_json(report / "report.json", {"cases": rows, "contract": contract})
    baseline = rows[0]
    eligible = [r for r in rows if r.get("fast") and r["batch_size"] > run["batch_size"]
                and r["num_workers"] > run["num_workers"]]
    selected = select_case(eligible)
    summary = {"cases": rows, "selected": selected, "best_any": select_case(rows), "contract": contract,
               "speedup": selected["samples_per_s"] / baseline["samples_per_s"] if baseline.get("ok") else None}
    p.write_json(report / "report.json", summary)
    if not baseline.get("ok") or selected["samples_per_s"] <= baseline["samples_per_s"]:
        raise RuntimeError("Tăng cả batch/worker chưa nhanh hơn baseline; không tự đổi run. Xem report.json")
    selected_run = dict(run, batch_size=selected["batch_size"], num_workers=selected["num_workers"],
                        val_num_workers=2, prefetch_factor=1, image_transfer_dtype="uint8",
                        cuda_prefetch=True, cuda_prefetch_threaded=True,
                        output_dir="outputs/pick_place_finetune_mp4_fast")
    config = ROOT / "configs/act/pick_place_finetune_mp4_fast.json"
    p.write_json(config, selected_run)
    from src.training.fork_run import fork_run
    fork_run(selected_run, model, data, splits, stats, contract, checkpoint)
    print(f"SELECTED batch={selected_run['batch_size']} workers={selected_run['num_workers']} "
          f"speedup={summary['speedup']}; MP4_FAST_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
