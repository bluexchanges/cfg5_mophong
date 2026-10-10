"""Create an explicit batch/worker experiment from a saved epoch boundary."""

import os
import shutil
from pathlib import Path

import torch

from src.data.common import ROOT, outside_raw, read_json
from src.training.pretrain import (atomic_torch_save, check_contract, checkpoint_metadata,
                                   resume_signature, write_json)


def fork_run(run, model_cfg, data_cfg, splits, stats, contract, source):
    """Keep optimizer/RNG/epoch; permit batch/workers, target and output changes.

    The original checkpoint is never edited. The child is a new experiment,
    with its own signature so subsequent ordinary --resume stays strict.
    """
    source = Path(source).resolve()
    out = outside_raw(ROOT / run["output_dir"], data_cfg)
    if out == source.parent:
        raise ValueError("Fork cần output_dir mới, khác thư mục checkpoint nguồn")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output fork phải trống: {out}")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    check_contract(saved, contract)
    if saved.get("kind") != "train_state" or saved.get("smoke") is not False:
        raise ValueError("Fork cần latest.pt train_state của full training")
    old_run = saved["metadata"]["run_config"]
    if saved["run_config_hash"] != resume_signature(old_run):
        raise ValueError("Metadata run config nguồn không khớp hash checkpoint")
    permitted = {"epochs", "batch_size", "num_workers", "cuda_prefetch", "output_dir", "gpu_count"}
    changed = {key for key in set(old_run) | set(run) if old_run.get(key) != run.get(key)}
    if changed - permitted:
        raise ValueError(f"Fork chỉ cho đổi epochs/batch_size/num_workers/cuda_prefetch/output_dir/gpu_count: {sorted(changed - permitted)}")
    if run["epochs"] <= saved["epoch"] + 1:
        raise ValueError("epochs mục tiêu phải lớn hơn số epoch nguồn đã lưu")
    history = read_json(source.parent / "history.json")
    history = [record for record in history if record["epoch"] <= saved["epoch"]]
    if not history or (history[-1]["epoch"], history[-1]["global_step"]) != (
            saved["epoch"], saved["global_step"]):
        raise ValueError("history.json nguồn không khớp epoch/global_step checkpoint")
    best_path = source.parent / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    check_contract(best, contract)
    if (best.get("kind") != "best_model" or best.get("smoke") is not False
            or best["epoch"] > saved["epoch"]
            or best["val"]["loss"] != saved["best_val_loss"]):
        raise ValueError("best.pt nguồn không khớp best_val_loss của latest.pt")
    del best
    provenance = {"source_checkpoint": str(source), "source_epoch": saved["epoch"],
                  "source_global_step": saved["global_step"],
                  "source_run_config_hash": saved["run_config_hash"],
                  "changes": {key: {"before": old_run.get(key), "after": run.get(key)}
                              for key in sorted(changed)},
                  "parent_lineage": saved["metadata"].get("fork", None)}
    saved["metadata"] = checkpoint_metadata(run, model_cfg, data_cfg, splits, stats)
    saved["metadata"]["fork"] = provenance
    saved["run_config_hash"] = resume_signature(run)
    for record in history:
        record.setdefault("batch_size", old_run["batch_size"])
        record.setdefault("num_workers", old_run["num_workers"])
        record.setdefault("gpu_count", old_run.get("gpu_count", 1))
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "best.pt.partial"
    try:
        shutil.copyfile(best_path, temporary)
        os.replace(temporary, out / "best.pt")
    finally:
        temporary.unlink(missing_ok=True)
    write_json(out / "history.json", history)
    write_json(out / "fork.json", provenance)
    # Write latest last. Its presence means the complete child is ready to resume.
    atomic_torch_save(saved, out / "latest.pt")
    print(f"Fork đã tạo: {out}; đã lưu {saved['epoch'] + 1} epoch; "
          f"tiếp epoch {saved['epoch'] + 2}; batch_size={run['batch_size']}; "
          f"num_workers={run['num_workers']}; "
          f"mục tiêu tổng {run['epochs']} epoch", flush=True)
    return out / "latest.pt"
