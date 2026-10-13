"""Offline evaluation of a learned frozen MCR+ACT checkpoint on one held-out split."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from pathlib import Path
import _bootstrap
import torch
from torch.utils.data import DataLoader
from src.data.common import ROOT, read_json, outside_raw
from src.data.feature_cache import FeatureACTDataset, require_ready
from src.data.staged_mcr import validate_frozen_checkpoint, atomic_json
from src.models.mcr_act import MCRACTPolicy
from src.training.pretrain import validate_with_fixed_seed
from train_mcr_act import CachedPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch/workers")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU job")
    root = ROOT / args.cache_root
    ctx = read_json(root / "context.json")
    cfg, split = ctx["data_config"], ctx["splits"]
    origin = validate_frozen_checkpoint(ROOT / args.checkpoint, ctx)
    output = outside_raw(ROOT / args.output, cfg)
    import fcntl
    with (root / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        require_ready(root, ctx, "frozen", split_name=args.split)
        saved = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=True)
        policy = MCRACTPolicy(saved["metadata"]["model_config"], cfg)
        policy.load_state_dict(saved["model"], strict=True)
        policy.to(args.device).eval()
        data = FeatureACTDataset(root, split, ctx["stats"], cfg, args.split,
                                 feature_contract=ctx["contract"], stage="frozen")
        options = {"multiprocessing_context": "spawn", "prefetch_factor": 1} if args.workers else {}
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=args.device == "cuda", **options)
        result = validate_with_fixed_seed(CachedPolicy(policy), loader, args.device, 42,
                                         max_batches=0, log_every_batches=100)
        payload = {"checkpoint": origin, "split": args.split, "episodes": len(data.episodes),
                   "metrics": result, "camera_dropout_active": False, "ACT_dropout_active": False,
                   "checkpoint_selection_performed": False, "offline_only": True}
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, payload)
        print("MCR_FROZEN_EVALUATION_OK", payload, flush=True)


if __name__ == "__main__":
    main()
