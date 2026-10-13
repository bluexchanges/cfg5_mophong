"""Two stages: train ACT from cached MCR features, then full fine-tune from JPEG."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from copy import deepcopy
from pathlib import Path
import _bootstrap
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.common import ROOT, digest, outside_raw, read_json
from src.data.feature_cache import FeatureACTDataset, require_ready
from src.data.sim_demos import file_hash
from src.models.mcr_act import MCRACTPolicy, PREPROCESS, MCR_SHA
from src.training.pretrain import (EpisodeShuffleSampler, atomic_torch_save, capture_rng,
    restore_rng, seed_everything, run_epoch, validate_with_fixed_seed, write_json)


class CachedPolicy(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base, self.config = base, base.config

    def forward(self, state, features, actions=None, is_pad=None, camera_valid=None):
        return self.base.forward_features(state, features, actions, is_pad, camera_valid=camera_valid)


def validate_initialization(saved, context):
    if saved.get("smoke") is not False:
        raise ValueError("Không dùng smoke checkpoint")
    meta = saved["metadata"]
    for key in ("state", "action"):
        if meta["normalization"][key] != context["stats"][key]:
            raise ValueError("Numeric normalization khác init checkpoint; không chuyển âm thầm")
    for key in ("camera_serials", "chunk_size", "control_hz"):
        if meta["data_config"][key] != context["data_config"][key]:
            raise ValueError(f"Init checkpoint khác {key}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--stage", choices=("frozen", "full"), required=True)
    p.add_argument("--init-checkpoint")
    p.add_argument("--resume")
    p.add_argument("--from-scratch", action="store_true", help="New ACT with pretrained MCR; frozen stage only")
    p.add_argument("--mcr-checkpoint", default="outputs/backbones/mcr_resnet50.pth")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-backbone", type=float, default=1e-6)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.init_checkpoint, args.resume, args.from_scratch))) != 1:
        p.error("Chọn đúng một --init-checkpoint, --resume hoặc --from-scratch")
    if args.from_scratch and args.stage != "frozen":
        p.error("--from-scratch chỉ dùng frozen stage")
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0 or args.lr <= 0 or args.lr_backbone <= 0:
        p.error("Epoch/batch/lr phải dương; workers>=0")
    if not args.smoke and (args.device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Full train yêu cầu CUDA; không chạy train nặng trên head node")
    root = ROOT / args.cache_root
    ctx = read_json(root / "context.json")
    cfg, split, stats = ctx["data_config"], ctx["splits"], ctx["stats"]
    names = split["splits"]["train"] + split["splits"]["val"]
    require_ready(root, ctx, args.stage)
    output = outside_raw(ROOT / args.output, cfg)
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".cache.lock").open("a") as cache_lock, (output / ".train.lock").open("w") as lock:
        fcntl.flock(cache_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        require_ready(root, ctx, args.stage)  # Recheck after acquiring the read lock.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and any((output / name).exists() for name in ("best.pt", "latest.pt", "history.json")):
            raise FileExistsError("Output có train state; dùng resume hoặc output mới")
        seed_everything(42)
        generator = torch.Generator().manual_seed(42)
        model_cfg = {**deepcopy(ctx["source_model_config"]), "backbone": "mcr_resnet50",
                     "backbone_weights_path": None, "feature_preprocess": PREPROCESS,
                     "lr": args.lr, "lr_backbone": 0. if args.stage == "frozen" else args.lr_backbone,
                     "cache_quantization": ctx["contract"]["quantization"]}
        contract = {"model_config_hash": digest(model_cfg), "data_config_hash": digest(cfg),
                    "split_hash": split["split_hash"], "stats_hash": digest(stats),
                    "feature_contract_hash": digest(ctx["contract"]), "manifest_hash": split["manifest_hash"],
                    "camera_serials": cfg["camera_serials"], "representation": cfg["representation"],
                    "upstream_commit": model_cfg["upstream_commit"]}
        runtime = {"stage": args.stage, "batch_size": args.batch_size, "workers": args.workers,
                   "seed": 42, "cache_context_hash": digest(ctx), "smoke": args.smoke}
        source = ROOT / (args.resume or args.init_checkpoint) if not args.from_scratch else None
        saved = torch.load(source, map_location="cpu", weights_only=True) if source else None
        if saved is not None:
            validate_initialization(saved, ctx)
        is_mcr = saved is not None and saved["metadata"]["model_config"]["backbone"] == "mcr_resnet50"
        if args.stage == "frozen" and is_mcr and (saved["metadata"].get("stage") != "frozen" or saved["metadata"].get("feature_contract") != ctx["contract"]):
            raise ValueError("Frozen stage cần cùng cache và backbone MCR gốc; checkpoint full đã đổi backbone không được dùng với cache cũ")
        if args.stage == "full" and not is_mcr:
            raise ValueError("Full fine-tune phải bắt đầu từ checkpoint MCR+ACT frozen đã học")
        policy = MCRACTPolicy(model_cfg, cfg, mcr_checkpoint=None if is_mcr else ROOT / args.mcr_checkpoint)
        if args.from_scratch:
            origin = {"method": "new ACT trained on current robot data with pretrained frozen MCR",
                      "backbone_sha256": MCR_SHA}
        elif is_mcr:
            policy.load_state_dict(saved["model"], strict=True)
            origin = {"method": "full MCR+ACT checkpoint transfer", "checkpoint_sha256": file_hash(source),
                      "checkpoint_epoch": saved["epoch"] + 1}
        else:
            if saved["epoch"] != 4:
                raise ValueError("ACT warmup nhánh này bắt đầu từ pretrain E5")
            origin = {**policy.initialize_act(saved), "checkpoint_sha256": file_hash(source),
                      "checkpoint_epoch": 5, "backbone_sha256": MCR_SHA}
        policy.to(args.device)
        optimizer = policy.configure_optimizer()
        start, history = 0, []
        if args.resume:
            if saved.get("contract") != contract or saved.get("runtime") != runtime or saved.get("kind") != "train_state":
                raise ValueError("Resume config/contract/stage khác")
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"])
            generator.set_state(saved["loader_generator_state"].cpu())
            start = saved["epoch"] + 1
            history = read_json(output / "history.json")
            if not history or history[-1]["epoch"] + 1 != start:
                raise ValueError("Resume history chưa đồng bộ checkpoint")
            origin = saved["metadata"]["initialization"]
        del saved
        options = {"prefetch_factor": 1, "persistent_workers": True, "multiprocessing_context": "spawn"} if args.workers else {}
        train_data = FeatureACTDataset(root, split, stats, cfg, "train", feature_contract=ctx["contract"], stage=args.stage)
        val_data = FeatureACTDataset(root, split, stats, cfg, "val", feature_contract=ctx["contract"], stage=args.stage)
        loader_seed = torch.Generator().manual_seed(0)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers,
                                  sampler=EpisodeShuffleSampler(train_data, generator), pin_memory=args.device == "cuda",
                                  generator=loader_seed, **options)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=min(args.workers, 2),
                                pin_memory=args.device == "cuda", **(options if args.workers else {}))
        train_policy = CachedPolicy(policy) if args.stage == "frozen" else policy
        metadata = {"model_config": model_cfg, "data_config": cfg, "normalization": stats,
                    "initialization": origin, "feature_contract": ctx["contract"], "stage": args.stage}
        best = min((r["val"]["loss"] for r in history), default=float("inf"))
        print(f"MCR_TRAIN stage={args.stage} train={len(train_data)} val={len(val_data)} workers={args.workers} batch={args.batch_size} start_epoch={start+1}", flush=True)
        for epoch in range(start, 1 if args.smoke else args.epochs):
            training = run_epoch(train_policy, train_loader, args.device, optimizer=optimizer,
                                 max_batches=1 if args.smoke else 0, log_every_batches=100)
            validation = validate_with_fixed_seed(train_policy, val_loader, args.device, 42,
                                                  max_batches=1 if args.smoke else 0, log_every_batches=100)
            row = {"epoch": epoch, "train": training, "val": validation, "smoke": args.smoke, "stage": args.stage}
            history.append(row)
            if validation["loss"] < best:
                best = validation["loss"]
                atomic_torch_save({"kind": "best_model", "model": policy.state_dict(), "epoch": epoch,
                                   "val": validation, "contract": contract, "metadata": metadata,
                                   "smoke": args.smoke}, output / "best.pt")
            record = {"kind": "train_state", "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "contract": contract, "runtime": runtime, "metadata": metadata,
                      "smoke": args.smoke, "rng": capture_rng(), "loader_generator_state": generator.get_state()}
            atomic_torch_save(record, output / "latest.pt")
            # Archive every epoch of this new branch, with inference weights only.
            atomic_torch_save({"kind": "epoch_model", "model": policy.state_dict(), "epoch": epoch,
                               "val": validation, "contract": contract, "metadata": metadata,
                               "smoke": args.smoke}, output / f"epoch_{epoch+1:03d}.pt")
            write_json(output / "history.json", history)
            print(row, flush=True)
        print("MCR_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
