"""Evaluate an ACT checkpoint on processed CFG5 validation without optimizer steps."""

import argparse
import json

import torch
import _bootstrap

from src.data.common import ROOT, outside_raw
from src.models.rh20t_act import RH20TACTPolicy
from src.training.pretrain import (check_contract, choose_device, load_run,
                                   make_loader, validate_with_fixed_seed, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/act/cfg5_all_pretrain.json")
    parser.add_argument("--splits")
    parser.add_argument("--stats")
    parser.add_argument("--processed-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="outputs/act_pretrain/validation.json")
    parser.add_argument("--smoke", action="store_true", help="Chỉ 1 batch, ghi rõ sample count")
    parser.add_argument("--allow-partial-smoke", action="store_true")
    args = parser.parse_args()
    if args.allow_partial_smoke and not args.smoke:
        parser.error("--allow-partial-smoke chỉ đi cùng --smoke")
    run, model_cfg, data_cfg, splits, stats, contract = load_run(
        ROOT / args.config, splits_path=args.splits, stats_path=args.stats,
        processed_root=args.processed_root)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    check_contract(checkpoint, contract)
    if bool(checkpoint["smoke"]) != args.smoke:
        raise ValueError("Checkpoint smoke/full không khớp chế độ validation")
    loader = make_loader(ROOT / run["processed_root"], splits, stats, data_cfg, "val",
                         run["batch_size"], run["num_workers"],
                         allow_partial_smoke=args.allow_partial_smoke)
    policy = RH20TACTPolicy(model_cfg, data_cfg).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    result = validate_with_fixed_seed(policy, loader, device, run["seed"],
                                      max_batches=1 if args.smoke else 0)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
              "split": "val", "all_val_samples": len(loader.dataset), "evaluated": result,
              "contract": contract, "smoke": args.smoke, "optimizer_steps": 0}
    write_json(outside_raw(ROOT / args.output, data_cfg), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
