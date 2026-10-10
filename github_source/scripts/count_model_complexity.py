"""Count ACT parameters and major FLOPs using shapes; no payload/training run."""

import argparse
import json
from pathlib import Path

import _bootstrap
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode

from src.data.common import ROOT, digest, load_config, read_json
from src.models.rh20t_act import RH20TACTPolicy


def count_pass(policy, data, batch, mode):
    policy.zero_grad(set_to_none=True)
    policy.train(mode == "train")
    state = torch.zeros(batch, 8)
    images = torch.zeros(batch, 8, 3, data["image_height"], data["image_width"])
    targets = torch.zeros(batch, policy.config["chunk_size"], 8) if mode != "inference" else None
    mask = torch.zeros(batch, policy.config["chunk_size"], dtype=torch.bool) if targets is not None else None
    with torch.set_grad_enabled(mode == "train"), FlopCounterMode(display=False) as counter:
        # The actual core runs. Bypass wrapper's data-value checks, which cannot
        # execute on fake tensors and do not add counted conv/matmul FLOPs.
        pred, _pad, (mu, logvar) = policy.model(state, images, None, targets, mask)
        forward = counter.get_total_flops()
        if mode == "train":
            # Dense shapes are independent of how many padding targets are
            # masked. The all-valid fixture permits shape-only loss/backward.
            l1 = (pred-targets).abs().mean()
            kl = (-0.5*(1+logvar-mu.square()-logvar.exp())).sum(dim=1).mean()
            (l1+policy.config["kl_weight"]*kl).backward()
        total = counter.get_total_flops()
    return {"mode": mode, "batch_size": batch, "output_shape": list(pred.shape),
            "forward_flops": forward, "backward_flops": total-forward,
            "total_flops": total,
            "operator_counts": {str(op): count for op, count in
                                counter.get_flop_counts()["Global"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/act/rh20t_all8.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT/"docs/audit/model_complexity.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size phải dương")
    cfg = read_json(args.config)
    if cfg["backbone_weights_path"]:
        raise ValueError("Dùng config kiến trúc không load model asset để đếm shapes")
    data = load_config(ROOT/cfg["data_config"])
    with FakeTensorMode(allow_non_fake_inputs=True):
        policy = RH20TACTPolicy(cfg, data)
        total_parameters = sum(p.numel() for p in policy.parameters())
        report = {"torch": str(torch.__version__), "model_config": str(args.config),
                  "model_config_hash": digest(cfg), "data_config_hash": digest(data),
                  "method": "FakeTensorMode + FlopCounterMode; multiply+add=2 FLOPs",
                  "limitations": "Major conv/matmul only; excludes optimizer, I/O, softmax, norms, elementwise ops. Not a GPU latency benchmark.",
                  "parameters": total_parameters,
                  "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                  "parameter_groups": {name: sum(p.numel() for p in module.parameters())
                                       for name, module in policy.model.named_children()},
                  "passes": [count_pass(policy, data, batch, mode)
                             for batch in sorted({1, args.batch_size})
                             for mode in ("inference", "validation", "train")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
