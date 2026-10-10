"""Pretrain project ACT on processed CFG5. Full runs require server CUDA."""

import argparse

import _bootstrap
from src.data.common import ROOT
from src.training.pretrain import choose_device, load_run, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/act/cfg5_all_pretrain.json")
    parser.add_argument("--splits", help="Server split sau khi đã đối chiếu membership")
    parser.add_argument("--stats", help="Train-only normalization tính lại trên server")
    parser.add_argument("--processed-root", help="Processed train+val trên server")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", help="latest.pt của chính run này")
    parser.add_argument("--init-checkpoint", help="best.pt pretrain để khởi tạo run fine-tune mới")
    parser.add_argument("--smoke", action="store_true", help="1 train batch + 1 val batch, 1 epoch; output riêng")
    parser.add_argument("--allow-partial-smoke", action="store_true",
                        help="Chỉ đi cùng --smoke để dùng processed/stats smoke riêng")
    args = parser.parse_args()
    run, model, data, splits, stats, contract = load_run(
        ROOT / args.config, splits_path=args.splits, stats_path=args.stats,
        processed_root=args.processed_root)
    device = choose_device(args.device,
                           require_cuda=run["require_cuda_for_full_run"] and not args.smoke)
    print(f"Device={device}; train={len(splits['splits']['train'])} episodes; "
          f"val={len(splits['splits']['val'])} episodes; smoke={args.smoke}", flush=True)
    train(run, model, data, splits, stats, contract, device=device,
          resume=args.resume, smoke=args.smoke,
          allow_partial_smoke=args.allow_partial_smoke, init_checkpoint=args.init_checkpoint)


if __name__ == "__main__":
    main()
