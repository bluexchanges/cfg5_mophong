"""Evaluate pretrain E5 on 10 locked validation demonstrations and MuJoCo scenes."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import _bootstrap
from src.data.common import ROOT
from src.evaluation.transfer_probe import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/act_pretrain/best.pt")
    p.add_argument("--expected-epoch", type=int, default=5)
    p.add_argument("--data-config", default="configs/data/pick_place_finetune8.json")
    p.add_argument("--splits", default="data/splits/pick_place_finetune_splits.json")
    p.add_argument("--manifest", default="data/splits/pick_place_finetune_manifest.json")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--mode", choices=("both", "offline"), default="both")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--output", default="outputs/epoch5_transfer_probe")
    args = p.parse_args()
    if min(args.batch_size, args.max_steps, args.episodes) < 1:
        p.error("batch/steps/episodes phải dương")
    args.checkpoint = ROOT / args.checkpoint
    evaluate(args)


if __name__ == "__main__":
    main()
