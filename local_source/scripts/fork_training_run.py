"""Tạo lượt chạy batch mới, giữ model/optimizer/RNG từ latest.pt cũ."""

import argparse

import _bootstrap
from src.data.common import ROOT
from src.training.fork_run import fork_run
from src.training.pretrain import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-checkpoint", required=True)
    parser.add_argument("--splits")
    parser.add_argument("--stats")
    parser.add_argument("--processed-root")
    args = parser.parse_args()
    run, model, data, splits, stats, contract = load_run(
        ROOT / args.config, splits_path=args.splits, stats_path=args.stats,
        processed_root=args.processed_root)
    fork_run(run, model, data, splits, stats, contract, ROOT / args.from_checkpoint)


if __name__ == "__main__":
    main()
