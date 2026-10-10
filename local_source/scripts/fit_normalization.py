import argparse
import _bootstrap
from src.data.common import load_config, outside_raw, read_json, write_json
from src.data.act_dataset import fit_normalization

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fit normalization chỉ từ train")
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--splits", default="data/splits/cfg5_splits.json")
    p.add_argument("--processed-root", default="data/processed/cfg5")
    p.add_argument("--output", default="data/splits/normalization.json")
    p.add_argument("--allow-partial-smoke", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    stats = fit_normalization(a.processed_root, read_json(a.splits), cfg, a.allow_partial_smoke)
    write_json(outside_raw(a.output, cfg), stats)
    print("Train episodes:", len(stats["episodes"]), "smoke_only:", stats["smoke_only"])
