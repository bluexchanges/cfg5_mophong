import argparse
import numpy as np
import _bootstrap
from src.data.common import load_config, read_json, outside_raw, write_json
from src.data.act_dataset import ACTDataset

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kiểm tra sample ACT; không cần torch")
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--splits", default="data/splits/cfg5_splits.json")
    p.add_argument("--stats", default="data/splits/normalization.json")
    p.add_argument("--processed-root", default="data/processed/cfg5")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--allow-partial-smoke", action="store_true")
    p.add_argument("--output", default="docs/audit/dataset_check.json")
    a = p.parse_args()
    cfg = load_config(a.config)
    dataset = ACTDataset(a.processed_root, read_json(a.splits), read_json(a.stats), cfg, a.split, a.allow_partial_smoke)
    checks = []
    for index in sorted(set([0, len(dataset)//2, len(dataset)-1])):
        sample = dataset[index]
        assert sample["images"].shape == (8, 3, cfg["image_height"], cfg["image_width"])
        assert sample["actions"].shape == (cfg["chunk_size"], 8)
        assert sample["is_pad"].dtype == np.bool_ and not sample["is_pad"][0]
        checks.append({"index": index, "episode": sample["episode"], "image_shape": list(sample["images"].shape),
                       "valid_actions": int((~sample["is_pad"]).sum()), "timestamp_ms": sample["timestamp_ms"]})
    result = {"split": a.split, "episodes_available": len(dataset.episodes), "sample_count": len(dataset),
              "smoke_allowed": a.allow_partial_smoke, "checks": checks, "model_executed": False}
    write_json(outside_raw(a.output, cfg), result)
    print(result)
