import argparse
from pathlib import Path
import _bootstrap
from src.data.common import load_config, read_json, require_contract
from src.data.cfg5 import preprocess_episode, verify_processed_episode
from src.data.splits import validate_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Decode 8 camera streaming và lưu numeric NPY ngoài raw")
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--manifest", default="data/splits/cfg5_manifest.json")
    p.add_argument("--splits", default="data/splits/cfg5_splits.json")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--output-root", default="data/processed/cfg5")
    p.add_argument("--limit-episodes", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--resume", action="store_true",
                   help="Bỏ qua episode hoàn chỉnh sau khi kiểm metadata/hash/shape; không bỏ qua .partial")
    a = p.parse_args()
    if a.limit_episodes is not None and a.limit_episodes <= 0:
        p.error("limit-episodes phải dương")
    if a.max_steps is not None and a.max_steps < 2:
        p.error("max-steps phải >= 2")
    cfg, manifest, split = load_config(a.config), read_json(a.manifest), read_json(a.splits)
    require_contract(manifest, cfg)
    require_contract(split, cfg)
    validate_splits(split)
    if split["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Manifest không khớp split")
    records = {r["episode"]: r for r in manifest["episodes"]}
    for episode in split["splits"][a.split][:a.limit_episodes]:
        if a.resume and (Path(a.output_root) / (episode + ".partial")).exists():
            raise FileExistsError(f"Episode còn thư mục .partial; kiểm ERROR.json trước: {episode}")
        if a.resume and (Path(a.output_root) / episode).is_dir():
            result = verify_processed_episode(records[episode], cfg, a.output_root, split["split_hash"], a.max_steps)
            print(episode, "steps=", result["steps"], "verified_skip=True", flush=True)
            continue
        result = preprocess_episode(records[episode], cfg, a.output_root, split["split_hash"], a.max_steps)
        print(episode, "steps=", result["steps"], "smoke=", result["smoke_only"], flush=True)
