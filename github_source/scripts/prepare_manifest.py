"""Khảo sát sâu các task/camera đã chọn; không ghi dữ liệu đã decode."""
import argparse
import json
import _bootstrap
from src.data.common import load_config, outside_raw, write_json
from src.data.cfg5 import build_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/cfg5_base8.json")
    parser.add_argument("--output", default="data/splits/cfg5_manifest.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = outside_raw(args.output, cfg)
    manifest = build_manifest(cfg)
    write_json(output, manifest)
    print(json.dumps({"accepted": len(manifest['episodes']), "rejected": len(manifest['rejected']), "output": str(output)}))
