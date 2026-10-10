import argparse
import _bootstrap
from src.data.common import load_config, outside_raw, read_json, write_json
from src.data.splits import create_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split theo nhóm trước khi tạo chunks")
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--manifest", default="data/splits/cfg5_manifest.json")
    p.add_argument("--output", default="data/splits/cfg5_splits.json")
    a = p.parse_args()
    cfg = load_config(a.config)
    result = create_splits(read_json(a.manifest), cfg)
    write_json(outside_raw(a.output, cfg), result)
    print(result["counts"])
