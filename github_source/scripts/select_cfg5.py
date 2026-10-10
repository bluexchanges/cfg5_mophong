import argparse
import _bootstrap
from src.data.common import load_config, read_json, write_json, outside_raw
from src.data.selection import select_manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Áp dụng allowlist episode vào manifest mới; không sửa raw/split cũ")
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--manifest", default="data/splits/cfg5_manifest.json")
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(); cfg = load_config(a.config)
    out = outside_raw(a.output, cfg)
    if out.exists(): raise FileExistsError("Chọn output manifest mới")
    result = select_manifest(read_json(a.manifest), read_json(a.selection), cfg)
    write_json(out, result)
    print("Selected episodes:", len(result["episodes"]), "→", out)
