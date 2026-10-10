"""Mở processed NPY bằng memmap, xuất8 ảnh một mốc và bảng số liệu để trình bày."""
import argparse
import html
from pathlib import Path
import cv2
import numpy as np
import _bootstrap
from src.data.common import load_config, read_json, outside_raw, require_contract, write_json


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--episode-dir", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", default="outputs/processed_preview")
    a = p.parse_args(); cfg = load_config(a.config)
    folder = Path(a.episode_dir); metadata = read_json(folder / "episode.json")
    require_contract(metadata, cfg)
    if metadata["camera_serials"] != cfg["camera_serials"]:
        raise ValueError("Camera order khác config")
    names = ["images", "state", "action", "timestamps_ms", "segment_ids", "frame_indices", "camera_timestamps_ms", "state_timestamps_ms"]
    arrays = {name: np.load(folder / (name+".npy"), mmap_mode="r", allow_pickle=False) for name in names}
    n = len(arrays["state"])
    if not 0 <= a.index < n: raise ValueError(f"index phải trong0..{n-1}")
    i = a.index; segment = arrays["segment_ids"][i]
    end = i+1
    while end < n and arrays["segment_ids"][end] == segment: end += 1
    valid = min(cfg["chunk_size"], end-i-1)
    out = outside_raw(a.output, cfg); out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for serial, rgb in zip(cfg["camera_serials"], arrays["images"][i]):
        tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(tile, serial, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        tiles.append(tile)
    if not cv2.imwrite(str(out / "eight_cameras.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])):
        raise OSError("Không ghi được PNG")
    summary = {"episode": metadata["episode"], "smoke_only": metadata["smoke_only"], "steps_saved": n,
               "index": i, "timestamp_ms": int(arrays["timestamps_ms"][i]), "segment": int(segment),
               "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key,value in arrays.items()},
               "state_xyz_wxyz_width_m": arrays["state"][i].tolist(),
               "valid_future_actions": valid, "future_actions_physical": arrays["action"][i+1:i+1+valid].tolist(),
               "is_pad": [j >= valid for j in range(cfg["chunk_size"])],
               "camera_age_ms": (arrays["timestamps_ms"][i]-arrays["camera_timestamps_ms"][i]).tolist(),
               "state_age_ms": int(arrays["timestamps_ms"][i]-arrays["state_timestamps_ms"][i]),
               "config_hash": metadata["config_hash"], "split_hash": metadata["split_hash"]}
    write_json(out / "sample.json", summary)
    rows = ''.join(f'<tr><td>{name}.npy</td><td>{value.shape}</td><td>{value.dtype}</td></tr>' for name,value in arrays.items())
    table = ''.join(f'<tr><td>{j}</td><td>{int(arrays["timestamps_ms"][j])}</td><td>{int(arrays["segment_ids"][j])}</td><td>{html.escape(str(arrays["state"][j].tolist()))}</td></tr>' for j in range(min(n, 50)))
    (out / "index.html").write_text(f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Processed CFG5</title>
<style>body{{font:16px system-ui;margin:30px;background:#f4f7fa;color:#173148}}img{{width:100%}}td,th{{padding:8px;border:1px solid #abc}}table{{border-collapse:collapse}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(metadata['episode'])}</h1><p>Smoke={metadata['smoke_only']}; lưu{n} bước; đang xem index{i}, segment{segment}, có{valid}/20 targets thật. Không có target thật ở cuối segment thì mốc đó không thành sample ACT.</p>
<img src="eight_cameras.png" alt="8 camera theo thứ tự config"><h2>File đã lưu</h2><table><tr><th>File</th><th>Shape</th><th>Dtype</th></tr>{rows}</table>
<h2>State tại mốc đang xem: xyz(m),wxyz,width(m)</h2><pre>{summary['state_xyz_wxyz_width_m']}</pre>
<p>Target bắt đầu t+1. <a href="sample.json">sample.json</a> chứa targets chưa normalize, mask và tuổi ảnh/state; is_pad=True là phần bỏ qua trong loss. Ảnh PNG giữ RGB nhìn được, không phải tensor đã normalize.</p>
<h2>Tối đa50 dòng đầu để đọc</h2><table><tr><th>Index</th><th>Timestamp(ms)</th><th>Segment</th><th>State vật lý8D</th></tr>{table}</table></html>''', encoding="utf-8")
    print(out / "index.html")
    print("Arrays:", summary["arrays"], "valid_future_actions:", valid)
