"""Xuất bảng HTML để đọc57 accepted/42 rejected và JSON chọn episode; không decode video."""
import argparse
from collections import Counter
import html
from pathlib import Path
import _bootstrap
from src.data.common import load_config, read_json, require_contract, outside_raw, write_json, episode_path
from src.data.splits import validate_splits


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--manifest", default="data/splits/cfg5_manifest.json")
    p.add_argument("--splits", default="data/splits/cfg5_splits.json")
    p.add_argument("--output", default="outputs/cfg5_review")
    p.add_argument("--refresh-html", action="store_true", help="Giữ selection.json hiện có, tạo lại link video sau khi move")
    a = p.parse_args()
    cfg, manifest, splits = load_config(a.config), read_json(a.manifest), read_json(a.splits)
    require_contract(manifest, cfg); require_contract(splits, cfg); validate_splits(splits)
    if splits["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("Manifest/split không khớp")
    out = outside_raw(a.output, cfg)
    out.mkdir(parents=True, exist_ok=True)
    selection = out / "selection.json"
    if selection.exists():
        if not a.refresh_html: raise FileExistsError("Không ghi đè selection đã có; chọn --output mới hoặc --refresh-html")
        if read_json(selection)["source_manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Selection hiện tại thuộc manifest khác; chọn output mới")
    else:
        write_json(selection, {"source_manifest_hash": manifest["manifest_hash"],
                              "note": "Giữ ID muốn dùng, xóa ID không muốn dùng. Đây là lựa chọn trước training.",
                              "episodes": [e["episode"] for e in manifest["episodes"]]})
    split_by_id = {ep: name for name, ids in splits["splits"].items() for ep in ids}
    rows = []
    total_steps = total_samples = full_chunks = 0
    for record in manifest["episodes"]:
        q = record["quality"]; lengths = q["segment_lengths"]
        samples = sum(n-1 for n in lengths)
        complete = sum(max(0, n-cfg["chunk_size"]) for n in lengths)
        total_steps += q["steps"]; total_samples += samples; full_chunks += complete
        ep = record["episode"]
        folder = episode_path(cfg, ep)
        videos = " · ".join(f'<a href="{(folder / ("cam_"+s) / "color.mp4").as_uri()}">{s}</a>' for s in cfg["camera_serials"])
        rows.append(f'<tr><td><code>{html.escape(ep)}</code><details><summary>Video raw8 camera</summary>{videos}</details></td>'
                    f'<td>{record["task"]}</td><td>{record["rating"]}</td><td>{split_by_id[ep]}</td>'
                    f'<td>{q["steps"]}</td><td>{q["segments"]}</td><td>{samples}</td><td>{complete}</td></tr>')
    rejected = "".join(f'<tr><td>{html.escape(e["episode"])}</td><td>{html.escape(e["reason"])}</td></tr>' for e in manifest["rejected"])
    estimated_bytes = total_steps * 8 * cfg["image_width"] * cfg["image_height"] * 3
    summary = {"accepted_episodes": len(rows), "rejected_episodes": len(manifest["rejected"]),
               "split_counts": splits["counts"], "accepted_per_task": dict(Counter(e["task"] for e in manifest["episodes"])),
               "timeline_steps": total_steps, "potential_samples": total_samples,
               "samples_with_full_chunk": full_chunks, "estimated_images_bytes": estimated_bytes,
               "manifest_hash": manifest["manifest_hash"], "decoded_in_this_command": False}
    write_json(out / "summary.json", summary)
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>CFG5 — chọn episode</title>
<style>body{{font:16px system-ui;margin:32px;background:#f5f7fa;color:#182d42}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ccd5dd;padding:10px;text-align:left}}th{{background:#dceaf3}}code{{font-size:13px}}details{{max-width:650px}}a{{color:#0663a5}}p{{max-width:1100px}}</style>
<h1>CFG5: đã lọc metadata, chưa preprocessing toàn bộ</h1>
<p>{len(rows)} accepted · {len(manifest['rejected'])} rejected · split {splits['counts']}. {total_steps} bước timeline; {total_samples} sample có ít nhất1 future target; chỉ{full_chunks} sample đủ20 targets. Ảnh NPY dự kiến{estimated_bytes/1e9:.2f} GB thập phân nếu xử lý tất cả.</p>
<p>Episode là một lượt thu, sample là một observation+chunk. Timeline kiểm được không đồng nghĩa ảnh đã decode/lưu. File này chỉ đọc manifest; video mở trực tiếp từ raw. FPS phát video không thay timestamp thật khi đồng bộ.</p>
<p>Mở <a href="selection.json">selection.json</a> bằng trình soạn thảo, giữ các ID muốn dùng. Không sửa JSON raw hoặc chuyển thư mục dataset. Nếu lọc lại phải tạo manifest/split/output mới. Chọn trước training; không dùng kết quả model trên test để lựa dữ liệu.</p>
<table><tr><th>Episode</th><th>Task</th><th>Rating</th><th>Split hiện tại</th><th>Bước</th><th>Segments</th><th>Samples</th><th>Đủ20</th></tr>{''.join(rows)}</table>
<h2>Episode bị loại và lý do đầu tiên</h2><table><tr><th>Episode</th><th>Lý do</th></tr>{rejected}</table></html>'''
    (out / "review.html").write_text(page, encoding="utf-8")
    print(summary)
    print(out / "review.html")
