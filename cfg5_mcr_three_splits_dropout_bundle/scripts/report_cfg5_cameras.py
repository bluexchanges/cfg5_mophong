"""One HTML report of missing RGB views, from the locked 1317-episode preparation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"task_(\d+)_user_\d+_scene_\d+_cfg_0005$")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def validate_records(records, cameras, expected=None):
    if len(cameras) != 8 or len(set(cameras)) != 8:
        raise ValueError("Need 8 unique camera serials")
    if expected is not None and len(records) != expected:
        raise ValueError(f"Expected {expected} episodes, got {len(records)}")
    seen = set()
    for row in records:
        match = NAME.fullmatch(row["episode"])
        if not match or row["episode"] in seen or int(match[1]) != row["task"]:
            raise ValueError("Duplicate/invalid episode/task: " + row["episode"])
        seen.add(row["episode"])
        quality = row["quality"]
        steps, counts = quality["steps"], quality["camera_valid_ticks"]
        if (type(steps) is not int or steps < 1 or len(counts) != 8
                or any(type(n) is not int or not 0 <= n <= steps for n in counts)
                or sum(counts) < steps):
            raise ValueError("Invalid camera coverage: " + row["episode"])
        if set(quality.get("camera_issues", {})) - set(cameras):
            raise ValueError("Unknown camera issue serial")


def verify_masks(records, cfg, source, split):
    import numpy as np
    for index, row in enumerate(records, 1):
        folder = source / row["episode"]
        meta = read(folder / "episode.json")
        if (meta["episode"] != row["episode"] or meta["config_hash"] != digest(cfg)
                or meta["split_hash"] != split["split_hash"]
                or meta["camera_serials"] != cfg["camera_serials"]
                or meta["quality"] != row["quality"]):
            raise ValueError("Light metadata differs: " + row["episode"])
        path = folder / "camera_valid.npy"
        if hash_file(path) != meta["numeric_sha256"]["camera_valid"]:
            raise ValueError("Camera mask checksum differs: " + row["episode"])
        mask = np.load(path, allow_pickle=False, mmap_mode="r")
        if (mask.dtype != np.bool_ or mask.shape != (row["quality"]["steps"], 8)
                or not mask.any(1).all()
                or mask.sum(0).tolist() != row["quality"]["camera_valid_ticks"]):
            raise ValueError("Camera mask differs: " + row["episode"])
        if index % 100 == 0 or index == len(records):
            print(f"CAMERA_REPORT_CHECK {index}/{len(records)}", flush=True)


def aggregate(records, cameras):
    groups = defaultdict(list)
    for row in records:
        groups[row["task"]].append(row)
    summaries = []
    for task, rows in sorted(groups.items()):
        steps = sum(row["quality"]["steps"] for row in rows)
        cells = []
        for cam in range(8):
            counts = [row["quality"]["camera_valid_ticks"][cam] for row in rows]
            cells.append({"serial": cameras[cam], "whole": sum(n == 0 for n in counts),
                          "partial": sum(0 < n < row["quality"]["steps"] for n, row in zip(counts, rows)),
                          "valid": sum(counts), "ticks": steps})
        summaries.append({"task": task, "episodes": len(rows), "steps": steps, "cameras": cells})
    return groups, summaries


def render_report(document, label, splits=None, check_files=False):
    cameras = document["config"]["camera_serials"]
    records = document["episodes"]
    validate_records(records, cameras)
    groups, summaries = aggregate(records, cameras)
    esc = lambda value: html.escape(str(value), quote=True)
    total_steps = sum(row["quality"]["steps"] for row in records)
    histogram = Counter(sum(n == 0 for n in row["quality"]["camera_valid_ticks"]) for row in records)
    missing_whole = sum(any(n == 0 for n in row["quality"]["camera_valid_ticks"]) for row in records)
    partial = sum(any(0 < n < row["quality"]["steps"] for n in row["quality"]["camera_valid_ticks"]) for row in records)
    membership = {} if splits is None else {name: kind for kind, names in splits["splits"].items() for name in names}
    rejected = document.get("rejected", document.get("excluded_episodes", document.get("blocked_episodes", [])))
    parts = ["""<!doctype html><html lang="vi"><meta charset="utf-8"><title>CFG5 — báo cáo camera</title>
<style>body{font:15px system-ui;margin:24px;color:#182535}h1{font-size:26px}small{color:#536171}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #dce2e8;padding:8px;text-align:left;vertical-align:top}th{background:#edf3fa;position:sticky;top:0}.bad{background:#fff0ef}.partial{background:#fff8de}details{margin:12px 0;border:1px solid #dce2e8;padding:10px}summary{cursor:pointer;font-weight:600}input{padding:10px;width:320px;max-width:90%}.note{background:#edf3fa;padding:14px;border-radius:6px}a{color:#1359ad}code{font-size:13px}footer{margin-top:24px;color:#536171}</style>
<h1>CFG5: task nào thiếu camera RGB?</h1>""",
        f"<p><b>Nguồn:</b> {esc(label)}<br><small>Tạo lúc {esc(datetime.now(timezone.utc).isoformat())}; không giải mã ảnh test.</small></p>",
        f"<p><b>{len(records):,} episode · {len(groups)} task · {total_steps:,} timestamp giữ lại.</b><br>"
        f"Episode có camera không dùng được cả đoạn: {missing_whole:,}. Episode có camera thiếu một phần: {partial:,}. Hai số có thể chồng nhau.</p>",
        "<div class='note'><b>Cách đọc:</b> <b>T</b> = số episode có 0 timestamp hợp lệ ở camera đó; <b>P</b> = số episode có camera hợp lệ một phần; <b>%</b> = tỷ lệ timestamp hợp lệ, có trọng số theo độ dài episode. Camera không hợp lệ có thể do thiếu file, video lỗi, lệch timestamp hoặc không phủ thời gian giữ lại. Không đánh đồng tất cả với mất camera vật lý. Đây là thống kê trên timeline đã căn 10Hz và mask của preprocessing, không phải đánh giá khả năng gắp.</div>",
        "<h2>Phân bố số camera không dùng được cả episode</h2><table><tr><th>Số camera</th><th>Số episode</th></tr>" +
        "".join(f"<tr><td>{n}</td><td>{histogram[n]}</td></tr>" for n in range(9)) + "</table>",
        "<h2>Tổng hợp theo task</h2><input id='filter' placeholder='Lọc task, ví dụ task_0039' aria-label='Lọc task'><div class='scroll'><table><thead><tr><th>Task</th><th>Episode</th><th>Ticks</th>" +
        "".join(f"<th>{esc(cam)}</th>" for cam in cameras) + "</tr></thead><tbody>"]
    for item in summaries:
        task_name = f"task_{item['task']:04d}"
        parts.append(f"<tr data-task='{task_name}'><td><a href='#{task_name}' onclick=\"document.getElementById('{task_name}').open=true\">{task_name}</a></td><td>{item['episodes']}</td><td>{item['steps']:,}</td>")
        for cell in item["cameras"]:
            css = "bad" if cell["whole"] else ("partial" if cell["partial"] else "")
            parts.append(f"<td class='{css}'>{cell['whole']} T / {cell['partial']} P<br>{100*cell['valid']/cell['ticks']:.1f}%</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div><h2>Chi tiết từng task / episode</h2>")
    for task, rows in sorted(groups.items()):
        task_name = f"task_{task:04d}"
        parts.append(f"<details id='{task_name}' data-task='{task_name}'><summary>{task_name} — {len(rows)} episode</summary><div class='scroll'><table><tr><th>Episode / split</th><th>Ticks</th><th>Camera không hợp lệ cả đoạn</th><th>Camera thiếu một phần</th><th>Nguyên nhân / file thiếu</th></tr>")
        for row in sorted(rows, key=lambda r: r["episode"]):
            quality, full, some = row["quality"], [], []
            for cam, count in zip(cameras, quality["camera_valid_ticks"]):
                if count == 0:
                    full.append(cam)
                elif count < quality["steps"]:
                    some.append(f"{cam}: {count}/{quality['steps']} ticks")
            reasons = [f"{cam}: {reason}" for cam, reason in quality.get("camera_issues", {}).items()]
            if check_files:
                raw = Path(row["path"])
                if not raw.is_dir():
                    raise FileNotFoundError("Raw episode unavailable: " + str(raw))
                for cam in cameras:
                    folder = raw / f"cam_{cam}"
                    if not folder.is_dir():
                        reasons.append(f"{cam}: thiếu thư mục camera")
                        continue
                    for filename in ("color.mp4", "timestamps.npy"):
                        path = folder / filename
                        if not path.is_file():
                            reasons.append(f"{cam}: thiếu {filename}")
                        elif path.stat().st_size == 0:
                            reasons.append(f"{cam}: {filename} rỗng")
            parts.append(f"<tr><td><code>{esc(row['episode'])}</code><br>{esc(membership.get(row['episode'], 'snapshot local; chưa gán split server'))}</td><td>{quality['steps']}</td><td>{'<br>'.join(map(esc, full)) or '—'}</td><td>{'<br>'.join(map(esc, some)) or '—'}</td><td>{'<br>'.join(map(esc, sorted(set(reasons)))) or 'Không có lỗi video/timestamp toàn camera trong audit; thiếu từng tick thể hiện ở cột bên.'}</td></tr>")
        parts.append("</table></div></details>")
    if rejected:
        parts.append("<h2>Episode ngoài bảng 1.317</h2><p>Các episode không có joint ghi gốc đã được người dùng đồng ý loại khỏi split; raw vẫn giữ.</p><ul>" +
                     "".join(f"<li>{esc(row['episode'])}: {esc(row.get('reason', 'missing recorded joint'))}</li>" for row in rejected) + "</ul>")
    parts.append("""<footer>Không thêm bộ lọc rating hoặc loại episode thiếu camera. Báo cáo không thay đổi dữ liệu, split hay mask.</footer>
<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.trim().toLowerCase();document.querySelectorAll('[data-task]').forEach(el=>el.hidden=!el.dataset.task.includes(q))})</script></html>""")
    return "\n".join(parts), {"episodes": len(records), "tasks": len(groups), "ticks": total_steps,
                              "whole_missing_episodes": missing_whole, "partial_missing_episodes": partial}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--base-config", default="configs/act/cfg5_robot_all_masked.json")
    p.add_argument("--manifest", default="data/splits/cfg5_robot_all_masked_manifest.json")
    p.add_argument("--audit-json", help="Generate an explicitly labelled local audit snapshot instead of locked server manifest")
    p.add_argument("--output", default="outputs/data_audit/cfg5_robot_all_camera_report.html")
    p.add_argument("--expected-episodes", type=int, default=1317)
    args = p.parse_args()
    repo = Path(args.repo).resolve(strict=True)
    splits = None
    if args.audit_json:
        document = read(args.audit_json)
        label = "SNAPSHOT AUDIT LOCAL — không chứng minh trạng thái raw/split server: " + str(Path(args.audit_json).resolve())
    else:
        run = read(repo / args.base_config)
        model = read(repo / run["model_config"])
        cfg = read(repo / model["data_config"])
        splits = read(repo / run["splits"])
        locked = read(repo / run["locked_membership"])
        document = read(repo / args.manifest)
        if (document["config_hash"] != digest(cfg) or document["config"] != cfg
                or splits["config_hash"] != digest(cfg)
                or splits["manifest_hash"] != document["manifest_hash"]):
            raise ValueError("Manifest/config/split contract differs")
        content = {k: v for k, v in document.items() if k != "manifest_hash"}
        if digest(content) != document["manifest_hash"]:
            raise ValueError("Manifest hash differs")
        names = [n for kind in ("train", "val", "test") for n in splits["splits"][kind]]
        if len(set(names)) != len(names) or set(names) != {r["episode"] for r in document["episodes"]}:
            raise ValueError("Split does not cover exactly the manifest episodes")
        if any(splits["splits"][kind] != locked["splits"][kind] for kind in ("train", "val", "test")):
            raise ValueError("Locked membership differs")
        validate_records(document["episodes"], cfg["camera_serials"], args.expected_episodes)
        verify_masks(document["episodes"], cfg, repo / run["processed_root"], splits)
        label = "Manifest + mask đã kiểm tra trên máy chạy job: " + str(repo / args.manifest)
    validate_records(document["episodes"], document["config"]["camera_serials"], args.expected_episodes)
    content, summary = render_report(document, label, splits, check_files=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = repo / output
    # Never write a report into original raw data.
    raw = (repo / document["config"]["dataset_root"]).resolve()
    if output.resolve().is_relative_to(raw):
        raise ValueError("Report output is inside raw data")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    print("CAMERA_REPORT_OK", summary, "output=" + str(output), flush=True)


if __name__ == "__main__":
    main()
