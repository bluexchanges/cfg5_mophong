"""Draw train/val loss SVGs from saved history, using only Python stdlib."""

import argparse
import html
import json
import math
from pathlib import Path


def history_path(repo):
    for name in ("act_pretrain_bs8_to80", "act_pretrain_bs16_w8_gpu2", "act_pretrain_fast", "act_pretrain_prefetch",
                 "act_pretrain_bs4_w2", "act_pretrain_bs4", "act_pretrain"):
        candidate = repo / "outputs" / name / "history.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Chưa tìm thấy history.json; có thể dùng --history /path/history.json")


def read_history(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("History trống")
    epochs = []
    for row in rows:
        epoch = row["epoch"] + 1
        if epochs and epoch <= epochs[-1]:
            raise ValueError("Epoch trong history phải tăng, không trùng")
        for split in ("train", "val"):
            if not math.isfinite(row[split]["loss"]):
                raise ValueError(f"Loss không hữu hạn ở E{epoch}")
        epochs.append(epoch)
    return rows


def chart(rows, splits):
    width, height = 1080, 550
    left, right, top, bottom = 90, 1020, 75, 420
    xmin, xmax = rows[0]["epoch"] + 1, rows[-1]["epoch"] + 1
    losses = [row[split]["loss"] for row in rows for split in splits]
    low, high = min(losses), max(losses)
    margin = max((high-low)*0.12, abs(high)*0.04, 0.005)
    low, high = max(0, low-margin), high+margin
    def x(epoch):
        return (left+right)/2 if xmax == xmin else left+(epoch-xmin)/(xmax-xmin)*(right-left)
    def y(value):
        return bottom-(value-low)/(high-low)*(bottom-top)
    def text(px, py, value, **attributes):
        attrs = ' '.join(f'{key.replace("_", "-")}="{value}"' for key, value in attributes.items())
        return f'<text x="{px:.1f}" y="{py:.1f}" {attrs}>{html.escape(str(value))}</text>'
    title = "Train và validation loss" if len(splits) == 2 else splits[0].capitalize()+" loss"
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="14" fill="#233044">',
           text(left, 32, title, font_size="23", font_weight="bold")]
    for index in range(6):
        value = low+(high-low)*index/5
        py = y(value)
        svg += [f'<line x1="{left}" x2="{right}" y1="{py:.1f}" y2="{py:.1f}" stroke="#e3e8ed"/>',
                text(left-12, py+5, f'{value:.3f}', text_anchor="end")]
    ticks = sorted({xmin, xmax, *[round(xmin+(xmax-xmin)*i/8) for i in range(9)]})
    for epoch in ticks:
        px = x(epoch)
        svg += [f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{bottom}" stroke="#f0f2f5"/>',
                text(px, bottom+27, f'E{epoch}', text_anchor="middle")]
    svg.append(text((left+right)/2, bottom+55, 'Epoch đã hoàn thành', text_anchor="middle"))
    # Mark batch changes without changing, averaging or smoothing the losses.
    for before, after in zip(rows, rows[1:]):
        if (before.get("batch_size") is not None and after.get("batch_size") is not None
                and before["batch_size"] != after["batch_size"]):
            px = x(after["epoch"]+1)
            svg += [f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{bottom}" stroke="#94a3b8" stroke-dasharray="5 5"/>',
                    text(px, top-15, f'BS {after["batch_size"]}', text_anchor="middle", font_size="12")]
    colors = {"train": "#2563eb", "val": "#dc2626"}
    for i, split in enumerate(splits):
        points = ' '.join(f'{x(r["epoch"]+1):.2f},{y(r[split]["loss"]):.2f}' for r in rows)
        color = colors[split]
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in rows:
            px, py = x(row["epoch"]+1), y(row[split]["loss"])
            caption = html.escape(f'E{row["epoch"]+1}: {split} loss={row[split]["loss"]:.6f}')
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"><title>{caption}</title></circle>')
        svg.append(text(right-190+i*100, 32, split, fill=color, font_weight="bold"))
    best = min(rows, key=lambda r:r["val"]["loss"])
    svg.append(text(left, 520, f'Best val: E{best["epoch"]+1} = {best["val"]["loss"]:.6f} | {len(rows)} epoch đã lưu; không làm mượt'))
    svg += ['</g>', '</svg>']
    return '\n'.join(svg)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.history or history_path(args.repo)
    rows = read_history(source)
    output = args.output_dir or args.repo / "docs/audit/training_plots"
    output.mkdir(parents=True, exist_ok=True)
    for filename, splits in (("loss_train_val.svg", ("train", "val")),
                             ("loss_train.svg", ("train",)), ("loss_val.svg", ("val",))):
        target = output / filename
        temporary = target.with_suffix('.svg.partial')
        temporary.write_text(chart(rows, splits), encoding="utf-8")
        temporary.replace(target)
        print(target)
    last = rows[-1]
    print(f'Nguồn: {source}; đã lưu {last["epoch"]+1} epoch')


if __name__ == "__main__":
    main()
