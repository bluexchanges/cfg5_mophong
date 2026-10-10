"""Kiểm chứng hàm tạo sample upstream bằng dữ liệu giả, không import/train ACT.

Cần NumPy. Chỉ thực thi ba hàm đã đọc trong snapshot có hash được xác minh.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "references/upstream"


def extract(repo, file, names):
    manifest = json.loads((UPSTREAM / "manifest.json").read_text(encoding="utf-8"))
    record = next(r for r in manifest if r["repo"] == repo)
    entry = next(f for f in record["files"] if f["path"] == file)
    path = UPSTREAM / repo.replace("/", "__") / file
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {f.name for f in functions} == set(names)
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    official = extract("rh20t/act_baseline", "dataset/rh20t.py", ["_get_input_output_frame_id_lists"])
    rows, masks = official["_get_input_output_frame_id_lists"](None, [1000, 1100, 1200], num_input=1, horizon=4)
    # Một observation và ba future targets: chỉ hai future targets thật tồn tại.
    expected_mask = [False, False, False, True]
    assert len(rows) == 3  # Upstream kéo cả episode ngắn thành horizon.
    assert rows[0] == [1000, 1100, 1200, 1200]
    assert masks[0].tolist() == [False] * 4  # Xác nhận hành vi lỗi, không coi là đúng.
    air = extract("AirExo/act-in-the-wild", "wholearm.py", ["sample_timestamps", "find_valid_timestamp_sequence"])
    stamps = air["sample_timestamps"]([1000, 1050, 1100, 1150, 1200], 1000, 1200)
    starts = [i for i, t in enumerate(stamps) if stamps[-1] - t >= 100]
    futures = air["find_valid_timestamp_sequence"](stamps, 0, 3, 1, 0.1)
    assert starts == [0, 1, 2]
    assert futures == [1100, 1200]
    report = {
        "scope": "Isolated upstream functions, synthetic data; no model execution or training.",
        "rh20t_padding": {"input_timestamps": [1000, 1100, 1200], "horizon": 4,
                          "generated_samples": len(rows), "first_sample": rows[0],
                          "upstream_mask": masks[0].tolist(), "correct_mask_for_first_sample": expected_mask,
                          "padding_bug_reproduced": masks[0].tolist() != expected_mask},
        "airexo_density": {"timestamps": stamps, "frequency_hz": 10, "eligible_start_indices": starts,
                           "future_timestamps_from_first": futures,
                           "note": "Start indices remain dense at original timestamps; freq controls future spacing."},
    }
    output = ROOT / "docs/audit/upstream_sampling_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
