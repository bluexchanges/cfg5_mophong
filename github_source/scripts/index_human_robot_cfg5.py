"""Lập chỉ mục liên hệ human–robot CFG5 sau khi MOVE, không sao chép episode."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import io
from pathlib import Path

import _bootstrap
from src.data.common import ROOT, digest, read_json, write_json
from src.data.splits import validate_splits


PLAN = ROOT / "docs/audit/cfg5_all_organization_plan.json"
VERIFIED = ROOT / "docs/audit/cfg5_all_organization_verified.json"
DATASET = ROOT / "RH20T_cfg5"
INDEX = ROOT / "data/index/cfg5_human_robot_pairs.json"
SPLIT = ROOT / "data/splits/cfg5_all_splits.json"
MANIFEST = ROOT / "data/splits/cfg5_all_manifest.json"
FIELDS = ("task", "human_episode", "human_path", "human_calib", "robot_episode",
          "robot_path", "robot_calib", "same_calibration", "robot_status",
          "robot_act_split", "match_basis", "act_label_status")


def write_csv(path: Path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key in FIELDS} for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = buffer.getvalue()
    if path.exists() and path.read_text(encoding="utf-8-sig") == content:
        return
    path.write_text(content, encoding="utf-8-sig", newline="")


def main():
    plan = read_json(PLAN)
    plan_hash = plan.pop("plan_hash")
    verified = read_json(VERIFIED)
    if digest(plan) != plan_hash or verified["plan_hash"] != plan_hash:
        raise ValueError("Plan và báo cáo verify không khớp")
    if verified["episode_count"] != len(plan["episodes"]):
        raise ValueError("Số episode verify không khớp")
    if Path(plan["dataset_root"]).resolve() != DATASET.resolve():
        raise ValueError("Dataset root không khớp")
    by_name = {record["episode"]: record for record in plan["episodes"]}
    if len(by_name) != len(plan["episodes"]):
        raise ValueError("Episode name bị trùng")
    split = read_json(SPLIT) if SPLIT.exists() else None
    split_by_episode = {}
    if split:
        validate_splits(split)
        if split["manifest_hash"] != read_json(MANIFEST)["manifest_hash"]:
            raise ValueError("Split không khớp full manifest")
        split_by_episode = {episode: name for name, episodes in split["splits"].items()
                            for episode in episodes}
    rows = []
    for human in plan["episodes"]:
        if human["kind"] != "human":
            continue
        if not (DATASET / human["destination_rel"]).is_dir():
            raise FileNotFoundError(human["destination_rel"])
        human_meta = read_json(DATASET / human["destination_rel"] / "metadata.json")
        robot = by_name.get(human["robot_counterpart"])
        if robot is not None:
            if robot["kind"] != "robot" or robot["task"] != human["task"]:
                raise ValueError("Counterpart sai loại hoặc task")
            if not (DATASET / robot["destination_rel"]).is_dir():
                raise FileNotFoundError(robot["destination_rel"])
        robot_meta = read_json(DATASET / robot["destination_rel"] / "metadata.json") if robot else {}
        if split and robot and robot["category"] == "training_data" and human["robot_counterpart"] not in split_by_episode:
            raise ValueError("Robot đủ điều kiện ACT nhưng thiếu split")
        human_calib = str(human_meta.get("calib", ""))
        robot_calib = str(robot_meta.get("calib", ""))
        rows.append({
            "task": human["task"],
            "human_episode": human["episode"],
            "human_path": human["destination_rel"],
            "human_calib": human_calib,
            "robot_episode": human["robot_counterpart"],
            "robot_path": robot["destination_rel"] if robot else "",
            "robot_calib": robot_calib,
            "same_calibration": ("yes" if human_calib and human_calib == robot_calib
                                 else "no" if robot else "unknown"),
            "robot_status": robot["category"] if robot else "absent_from_local_cfg5",
            "robot_act_split": split_by_episode.get(human["robot_counterpart"], "not_in_act_split"),
            "match_basis": "same_task_user_scene_cfg_name_only" if robot else "none",
            "act_label_status": ("robot_has_own_8d_targets_not_human_frame_labels"
                                 if robot and robot["category"] == "training_data"
                                 else "no_act_ready_robot_target"),
        })
    rows.sort(key=lambda row: (row["task"], row["human_episode"]))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    for task, task_rows in grouped.items():
        write_csv(DATASET / "missing_data" / task / "human_robot_index.csv", task_rows)
    write_csv(DATASET / "human_robot_index.csv", rows)
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "organization_plan_hash": plan_hash,
        "split_hash": split["split_hash"] if split else None,
        "dataset_root": str(DATASET),
        "relationship_warning": "Tên task/user/scene/cfg là ứng viên liên hệ theo cây dữ liệu; chưa xác minh cùng lần thao tác hay đồng bộ thời gian. Robot trajectory không cung cấp nhãn action theo từng frame human.",
        "counts": {"human": len(rows), "by_robot_status": dict(Counter(r["robot_status"] for r in rows)),
                   "by_calibration_match": dict(Counter(r["same_calibration"] for r in rows)),
                   "by_robot_act_split": dict(Counter(r["robot_act_split"] for r in rows)),
                   "tasks_with_human": len(grouped)},
        "rows": rows,
    }
    write_json(INDEX, report)
    print(f"Indexed {len(rows)} human records across {len(grouped)} tasks; {report['counts']['by_robot_status']}")


if __name__ == "__main__":
    main()
