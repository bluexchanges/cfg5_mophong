import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    cfg = read_json(path)
    if len(cfg["camera_serials"]) != 8 or len(set(cfg["camera_serials"])) != 8:
        raise ValueError("Config phải có đúng 8 camera khác nhau")
    if not cfg["require_all_cameras"]:
        raise ValueError("Adapter này yêu cầu đủ 8 view; chưa hỗ trợ camera masking")
    if cfg["representation"] not in ("rh20t_aligned_tcp_base_wxyz_gripper_m",
                                     "tcp_base_xyz_wxyz_gripper_width_m"):
        raise ValueError("Representation không được hỗ trợ")
    offset = cfg.get("action_target_offset", 1)
    if type(offset) is not int or offset not in (0, 1):
        raise ValueError("action_target_offset phải là 0 hoặc 1")
    if cfg["representation"] == "tcp_base_xyz_wxyz_gripper_width_m" and offset != 0:
        raise ValueError("Demo pre-step phải dùng command cùng tick, offset=0")
    if cfg["state_camera_serial"] not in cfg["camera_serials"]:
        raise ValueError("State camera phải thuộc rig")
    for key in ("image_width", "image_height", "control_hz", "chunk_size", "min_segment_steps"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} phải dương")
    return cfg


def data_root(cfg):
    return (ROOT / cfg["dataset_root"]).resolve()


def episode_path(cfg, name):
    """Read pre/post move layouts without making a second dataset copy."""
    if Path(name).name != name or not name.startswith("task_"):
        raise ValueError("Episode ID không hợp lệ")
    root = data_root(cfg)
    old = root / name
    task = name.split("_", 2)
    nested = root / "training_data" / f"task_{task[1]}" / name
    if old.is_dir() and nested.is_dir():
        raise ValueError(f"Episode xuất hiện ở hai vị trí: {name}")
    if nested.is_dir():
        return nested
    if old.is_dir():
        return old
    raise FileNotFoundError(f"Không tìm thấy training episode: {name}")


def outside_raw(path, cfg):
    path = Path(path).resolve()
    roots = [data_root(cfg), *(Path(root).resolve() for root in cfg.get("raw_sources", []))]
    if any(path == root or path.is_relative_to(root) for root in roots):
        raise ValueError("Không ghi output vào dataset gốc")
    return path


def require_contract(record, cfg):
    if record["config_hash"] != digest(cfg):
        raise ValueError("Config không khớp artifact; tạo lại manifest/split/processed với cùng config")
