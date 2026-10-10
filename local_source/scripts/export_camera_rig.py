import argparse
import _bootstrap
from src.data.common import load_config, outside_raw, write_json
from src.simulation.camera_rig import build_rig

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Xuất đúng 8 camera từ calibration CFG5, không tạo pose giả")
    p.add_argument("--config", default="configs/data/cfg5_base8.json")
    p.add_argument("--calibration", default="1640071679624")
    p.add_argument("--output", default="configs/mujoco/cfg5_camera_rig.json")
    a = p.parse_args()
    cfg = load_config(a.config)
    rig = build_rig(cfg, a.calibration)
    write_json(outside_raw(a.output, cfg), rig)
    print("Cameras:", len(rig["cameras"]), "calibration:", rig["calibration_id"])
