"""Một nguồn công thức cho độ cao khuôn, target demo và evaluator (đơn vị m)."""
import numpy as np


def seating_dimensions(cfg):
    floor = cfg["table_surface_z_m"] + cfg["mold_floor_thickness_m"]
    mouth = floor + cfg["hole_depth_m"]
    return {"hole_floor_z_m": floor, "hole_mouth_z_m": mouth,
            "upright_seated_center_z_m": floor + cfg["peg_length_m"]/2,
            "upright_seated_top_z_m": floor + cfg["peg_length_m"],
            "upright_protrusion_m": cfg["peg_length_m"] - cfg["hole_depth_m"],
            "radial_clearance_m": cfg["hole_radius_m"] - cfg["peg_radius_m"]}


def mold_floor_bounds(cfg):
    offsets = np.asarray(cfg["hole_offsets_xy_m"], dtype=float)
    half = np.asarray(cfg["mold_cell_half_xy_m"], dtype=float)
    return offsets.min(axis=0)-half, offsets.max(axis=0)+half
