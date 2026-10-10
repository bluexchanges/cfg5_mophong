"""Từ chối kích thước/ngưỡng mâu thuẫn trước khi build hoặc chấm success."""
import numpy as np
from src.simulation.task_geometry import seating_dimensions, mold_floor_bounds


def _array(cfg,key,shape):
    value=np.asarray(cfg[key],dtype=float)
    if value.shape!=shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{key}: cần shape {shape}, số hữu hạn")
    return value


def validate_sim_config(cfg):
    if cfg.get("schema_version")!=2 or cfg.get("task")!="three_color_matched_pegs":
        raise ValueError("Cần schema_version=2 cho task ba chốt; config hộp cũ không được hỗ trợ")
    if cfg.get("placement_goal")!="bottom_seated_with_protrusion" or cfg.get("height_reference")!="peg_top_relative_to_hole_mouth" or cfg.get("protrusion_target_rule")!="peg_length_m - hole_depth_m":
        raise ValueError("Phải khai báo độ nhô từ đỉnh vật tới miệng lỗ, mục tiêu = chiều cao vật - độ sâu lỗ")
    if "seated_height_tolerance_m" in cfg:
        raise ValueError("Dùng max_bottom_gap_m và max_floor_penetration_m riêng")
    positive=("timestep_s","control_hz","evaluation_hz","peg_radius_m","peg_length_m","peg_mass_kg","hole_radius_m","hole_depth_m","mold_floor_thickness_m","max_bottom_gap_m","protrusion_tolerance_m","min_floor_support_force_n","stable_seconds","stable_linear_speed_m_s","stable_angular_speed_rad_s","grasp_hold_seconds","grasp_contact_force_n","lift_height_m","episode_seconds")
    for key in positive:
        if not np.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(f"{key} phải dương và hữu hạn")
    for key in ("max_floor_penetration_m","placement_margin_m","initial_xy_jitter_m","drop_table_height_tolerance_m","release_clearance_m"):
        if not np.isfinite(cfg[key]) or cfg[key]<0: raise ValueError(f"{key} phải không âm và hữu hạn")
    if not np.isfinite(cfg["table_surface_z_m"]): raise ValueError("table_surface_z_m không hữu hạn")
    if not 0<cfg["tilt_tolerance_deg"]<90: raise ValueError("tilt_tolerance_deg ngoài (0,90)")
    geom=seating_dimensions(cfg)
    if geom["radial_clearance_m"]<=cfg["placement_margin_m"]:
        raise ValueError("Lỗ phải lớn hơn chốt và còn clearance sau margin")
    if geom["upright_protrusion_m"]<=0:
        raise ValueError("placement_goal này yêu cầu peg_length_m > hole_depth_m")
    if cfg["protrusion_tolerance_m"]>=min(geom["upright_protrusion_m"],cfg["hole_depth_m"]):
        raise ValueError("Sai số độ nhô phải nhỏ hơn độ nhô mục tiêu và độ sâu lỗ")
    if cfg["max_bottom_gap_m"]>=cfg["hole_depth_m"]:
        raise ValueError("Khe hở đáy không được lớn tới mức vật chưa cắm vẫn đạt")
    if cfg["max_floor_penetration_m"]>=min(cfg["mold_floor_thickness_m"],cfg["max_bottom_gap_m"]):
        raise ValueError("Giới hạn lún phải nhỏ hơn độ dày đáy và max_bottom_gap_m")
    if cfg["min_floor_support_force_n"]>=cfg["peg_mass_kg"]*9.81:
        raise ValueError("Ngưỡng lực đỡ không được lớn hơn trọng lượng vật")
    for key in ("control_hz","evaluation_hz"):
        ratio=1/(cfg[key]*cfg["timestep_s"])
        if ratio<1 or not np.isclose(ratio,round(ratio),atol=1e-9,rtol=0):
            raise ValueError(f"{key}: chu kỳ phải là số nguyên physics steps")
    ratio=cfg["evaluation_hz"]/cfg["control_hz"]
    if ratio<1 or not np.isclose(ratio,round(ratio),atol=1e-9,rtol=0):
        raise ValueError("evaluation_hz phải là bội nguyên của control_hz")
    if cfg["stable_seconds"]<1/cfg["evaluation_hz"] or cfg["grasp_hold_seconds"]<1/cfg["evaluation_hz"]:
        raise ValueError("Thời gian giữ/ổn định không được ngắn hơn chu kỳ đánh giá")
    n=cfg["hole_wall_segments"]
    if not isinstance(n,int) or isinstance(n,bool) or n<8: raise ValueError("hole_wall_segments cần integer >=8")
    half=_array(cfg,"mold_cell_half_xy_m",(2,))
    if np.any(half<=cfg["hole_radius_m"]/np.cos(np.pi/n)):
        raise ValueError("Thành khuôn không còn độ dày ngoài lỗ")
    offsets=_array(cfg,"hole_offsets_xy_m",(3,2))
    for i in range(3):
        for j in range(i):
            if np.all(np.abs(offsets[i]-offsets[j])<2*half-1e-10):
                raise ValueError("Các cell khuôn chồng lên nhau, có thể lấp lỗ")
    center=_array(cfg,"mold_center_xy_m",(2,))
    table_center=_array(cfg,"table_center_xy_m",(2,))
    table_half=_array(cfg,"table_half_size_m",(3,))
    if np.any(table_half<=0): raise ValueError("Table size phải dương")
    lower,upper=mold_floor_bounds(cfg)
    if np.any(center+lower<table_center-table_half[:2]) or np.any(center+upper>table_center+table_half[:2]):
        raise ValueError("Khuôn nằm ngoài mặt bàn")
    initial=_array(cfg,"peg_initial_xy_m",(3,2))
    _array(cfg,"peg_colors_rgba",(3,4))
    extent=cfg["peg_radius_m"]+cfg["initial_xy_jitter_m"]
    if np.any(initial-extent<table_center-table_half[:2]) or np.any(initial+extent>table_center+table_half[:2]):
        raise ValueError("Chốt/jitter ngoài mặt bàn")
    for xy in initial:
        if np.all(xy+extent>center+lower) and np.all(xy-extent<center+upper):
            raise ValueError("Chốt khởi tạo đè lên khuôn")
    for i in range(3):
        for j in range(i):
            if np.linalg.norm(initial[i]-initial[j])<=2*(cfg["peg_radius_m"]+np.sqrt(2)*cfg["initial_xy_jitter_m"]):
                raise ValueError("Jitter có thể làm các chốt chồng nhau")
    demo=cfg["demo"]
    grasp=demo["grasp_height_above_surface_m"]
    if not cfg["hole_depth_m"]<grasp<cfg["peg_length_m"]:
        raise ValueError("Điểm gắp phải nằm trên miệng lỗ khi seated và trong chiều dài chốt")
    low=_array(cfg,"workspace_min_m",(3,)); high=_array(cfg,"workspace_max_m",(3,))
    if np.any(low>=high): raise ValueError("Workspace min phải nhỏ hơn max")
    heights=[cfg["table_surface_z_m"]+grasp,geom["hole_floor_z_m"]+grasp,demo["hover_z_m"]]
    if min(heights)<low[2] or max(heights)>high[2]: raise ValueError("Waypoint cao độ nằm ngoài workspace")
    if demo["hover_z_m"]<=geom["hole_mouth_z_m"]+grasp+demo["insertion_entry_clearance_m"]:
        raise ValueError("Hover phải cao hơn waypoint trước khi cắm")
    return cfg
