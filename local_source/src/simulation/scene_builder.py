"""Sinh scene có lỗ rỗng thật từ các convex wedge; upstream Panda không sửa."""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from src.data.common import ROOT, read_json, write_json, digest, outside_raw
from src.simulation.config_validation import validate_sim_config
from src.simulation.task_geometry import mold_floor_bounds


def nums(x):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(x).ravel())


def quat(matrix):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(matrix, dtype=float).ravel())
    return q


def load_sim_config(path="configs/mujoco/panda_three_pegs.json"):
    seen=set()
    path=(ROOT/path).resolve()
    while True:
        if path in seen: raise ValueError("Config alias tạo vòng lặp")
        seen.add(path)
        cfg=read_json(path)
        if cfg.get("config_type")!="alias": break
        allowed={"schema_version","config_type","active_config","description_vi"}
        if set(cfg)-allowed: raise ValueError("Alias không được chứa geometry/tolerance riêng")
        path=(ROOT/cfg["active_config"]).resolve()
    return validate_sim_config(cfg)


def build_scene(cfg):
    validate_sim_config(cfg)
    source = ROOT/cfg["robot_xml"]
    if not source.exists():
        raise FileNotFoundError("Chạy scripts/download_panda_model.py trước")
    manifest = read_json(source.parent/"manifest.json")
    for record in manifest["files"]:
        path = source.parent/record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Model source hash sai: {path}")
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "panda_three_pegs")
    root.find("compiler").set("meshdir", "../models/franka_emika_panda/assets")
    option = root.find("option")
    option.set("timestep", str(cfg["timestep_s"]))
    option.set("iterations", str(cfg["solver_iterations"]))
    option.set("cone", "elliptic")
    root.remove(root.find("keyframe"))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(cfg["render"]["preview_width"]), offheight=str(cfg["render"]["preview_height"]))
    ET.SubElement(visual, "map", znear="0.001", zfar="5")
    ET.SubElement(visual, "headlight", ambient="0.2 0.2 0.2", diffuse="0.3 0.3 0.3")
    world, assets = root.find("worldbody"), root.find("asset")
    world.find("light").set("diffuse", "0.4 0.4 0.4")
    contact = dict(solref=nums(cfg["contact_solref"]), solimp=nums(cfg["contact_solimp"]), condim="4", friction="0.8 0.005 0.0001")
    default = root.find("default")
    ET.SubElement(default, "geom", **contact)
    for body in root.findall(".//body"):
        if cfg["gravity_compensation"]:
            body.set("gravcomp", "1")
        if body.get("name") in ("left_finger", "right_finger"):
            for index, geom in enumerate(body.findall("geom")):
                geom.set("name", f"{body.get('name')}_geom_{index}")
                if geom.get("class") != "visual":
                    geom.set("friction", nums(cfg["finger_friction"]))
    root.find("./actuator/general[@name='actuator8']").set("forcerange", nums([-cfg["gripper_force_limit_n"], cfg["gripper_force_limit_n"]]))
    hand = root.find(".//body[@name='hand']")
    tcp = ET.SubElement(hand, "body", name="policy_tcp_frame", pos=nums(cfg["tcp_in_hand_pos_m"]), quat=nums(cfg["tcp_in_hand_quat_wxyz"]))
    ET.SubElement(tcp, "site", name="policy_tcp", size="0.002", rgba="1 0.7 0 0", group="4")
    rig = read_json(ROOT/cfg["camera_rig"])
    for camera in rig["cameras"]:
        parent = world if camera["parent_frame"] == "robot_base" else tcp
        matrix = np.asarray(camera["T_parent_from_camera_mujoco"])
        k = np.asarray(camera["K_pixels"])
        w,h = camera["resolution_wh"]
        ET.SubElement(parent, "camera", name=camera["name"], pos=nums(matrix[:3,3]), quat=nums(quat(matrix[:3,:3])),
                      resolution=nums([w,h]), sensorsize="0.0064 0.0036", focalpixel=nums([k[0,0], k[1,1]]),
                      principalpixel=nums([w/2-k[0,2], h/2-k[1,2]]))
    ET.SubElement(world,"camera",name="overview",pos="1.05 -0.95 0.82",xyaxes="0.85 0.53 0 -0.27 0.43 0.86",fovy="43")
    ET.SubElement(world,"camera",name="task_closeup",pos="0.45 -0.38 0.43",xyaxes="1 0 0 0 0.76 0.65",fovy="38")
    ET.SubElement(world,"light",pos="0.5 -0.4 1.2",dir="0 0 -1",diffuse="0.3 0.3 0.3")
    surface = cfg["table_surface_z_m"]
    half = cfg["table_half_size_m"]
    ET.SubElement(world,"geom",name="table",type="box",pos=nums([*cfg["table_center_xy_m"],surface-half[2]]),size=nums(half),rgba="0.65 0.70 0.73 1")
    ET.SubElement(world,"geom",name="floor",type="plane",pos="0 0 -0.75",size="2 2 0.05",rgba="0.17 0.20 0.24 1")
    center = cfg["mold_center_xy_m"]
    floor = cfg["mold_floor_thickness_m"]
    depth = cfg["hole_depth_m"]
    mold = ET.SubElement(world,"body",name="mold",pos=nums([*center,surface]))
    cellx,celly = cfg["mold_cell_half_xy_m"]
    lower,upper=mold_floor_bounds(cfg)
    ET.SubElement(mold,"geom",name="mold_floor",type="box",size=nums([*((upper-lower)/2),floor/2]),pos=nums([*((lower+upper)/2),floor/2]),rgba="0.24 0.28 0.32 1")
    n=cfg["hole_wall_segments"]
    # Inner regular polygon has apothem hole_radius, so opening never smaller than requested.
    inner=cfg["hole_radius_m"]/math.cos(math.pi/n)
    for i,((x,y),color) in enumerate(zip(cfg["hole_offsets_xy_m"],cfg["peg_colors_rgba"])):
        for j in range(n):
            points=[]
            for angle in (2*math.pi*j/n,2*math.pi*(j+1)/n):
                direction=np.array([math.cos(angle),math.sin(angle)])
                outer=min(cellx/max(abs(direction[0]),1e-12),celly/max(abs(direction[1]),1e-12))
                points.extend([direction*inner,direction*outer])
            vertices=[[p[0],p[1],z] for z in (floor,floor+depth) for p in points]
            name=f"hole_{i}_wall_{j}"
            ET.SubElement(assets,"mesh",name=name,vertex=nums(vertices))
            ET.SubElement(mold,"geom",name=name,type="mesh",mesh=name,pos=nums([x,y,0]),rgba=nums(np.array(color)*[0.65,0.65,0.65,1]))
        ET.SubElement(mold,"site",name=f"hole_{i}",pos=nums([x,y,floor+depth]),size="0.001",rgba="0 0 0 0",group="4")
    for i,(xy,color) in enumerate(zip(cfg["peg_initial_xy_m"],cfg["peg_colors_rgba"])):
        peg=ET.SubElement(world,"body",name=f"peg_{i}",pos=nums([*xy,surface+cfg["peg_length_m"]/2+0.0001]))
        ET.SubElement(peg,"freejoint",name=f"peg_{i}_free")
        ET.SubElement(peg,"geom",name=f"peg_{i}_geom",type="cylinder",size=nums([cfg["peg_radius_m"],cfg["peg_length_m"]/2]),mass=str(cfg["peg_mass_kg"]),rgba=nums(color),**contact)
    output=outside_raw(ROOT/cfg["scene_xml"],read_json(ROOT/cfg["data_config"]))
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(tree,space="  ")
    tree.write(output,encoding="utf-8",xml_declaration=True)
    write_json(output.with_suffix(".manifest.json"),{"config_hash":digest(cfg),"source_commit":manifest["commit"],"source_xml_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"camera_rig_sha256":hashlib.sha256((ROOT/cfg["camera_rig"]).read_bytes()).hexdigest(),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"generated_xml_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"note":"Generated derivative; source Panda XML unchanged. Virtual TCP mapping is explicit in config."})
    return output
