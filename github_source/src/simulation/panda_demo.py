"""Scripted privileged controller demo. Không sử dụng ACT/checkpoint."""
import numpy as np
from src.simulation.task_geometry import seating_dimensions


def run_demo(env,callback=None):
    cfg=env.cfg; demo=cfg["demo"]; hz=cfg["control_hz"]
    geometry=seating_dimensions(cfg)
    trace=[]; stages=[]
    quat=np.asarray(demo["down_quaternion_wxyz"])

    def tick(position,width,stage):
        if env.done: return
        action=np.r_[position,quat,width]
        _,_,terminated,truncated,info=env.step(action)
        trace.append({"step":env.steps,"stage":stage,"action":action.tolist(),"tcp":env.tcp_pose().tolist(),"width":env.gripper_width(),"objects":[o["position"].tolist() for o in env.object_states()],"contacts":env.finger_contacts().tolist()})
        if callback: callback(env,stage)
        if truncated: raise RuntimeError("Demo vượt giới hạn thời gian episode")

    def hold(position,width,seconds,stage):
        for _ in range(max(1,round(seconds*hz))): tick(position,width,stage)

    def move(position,width,stage,speed=None):
        if env.done: return
        start=env.tcp_pose()[:3].copy()
        position=np.asarray(position)
        duration=np.linalg.norm(position-start)/(speed or demo["move_speed_m_s"])
        count=max(1,int(np.ceil(duration*hz)))
        for i in range(1,count+1): tick(start+(position-start)*(i/count),width,stage)
        if env.done: return
        hold(position,width,demo["waypoint_settle_seconds"],stage+"_settle")
        error=float(np.linalg.norm(env.tcp_pose()[:3]-position))
        stages.append({"stage":stage,"position_error_m":error,"time_s":float(env.data.time-env.start_time)})
        print(stage,"error_mm",round(error*1000,3),flush=True)

    failure=None
    try:
        for i in range(3):
            source=env.object_states()[i]["position"]
            xy=source[:2].copy()
            grasp_z=cfg["table_surface_z_m"]+demo["grasp_height_above_surface_m"]
            move([*xy,demo["hover_z_m"]],demo["open_width_m"],f"peg{i}_approach")
            move([*xy,grasp_z],demo["open_width_m"],f"peg{i}_descend")
            hold([*xy,grasp_z],demo["closed_width_m"],demo["grasp_seconds"],f"peg{i}_grasp")
            if not env.metrics["objects"][i]["grasp_success"]:
                raise RuntimeError(f"peg{i}: không xác nhận tiếp xúc hai ngón")
            move([*xy,demo["hover_z_m"]],demo["closed_width_m"],f"peg{i}_lift")
            if not env.metrics["objects"][i]["lift_success"]:
                raise RuntimeError(f"peg{i}: nâng chưa đạt")
            # Đo offset sau grasp chỉ cho scripted oracle; không đưa privileged state vào ACT.
            offset=env.tcp_pose()[:3]-env.object_states()[i]["position"]
            hole=np.asarray(cfg["mold_center_xy_m"])+cfg["hole_offsets_xy_m"][i]
            target_xy=hole+offset[:2]
            move([*target_xy,demo["hover_z_m"]],demo["closed_width_m"],f"peg{i}_transfer")
            seated_z=geometry["upright_seated_center_z_m"]+offset[2]
            entry_z=seated_z+cfg["hole_depth_m"]+demo["insertion_entry_clearance_m"]
            move([*target_xy,entry_z],demo["closed_width_m"],f"peg{i}_preinsert")
            # Recompute grip offset after the transfer has settled (oracle demo only).
            offset=env.tcp_pose()[:3]-env.object_states()[i]["position"]
            target_xy=hole+offset[:2]
            seated_z=geometry["upright_seated_center_z_m"]+offset[2]
            move([*target_xy,seated_z],demo["closed_width_m"],f"peg{i}_insert",demo["insertion_speed_m_s"])
            hold([*target_xy,seated_z],demo["open_width_m"],demo["release_seconds"],f"peg{i}_release")
            if env.done: break
            move([*target_xy,demo["hover_z_m"]],demo["open_width_m"],f"peg{i}_retreat")
            hold([*target_xy,demo["hover_z_m"]],demo["open_width_m"],cfg["stable_seconds"],f"peg{i}_verify")
            if not env.metrics["objects"][i]["task_success"]:
                raise RuntimeError(f"peg{i}: placement không đạt: {env.metrics['objects'][i]['failure_reason']}")
    except RuntimeError as exc:
        failure=str(exc)
    return {"mode":"scripted_controller_privileged_state","act_model_used":False,"checkpoint":None,
            "seed":env.seed,"sim_time_s":float(env.data.time-env.start_time),"failure":failure,
            "stages":stages,"metrics":env.metrics,"trace":trace}
