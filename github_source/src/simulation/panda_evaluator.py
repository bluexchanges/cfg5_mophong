"""Evaluator có lịch sử grasp/lift và hình học seated/released/stable đúng lỗ."""
import numpy as np
from src.simulation.task_geometry import seating_dimensions


def placement_geometry(position,axis,linear_speed,angular_speed,hole_xy,cfg):
    position=np.asarray(position,dtype=float)
    axis=np.asarray(axis,dtype=float)
    hole_xy=np.asarray(hole_xy,dtype=float)
    if position.shape!=(3,) or axis.shape!=(3,) or hole_xy.shape!=(2,) or not np.all(np.isfinite(np.r_[position,axis,hole_xy,linear_speed,angular_speed])):
        raise ValueError("Pose/speed/hole không hữu hạn hoặc sai shape")
    if not np.isclose(np.linalg.norm(axis),1,atol=1e-6) or min(linear_speed,angular_speed)<0:
        raise ValueError("Axis phải là unit vector và speed không âm")
    if axis[2]<0: axis=-axis  # Cylinder đối xứng đầu/cuối.
    sine=float(np.linalg.norm(axis[:2]))
    tilt=float(np.arctan2(sine,axis[2]))
    radial=float(np.linalg.norm(position[:2]-hole_xy))
    radius=cfg["peg_radius_m"]; half=cfg["peg_length_m"]/2
    bottom=float(position[2]-half*axis[2]-radius*sine)
    top=float(position[2]+half*axis[2]+radius*sine)
    dimensions=seating_dimensions(cfg)
    floor=dimensions["hole_floor_z_m"]; mouth=dimensions["hole_mouth_z_m"]
    # Chỉ xét phần cylinder giao slab đáy..miệng lỗ. Disk bán kính r bao
    # cross-section xiên; khoảng s nới bởi r*sin(tilt), nên bound bảo thủ.
    intersects=False
    envelope=radial+radius+cfg["peg_length_m"]
    if axis[2]>1e-8:
        s0=max(-half,(floor-position[2]-radius*sine)/axis[2])
        s1=min(half,(mouth-position[2]+radius*sine)/axis[2])
        intersects=s0<=s1
        if intersects:
            endpoints=position[:2]-hole_xy+np.outer([s0,s1],axis[:2])
            envelope=float(np.max(np.linalg.norm(endpoints,axis=1))+radius)
    gap=bottom-floor
    penetration=max(0.0,-gap)
    insertion=mouth-bottom
    protrusion=top-mouth
    expected_protrusion=dimensions["upright_protrusion_m"]
    protrusion_error=protrusion-expected_protrusion
    return {"contained":bool(intersects and envelope<=cfg["hole_radius_m"]-cfg["placement_margin_m"]+1e-12),
            "height_ok":bool(abs(protrusion_error)<=cfg["protrusion_tolerance_m"]+1e-12),
            "bottom_seated":bool(-cfg["max_floor_penetration_m"]-1e-12<=gap<=cfg["max_bottom_gap_m"]+1e-12),
            "insertion_ok":bool(insertion>=cfg["hole_depth_m"]-cfg["max_bottom_gap_m"]-1e-12 and top>mouth),
            "tilt_ok":bool(tilt<=np.deg2rad(cfg["tilt_tolerance_deg"])),
            "stable":bool(linear_speed<=cfg["stable_linear_speed_m_s"] and angular_speed<=cfg["stable_angular_speed_rad_s"]),
            "radial_error_m":radial,"tilt_deg":float(np.rad2deg(tilt)),"bottom_height_error_m":gap,
            "peg_bottom_z_m":bottom,"peg_top_z_m":top,"hole_floor_z_m":floor,"hole_mouth_z_m":mouth,
            "floor_penetration_m":penetration,"insertion_depth_m":insertion,"protrusion_above_mouth_m":protrusion,
            "expected_protrusion_m":expected_protrusion,"protrusion_error_m":protrusion_error,
            "inserted_section_envelope_radius_m":float(envelope)}


class PandaEvaluator:
    def __init__(self,cfg,initial_positions):
        self.cfg=cfg
        self.initial=np.asarray(initial_positions).copy()
        self.grasp_time=np.zeros(3)
        self.place_time=np.zeros(3)
        self.grasped=np.zeros(3,dtype=bool)
        self.lifted=np.zeros(3,dtype=bool)
        self.dropped=np.zeros(3,dtype=bool)
        self.previous_grip=np.zeros(3,dtype=bool)
        self.previous_place=np.zeros(3,dtype=bool)

    def update(self,objects,contacts,tcp,width,dt,episode_length,*,robot_touching,floor_support_forces_n):
        cfg=self.cfg
        if not np.isfinite(dt) or dt<0: raise ValueError("dt phải không âm và hữu hạn")
        if len(objects)!=3 or np.asarray(contacts).shape!=(3,2) or np.asarray(robot_touching).shape!=(3,):
            raise ValueError("Cần trạng thái/contact của đúng 3 chốt")
        support=np.asarray(floor_support_forces_n,dtype=float)
        if support.shape!=(3,) or not np.all(np.isfinite(support)) or np.any(support<0):
            raise ValueError("Lực đỡ đáy phải là 3 số hữu hạn không âm")
        results=[]
        for i,obj in enumerate(objects):
            hold=bool(contacts[i][0] and contacts[i][1])
            self.grasp_time[i]=self.grasp_time[i]+dt if hold and self.previous_grip[i] else 0
            self.previous_grip[i]=hold
            self.grasped[i]|=self.grasp_time[i]>=cfg["grasp_hold_seconds"]-1e-9
            self.lifted[i]|=self.grasped[i] and hold and obj["position"][2]-self.initial[i,2]>=cfg["lift_height_m"]
            hole=np.asarray(cfg["mold_center_xy_m"])+cfg["hole_offsets_xy_m"][i]
            geo=placement_geometry(obj["position"],obj["axis"],obj["linear_speed"],obj["angular_speed"],hole,cfg)
            released=not robot_touching[i] and (width>2*cfg["peg_radius_m"]+cfg["release_clearance_m"] or np.linalg.norm(np.asarray(tcp)-obj["position"])>cfg["peg_length_m"])
            supported=bool(support[i]>=cfg["min_floor_support_force_n"])
            place=all(geo[k] for k in ("contained","height_ok","bottom_seated","insertion_ok","tilt_ok","stable")) and released and supported
            self.place_time[i]=self.place_time[i]+dt if place and self.previous_place[i] else 0
            self.previous_place[i]=place
            success=self.place_time[i]>=cfg["stable_seconds"]-1e-9
            if self.lifted[i] and not robot_touching[i] and not geo["contained"] and geo["peg_bottom_z_m"]<cfg["table_surface_z_m"]+cfg["drop_table_height_tolerance_m"]:
                self.dropped[i]=True
            reasons=[k for k in ("contained","height_ok","bottom_seated","insertion_ok","tilt_ok","stable") if not geo[k]]
            if not released: reasons.append("not_released")
            if not supported: reasons.append("no_floor_support")
            if place and not success: reasons.append("stability_time")
            if not self.grasped[i]: reasons.append("missing_grasp")
            if not self.lifted[i]: reasons.append("missing_lift")
            results.append({"object":f"peg_{i}","target_hole":i,"grasp_success":bool(self.grasped[i]),"lift_success":bool(self.lifted[i]),"place_success":bool(success),"task_success":bool(success and self.grasped[i] and self.lifted[i]),"object_dropped":bool(self.dropped[i]),"object_displacement_m":float(np.linalg.norm(obj["position"]-self.initial[i])),"stable_seconds":float(self.place_time[i]),"contact_left_right":list(map(bool,contacts[i])),"released":bool(released),"floor_supported":supported,"floor_support_force_n":float(support[i]),"failure_reason":reasons,**geo})
        return {"objects":results,"grasp_success":all(r["grasp_success"] for r in results),"lift_success":all(r["lift_success"] for r in results),"place_success":all(r["place_success"] for r in results),"task_success":all(r["task_success"] for r in results),"episode_length":episode_length,"object_dropped":any(r["object_dropped"] for r in results)}
