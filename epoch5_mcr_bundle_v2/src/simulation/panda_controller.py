"""Damped least-squares IK + Panda position actuators; không teleport robot/vật."""
import numpy as np
import mujoco
from src.data.geometry import pose_matrix, normalize_quaternion


def rotation_error(target, current):
    q=np.empty(4)
    mujoco.mju_mat2Quat(q,(target@current.T).ravel())
    if q[0] < 0:
        q=-q
    norm=np.linalg.norm(q[1:])
    return 2*q[1:] if norm<1e-10 else q[1:]/norm*(2*np.arctan2(norm,q[0]))


def validate_action(action,cfg):
    action=np.asarray(action,dtype=float)
    if action.shape!=(8,) or not np.all(np.isfinite(action)):
        raise ValueError("Action cần 8 số hữu hạn: xyz,wxyz,width_m")
    if np.any(action[:3]<cfg["workspace_min_m"]) or np.any(action[:3]>cfg["workspace_max_m"]):
        raise ValueError("TCP ngoài workspace")
    if not cfg["gripper_width_range_m"][0]<=action[7]<=cfg["gripper_width_range_m"][1]:
        raise ValueError("Gripper width ngoài giới hạn Panda")
    result=action.copy()
    result[3:7]=normalize_quaternion(result[3:7])
    return result


class PandaController:
    def __init__(self,model,cfg):
        self.model,self.cfg=model,cfg
        self.ik_data=mujoco.MjData(model)
        self.site=model.site("policy_tcp").id
        self.joints=np.array([model.joint(f"joint{i}").id for i in range(1,8)])
        self.qadr=model.jnt_qposadr[self.joints]
        self.dofs=model.jnt_dofadr[self.joints]
        self.actuators=np.array([model.actuator(f"actuator{i}").id for i in range(1,8)])
        self.gripper=model.actuator("actuator8").id
        self.limits=model.jnt_range[self.joints]
        self.last_ik={}

    def inverse_kinematics(self,data,pose):
        settings=self.cfg["controller"]
        target=pose_matrix(pose)
        work=self.ik_data
        work.qpos[:]=data.qpos
        jp=np.zeros((3,self.model.nv)); jr=np.zeros_like(jp)
        for iteration in range(settings["ik_iterations"]):
            mujoco.mj_forward(self.model,work)
            pe=target[:3,3]-work.site_xpos[self.site]
            re=rotation_error(target[:3,:3],work.site_xmat[self.site].reshape(3,3))
            if np.linalg.norm(pe)<settings["ik_position_tolerance_m"] and np.linalg.norm(re)<settings["ik_rotation_tolerance_rad"]:
                break
            mujoco.mj_jacSite(self.model,work,jp,jr,self.site)
            jac=np.vstack([jp[:,self.dofs],jr[:,self.dofs]])
            error=np.r_[pe,re]
            delta=jac.T@np.linalg.solve(jac@jac.T+settings["ik_damping"]**2*np.eye(6),error)
            delta*=min(1,settings["ik_max_joint_delta_rad"]/max(np.max(np.abs(delta)),1e-12))
            work.qpos[self.qadr]=np.clip(work.qpos[self.qadr]+delta,self.limits[:,0]+1e-5,self.limits[:,1]-1e-5)
        mujoco.mj_forward(self.model,work)
        self.last_ik={"position_error_m":float(np.linalg.norm(target[:3,3]-work.site_xpos[self.site])),"rotation_error_rad":float(np.linalg.norm(rotation_error(target[:3,:3],work.site_xmat[self.site].reshape(3,3)))),"iterations":iteration+1}
        return work.qpos[self.qadr].copy()

    def set_target(self,data,action):
        action=validate_action(action,self.cfg)
        position=data.site_xpos[self.site]
        delta=action[:3]-position
        length=np.linalg.norm(delta)
        bounded=action.copy()
        bounded[:3]=position+delta*min(1,self.cfg["controller"]["max_translation_per_step_m"]/max(length,1e-12))
        self.target_q=self.inverse_kinematics(data,bounded[:7])
        self.target_width=float(action[7])

    def substep(self,data):
        max_delta=self.cfg["controller"]["max_joint_speed_rad_s"]*self.cfg["timestep_s"]
        current=data.ctrl[self.actuators]
        data.ctrl[self.actuators]=current+np.clip(self.target_q-current,-max_delta,max_delta)
        data.ctrl[self.gripper]=self.target_width/0.08*255

