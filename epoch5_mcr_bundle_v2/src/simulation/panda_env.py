"""PandaEnv: reset/step 5-tuple; API riêng, không đăng ký Gymnasium."""
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from src.data.common import ROOT, read_json, digest
from src.simulation.scene_builder import load_sim_config, build_scene, quat
from src.simulation.panda_controller import PandaController
from src.simulation.panda_evaluator import PandaEvaluator


class PandaEnv:
    def __init__(self,config_path="configs/mujoco/panda_three_pegs.json",render_observations=True):
        self.cfg=load_sim_config(config_path)
        path=ROOT/self.cfg["scene_xml"]
        record=read_json(path.with_suffix(".manifest.json")) if path.with_suffix(".manifest.json").exists() else {}
        inputs={"source_xml_sha256":ROOT/self.cfg["robot_xml"],"camera_rig_sha256":ROOT/self.cfg["camera_rig"],"generator_sha256":Path(__file__).with_name("scene_builder.py"),"generated_xml_sha256":path}
        current=record.get("config_hash")==digest(self.cfg) and all(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==record.get(k) for k,p in inputs.items())
        if not current:
            path=build_scene(self.cfg)
        self.model=mujoco.MjModel.from_xml_path(str(path))
        self.data=mujoco.MjData(self.model)
        self.controller=PandaController(self.model,self.cfg)
        self.substeps=round(1/self.cfg["control_hz"]/self.cfg["timestep_s"])
        self.eval_substeps=round(1/self.cfg["evaluation_hz"]/self.cfg["timestep_s"])
        if self.substeps % self.eval_substeps:
            raise ValueError("Evaluation rate phải chia đều control period")
        self.camera_names=["cam_"+s for s in read_json(ROOT/self.cfg["camera_rig"])["camera_order"]]
        self.render_observations=render_observations
        self._renderers={}; self._viewer=None; self.closed=False; self.ready=False
        self.finger_joints=[self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1,2)]
        self.finger_bodies=[self.model.body(n).id for n in ("left_finger","right_finger")]
        robot_root=self.model.body("link0").id
        self.robot_bodies={robot_root}
        for body in range(robot_root+1,self.model.nbody):
            if self.model.body_parentid[body] in self.robot_bodies: self.robot_bodies.add(body)
        self.mold_floor_geom=self.model.geom("mold_floor").id
        self.peg_geoms=[self.model.geom(f"peg_{i}_geom").id for i in range(3)]
        self.peg_qadr=[self.model.joint(f"peg_{i}_free").qposadr[0] for i in range(3)]

    def _check_open(self):
        if self.closed: raise RuntimeError("Environment đã đóng")

    def reset(self,seed=42,options=None):
        self._check_open()
        if options: raise ValueError("Reset options chưa được hỗ trợ")
        rng=np.random.default_rng(seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.controller.qadr]=self.cfg["home_joints_rad"]
        self.data.qpos[self.finger_joints]=0.04
        self.data.ctrl[self.controller.actuators]=self.cfg["home_joints_rad"]
        self.data.ctrl[self.controller.gripper]=255
        for i,adr in enumerate(self.peg_qadr):
            self.data.qpos[adr:adr+2]=np.asarray(self.cfg["peg_initial_xy_m"][i])+rng.uniform(-self.cfg["initial_xy_jitter_m"],self.cfg["initial_xy_jitter_m"],2)
        mujoco.mj_forward(self.model,self.data)
        for _ in range(round(self.cfg["reset_settle_seconds"]/self.cfg["timestep_s"])):
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.start_time=float(self.data.time); self.steps=0; self.done=False; self.ready=True
        self.seed=seed
        self._last_observation_quaternion=None
        self.evaluator=PandaEvaluator(self.cfg,[o["position"] for o in self.object_states()])
        self.metrics=self._evaluate(0)
        return self.observation(),self.info()

    def tcp_pose(self):
        self._check_open()
        site=self.data.site("policy_tcp")
        return np.r_[site.xpos,quat(site.xmat.reshape(3,3))]

    def gripper_width(self):
        return float(np.sum(self.data.qpos[self.finger_joints]))

    def object_states(self):
        result=[]
        for i in range(3):
            body=self.data.body(f"peg_{i}")
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,body.id,velocity,0)
            result.append({"position":body.xpos.copy(),"quaternion":body.xquat.copy(),"axis":body.xmat.reshape(3,3)[:,2].copy(),"linear_speed":float(np.linalg.norm(velocity[3:])),"angular_speed":float(np.linalg.norm(velocity[:3]))})
        return result

    def contact_state(self):
        contacts=np.zeros((3,2),dtype=bool)
        touching=np.zeros(3,dtype=bool)
        support=np.zeros(3,dtype=float)
        force=np.zeros(6)
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            for a,b in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
                if a not in self.peg_geoms or b<0: continue
                peg=self.peg_geoms.index(a)
                body=self.model.geom_bodyid[b]
                mujoco.mj_contactForce(self.model,self.data,i,force)
                if body in self.robot_bodies and c.dist<=0:
                    touching[peg]=True  # Không dùng ngưỡng grasp để kết luận đã nhả.
                if body in self.finger_bodies and force[0]>self.cfg["grasp_contact_force_n"]:
                    contacts[peg,self.finger_bodies.index(body)]=True
                if b==self.mold_floor_geom:
                    # Contact normal hướng geom1 -> geom2; lấy thành phần đỡ +Z lên peg.
                    normal_z=c.frame[2]*(1 if c.geom2==a else -1)
                    support[peg]+=max(0.0,float(force[0]*normal_z))
        return {"grasp_contacts":contacts,"robot_touching":touching,"floor_support_forces_n":support}

    def finger_contacts(self):
        return self.contact_state()["grasp_contacts"]

    def _evaluate(self,dt):
        contact=self.contact_state()
        return self.evaluator.update(self.object_states(),contact["grasp_contacts"],self.tcp_pose()[:3],self.gripper_width(),dt,self.steps,
                                     robot_touching=contact["robot_touching"],floor_support_forces_n=contact["floor_support_forces_n"])

    def observation(self):
        self._check_open()
        if not self.ready: raise RuntimeError("Cần reset trước")
        pose=self.tcp_pose()
        if self._last_observation_quaternion is None:
            if pose[3]<0: pose[3:]*=-1
        elif np.dot(pose[3:],self._last_observation_quaternion)<0:
            pose[3:]*=-1
        self._last_observation_quaternion=pose[3:].copy()
        return {"images":self.render_all() if self.render_observations else None,
                "state":np.r_[pose,self.gripper_width()].astype(np.float32),
                "camera_names":self.camera_names.copy(),"timestamp_s":float(self.data.time-self.start_time)}

    def info(self):
        return {"seed":self.seed,"metrics":self.metrics,"controller":self.controller.last_ik.copy(),
                "sim_time_s":float(self.data.time-self.start_time),"privileged_object_states":[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o.items()} for o in self.object_states()]}

    def step(self,action):
        self._check_open()
        if not self.ready or self.done: raise RuntimeError("Cần reset trước step hoặc sau episode kết thúc")
        self.controller.set_target(self.data,action)
        self.steps+=1
        for index in range(self.substeps):
            self.controller.substep(self.data)
            mujoco.mj_step(self.model,self.data)
            if (index+1)%self.eval_substeps==0:
                mujoco.mj_forward(self.model,self.data)
                self.metrics=self._evaluate(self.eval_substeps*self.cfg["timestep_s"])
        mujoco.mj_forward(self.model,self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise FloatingPointError("MuJoCo sinh NaN/Inf")
        for warning in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC):
            if self.data.warning[warning].number:
                raise FloatingPointError(f"MuJoCo warning: {warning}")
        terminated=bool(self.metrics["task_success"])
        truncated=bool(not terminated and self.data.time-self.start_time>=self.cfg["episode_seconds"])
        self.done=terminated or truncated
        reward=float(sum(o["place_success"] for o in self.metrics["objects"])/3)
        return self.observation(),reward,terminated,truncated,self.info()

    def render(self,camera="overview",width=None,height=None):
        self._check_open()
        width=width or self.cfg["render"]["width"]
        height=height or self.cfg["render"]["height"]
        key=(width,height)
        if key not in self._renderers:
            self._renderers[key]=mujoco.Renderer(self.model,height=height,width=width)
        renderer=self._renderers[key]
        renderer.update_scene(self.data,camera=camera)
        return renderer.render().copy()

    def render_all(self):
        return np.stack([self.render(n) for n in self.camera_names])

    def show_viewer(self):
        self._check_open()
        import mujoco.viewer
        if self._viewer is None:
            self._viewer=mujoco.viewer.launch_passive(self.model,self.data)
        self._viewer.sync()
        return self._viewer

    def close(self):
        if self.closed: return
        for renderer in self._renderers.values(): renderer.close()
        if self._viewer is not None: self._viewer.close()
        self._renderers.clear(); self.closed=True

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
