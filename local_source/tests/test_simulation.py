import unittest
import numpy as np
import mujoco
from src.simulation.panda_env import PandaEnv
from src.simulation.panda_evaluator import placement_geometry, PandaEvaluator
from src.simulation.panda_controller import validate_action
from src.simulation.scene_builder import load_sim_config
from src.simulation.camera_rig import camera_world_poses
from src.data.common import ROOT, read_json
from src.data.geometry import pose_matrix


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.env=PandaEnv(render_observations=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=42)
        self.cfg=self.env.cfg

    def test_model_names_limits_reset_and_close(self):
        env=self.env
        self.assertEqual(env.model.nu,8)
        self.assertEqual(len(env.camera_names),8)
        self.assertTrue(np.all(env.controller.limits[:,1]>env.controller.limits[:,0]))
        a=env.data.qpos.copy()
        env.step(np.r_[env.tcp_pose(),.03])
        env.reset(seed=42)
        np.testing.assert_allclose(env.data.qpos,a,atol=1e-12)
        env.close(); env.close()
        with self.assertRaises(RuntimeError): env.reset()

    def test_action_validation(self):
        a=np.r_[self.env.tcp_pose(),.04]
        for bad in (a[:7],a*np.nan,np.r_[a[:3],np.zeros(4),.04],np.r_[2,a[1:]],np.r_[a[:7],.09]):
            with self.assertRaises(ValueError): validate_action(bad,self.cfg)
        result=validate_action(a,self.cfg)
        self.assertAlmostEqual(np.linalg.norm(result[3:7]),1)

    def test_robot_moves_and_gripper_opens_closes(self):
        env=self.env
        before=env.tcp_pose()
        target=np.r_[before[:3]+[.02,0,-.02],before[3:],.01]
        for _ in range(12): env.step(target)
        self.assertLess(np.linalg.norm(env.tcp_pose()[:3]-target[:3]),.001)
        self.assertLess(env.gripper_width(),.015)
        target[-1]=.07
        for _ in range(10): env.step(target)
        self.assertGreater(env.gripper_width(),.065)
        self.assertTrue(np.all(env.data.qpos[env.controller.qadr]>=env.controller.limits[:,0]))
        self.assertTrue(np.all(env.data.qpos[env.controller.qadr]<=env.controller.limits[:,1]))

    def test_camera_mounts_follow_actual_kinematics(self):
        env=self.env
        rig=read_json(ROOT/self.cfg["camera_rig"])
        expected=camera_world_poses(rig,np.eye(4),pose_matrix(env.tcp_pose()))
        before={n:env.data.camera(n).xpos.copy() for n in env.camera_names}
        for name in env.camera_names:
            np.testing.assert_allclose(env.data.camera(name).xpos,expected[name][:3,3],atol=1e-8)
            np.testing.assert_allclose(env.data.camera(name).xmat.reshape(3,3),expected[name][:3,:3],atol=1e-8)
        target=np.r_[env.tcp_pose(),.04]; target[0]+=.02
        for _ in range(8): env.step(target)
        for c in rig["cameras"]:
            movement=np.linalg.norm(env.data.camera(c["name"]).xpos-before[c["name"]])
            if c["parent_frame"]=="robot_base": self.assertLess(movement,1e-10)
            else: self.assertGreater(movement,.01)

    def test_holes_are_physical_cavities(self):
        # Fixture placement chỉ dùng để test collision, không phải controller demo.
        env=self.env
        for i,offset in enumerate(([0,0],[.01,0])):
            xy=np.asarray(self.cfg["mold_center_xy_m"])+self.cfg["hole_offsets_xy_m"][i]+offset
            adr=env.peg_qadr[i]
            env.data.qpos[adr:adr+7]=np.r_[xy,.055,1,0,0,0]
        env.data.qvel[:]=0
        mujoco.mj_forward(env.model,env.data)
        for _ in range(700): mujoco.mj_step(env.model,env.data)
        mujoco.mj_forward(env.model,env.data)
        positions=[o["position"] for o in env.object_states()]
        self.assertAlmostEqual(positions[0][2],self.cfg["mold_floor_thickness_m"]+self.cfg["peg_length_m"]/2,delta=.0003)
        self.assertGreater(positions[1][2],positions[0][2]+.01)
        support=env.contact_state()["floor_support_forces_n"]
        self.assertGreater(support[0],.02)
        self.assertLess(support[1],1e-8)  # Nằm trên mặt khuôn không phải chạm đáy lỗ.

    def test_evaluator_geometry_and_history(self):
        cfg=self.cfg
        z=cfg["mold_floor_thickness_m"]+cfg["peg_length_m"]/2
        holes=np.asarray(cfg["mold_center_xy_m"])+np.asarray(cfg["hole_offsets_xy_m"])
        good=placement_geometry([*holes[0],z],[0,0,1],0,0,holes[0],cfg)
        self.assertTrue(all(good[k] for k in ("contained","height_ok","tilt_ok","stable")))
        cases=[([*(holes[0]+[.002,0]),z],[0,0,1],0,0,"contained"),([*holes[0],z+.01],[0,0,1],0,0,"height_ok"),([*holes[0],z],[.2,0,np.sqrt(.96)],0,0,"tilt_ok"),([*holes[0],z],[0,0,1],.1,0,"stable"),([*holes[1],z],[0,0,1],0,0,"contained")]
        for p,a,v,w,reason in cases: self.assertFalse(placement_geometry(p,a,v,w,holes[0],cfg)[reason])
        objects=[{"position":np.r_[xy,z],"axis":np.array([0,0,1]),"linear_speed":0,"angular_speed":0} for xy in holes]
        evaluator=PandaEvaluator(cfg,[o["position"] for o in objects])
        contact_state={"robot_touching":[False]*3,"floor_support_forces_n":[.049]*3}
        for _ in range(6): result=evaluator.update(objects,[[False,False]]*3,[.3,0,.3],.08,.1,6,**contact_state)
        self.assertTrue(result["place_success"])
        self.assertFalse(result["task_success"])  # Direct placement is not grasp/lift success.
        evaluator.grasped[:]=True; evaluator.lifted[:]=True
        result=evaluator.update(objects,[[False,False]]*3,[.3,0,.3],.08,.1,7,**contact_state)
        self.assertTrue(result["task_success"])
        contact_state["robot_touching"]=[True]*3
        result=evaluator.update(objects,[[True,True]]*3,[.53,0,z],.001,.1,8,**contact_state)
        self.assertFalse(result["place_success"])


if __name__=="__main__": unittest.main()
