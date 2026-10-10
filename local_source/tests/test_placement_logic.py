import copy
import unittest
import numpy as np
from src.data.common import ROOT, read_json
from src.simulation.scene_builder import load_sim_config
from src.simulation.config_validation import validate_sim_config
from src.simulation.task_geometry import seating_dimensions, mold_floor_bounds
from src.simulation.panda_evaluator import placement_geometry, PandaEvaluator


class PlacementLogicTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_sim_config()
        self.d=seating_dimensions(self.cfg)
        self.holes=np.asarray(self.cfg["mold_center_xy_m"])+np.asarray(self.cfg["hole_offsets_xy_m"])

    def geometry(self,bottom_offset=0,axis=None,cfg=None,xy=None):
        cfg=cfg or self.cfg
        axis=np.array([0,0,1]) if axis is None else np.asarray(axis)
        d=seating_dimensions(cfg)
        halfheight=cfg["peg_length_m"]/2*abs(axis[2])+cfg["peg_radius_m"]*np.linalg.norm(axis[:2])
        position=[*(self.holes[0] if xy is None else xy),d["hole_floor_z_m"]+halfheight+bottom_offset]
        return placement_geometry(position,axis,0,0,self.holes[0],cfg)

    def fixture(self):
        objects=[{"position":np.r_[xy,self.d["upright_seated_center_z_m"]],"axis":np.array([0,0,1]),"linear_speed":0,"angular_speed":0} for xy in self.holes]
        return objects,PandaEvaluator(self.cfg,[o["position"] for o in objects])

    def update(self,evaluator,objects,dt=.1,support=None,touch=None,grip=None):
        return evaluator.update(objects,grip or [[False,False]]*3,[.3,0,.3],.08,dt,1,
            robot_touching=[False]*3 if touch is None else touch,
            floor_support_forces_n=[.049]*3 if support is None else support)

    def test_legacy_file_is_alias_not_second_geometry(self):
        alias=read_json(ROOT/"configs/mujoco/precision_pick_place.json")
        self.assertEqual(alias["config_type"],"alias")
        self.assertNotIn("object_size_m",alias)
        self.assertEqual(load_sim_config("configs/mujoco/precision_pick_place.json"),self.cfg)

    def test_explicit_25_minus_10_and_35_minus_12(self):
        cfg=copy.deepcopy(self.cfg); cfg["peg_length_m"]=.025; cfg["hole_depth_m"]=.010
        result=self.geometry(cfg=cfg)
        self.assertAlmostEqual(result["expected_protrusion_m"],.015)
        self.assertAlmostEqual(result["protrusion_above_mouth_m"],.015)
        self.assertTrue(result["height_ok"])
        result=self.geometry()
        self.assertAlmostEqual(result["expected_protrusion_m"],.023)
        self.assertAlmostEqual(result["protrusion_above_mouth_m"],.023)
        self.assertAlmostEqual(self.d["hole_floor_z_m"],.004)
        self.assertAlmostEqual(self.d["hole_mouth_z_m"],.016)
        self.assertAlmostEqual(result["peg_top_z_m"],.039)

    def test_protrusion_tolerance_is_error_around_target(self):
        for offset,expected in ((.0005,True),(.00051,False),(-.0005,True),(-.00051,False)):
            result=self.geometry(bottom_offset=offset)
            self.assertEqual(result["height_ok"],expected)
            self.assertAlmostEqual(result["protrusion_error_m"],offset)

    def test_negative_penetration_limit_is_separate(self):
        result=self.geometry(bottom_offset=-.0002)
        self.assertTrue(result["height_ok"])
        self.assertFalse(result["bottom_seated"])
        self.assertTrue(self.geometry(bottom_offset=-.00005)["bottom_seated"])

    def test_tilted_protruding_part_is_not_required_inside_aperture(self):
        theta=np.deg2rad(2)
        axis=np.array([np.sin(theta),0,np.cos(theta)])
        halfheight=self.cfg["peg_length_m"]/2*axis[2]+self.cfg["peg_radius_m"]*axis[0]
        center_z=self.d["hole_floor_z_m"]+halfheight
        mid_z=(self.d["hole_floor_z_m"]+self.d["hole_mouth_z_m"])/2
        xy=self.holes[0]+axis[:2]*(center_z-mid_z)/axis[2]
        result=self.geometry(axis=axis,xy=xy)
        old_envelope=np.linalg.norm(xy-self.holes[0])+self.cfg["peg_radius_m"]+self.cfg["peg_length_m"]/2*np.sin(theta)
        self.assertGreater(old_envelope,self.cfg["hole_radius_m"])
        self.assertTrue(result["contained"])
        self.assertTrue(result["tilt_ok"])
        self.assertTrue(result["height_ok"])

    def test_tipped_or_wrong_hole_never_passes(self):
        result=self.geometry(axis=[1,0,0])
        self.assertFalse(result["height_ok"])
        self.assertFalse(result["tilt_ok"])
        self.assertFalse(result["contained"])
        self.assertFalse(self.geometry(xy=self.holes[1])["contained"])

    def test_fallen_object_at_correct_top_height_still_fails(self):
        # Đặt tâm cao lên để chốt nằm ngang có đúng đỉnh mục tiêu: riêng height sẽ đạt.
        center_z=self.d["upright_seated_top_z_m"]-self.cfg["peg_radius_m"]
        result=placement_geometry([*self.holes[0],center_z],[1,0,0],0,0,self.holes[0],self.cfg)
        self.assertTrue(result["height_ok"])
        self.assertFalse(result["tilt_ok"])
        self.assertFalse(result["insertion_ok"])
        self.assertFalse(result["contained"])

    def test_floating_inside_height_tolerance_fails_without_support(self):
        objects,evaluator=self.fixture()
        for obj in objects: obj["position"][2]+=.0003
        for _ in range(10): result=self.update(evaluator,objects,support=[0]*3)
        self.assertTrue(result["objects"][0]["height_ok"])
        self.assertFalse(result["place_success"])
        self.assertIn("no_floor_support",result["objects"][0]["failure_reason"])

    def test_weak_robot_contact_is_still_not_released(self):
        objects,evaluator=self.fixture()
        for _ in range(10): result=self.update(evaluator,objects,touch=[True]*3)
        self.assertFalse(result["place_success"])
        self.assertIn("not_released",result["objects"][0]["failure_reason"])

    def test_stability_counts_elapsed_intervals_not_first_sample(self):
        objects,evaluator=self.fixture()
        first=self.update(evaluator,objects,dt=1.0)
        self.assertFalse(first["place_success"])
        self.assertEqual(first["objects"][0]["stable_seconds"],0)
        for _ in range(5): result=self.update(evaluator,objects)
        self.assertTrue(result["place_success"])
        self.assertFalse(result["task_success"])
        self.assertIn("missing_grasp",result["objects"][0]["failure_reason"])
        self.update(evaluator,objects,support=[0]*3)
        result=self.update(evaluator,objects)
        self.assertFalse(result["place_success"])
        self.assertEqual(result["objects"][0]["stable_seconds"],0)

    def test_config_rejects_impossible_geometry_and_thresholds(self):
        invalid={"hole_radius_m":.002,"hole_depth_m":.04,"max_bottom_gap_m":.02,
                 "protrusion_tolerance_m":.023,"max_floor_penetration_m":.004,
                 "placement_margin_m":.0006,"peg_length_m":-.01,"evaluation_hz":75,
                 "hole_wall_segments":3,"min_floor_support_force_n":1.0}
        for key,value in invalid.items():
            with self.subTest(key=key):
                cfg=copy.deepcopy(self.cfg); cfg[key]=value
                with self.assertRaises(ValueError): validate_sim_config(cfg)
        cfg=copy.deepcopy(self.cfg);cfg["hole_offsets_xy_m"][1]=cfg["hole_offsets_xy_m"][0]
        with self.assertRaises(ValueError): validate_sim_config(cfg)

    def test_floor_bounds_follow_offsets_in_both_axes(self):
        cfg=copy.deepcopy(self.cfg)
        cfg["hole_offsets_xy_m"]=[[.01,-.075],[.01,0],[.01,.075]]
        lower,upper=mold_floor_bounds(cfg)
        np.testing.assert_allclose(lower,[-.01,-.1125])
        np.testing.assert_allclose(upper,[.03,.1125])

    def test_invalid_axis_or_nonfinite_is_rejected(self):
        with self.assertRaises(ValueError): self.geometry(axis=[0,0,0])
        with self.assertRaises(ValueError): self.geometry(axis=[0,0,float('nan')])


if __name__=="__main__": unittest.main()
