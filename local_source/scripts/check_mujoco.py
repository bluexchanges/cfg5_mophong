import argparse
import copy
import platform
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
import _bootstrap
from src.data.common import ROOT, read_json, write_json, digest, outside_raw
from src.simulation.scene_builder import build_scene, load_sim_config
from src.simulation.panda_env import PandaEnv
from src.simulation.panda_observation import to_act_observation


def verify_projection(cfg,rig):
    source=ET.parse(ROOT/cfg["scene_xml"])
    root=ET.Element("mujoco")
    visual=ET.SubElement(root,"visual")
    ET.SubElement(visual,"global",offwidth="640",offheight="360")
    world=ET.SubElement(root,"worldbody")
    for c in rig["cameras"]:
        camera=copy.deepcopy(source.find(f".//camera[@name='{c['name']}']"))
        camera.set("pos","0 0 0");camera.set("quat","1 0 0 0")
        world.append(camera)
    ET.SubElement(world,"geom",name="spot",type="sphere",size=".003",pos=".02 -.01 -.5")
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    result=[]
    with mujoco.Renderer(model,height=360,width=640) as renderer:
        renderer.enable_segmentation_rendering()
        for c in rig["cameras"]:
            renderer.update_scene(data,camera=c["name"])
            seg=renderer.render()
            y,x=np.where((seg[:,:,0]==model.geom("spot").id)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)))
            if not len(x): raise AssertionError("Không render được projection marker")
            k=np.asarray(c["K_pixels"])
            predicted=np.array([k[0,0]*.02/.5+k[0,2],k[1,1]*.01/.5+k[1,2]])
            observed=np.array([x.mean()+.5,y.mean()+.5])
            error=float(np.linalg.norm(predicted-observed))
            assert error<.5,(c["name"],error)
            result.append({"camera":c["name"],"expected_uv":predicted.tolist(),"rendered_uv":observed.tolist(),"error_pixels":error})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/mujoco/panda_three_pegs.json")
    p.add_argument("--output",default="docs/audit/mujoco_check.json")
    args=p.parse_args()
    cfg=load_sim_config(args.config)
    build_scene(cfg)
    rig=read_json(ROOT/cfg["camera_rig"])
    projections=verify_projection(cfg,rig)
    folder=ROOT/"outputs/stage3"; folder.mkdir(parents=True,exist_ok=True)
    with PandaEnv(args.config) as env:
        obs,info=env.reset(seed=42)
        assert obs["images"].shape==(8,360,640,3) and obs["images"].dtype==np.uint8
        assert obs["state"].shape==(8,) and obs["state"].dtype==np.float32
        data_cfg=read_json(ROOT/cfg["data_config"])
        # Identity statistics test adapter mechanics, not trained normalization.
        adapter=to_act_observation(obs,{"state":{"mean":[0]*8,"std":[1]*8}},data_cfg)
        assert adapter["images"].shape==(8,3,360,640)
        tiles=[]
        for name,rgb in zip(env.camera_names,obs["images"]):
            frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame,(0,0),(640,27),(25,25,25),-1)
            cv2.putText(frame,name,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            tiles.append(frame)
        cv2.imwrite(str(folder/"eight_cameras.png"),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
        for cam in ("overview","task_closeup"):
            cv2.imwrite(str(folder/(cam+".png")),cv2.cvtColor(env.render(cam,1280,720),cv2.COLOR_RGB2BGR))
        report={"mujoco_version":mujoco.__version__,"python":platform.python_version(),"config_hash":digest(cfg),"nq":env.model.nq,"nv":env.model.nv,"actuators":env.model.nu,"policy_cameras":env.camera_names,"debug_cameras":["overview","task_closeup"],"rgb_shape":list(obs["images"].shape),"rgb_dtype":str(obs["images"].dtype),"policy_images_shape":list(adapter["images"].shape),"projection_checks":projections,"headless_render_passed":True,"act_model_used":False}
        write_json(outside_raw(ROOT/args.output,data_cfg),report)
        print("MuJoCo",mujoco.__version__,"RGB",obs["images"].shape,"projection max error pixels",max(v["error_pixels"] for v in projections))
