import argparse
from pathlib import Path
import cv2
import numpy as np
import _bootstrap
from src.data.common import ROOT, read_json, write_json, digest, outside_raw
from src.simulation.scene_builder import load_sim_config
from src.simulation.panda_env import PandaEnv
from src.simulation.panda_demo import run_demo

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Controller demo 3 chốt; không phải ACT rollout")
    p.add_argument("--config",default="configs/mujoco/panda_three_pegs.json")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",default="outputs/three_pegs_demo")
    p.add_argument("--video",action="store_true")
    p.add_argument("--viewer",action="store_true")
    args=p.parse_args()
    cfg=load_sim_config(args.config)
    out=outside_raw(Path(args.output),read_json(ROOT/cfg["data_config"])); out.mkdir(parents=True,exist_ok=True)
    writer=None
    try:
        with PandaEnv(args.config,render_observations=False) as env:
            env.reset(seed=args.seed)
            if args.video:
                writer=cv2.VideoWriter(str(out/"controller_demo.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),env.cfg["control_hz"],(1280,360))
                if not writer.isOpened(): raise RuntimeError("Không mở được video writer")
            def callback(e,stage):
                if args.viewer: e.show_viewer()
                if writer:
                    rgb=np.hstack([e.render("overview"),e.render("task_closeup")])
                    frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    cv2.putText(frame,stage,(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
                    writer.write(frame)
            result=run_demo(env,callback)
            trace=result.pop("trace")
            result["config_hash"]=digest(env.cfg)
            result["config"]=env.cfg
            write_json(out/"result.json",result)
            write_json(out/"trace.json",trace)
            cv2.imwrite(str(out/"final_closeup.png"),cv2.cvtColor(env.render("task_closeup",1280,720),cv2.COLOR_RGB2BGR))
            print("Task success:",result["metrics"]["task_success"],"failure:",result["failure"])
    finally:
        if writer: writer.release()
