"""Stage 3: evaluator/controller smoke multi-seed; chưa phải ACT evaluation."""
import argparse
import csv
from pathlib import Path
import _bootstrap
from src.data.common import ROOT, read_json, write_json, digest, outside_raw
from src.simulation.scene_builder import load_sim_config
from src.simulation.panda_env import PandaEnv
from src.simulation.panda_demo import run_demo

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["controller"],default="controller")
    parser.add_argument("--seeds",type=int,nargs="+",default=[42])
    parser.add_argument("--config",default="configs/mujoco/panda_three_pegs.json")
    parser.add_argument("--output",default="outputs/controller_evaluation")
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds): parser.error("Seeds phải khác nhau")
    cfg=load_sim_config(args.config)
    out=outside_raw(Path(args.output),read_json(ROOT/cfg["data_config"]))
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with PandaEnv(args.config,render_observations=False) as env:
        for seed in args.seeds:
            env.reset(seed=seed)
            result=run_demo(env)
            result.pop("trace")
            result["config_hash"]=digest(cfg)
            write_json(out/f"episode_{seed}.json",result)
            rows.append({"seed":seed,"task_success":result["metrics"]["task_success"],"placed_objects":sum(o["place_success"] for o in result["metrics"]["objects"]),"sim_time_s":result["sim_time_s"],"failure":result["failure"]})
    with (out/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    success=sum(r["task_success"] for r in rows)
    summary={"mode":"scripted_controller_privileged_state","act_model_used":False,"seeds":args.seeds,"episodes":len(rows),"successful_episodes":success,"success_rate":success/len(rows),"config_hash":digest(cfg),"scope":"engineering smoke checks; not ACT or statistical generalization evidence"}
    write_json(out/"summary.json",summary)
    print(summary)
