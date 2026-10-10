import numpy as np
import _bootstrap
from src.simulation.panda_env import PandaEnv

if __name__=="__main__":
    with PandaEnv() as env:
        obs,info=env.reset(seed=42)
        print("RGB:",obs["images"].shape,"state:",obs["state"].shape)
        action=obs["state"].astype(float).copy()
        action[0]+=0.01
        action[7]=0.04
        for _ in range(10):
            obs,reward,terminated,truncated,info=env.step(action)
            if terminated or truncated: break
        print("TCP:",obs["state"][:3],"gripper width (m):",obs["state"][-1])
