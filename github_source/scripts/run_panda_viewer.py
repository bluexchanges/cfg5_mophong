import time
import argparse
import _bootstrap
from src.simulation.panda_env import PandaEnv

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,help="Tự đóng viewer sau khoảng thời gian này để smoke test")
    args=p.parse_args()
    with PandaEnv(render_observations=False) as env:
        env.reset()
        viewer=env.show_viewer()
        print("Đóng cửa sổ để thoát; chạy run_pick_place_demo.py --viewer để xem controller.")
        started=time.monotonic()
        while viewer.is_running() and (args.seconds is None or time.monotonic()-started<args.seconds):
            viewer.sync()
            time.sleep(0.03)
        print("Viewer smoke/close completed")
