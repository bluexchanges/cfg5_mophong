import _bootstrap
from src.simulation.scene_builder import build_scene, load_sim_config

if __name__ == "__main__":
    print(build_scene(load_sim_config()))
