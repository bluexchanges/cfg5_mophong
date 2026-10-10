"""Đổi observation vật lý sang input ACT NumPy; không gồm privileged objects."""
import numpy as np


def to_act_observation(obs,stats,data_cfg):
    if obs["images"] is None:
        raise ValueError("Cần render_observations=True cho ACT")
    expected=["cam_"+s for s in data_cfg["camera_serials"]]
    if obs["camera_names"]!=expected:
        raise ValueError("Camera order không khớp pretraining")
    rgb=np.asarray(obs["images"])
    if rgb.shape!=(8,data_cfg["image_height"],data_cfg["image_width"],3) or rgb.dtype!=np.uint8:
        raise ValueError("RGB shape/dtype không khớp")
    mean=np.asarray(stats["state"]["mean"])
    std=np.asarray(stats["state"]["std"])
    if np.any(std<=0): raise ValueError("Statistics std phải dương")
    images=(rgb.astype(np.float32)/255-np.asarray(data_cfg["image_normalization_mean"],dtype=np.float32))/np.asarray(data_cfg["image_normalization_std"],dtype=np.float32)
    state=(np.asarray(obs["state"])-mean)/std
    if not np.all(np.isfinite(state)): raise ValueError("State không hữu hạn")
    return {"images":np.ascontiguousarray(images.transpose(0,3,1,2)),"state":state.astype(np.float32)}


def denormalize_action(prediction,stats):
    result=np.asarray(prediction,dtype=float)*np.asarray(stats["action"]["std"])+np.asarray(stats["action"]["mean"])
    if result.shape!=(8,) or not np.all(np.isfinite(result)):
        raise ValueError("Prediction action không hợp lệ")
    # Controller validate_action sẽ normalize quaternion và kiểm workspace/width.
    return result
