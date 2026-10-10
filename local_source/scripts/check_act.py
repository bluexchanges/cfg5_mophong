"""Full ACT inference trên train smoke; backward nhỏ kiểm graph, không optimizer.step."""
import argparse
from copy import deepcopy
import gc
import time
import platform
import torch
import torchvision
from torch.nn import functional as F
import _bootstrap
from src.data.common import ROOT, read_json, load_config, write_json, outside_raw, digest
from src.data.act_dataset import ACTDataset
from src.models.rh20t_act import RH20TACTPolicy, verify_upstream


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/act/rh20t_base8.json")
    p.add_argument("--processed-root", default="data/processed/smoke_cfg5")
    p.add_argument("--stats", default="data/splits/smoke_normalization.json")
    p.add_argument("--splits", default="data/splits/cfg5_splits.json")
    p.add_argument("--output", default="docs/audit/act_integration_check.json")
    a = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(42)
    cfg = read_json(a.config); data_cfg = load_config(ROOT / cfg["data_config"])
    stats = read_json(a.stats); splits = read_json(a.splits)
    dataset = ACTDataset(a.processed_root, splits, stats, data_cfg, "train", allow_partial_smoke=True)
    sample = dataset[0]
    state = torch.from_numpy(sample["state"]).unsqueeze(0)
    images = torch.from_numpy(sample["images"]).unsqueeze(0)
    actions = torch.from_numpy(sample["actions"]).unsqueeze(0)
    mask = torch.from_numpy(sample["is_pad"]).unsqueeze(0)
    verify_upstream(); started = time.perf_counter()
    policy = RH20TACTPolicy(cfg, data_cfg).eval()
    count = sum(p.numel() for p in policy.parameters())
    print(f"Full model {count:,} parameters; inference {list(images.shape)}", flush=True)
    with torch.inference_mode():
        prediction = policy(state, images)
    assert prediction.shape == (1, 20, 8) and torch.isfinite(prediction).all()
    full_seconds = time.perf_counter() - started
    del policy; gc.collect()
    small_cfg, small_data = deepcopy(cfg), deepcopy(data_cfg)
    small_cfg.update(hidden_dim=64, dim_feedforward=128, enc_layers=1, dec_layers=1, nheads=4, dropout=0.0)
    small_data.update(image_height=64, image_width=96)
    small_images = F.interpolate(images.flatten(0,1), size=(64,96), mode="bilinear", align_corners=False).reshape(1,8,3,64,96)
    policy = RH20TACTPolicy(small_cfg, small_data).train()
    before = policy.model.action_head.weight.detach().clone()
    losses = policy(state, small_images, actions, mask)
    losses["loss"].backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert policy.model.action_head.weight.grad.abs().sum() > 0
    assert policy.model.transformer.camera_embedding.weight.grad.abs().sum() > 0
    assert torch.equal(before, policy.model.action_head.weight)  # no optimizer step
    report = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
              "device": "cpu", "cuda_available_in_this_build": torch.cuda.is_available(),
              "seed": 42, "upstream_commit": cfg["upstream_commit"], "config_hash": digest(cfg),
              "data_config_hash": digest(data_cfg), "split_hash": splits["split_hash"],
              "sample_episode": sample["episode"], "sample_timestamp_ms": sample["timestamp_ms"],
              "samples_used": 1, "full_model_parameters": count,
              "full_model_input_shape": list(images.shape), "full_model_output_shape": list(prediction.shape),
              "full_model_forward_finite": True, "full_model_forward_seconds": full_seconds,
              "backward_smoke_architecture": {k: small_cfg[k] for k in ["hidden_dim", "dim_feedforward", "enc_layers", "dec_layers", "nheads"]},
              "backward_smoke_image_shape": list(small_images.shape), "backward_finite": True,
              "valid_action_steps": int(losses["valid_action_steps"]), "valid_action_elements": int(losses["valid_action_elements"]),
              "smoke_loss_only": {k: float(losses[k].detach()) for k in ["loss", "l1", "kl"]},
              "optimizer_steps": 0, "training_run": False, "act_pretrained_weights_loaded": False,
              "backbone_pretrained_weights_loaded": bool(cfg["backbone_weights_path"]), "simulation_policy_rollout": False,
              "scope": "Full configured ACT inference on one actual train-smoke sample; reduced architecture/resolution backward regression, no optimization or pretraining."}
    write_json(outside_raw(a.output, data_cfg), report)
    print("PASS: full inference + reduced backward; optimizer_steps=0", flush=True)
