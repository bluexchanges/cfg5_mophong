"""Check official MCR weights, frozen spatial extraction, and full-build storage before preprocessing."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import json
from pathlib import Path
import shutil
import time
import _bootstrap
import torch
from src.data.common import ROOT, outside_raw
from src.data.feature_cache import context, cache_contract
from src.models.mcr_act import MCRACTPolicy, MCR_SHA, PREPROCESS
from src.data.sim_demos import file_hash
from prepare_mcr_cache import estimate, home_usage


def check_backbone(model, cfg, checkpoint, device):
    path = Path(checkpoint)
    if not path.is_file() or path.stat().st_size != 94345861 or file_hash(path) != MCR_SHA:
        raise ValueError("MCR checkpoint absent or official size/SHA256 differs")
    model = {**model, "backbone": "mcr_resnet50", "backbone_weights_path": None,
             "feature_preprocess": PREPROCESS, "lr_backbone": 0.}
    policy = MCRACTPolicy(model, cfg, mcr_checkpoint=path).to(device)
    policy.train()  # Verify frozen backbone stays in eval even when ACT enters training mode.
    backbone = policy.model.backbones
    if any(p.requires_grad for p in backbone.parameters()) or any(m.training for m in backbone.modules()):
        raise ValueError("ResNet parameters/BatchNorm are not frozen")
    buffers = {k: v.clone() for k, v in backbone.named_buffers()}
    rgb = torch.zeros((1, 8, 3, 360, 640), dtype=torch.uint8, device=device)
    rgb[:, 0] = 160
    valid = torch.zeros((1, 8), dtype=torch.bool, device=device)
    valid[:, :2] = True
    started = time.perf_counter()
    with torch.no_grad():
        first = policy.extract_features(rgb, camera_valid=valid)
        second = policy.extract_features(rgb, camera_valid=valid)
    if device == "cuda":
        torch.cuda.synchronize()
    if first.shape != (1, 8, 2048, 7, 7) or not torch.isfinite(first).all():
        raise ValueError("MCR output shape or finite check failed")
    if first.requires_grad or torch.count_nonzero(first[:, 2:]).item() != 0:
        raise ValueError("Missing-camera mask or extraction gradient check failed")
    if not torch.allclose(first, second, rtol=1e-6, atol=1e-6):
        raise ValueError("Repeated eval extraction differs")
    if any(not torch.equal(buffers[k], v) for k, v in backbone.named_buffers()):
        raise ValueError("Backbone running statistics changed")
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError("ACT was accidentally frozen too")
    return {"ok": True, "checkpoint": str(path), "sha256": MCR_SHA,
            "bytes": path.stat().st_size, "architecture": "mcr_resnet50",
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "feature_shape": list(first.shape), "backbone_trainable_parameters": 0,
            "act_trainable_parameters": trainable, "batchnorm_buffers_unchanged": True,
            "masked_cameras_zero": True, "repeat_allclose": True,
            "synthetic_check_seconds": time.perf_counter() - started,
            "note": "Synthetic plumbing/freeze check, not robot task quality or training throughput"}


def storage_budget(used, additional, quota, free, reserve=10 * 2**30):
    return {"home_used_GiB": used / 2**30, "pending_estimated_GiB": additional / 2**30,
            "quota_budget_GiB": quota / 2**30, "reserve_GiB": reserve / 2**30,
            "filesystem_free_GiB": free / 2**30,
            "additional_space_needed_GiB": max(0, used + additional + reserve - quota) / 2**30,
            "full_build_fits": used + additional + reserve <= quota and free >= additional + reserve,
            "scope": "Entire shared user home; 1.8 decimal TB script budget, not a quota query"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/act/cfg5_robot_all_masked.json")
    parser.add_argument("--cache-root", default="data/features/cfg5_robot_all_mcr_features")
    parser.add_argument("--checkpoint", default="outputs/backbones/mcr_resnet50.pth")
    parser.add_argument("--output", default="outputs/mcr_checks/mcr_frozen_check.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--home-quota-tb", type=float, default=1.8)
    parser.add_argument("--require-build-space", action="store_true")
    parser.add_argument("--include-test", action="store_true", help="Include separately held-out frozen test features in disk budget")
    args = parser.parse_args()
    if args.home_quota_tb <= 0:
        parser.error("Positive home budget required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run GPU sbatch")
    run, model, cfg, split, stats = context(args.base_config)
    if (cfg["representation"] != "panda_joint_positions_rad_gripper_width_m"
            or run.get("cache_source_backend") != "cfg5_masked_raw_mp4_v1"):
        raise ValueError("Wrong dataset/representation; expected new masked joint CFG5")
    names = [name for kind in ("train", "val", "test") for name in split["splits"][kind]]
    if len(names) != 1317 or len(set(names)) != 1317:
        raise ValueError("Expected locked 1317-episode split")
    torch.set_num_threads(2)
    result = check_backbone(model, cfg, ROOT / args.checkpoint, args.device)
    print("MCR_RESNET_CHECK_OK", result, flush=True)
    source = ROOT / run["processed_root"]
    root = outside_raw(ROOT / args.cache_root, cfg)
    contract = cache_contract(cfg, split, stats, 90)
    cache_names = split["splits"]["train"] + split["splits"]["val"] + (split["splits"]["test"] if args.include_test else [])
    estimate_result = estimate(root, source, cache_names,
                               cfg, split, contract, features_only=True)
    used = home_usage()
    location = root if root.exists() else ROOT
    budget = storage_budget(used, estimate_result["pending_bytes"], int(args.home_quota_tb * 10**12),
                            shutil.disk_usage(location).free)
    output = outside_raw(ROOT / args.output, cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"resnet": result, "split_counts": {k: len(v) for k, v in split["splits"].items()},
               "cache_estimate": estimate_result, "storage": budget,
               "test_features_in_build_scope": args.include_test, "features_only": True,
               "test_used_for_training": False}
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print("MCR_STORAGE_CHECK", budget, "report=" + str(output), flush=True)
    if args.require_build_space and not budget["full_build_fits"]:
        raise OSError(f"ResNet OK, nhưng chưa đủ chỗ build feature: cần giải phóng thêm ít nhất "
                      f"{budget['additional_space_needed_GiB']:.1f} GiB theo ngân sách home; không tạo cache")
    print("MCR_PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
