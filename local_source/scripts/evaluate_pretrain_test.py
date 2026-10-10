"""Freeze the validation-selected model, preprocess held-out test, and evaluate it."""

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_selection(frozen):
    from src.data.common import digest, require_contract
    from src.data.splits import validate_splits
    data, splits, stats = frozen["data_config"], frozen["splits"], frozen["normalization"]
    validate_splits(splits)
    for record in (splits, stats, frozen["manifest"]):
        require_contract(record, data)
    expected = {"model_config_hash": digest(frozen["model_config"]),
                "data_config_hash": digest(data), "split_hash": splits["split_hash"],
                "stats_hash": digest(stats), "manifest_hash": splits["manifest_hash"],
                "camera_serials": data["camera_serials"], "representation": data["representation"],
                "upstream_commit": frozen["model_config"]["upstream_commit"]}
    if expected != frozen["contract"] or frozen["manifest"]["manifest_hash"] != splits["manifest_hash"]:
        raise ValueError("Artifacts test đã khóa không khớp contract")
    if stats["source_split"] != "train" or stats["smoke_only"] or set(stats["episodes"]) != set(splits["splits"]["train"]):
        raise ValueError("Normalization phải giữ nguyên và chỉ fit trên toàn bộ train")
    if len(splits["splits"]["test"]) != 66:
        raise ValueError("Cần đúng 66 test episode đã khóa")


class InferenceMetrics:
    """Weight by real future targets; report physical units after denormalization."""

    def __init__(self, stats):
        import numpy as np
        self.mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
        self.std = np.asarray(stats["action"]["std"], dtype=np.float64)
        self.steps = self.samples = self.batches = self.quaternions = 0
        self.episodes = set()
        self.normalized_sum = np.zeros(8)
        self.physical_sum = np.zeros(8)
        self.position_sum = self.angle_sum = 0.0

    def add(self, prediction, target, is_pad, episodes):
        import numpy as np
        valid = ~np.asarray(is_pad, dtype=bool)
        predicted, expected = np.asarray(prediction)[valid], np.asarray(target)[valid]
        if predicted.shape != expected.shape or predicted.shape[1:] != (8,):
            raise ValueError("Prediction/target phải có 8D và shape khớp")
        if not len(predicted) or not np.isfinite(predicted).all() or not np.isfinite(expected).all():
            raise ValueError("Không có target thật hoặc prediction/target chứa NaN/Inf")
        self.steps += len(predicted)
        self.samples += len(valid)
        self.batches += 1
        self.episodes.update(episodes)
        self.normalized_sum += np.abs(predicted - expected).sum(axis=0)
        predicted = predicted * self.std + self.mean
        expected = expected * self.std + self.mean
        self.physical_sum += np.abs(predicted - expected).sum(axis=0)
        self.position_sum += np.linalg.norm(predicted[:, :3] - expected[:, :3], axis=1).sum()
        p, q = predicted[:, 3:7], expected[:, 3:7]
        pn, qn = np.linalg.norm(p, axis=1), np.linalg.norm(q, axis=1)
        good = (pn > 1e-8) & (qn > 1e-8)
        dot = np.abs((p[good] * q[good]).sum(axis=1) / (pn[good] * qn[good]))
        self.angle_sum += np.degrees(2 * np.arccos(np.clip(dot, 0, 1))).sum()
        self.quaternions += int(good.sum())

    def summary(self):
        if not self.steps:
            raise ValueError("Test không có target hợp lệ")
        return {"normalized_l1": float(self.normalized_sum.sum() / (self.steps * 8)),
                "normalized_mae_per_dimension": (self.normalized_sum / self.steps).tolist(),
                "position_mean_distance_mm": float(self.position_sum / self.steps * 1000),
                "position_mae_xyz_mm": (self.physical_sum[:3] / self.steps * 1000).tolist(),
                "gripper_mae_mm": float(self.physical_sum[7] / self.steps * 1000),
                "orientation_mean_angle_deg": float(self.angle_sum / self.quaternions)
                if self.quaternions else None,
                "orientation_valid_targets": self.quaternions,
                "orientation_invalid_targets": self.steps - self.quaternions,
                "samples": self.samples, "batches": self.batches,
                "episodes": len(self.episodes), "valid_action_steps": self.steps,
                "valid_action_elements": self.steps * 8}


def selected_record(history, saved, allow_early_stop=False):
    if not history or (not allow_early_stop and history[-1]["epoch"] + 1 != 80):
        raise ValueError("Cần lưu đủ 80 epoch trước khi khóa checkpoint test")
    winner = min(history, key=lambda r: r["val"]["loss"])
    if saved["epoch"] == winner["epoch"] and math.isclose(
            saved["val"]["loss"], winner["val"]["loss"], abs_tol=1e-10, rel_tol=0):
        return winner, history[-1]["epoch"] + 1
    # Cancellation may happen after atomic best save but before history commit.
    # A newer complete best checkpoint is usable for inference without an optimizer.
    if allow_early_stop and saved["epoch"] == history[-1]["epoch"] + 1 and saved["val"]["loss"] < winner["val"]["loss"]:
        return {"epoch": saved["epoch"], "val": saved["val"]}, saved["epoch"] + 1
    raise ValueError("best.pt không khớp checkpoint có val loss nhỏ nhất trong history")


def freeze(repo, output, allow_early_stop=False):
    import torch
    from src.data.common import ROOT, outside_raw, read_json, require_contract
    from src.training.pretrain import check_contract, load_run, write_json
    selection = output / "selection.json"
    if selection.exists():
        frozen = read_json(selection)
        validate_selection(frozen)
        outside_raw(output, frozen["data_config"])
        if file_hash(output / "best.pt") != frozen["checkpoint_sha256"]:
            raise ValueError("Checkpoint test đã khóa đổi nội dung")
        return frozen
    config = ROOT / "configs/act/cfg5_all_pretrain_bs8_80epoch.json"
    run, model, data, splits, stats, contract = load_run(config)
    outside_raw(output, data)
    history = read_json(ROOT / run["output_dir"] / "history.json")
    if not history or (not allow_early_stop and history[-1]["epoch"] + 1 != 80):
        raise ValueError("Cần lưu đủ 80 epoch trước khi khóa checkpoint test")
    if any(r.get("smoke") is not False for r in history):
        raise ValueError("History phải là full training")
    if stats["source_split"] != "train" or stats["smoke_only"]:
        raise ValueError("Chỉ dùng normalization train đầy đủ")
    if len(splits["splits"]["test"]) != 66:
        raise ValueError("Cần đúng 66 test episode đã khóa")
    source = ROOT / run["output_dir"] / "best.pt"
    saved = torch.load(source, map_location="cpu", weights_only=True)
    check_contract(saved, contract)
    if saved.get("kind") != "best_model" or saved.get("smoke") is not False:
        raise ValueError("Cần best.pt full training")
    winner, completed_epochs = selected_record(history, saved, allow_early_stop)
    del saved
    manifest = read_json(ROOT / "data/splits/cfg5_server_manifest.json")
    require_contract(manifest, data)
    if manifest["manifest_hash"] != splits["manifest_hash"]:
        raise ValueError("Manifest không khớp split")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "best.pt.partial"
    shutil.copyfile(source, temporary)
    temporary.replace(output / "best.pt")
    frozen = {"selection_rule": "minimum validation loss after user requested early stop"
              if allow_early_stop else "minimum validation loss after completed epoch 80",
              "training_completed_epochs": completed_epochs, "stopped_early": allow_early_stop,
              "checkpoint_epoch": winner["epoch"] + 1,
              "selection_val_loss": winner["val"]["loss"],
              "checkpoint_sha256": file_hash(output / "best.pt"),
              "run_config": run, "model_config": model, "data_config": data,
              "splits": splits, "normalization": stats, "manifest": manifest,
              "contract": contract, "seed": run["seed"],
              "inference": "images/state only, ACT prior latent z=0",
              "action_semantics": "future measured TCP base pose + gripper command width",
              "optimizer_steps": 0}
    validate_selection(frozen)
    write_json(selection, frozen)
    print(f"ĐÃ KHÓA best epoch {winner['epoch']+1}, val={winner['val']['loss']:.6f}", flush=True)
    return frozen


def preprocess_test(frozen, output):
    from src.data.cfg5 import preprocess_episode, verify_processed_episode
    from src.training.pretrain import write_json
    records = {r["episode"]: r for r in frozen["manifest"]["episodes"]}
    names = frozen["splits"]["splits"]["test"]
    root = output / "processed_test"
    for index, name in enumerate(names, 1):
        if (root / (name + ".partial")).exists():
            raise FileExistsError(f"Kiểm ERROR.json của {name}.partial trước khi tiếp tục")
        function = verify_processed_episode if (root / name).is_dir() else preprocess_episode
        result = function(records[name], frozen["data_config"], root, frozen["splits"]["split_hash"])
        print(f"PREPROCESS_TEST {index}/{len(names)} {name} steps={result['steps']}", flush=True)
    write_json(output / "preprocessing.json", {"split": "test", "episodes": names})
    return root


def evaluate_batches(policy, loader, stats, kl_weight, device, seed):
    import torch
    from time import perf_counter
    from src.training.pretrain import Metrics, move_batch, seed_everything
    seed_everything(seed)
    policy.eval()
    inference = InferenceMetrics(stats)
    reconstruction = Metrics()
    started = perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            tensors = move_batch(batch, device)
            predicted = policy(tensors["state"], tensors["images"])
            inference.add(predicted.cpu().numpy(), batch["actions"].numpy(),
                          batch["is_pad"].numpy(), batch["episode"])
            result = policy(tensors["state"], tensors["images"],
                            tensors["actions"], tensors["is_pad"])
            reconstruction.add(result, batch)
            if index % 100 == 0:
                print(f"test batch={index}/{len(loader)} samples={inference.samples}", flush=True)
    return inference, reconstruction.summary(kl_weight), perf_counter() - started


def evaluate(frozen, output, processed):
    import torch
    from src.models.rh20t_act import RH20TACTPolicy
    from src.training.pretrain import check_contract, make_loader, write_json
    if not torch.cuda.is_available():
        raise RuntimeError("Test full ACT cần CUDA trong job GPU")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    check_contract(saved, frozen["contract"])
    policy = RH20TACTPolicy(frozen["model_config"], frozen["data_config"]).to(device)
    policy.load_state_dict(saved["model"], strict=True)
    del saved
    loader = make_loader(processed, frozen["splits"], frozen["normalization"],
                         frozen["data_config"], "test", 8, 2)
    inference, reconstruction, seconds = evaluate_batches(
        policy, loader, frozen["normalization"], frozen["model_config"]["kl_weight"],
        device, frozen["seed"])
    if inference.samples != len(loader.dataset) or inference.episodes != set(frozen["splits"]["splits"]["test"]):
        raise ValueError("Test chưa đánh giá đủ toàn bộ sample/66 episode")
    if file_hash(output / "best.pt") != frozen["checkpoint_sha256"]:
        raise ValueError("Checkpoint đổi trong lúc test")
    report = {"split": "test", "smoke": False, "optimizer_steps": 0,
              "training_completed_epochs": frozen.get("training_completed_epochs", 80),
              "stopped_early": frozen.get("stopped_early", False),
              "checkpoint_epoch": frozen["checkpoint_epoch"],
              "checkpoint_sha256": frozen["checkpoint_sha256"], "contract": frozen["contract"],
              "seed": frozen["seed"], "batch_size": 8, "num_workers": 2,
              "inference": inference.summary(),
              "posterior_reconstruction": reconstruction,
              "evaluation_seconds": seconds,
              "notes": "Inference uses only images/state and z=0. Reconstruction uses target actions in CVAE posterior. Physical metrics are pooled over valid future targets; no simulation rollout or success rate."}
    write_json(output / "test_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("CFG5_PRETRAIN_TEST_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stopped-early", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    output = repo / "outputs/act_pretrain_bs8_to80" / ("test_final" if args.stopped_early else "test_epoch80")
    from src.data.common import read_json
    frozen = freeze(repo, output, args.stopped_early)
    if (output / "test_report.json").exists():
        report = read_json(output / "test_report.json")
        if report["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
            raise ValueError("Report cũ không khớp checkpoint đã khóa")
        print("Test đã hoàn tất; đọc lại báo cáo đã lưu:", flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    evaluate(frozen, output, preprocess_test(frozen, output))


if __name__ == "__main__":
    main()
