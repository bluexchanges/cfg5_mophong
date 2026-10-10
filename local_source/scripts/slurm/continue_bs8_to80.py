"""Resume to 80 epochs on one GPU, batch 8, preserving the fast40 workers."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
OUTPUT = "outputs/act_pretrain_bs8_to80"
CONFIG = "configs/act/cfg5_all_pretrain_bs8_80epoch.json"


def target_config(old):
    if old.get("gpu_count", 1) != 1:
        raise ValueError("Nguồn phải là lượt 1 GPU trước đây")
    if type(old["num_workers"]) is not int or not 0 <= old["num_workers"] <= 4:
        raise ValueError("Số worker nguồn không thuộc cấu hình fast40 cũ (0–4)")
    if old["save_every_epochs"] != 1:
        raise ValueError("Cần checkpoint mỗi epoch")
    # Do not inject optional keys into older config signatures.
    return dict(old, epochs=80, batch_size=8, output_dir=OUTPUT)


def load_source():
    import torch
    from src.training.pretrain import resume_signature
    child = Path(OUTPUT) / "latest.pt"
    source = child if child.exists() else Path("outputs/act_pretrain_fast/latest.pt")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("kind") != "train_state" or saved.get("smoke") is not False:
        raise ValueError("Nguồn phải là latest.pt của full training")
    old = saved["metadata"]["run_config"]
    if saved["run_config_hash"] != resume_signature(old):
        raise ValueError("Hash config nguồn không khớp")
    if saved["epoch"] + 1 < 40:
        raise ValueError("Nguồn chưa lưu đủ 40 epoch")
    run = target_config(old)
    if source == child and resume_signature(run) != saved["run_config_hash"]:
        raise ValueError("Nhánh BS8 hiện có khác cấu hình; không ghi đè")
    return source, saved, run


def main():
    import torch
    from src.training import pretrain
    from src.training.fork_run import fork_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 1, "Cần đúng 1 GPU CUDA được Slurm cấp"
    source, saved, run = load_source()
    pretrain.write_json(CONFIG, run)
    loaded = pretrain.load_run(CONFIG)
    run, model, data, splits, stats, contract = loaded
    pretrain.check_contract(saved, contract)
    epoch = saved["epoch"] + 1
    if epoch >= 80:
        print("Đã lưu đủ 80 epoch", flush=True)
        return
    if args.probe:
        generator = torch.Generator()
        generator.set_state(saved["loader_generator_state"].cpu())
        pretrain.seed_everything(run["seed"])
        policy = pretrain.RH20TACTPolicy(model, data).cuda()
        policy.load_state_dict(saved["model"])
        optimizer = policy.configure_optimizer()
        optimizer.load_state_dict(saved["optimizer"])
        pretrain.restore_rng(saved["rng"])
        del saved
        loader = pretrain.make_loader(pretrain.ROOT / run["processed_root"], splits, stats,
                                     data, "train", 8, run["num_workers"], generator=generator)
        torch.cuda.reset_peak_memory_stats()
        result = pretrain.run_epoch(policy, loader, "cuda", optimizer=optimizer,
                                   max_batches=3, cuda_prefetch=run.get("cuda_prefetch", False),
                                   grad_clip_norm=run["grad_clip_norm"])
        result["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        print("PROBE BS8 (3 batch, không lưu trọng số): " + json.dumps(result), flush=True)
        return
    del saved
    if source.parent != Path(OUTPUT):
        source = fork_run(*loaded, source)
    print(f"Tiếp từ {epoch} epoch đã lưu; batch=8; worker cũ={run['num_workers']}; 1 GPU", flush=True)
    pretrain.train(*loaded, device=torch.device("cuda"), resume=source)
    print("ACT_BS8_TO80_OK", flush=True)


if __name__ == "__main__":
    main()
