"""Fork the completed fast40 run, or resume its BS16/W8/two-GPU child."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

OUTPUT = "outputs/act_pretrain_bs16_w8_gpu2"
CONFIG = "configs/act/cfg5_all_pretrain_bs16_w8_gpu2_80epoch.json"


def target_config(old):
    run = dict(old)
    run.update(epochs=80, batch_size=16, num_workers=8, gpu_count=2,
               output_dir=OUTPUT, cuda_prefetch=True)
    if run["save_every_epochs"] != 1:
        raise ValueError("Lượt này yêu cầu checkpoint mỗi epoch")
    return run


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
        raise ValueError("Nhánh hiện có khác batch/worker/GPU/config; không ghi đè")
    return source, saved, run


def main():
    import torch
    from src.training import pretrain
    from src.training.fork_run import fork_run
    from src.training.multi_gpu import ParallelACT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 2, "Slurm phải cấp đúng 2 GPU CUDA nhìn thấy"
    source, saved, run = load_source()
    pretrain.write_json(CONFIG, run)
    loaded = pretrain.load_run(CONFIG)
    run, model, data, splits, stats, contract = loaded
    pretrain.check_contract(saved, contract)
    if saved["epoch"] + 1 >= 80:
        print("Đã lưu đủ 80 epoch", flush=True)
        return
    if args.probe:
        generator = torch.Generator()
        generator.set_state(saved["loader_generator_state"].cpu())
        pretrain.seed_everything(run["seed"])
        policy = ParallelACT(pretrain.RH20TACTPolicy(model, data).cuda(), 2)
        policy.load_state_dict(saved["model"])
        optimizer = policy.configure_optimizer()
        optimizer.load_state_dict(saved["optimizer"])
        pretrain.restore_rng(saved["rng"])
        del saved
        loader = pretrain.make_loader(pretrain.ROOT / run["processed_root"], splits, stats,
                                     data, "train", 16, 8, generator=generator)
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        result = pretrain.run_epoch(policy, loader, "cuda:0", optimizer=optimizer,
                                   max_batches=3, cuda_prefetch=True,
                                   grad_clip_norm=run["grad_clip_norm"])
        result["gpu_peak_reserved_mib"] = [torch.cuda.max_memory_reserved(i) / 2**20
                                           for i in range(2)]
        print("PROBE (3 batch, không lưu trọng số): " + json.dumps(result), flush=True)
        return
    epoch = saved["epoch"] + 1
    del saved  # Release the large CPU checkpoint before fork_run loads it.
    if source.parent != Path(OUTPUT):
        source = fork_run(*loaded, source)
    print(f"Tiếp từ {epoch} epoch đã lưu; batch tổng=16; 8 workers; 2 GPUs", flush=True)
    pretrain.train(*loaded, device=torch.device("cuda:0"), resume=source)
    print("ACT_BS16_GPU2_TO80_OK", flush=True)


if __name__ == "__main__":
    main()
