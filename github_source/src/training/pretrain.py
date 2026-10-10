"""CFG5 ACT pretraining, validation and reproducible epoch-boundary resume."""

import json
import hashlib
import os
import platform
import random
from contextlib import closing
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from src.data.act_dataset import ACTDataset
from src.data.common import ROOT, digest, load_config, outside_raw, read_json
from src.models.rh20t_act import RH20TACTPolicy


def load_run(config_path, *, splits_path=None, stats_path=None, processed_root=None):
    run = read_json(config_path)
    required = {"model_config", "splits", "locked_membership", "stats", "processed_root", "output_dir",
                "seed", "epochs", "batch_size", "num_workers", "save_every_epochs",
                "log_every_batches", "grad_clip_norm", "require_cuda_for_full_run"}
    if not required.issubset(run):
        raise ValueError(f"Thiếu trường run config: {sorted(required - set(run))}")
    for key in ("seed", "epochs", "batch_size", "num_workers", "save_every_epochs", "log_every_batches"):
        if not isinstance(run[key], int) or run[key] < (0 if key in ("seed", "num_workers") else 1):
            raise ValueError(f"Run config {key} không hợp lệ")
    if not isinstance(run["grad_clip_norm"], (float, int)) or run["grad_clip_norm"] <= 0:
        raise ValueError("grad_clip_norm phải dương")
    if not isinstance(run["require_cuda_for_full_run"], bool):
        raise ValueError("require_cuda_for_full_run phải boolean")
    run = dict(run)
    if not isinstance(run.get("cuda_prefetch", False), bool):
        raise ValueError("cuda_prefetch phải là boolean")
    gpu_count = run.get("gpu_count", 1)
    if type(gpu_count) is not int or gpu_count not in (1, 2):
        raise ValueError("gpu_count chỉ hỗ trợ 1 hoặc 2")
    if splits_path is not None:
        run["splits"] = splits_path
    if stats_path is not None:
        run["stats"] = stats_path
    if processed_root is not None:
        run["processed_root"] = processed_root
    model_cfg = read_json(ROOT / run["model_config"])
    data_cfg = load_config(ROOT / model_cfg["data_config"])
    splits = read_json(ROOT / run["splits"])
    locked = read_json(ROOT / run["locked_membership"])
    for name in ("train", "val", "test"):
        if splits["splits"][name] != locked["splits"][name]:
            raise ValueError(f"Server split {name} khác membership đã khóa")
    stats = read_json(ROOT / run["stats"])
    contract = {"model_config_hash": digest(model_cfg), "data_config_hash": digest(data_cfg),
                "split_hash": splits["split_hash"], "stats_hash": digest(stats),
                "manifest_hash": splits["manifest_hash"],
                "camera_serials": data_cfg["camera_serials"],
                "representation": data_cfg["representation"],
                "upstream_commit": model_cfg["upstream_commit"]}
    return run, model_cfg, data_cfg, splits, stats, contract


class EpisodeShuffleSampler(Sampler):
    """Shuffle episodes and their samples while keeping memmap access local."""

    def __init__(self, dataset, generator):
        self.generator = generator
        groups = {}
        for index, (episode, _start, _end) in enumerate(dataset.index):
            groups.setdefault(episode, []).append(index)
        self.groups = list(groups.values())

    def __len__(self):
        return sum(len(group) for group in self.groups)

    def __iter__(self):
        for group_id in torch.randperm(len(self.groups), generator=self.generator).tolist():
            group = self.groups[group_id]
            for local_id in torch.randperm(len(group), generator=self.generator).tolist():
                yield group[local_id]


def make_loader(root, splits, stats, data_cfg, split_name, batch_size, num_workers,
                *, generator=None, allow_partial_smoke=False):
    dataset = ACTDataset(root, splits, stats, data_cfg, split_name, allow_partial_smoke)
    sampler = EpisodeShuffleSampler(dataset, generator) if split_name == "train" else None
    worker_options = ({"prefetch_factor": 2, "persistent_workers": True,
                       "multiprocessing_context": "spawn"} if num_workers else {})
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      generator=generator, **worker_options)


def move_batch(batch, device):
    return {key: batch[key].to(device, non_blocking=True) for key in
            ("state", "images", "actions", "is_pad")}


def device_batches(batches, device, *, cuda_prefetch=False):
    """Stage one batch ahead, preserving CPU sources until async copies finish.

    wait_stream is enqueued before the NEXT copy, so compute waits only for
    the current batch. record_stream protects destination allocations used
    by the compute stream. Caller must close this generator on early exit.
    """
    if not cuda_prefetch:
        for batch in batches:
            # Serial path keeps transfer inside run_epoch's step timer.
            yield batch, None
        return
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("cuda_prefetch cần thiết bị CUDA")
    stream = torch.cuda.Stream(device=device)
    iterator = iter(batches)
    pending = []

    def stage():
        batch = next(iterator, None)
        if batch is None:
            return None
        if not all(batch[key].is_pinned() for key in
                   ("state", "images", "actions", "is_pad")):
            raise ValueError("cuda_prefetch cần DataLoader pin_memory=True")
        pending[:] = [(cpu, event) for cpu, event in pending
                      if event is None or not event.query()]
        if len(pending) >= 2:
            pending[0][1].synchronize()
            del pending[0]
        # Keep a reference even if an exception occurs during a partial copy.
        pending.append((batch, None))
        with torch.cuda.stream(stream):
            tensors = move_batch(batch, device)
            ready = torch.cuda.Event()
            ready.record(stream)
        pending[-1] = (batch, ready)
        return batch, tensors

    try:
        current = stage()
        while current is not None:
            compute = torch.cuda.current_stream(device)
            compute.wait_stream(stream)
            for tensor in current[1].values():
                tensor.record_stream(compute)
            # The next copy may overlap compute after yield.
            following = stage()
            pending[:] = [(cpu, event) for cpu, event in pending
                          if event is None or not event.query()]
            yield current
            current = following
    finally:
        stream.synchronize()
        pending.clear()


class Metrics:
    def __init__(self):
        self.l1_sum = 0.0
        self.kl_sum = 0.0
        self.valid_elements = 0
        self.samples = 0
        self.batches = 0
        self.episodes = set()

    def add(self, result, batch):
        elements = int(result["valid_action_elements"].detach())
        samples = len(batch["state"])
        self.l1_sum += float(result["l1"].detach()) * elements
        self.kl_sum += float(result["kl"].detach()) * samples
        self.valid_elements += elements
        self.samples += samples
        self.batches += 1
        self.episodes.update(batch["episode"])

    def summary(self, kl_weight):
        if not self.valid_elements or not self.samples:
            raise ValueError("Không có action thật để tổng hợp metric")
        l1 = self.l1_sum / self.valid_elements
        kl = self.kl_sum / self.samples
        return {"loss": l1 + kl_weight * kl, "l1": l1, "kl": kl,
                "samples": self.samples, "batches": self.batches,
                "episodes": len(self.episodes), "valid_action_steps": self.valid_elements // 8,
                "valid_action_elements": self.valid_elements}


def run_epoch(policy, loader, device, *, optimizer=None, max_batches=0,
              grad_clip_norm=1.0, log_every_batches=0, cuda_prefetch=False):
    if max_batches < 0:
        raise ValueError("max_batches không được âm")
    training = optimizer is not None
    policy.train(training)
    metrics = Metrics()
    started = wait_started = perf_counter()
    wait_seconds = step_seconds = window_wait = window_step = 0.0
    window_batches = window_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    source = islice(loader, max_batches) if max_batches else loader
    with context, closing(device_batches(source, device, cuda_prefetch=cuda_prefetch)) as batches:
        for batch_number, (batch, tensors) in enumerate(batches, start=1):
            fetched = perf_counter()
            wait = fetched - wait_started
            wait_seconds += wait
            window_wait += wait
            if tensors is None:
                tensors = move_batch(batch, device)
            result = policy(tensors["state"], tensors["images"],
                            tensors["actions"], tensors["is_pad"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                result["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm,
                                                       error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Gradient không hữu hạn")
                optimizer.step()
            metrics.add(result, batch)
            # Metrics reads CUDA scalars after optimizer.step on the same stream.
            # This host interval includes H2D, compute and scalar synchronization;
            # With prefetch, next H2D was enqueued by the iterator and can
            # overlap compute. Neither interval is a kernel-only measurement.
            step = perf_counter() - fetched
            step_seconds += step
            window_step += step
            window_batches += 1
            window_samples += len(batch["state"])
            if log_every_batches and batch_number % log_every_batches == 0:
                print(f"{'train' if training else 'val'} batch={batch_number} "
                      f"samples={metrics.samples} valid_steps={metrics.valid_elements // 8} "
                      f"data_wait_ms={1000 * window_wait / window_batches:.1f} "
                      f"step_ms={1000 * window_step / window_batches:.1f} "
                      f"samples_per_s={window_samples / max(window_wait + window_step, 1e-9):.2f}",
                      flush=True)
                window_wait = window_step = 0.0
                window_batches = window_samples = 0
            wait_started = perf_counter()
    result = metrics.summary(policy.config["kl_weight"])
    elapsed = perf_counter() - started
    result["timing"] = {"epoch_seconds": elapsed, "data_wait_seconds": wait_seconds,
                        "step_seconds": step_seconds,
                        "samples_per_s": metrics.samples / max(elapsed, 1e-9)}
    return result


def capture_rng():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (np_state[0], np_state[1].tolist(),
            np_state[2], np_state[3], np_state[4]), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint CUDA không thể resume trên runtime thiếu CUDA")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def validate_with_fixed_seed(policy, loader, device, seed, *, max_batches=0,
                             log_every_batches=0, cuda_prefetch=False):
    before = capture_rng()
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return run_epoch(policy, loader, device, max_batches=max_batches,
                         log_every_batches=log_every_batches, cuda_prefetch=cuda_prefetch)
    finally:
        restore_rng(before)


def check_contract(checkpoint, contract):
    if checkpoint.get("contract") != contract:
        raise ValueError("Checkpoint không khớp model/data/split/stats/camera contract")


def initialize_finetune(policy, checkpoint, model_cfg, data_cfg, stats, checkpoint_sha256):
    """New optimizer/run; keep the pretrained model's numeric normalization."""
    if checkpoint.get("smoke") is not False or checkpoint.get("kind") != "best_model":
        raise ValueError("Fine-tune cần best_model full đã khóa bằng validation")
    source = checkpoint["metadata"]
    ignored = {"data_config", "note", "lr", "lr_backbone", "weight_decay", "kl_weight"}
    architecture = lambda cfg: {k: v for k, v in cfg.items() if k not in ignored}
    if architecture(source["model_config"]) != architecture(model_cfg):
        raise ValueError("Kiến trúc fine-tune khác checkpoint")
    if (data_cfg.get("source_type") != "mujoco_scripted_controller"
            or data_cfg.get("action_target_offset") != 0
            or data_cfg["representation"] != "tcp_base_xyz_wxyz_gripper_width_m"):
        raise ValueError("Initializer này dành cho demo MuJoCo command cùng tick")
    for key in ("camera_serials", "image_width", "image_height", "chunk_size",
                "image_normalization_mean", "image_normalization_std"):
        if source["data_config"][key] != data_cfg[key]:
            raise ValueError(f"Data fine-tune khác checkpoint: {key}")
    origin = stats.get("inherited_from", {})
    if (origin.get("checkpoint_sha256") != checkpoint_sha256
            or origin.get("stats_hash") != digest(source["normalization"])
            or stats.get("normalization_origin") != "pretraining_train_checkpoint"):
        raise ValueError("Fine-tune phải kế thừa normalization từ đúng checkpoint")
    for key in ("state", "action"):
        if stats[key] != source["normalization"][key]:
            raise ValueError("Numeric normalization đã đổi; cần migration weights riêng")
    policy.load_state_dict(checkpoint["model"], strict=True)
    return {"method": "pretrained_weights_fresh_optimizer", "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint["epoch"] + 1,
            "source_contract": checkpoint["contract"],
            "target_action_semantics": "recorded command at observation tick"}


def resume_signature(run):
    """An epoch target may be extended without changing the training method."""
    return digest({key: value for key, value in run.items() if key != "epochs"})


def checkpoint_metadata(run, model_cfg, data_cfg, splits, stats):
    return {"run_config": run, "model_config": model_cfg, "data_config": data_cfg,
            "normalization": stats, "split_hash": splits.get("split_hash"),
            "manifest_hash": splits.get("manifest_hash"),
            "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "numpy": np.__version__, "torch_cuda": torch.version.cuda}}


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested="auto", *, require_cuda=False):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device phải là auto, cpu hoặc cuda")
    chosen = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if chosen == "auto":
        chosen = "cpu"
    if chosen == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch không thấy CUDA; kiểm torch/driver/GPU trên server")
    if require_cuda and chosen != "cuda":
        raise RuntimeError("Full pretraining yêu cầu CUDA theo run config; smoke có thể dùng CPU")
    return torch.device(chosen)


def train(run, model_cfg, data_cfg, splits, stats, contract, *, device, resume=None,
          smoke=False, allow_partial_smoke=False, init_checkpoint=None):
    if resume and init_checkpoint:
        raise ValueError("Chọn --resume hoặc --init-checkpoint, không dùng cả hai")
    if allow_partial_smoke and not smoke:
        raise ValueError("Partial/smoke data chỉ dùng với --smoke")
    if stats["smoke_only"] and not allow_partial_smoke:
        raise ValueError("Không dùng smoke normalization cho full pretraining")
    out = outside_raw(ROOT / run["output_dir"], data_cfg)
    if smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    if not resume and any((out / name).exists() for name in ("latest.pt", "best.pt", "history.json")):
        raise FileExistsError(f"Output đã có checkpoint/history: {out}; dùng --resume hoặc output_dir mới")
    generator = torch.Generator()
    seed_everything(run["seed"])
    generator.manual_seed(run["seed"])
    train_loader = make_loader(ROOT / run["processed_root"], splits, stats, data_cfg,
                               "train", run["batch_size"], run["num_workers"],
                               generator=generator, allow_partial_smoke=allow_partial_smoke)
    val_loader = make_loader(ROOT / run["processed_root"], splits, stats, data_cfg,
                             "val", run["batch_size"], run["num_workers"],
                             allow_partial_smoke=allow_partial_smoke)
    policy = RH20TACTPolicy(model_cfg, data_cfg).to(device)
    if run.get("gpu_count", 1) > 1:
        if torch.device(device).type != "cuda":
            raise ValueError("Multi-GPU cần device=cuda")
        from src.training.multi_gpu import ParallelACT
        policy = ParallelACT(policy, run["gpu_count"])
    optimizer = policy.configure_optimizer()
    metadata = checkpoint_metadata(run, model_cfg, data_cfg, splits, stats)
    start_epoch, global_step, best_loss = 0, 0, float("inf")
    if init_checkpoint:
        checksum = hashlib.sha256()
        with Path(init_checkpoint).open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                checksum.update(block)
        saved = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        metadata["initialization"] = initialize_finetune(
            policy, saved, model_cfg, data_cfg, stats, checksum.hexdigest())
        print(f"Fine-tune từ pretrain epoch {saved['epoch'] + 1}; optimizer mới; epoch bắt đầu 1", flush=True)
        del saved
    elif data_cfg.get("source_type") == "mujoco_scripted_controller" and not resume:
        raise ValueError("Fine-tune cần --init-checkpoint; resume dùng latest.pt của run fine-tune")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        check_contract(saved, contract)
        if saved.get("kind") != "train_state" or saved.get("smoke") != smoke:
            raise ValueError("Resume cần train_state của cùng chế độ smoke/full")
        if saved["run_config_hash"] != resume_signature(run):
            raise ValueError("Run config đã đổi; không resume cùng thí nghiệm")
        policy.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        generator.set_state(saved["loader_generator_state"].cpu())
        start_epoch = saved["epoch"] + 1
        global_step = saved["global_step"]
        best_loss = saved["best_val_loss"]
        if "fork" in saved["metadata"]:
            metadata["fork"] = saved["metadata"]["fork"]
        if "initialization" in saved["metadata"]:
            metadata["initialization"] = saved["metadata"]["initialization"]
    max_epochs = 1 if smoke else run["epochs"]
    print(f"Đã lưu {start_epoch} epoch; chạy tới tổng {max_epochs}; "
          f"batch_size={run['batch_size']}; num_workers={run['num_workers']}; "
          f"gpu_count={run.get('gpu_count', 1)}; "
          f"cuda_prefetch={run.get('cuda_prefetch', False)}; global_step={global_step}", flush=True)
    history_path = out / "history.json"
    history = read_json(history_path) if resume and history_path.exists() else []
    if resume:
        history = [record for record in history if record["epoch"] < start_epoch]
    for epoch in range(start_epoch, max_epochs):
        train_limit = 1 if smoke else 0
        val_limit = 1 if smoke else 0
        train_result = run_epoch(policy, train_loader, device, optimizer=optimizer,
                                 max_batches=train_limit, grad_clip_norm=run["grad_clip_norm"],
                                 log_every_batches=run["log_every_batches"],
                                 cuda_prefetch=run.get("cuda_prefetch", False))
        global_step += train_result["batches"]
        val_result = validate_with_fixed_seed(policy, val_loader, device, run["seed"],
                                              max_batches=val_limit,
                                              log_every_batches=run["log_every_batches"],
                                              cuda_prefetch=run.get("cuda_prefetch", False))
        record = {"epoch": epoch, "global_step": global_step, "train": train_result,
                  "val": val_result, "smoke": smoke, "batch_size": run["batch_size"],
                  "num_workers": run["num_workers"],
                  "gpu_count": run.get("gpu_count", 1),
                  "cuda_prefetch": run.get("cuda_prefetch", False)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_result["loss"] < best_loss:
            best_loss = val_result["loss"]
            atomic_torch_save({"kind": "best_model", "model": policy.state_dict(),
                               "contract": contract, "epoch": epoch,
                               "val": val_result, "smoke": smoke,
                               "metadata": metadata}, out / "best.pt")
        if (epoch + 1) % run["save_every_epochs"] == 0 or epoch + 1 == max_epochs:
            atomic_torch_save({"kind": "train_state", "model": policy.state_dict(),
                               "optimizer": optimizer.state_dict(), "contract": contract,
                               "run_config_hash": resume_signature(run), "epoch": epoch,
                               "global_step": global_step, "best_val_loss": best_loss,
                               "rng": capture_rng(),
                               "loader_generator_state": generator.get_state(),
                               "smoke": smoke, "metadata": metadata}, out / "latest.pt")
        write_json(history_path, history)
    return history
