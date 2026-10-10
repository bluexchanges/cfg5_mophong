"""MP4 CPU decode, bounded uint8 episode cache, one episode owner per worker."""
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, default_collate, get_worker_info

from .act_dataset import ACTDataset
from .common import digest, outside_raw, read_json, require_contract, write_json
from .sim_demos import fingerprint, file_hash, inspect_episode, load_trajectory, read_camera_times

BACKEND = "mp4_episode_cache_v1"
NUMERIC_KEYS = ("state", "action", "timestamps_ms", "segment_ids")


def validate_source(record, cfg):
    raw = Path(record["path"]).resolve()
    if raw.name != record["episode"] or not any(
            raw.is_relative_to(Path(root).resolve()) for root in cfg["raw_sources"]):
        raise ValueError(f"Raw path ngoài nguồn đã khóa: {raw}")
    if fingerprint(raw, cfg["camera_serials"]) != record["source_fingerprint"]:
        raise ValueError(f"Raw thay đổi sau split: {raw}")
    return raw


def verify_light(record, cfg, root, split_hash):
    raw = validate_source(record, cfg)
    folder = Path(root) / record["episode"]
    if folder.with_name(folder.name + ".partial").exists():
        raise FileExistsError(f"Metadata dở; đọc ERROR.json trước: {folder}.partial")
    meta = read_json(folder / "episode.json")
    require_contract(meta, cfg)
    if (meta.get("image_storage") != BACKEND or meta.get("split_hash") != split_hash
            or meta.get("camera_serials") != cfg["camera_serials"]
            or meta.get("source_fingerprint") != record["source_fingerprint"]
            or meta.get("raw_path") != str(raw) or meta.get("smoke_only") is not False
            or meta.get("steps") != record["quality"]["steps"]):
        raise ValueError(f"Light metadata contract sai: {folder}")
    if (folder / "images.npy").exists():
        raise ValueError("MP4 root không được chứa images.npy")
    _, times, state, action, segments = load_trajectory(raw, cfg)
    for key, expected in zip(NUMERIC_KEYS, (state, action, times, segments)):
        path = folder / f"{key}.npy"
        if file_hash(path) != meta["numeric_sha256"][key]:
            raise ValueError(f"Numeric file đổi: {path}")
        actual = np.load(path, allow_pickle=False)
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"Numeric array khác raw: {path}")
    return meta


def prepare_light(record, cfg, root, split_hash):
    """Write only aligned numeric arrays + raw reference; never write decoded images."""
    raw = validate_source(record, cfg)
    inspected = inspect_episode(raw, cfg)
    if inspected != record:
        raise ValueError(f"Audit record khác manifest: {raw}")
    root = outside_raw(root, cfg)
    final = root / record["episode"]
    partial = final.with_name(final.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError(f"Output đã có: {final}; dùng --resume hoặc đọc .partial/ERROR.json")
    partial.mkdir(parents=True)
    try:
        _, times, state, action, segments = load_trajectory(raw, cfg)
        numeric = dict(zip(NUMERIC_KEYS, (state, action, times, segments)))
        for key, array in numeric.items():
            np.save(partial / f"{key}.npy", array, allow_pickle=False)
        meta = {"schema_version": 1, "episode": record["episode"], "steps": len(times),
                "config_hash": digest(cfg), "split_hash": split_hash,
                "camera_serials": cfg["camera_serials"], "source_fingerprint": record["source_fingerprint"],
                "representation": cfg["representation"], "smoke_only": False,
                "action_target_offset": 0, "image_storage": BACKEND, "raw_path": str(raw),
                "numeric_sha256": {key: file_hash(partial / f"{key}.npy") for key in numeric},
                "quality": record["quality"],
                "action_semantics": "recorded command at observation tick; no measured future proxy"}
        write_json(partial / "episode.json", meta)
        partial.rename(final)
        return meta
    except Exception as exc:
        write_json(partial / "ERROR.json", {"error": str(exc), "incomplete": True})
        raise


class MP4ACTDataset(ACTDataset):
    def __init__(self, *args, cache_gib=3.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(cache_gib) or cache_gib <= 0:
            raise ValueError("cache_gib phải dương và hữu hạn")
        if self.cfg.get("source_type") != "mujoco_scripted_controller" or self.target_offset != 0:
            raise ValueError("MP4 adapter chỉ hỗ trợ demo pre-step đã audit")
        self.cache_bytes = int(cache_gib * 2**30)
        self.max_episode_bytes = 0
        for episode in self.episodes:
            meta = read_json(self.root / episode / "episode.json")
            if meta.get("image_storage") != BACKEND or meta.get("smoke_only") is not False:
                raise ValueError(f"Cần light metadata full: {episode}")
            size = meta["steps"] * len(self.cfg["camera_serials"]) * self.cfg["image_height"] * self.cfg["image_width"] * 3
            if size > self.cache_bytes:
                raise MemoryError(f"{episode}: {size / 2**30:.2f} GiB vượt cache {cache_gib} GiB")
            self.max_episode_bytes = max(self.max_episode_bytes, size)

    def _load_arrays(self, episode):
        # ACTDataset discards the previous cache BEFORE allocating this episode.
        folder = self.root / episode
        meta = read_json(folder / "episode.json")
        record = {"episode": episode, "path": meta["raw_path"],
                  "source_fingerprint": meta["source_fingerprint"], "quality": meta["quality"]}
        verify_light(record, self.cfg, self.root, meta["split_hash"])
        count = meta["steps"]
        shape = (count, len(self.cfg["camera_serials"]), self.cfg["image_height"], self.cfg["image_width"], 3)
        if math.prod(shape) > self.cache_bytes:
            raise MemoryError(f"Cache quá giới hạn: {episode}")
        images = np.empty(shape, dtype=np.uint8)
        arrays = {key: np.load(folder / f"{key}.npy", allow_pickle=False, mmap_mode="r")
                  for key in ("state", "action", "timestamps_ms")}
        cv2.setNumThreads(1)
        started = time.perf_counter()
        for camera_index, camera in enumerate(self.cfg["camera_serials"]):
            camera_root = Path(meta["raw_path"]) / f"cam_{camera}"
            if not np.array_equal(read_camera_times(camera_root / "timestamps.npy"), arrays["timestamps_ms"]):
                raise ValueError(f"Camera timestamp lệch: {episode}/{camera}")
            cap = cv2.VideoCapture(str(camera_root / "color.mp4"))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Không mở video: {episode}/{camera}")
                for row in range(count):
                    ok, frame = cap.read()
                    if not ok or frame.shape != shape[2:]:
                        raise ValueError(f"Video lỗi: {episode}/{camera} frame={row}")
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=images[row, camera_index])
                if cap.read()[0]:
                    raise ValueError(f"Video dư frame: {episode}/{camera}")
            finally:
                cap.release()
        # Verify source again after decode to catch concurrent replacement.
        validate_source(record, self.cfg)
        worker = get_worker_info()
        print(f"MP4_CACHE worker={worker.id if worker else 0} episode={episode} "
              f"steps={count} cache_GiB={images.nbytes / 2**30:.3f} "
              f"decode_s={time.perf_counter() - started:.3f}", flush=True)
        arrays["images"] = images
        return arrays


class EpisodeBatchStream(IterableDataset):
    """Shard EPISODES, not frame indices. Every valid sample appears exactly once.

    Batch tails are kept per episode (1..batch_size); no dropped or duplicate samples.
    Shared seed is set by the main process before each iterator, including persistent workers.
    """
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size phải >=1")
        self.base, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.epoch_seed = torch.zeros((), dtype=torch.int64).share_memory_()
        groups = {}
        for index, (episode, _, _) in enumerate(dataset.index):
            groups.setdefault(episode, []).append(index)
        self.groups = list(groups.items())

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for _, indices in self.groups)

    def __iter__(self):
        info = get_worker_info()
        worker_id, workers = (info.id, info.num_workers) if info else (0, 1)
        seed = int(self.epoch_seed.item())
        group_ids = list(range(len(self.groups)))
        if self.shuffle:
            random.Random(seed).shuffle(group_ids)
        try:
            for group_id in group_ids[worker_id::workers]:
                _, original = self.groups[group_id]
                indices = list(original)
                if self.shuffle:
                    random.Random(seed + group_id + 1).shuffle(indices)
                for start in range(0, len(indices), self.batch_size):
                    yield default_collate([self.base[i] for i in indices[start:start+self.batch_size]])
                self.base._cache = None
        finally:
            self.base._cache = None


class MP4DataLoader(DataLoader):
    def __init__(self, *args, epoch_generator=None, **kwargs):
        self.epoch_generator = epoch_generator
        super().__init__(*args, **kwargs)

    def __iter__(self):
        # One generator draw per epoch, recorded by existing checkpoint/resume logic.
        seed = (int(torch.randint(0, 2**62, (), generator=self.epoch_generator).item())
                if self.dataset.shuffle else 0)
        self.dataset.epoch_seed.fill_(seed)
        return super().__iter__()


def make_mp4_loader(root, splits, stats, cfg, split_name, batch_size, num_workers,
                    *, generator=None, allow_partial_smoke=False, prefetch_factor=1, cache_gib=3.5):
    base = MP4ACTDataset(root, splits, stats, cfg, split_name, allow_partial_smoke, cache_gib=cache_gib)
    stream = EpisodeBatchStream(base, batch_size, split_name == "train")
    options = ({"prefetch_factor": prefetch_factor, "persistent_workers": True,
                "multiprocessing_context": "spawn"} if num_workers else {})
    print(f"MP4_LOADER split={split_name} workers={num_workers} samples={len(base)} "
          f"batches={len(stream)} batch_size_max={batch_size} "
          f"max_episode_cache_GiB={base.max_episode_bytes / 2**30:.3f}", flush=True)
    # Worker seed draws must not consume the checkpointed epoch generator:
    # persistent workers and a newly resumed loader draw base seeds differently.
    return MP4DataLoader(stream, batch_size=None, num_workers=num_workers,
                         pin_memory=torch.cuda.is_available(),
                         epoch_generator=generator, generator=torch.Generator().manual_seed(0), **options)
