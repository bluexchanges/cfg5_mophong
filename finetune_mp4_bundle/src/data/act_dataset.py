"""Map-style NumPy dataset, dùng trực tiếp với torch DataLoader ở lượt training."""
from pathlib import Path

import numpy as np

from .common import digest, read_json, require_contract
from .splits import validate_splits


def fit_normalization(processed_root, split, cfg, allow_partial_smoke=False):
    validate_splits(split)
    require_contract(split, cfg)
    moments = {name: {"count": 0, "sum": np.zeros(8), "sum_squares": np.zeros(8)} for name in ("state", "action")}
    used, missing = [], []
    smoke = False
    for episode in split["splits"]["train"]:
        folder = Path(processed_root) / episode
        if not (folder / "episode.json").exists():
            missing.append(episode)
            continue
        meta = read_json(folder / "episode.json")
        require_contract(meta, cfg)
        if meta["split_hash"] != split["split_hash"]:
            raise ValueError("Processed data thuộc split khác")
        smoke |= meta["smoke_only"]
        segment_ids = np.load(folder / "segment_ids.npy", mmap_mode="r", allow_pickle=False)
        target_indices = (np.arange(len(segment_ids)) if cfg.get("action_target_offset", 1) == 0
                          else np.flatnonzero(segment_ids[1:] == segment_ids[:-1]) + 1)
        for name in moments:
            data = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            indices = np.arange(len(data)) if name == "state" else target_indices
            for offset in range(0, len(indices), 4096):
                part = np.asarray(data[indices[offset:offset+4096]], dtype=np.float64)
                if not np.all(np.isfinite(part)):
                    raise ValueError("NaN/Inf trong processed data")
                moments[name]["count"] += len(part)
                moments[name]["sum"] += part.sum(axis=0)
                moments[name]["sum_squares"] += (part * part).sum(axis=0)
        used.append(episode)
    if (missing or smoke) and not allow_partial_smoke:
        raise ValueError("Train chưa xử lý đủ hoặc chứa smoke data; không tạo statistics chính thức")
    if not used:
        raise ValueError("Chưa có train episode để tính statistics")
    stats = {"config_hash": digest(cfg), "split_hash": split["split_hash"], "source_split": "train",
             "episodes": used, "missing_train_episodes": missing, "smoke_only": bool(missing or smoke)}
    for name, moment in moments.items():
        if moment["count"] == 0:
            raise ValueError("Không có target hợp lệ")
        mean = moment["sum"] / moment["count"]
        variance = np.maximum(moment["sum_squares"] / moment["count"] - mean**2, 0)
        std = np.maximum(np.sqrt(variance), cfg["normalization_std_floor"])
        stats[name] = {"mean": mean.tolist(), "std": std.tolist(), "count": moment["count"]}
    return stats


class ACTDataset:
    def __init__(self, processed_root, split, stats, cfg, split_name="train", allow_partial_smoke=False):
        validate_splits(split)
        require_contract(split, cfg)
        require_contract(stats, cfg)
        if stats["source_split"] != "train" or stats["split_hash"] != split["split_hash"]:
            raise ValueError("Normalization phải được fit trên train của cùng split")
        if not set(stats["episodes"]).issubset(split["splits"]["train"]):
            raise ValueError("Normalization chứa episode ngoài train")
        if stats["smoke_only"] and not allow_partial_smoke:
            raise ValueError("Smoke statistics không được dùng cho thí nghiệm chính thức")
        self.root, self.cfg, self.stats = Path(processed_root), cfg, stats
        self.target_offset = cfg.get("action_target_offset", 1)
        self.index, self.episodes, self._cache = [], [], None
        for episode in split["splits"][split_name]:
            folder = self.root / episode
            if not (folder / "episode.json").exists():
                if allow_partial_smoke:
                    continue
                raise FileNotFoundError(folder)
            meta = read_json(folder / "episode.json")
            require_contract(meta, cfg)
            if meta["split_hash"] != split["split_hash"] or meta["camera_serials"] != cfg["camera_serials"]:
                raise ValueError("Processed contract không khớp")
            if meta["smoke_only"] and not allow_partial_smoke:
                raise ValueError("Không dùng smoke data cho full training")
            segments = np.load(folder / "segment_ids.npy", allow_pickle=False, mmap_mode="r")
            ends = np.r_[np.flatnonzero(np.diff(segments) != 0) + 1, len(segments)]
            start = 0
            for end in ends:
                self.index.extend((episode, i, int(end)) for i in range(start, int(end)-self.target_offset))
                start = int(end)
            self.episodes.append(episode)
        if not self.index:
            raise ValueError("Dataset không có observation + future action hợp lệ")

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        result = self.__dict__.copy()
        result["_cache"] = None
        return result

    def _load_arrays(self, episode):
        return {key: np.load(self.root / episode / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                for key in ("images", "state", "action", "timestamps_ms")}

    def __getitem__(self, index):
        episode, start, end = self.index[index]
        if self._cache is None or self._cache[0] != episode:
            self._cache = None
            self._cache = (episode, self._load_arrays(episode))
        arrays = self._cache[1]
        horizon = self.cfg["chunk_size"]
        first_target = start + self.target_offset
        length = min(horizon, end-first_target)
        targets = np.zeros((horizon, 8), dtype=np.float32)
        targets[:length] = (arrays["action"][first_target:first_target+length] - np.asarray(self.stats["action"]["mean"])) / np.asarray(self.stats["action"]["std"])
        mask = np.arange(horizon) >= length
        state = (arrays["state"][start] - np.asarray(self.stats["state"]["mean"])) / np.asarray(self.stats["state"]["std"])
        rgb = np.asarray(arrays["images"][start], dtype=np.float32) / 255.0
        rgb = (rgb - np.asarray(self.cfg["image_normalization_mean"], dtype=np.float32)) / np.asarray(self.cfg["image_normalization_std"], dtype=np.float32)
        images = np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))
        if not all(np.all(np.isfinite(x)) for x in (targets, state, images)):
            raise ValueError("NaN/Inf trong sample ACT")
        return {"images": images, "state": state.astype(np.float32), "actions": targets,
                "is_pad": mask, "episode": episode, "timestamp_ms": int(arrays["timestamps_ms"][start])}
