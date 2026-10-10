from collections import defaultdict
import math
import random

from .common import digest, require_contract


def target_counts(total, ratios):
    """Largest-remainder episode counts; keep each of 3 splits nonempty."""
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))[:total - sum(counts)]:
        counts[index] += 1
    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            if counts[donor] <= 1:
                raise ValueError("Không thể tạo 3 split không rỗng")
            counts[donor] -= 1
            counts[index] += 1
    return counts


def random_exact_group_assignment(items, desired, seed):
    """Randomize groups, then sample a feasible *exact* episode partition.

    DP tracks remaining val/test counts; train is the remaining episodes.
    No episode or (task,user,calib) group is ever split between partitions.
    """
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    val_target, test_target = desired[1:]
    test_mask = (1 << (test_target + 1)) - 1
    suffix = [[0] * (val_target + 1) for _ in range(len(items) + 1)]
    suffix[-1][0] = 1  # bit 0: zero test episodes required
    for pos in range(len(items) - 1, -1, -1):
        size = len(items[pos][1])
        after = suffix[pos + 1]
        current = suffix[pos]
        for val in range(val_target + 1):
            bits = after[val] | ((after[val] << size) & test_mask)
            if val >= size:
                bits |= after[val - size]
            current[val] = bits
    if not (suffix[0][val_target] >> test_target) & 1:
        raise ValueError(f"Không thể đạt đúng số episode {desired} khi giữ nguyên các group")

    remaining = desired[:]
    assigned = {name: [] for name in ("train", "val", "test")}
    for pos, (key, episodes) in enumerate(items):
        size = len(episodes)
        after = suffix[pos + 1]
        feasible = []
        if remaining[0] >= size and (after[remaining[1]] >> remaining[2]) & 1:
            feasible.append(0)
        if remaining[1] >= size and (after[remaining[1] - size] >> remaining[2]) & 1:
            feasible.append(1)
        if remaining[2] >= size and (after[remaining[1]] >> (remaining[2] - size)) & 1:
            feasible.append(2)
        if not feasible:
            raise AssertionError("DP reconstruction không có bước khả thi")
        chosen = rng.choices(feasible, weights=[remaining[i] for i in feasible], k=1)[0]
        remaining[chosen] -= size
        assigned[("train", "val", "test")[chosen]].append((key, episodes))
    if remaining != [0, 0, 0]:
        raise AssertionError("DP reconstruction sai tổng số episode")
    return assigned


def create_splits(manifest, cfg):
    require_contract(manifest, cfg)
    groups = defaultdict(list)
    for episode in manifest["episodes"]:
        key = "|".join(str(episode[field]) for field in cfg["split_group_fields"])
        groups[key].append(episode["episode"])
    if len(groups) < 3:
        raise ValueError("Ít hơn 3 nhóm độc lập; không tạo train/val/test giả bằng cách tách frame")
    ratios = cfg["split_ratios"]
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios)-1) > 1e-8:
        raise ValueError("Cần 3 split ratios dương, tổng bằng 1")
    items = sorted(groups.items())
    names = ["train", "val", "test"]
    desired = target_counts(len(manifest["episodes"]), ratios)
    assigned = random_exact_group_assignment(items, desired, cfg["seed"])
    assignment = {name: [episode for _, episodes in assigned[name] for episode in episodes] for name in names}
    assigned_groups = {name: [key for key, _ in assigned[name]] for name in names}
    result = {"schema_version": 1, "config_hash": digest(cfg), "manifest_hash": manifest["manifest_hash"],
              "seed": cfg["seed"], "group_fields": cfg["split_group_fields"], "groups": assigned_groups,
              "splits": {name: sorted(ids) for name, ids in assignment.items()},
              "counts": {name: len(ids) for name, ids in assignment.items()},
              "target_episode_counts": dict(zip(names, desired)),
              "assignment_method": "seeded_random_exact_episode_counts_group_constrained_dp",
              "note": "Groups are randomly assigned without task pinning. Exact counts follow largest remainder; calibration/camera backgrounds may be shared across different task-user groups."}
    result["split_hash"] = digest(result)
    validate_splits(result)
    return result


def validate_splits(split):
    ids, groups = set(), set()
    for name in ("train", "val", "test"):
        current = split["splits"][name]
        current_groups = split["groups"][name]
        if not current or len(set(current)) != len(current) or ids.intersection(current) or groups.intersection(current_groups):
            raise ValueError("Split trống hoặc có episode/group leakage")
        if split["counts"][name] != len(current):
            raise ValueError("Split count khác số episode thực")
        ids.update(current)
        groups.update(current_groups)
    if "target_episode_counts" in split and split["counts"] != split["target_episode_counts"]:
        raise ValueError("Split không đạt đúng tỷ lệ episode đã khóa")
    content = {k: v for k, v in split.items() if k != "split_hash"}
    if digest(content) != split["split_hash"]:
        raise ValueError("Split hash sai")
