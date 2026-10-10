"""Metric inference đi kèm bundle E5; cùng công thức của evaluator pretrain cũ."""
import numpy as np


class InferenceMetrics:
    """Weight by real future targets; report physical units after denormalization."""

    def __init__(self, stats):
        self.mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
        self.std = np.asarray(stats["action"]["std"], dtype=np.float64)
        self.steps = self.samples = self.batches = self.quaternions = 0
        self.episodes = set()
        self.normalized_sum = np.zeros(8)
        self.physical_sum = np.zeros(8)
        self.position_sum = self.angle_sum = 0.0

    def add(self, prediction, target, is_pad, episodes):
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
