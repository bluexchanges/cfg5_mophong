"""Regression: bundled metric agrees with the previously used test evaluator."""
import unittest
import numpy as np
from src.evaluation.transfer_metrics import InferenceMetrics
from scripts.evaluate_pretrain_test import InferenceMetrics as PreviousMetrics


class MetricTests(unittest.TestCase):
    def test_old_and_bundled_metrics_with_padding_and_quaternion_sign(self):
        stats = {"action": {"mean": [0.] * 8, "std": [1.] * 8}}
        truth = np.zeros((2, 3, 8), np.float32)
        truth[:, :, 3] = 1
        pred = truth.copy()
        pred[0, 0, 0] = .01
        pred[1, 0, 7] = .002
        pred[1, 1, 3] = -1
        pad = np.array([[False, True, True], [False, False, True]])
        pred[pad] = np.nan
        a, b = InferenceMetrics(stats), PreviousMetrics(stats)
        for metric in (a, b):
            metric.add(pred, truth, pad, ["episode_a", "episode_b"])
        self.assertEqual(a.summary(), b.summary())
        self.assertAlmostEqual(a.summary()["position_mean_distance_mm"], 10 / 3, places=6)
        self.assertEqual(a.summary()["valid_action_elements"], 24)
        self.assertEqual(a.summary()["orientation_mean_angle_deg"], 0)
