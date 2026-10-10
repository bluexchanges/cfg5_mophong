import json
import unittest

from scripts.estimate_mp4_training_time import forecast, parse_log, slurm_seconds


def epoch(index, train=1000, val=100):
    return {"epoch": index, "smoke": False,
            "train": {"samples": 10000, "timing": {"epoch_seconds": train}},
            "val": {"samples": 1000, "timing": {"epoch_seconds": val}}}


class TrainingETATests(unittest.TestCase):
    def log(self, extra):
        return parse_log(["MP4_LOADER split=train workers=16 samples=10000",
                          "MP4_LOADER split=val workers=2 samples=1000",
                          "Đã lưu 1 epoch; chạy tới tổng 3;", *extra])

    def test_partial_train_plus_validation_and_future_epochs(self):
        log = self.log(["train batch=100 samples=2000 valid_steps=40000 samples_per_s=20.00"])
        result = forecast({"epochs": 3}, [epoch(0)], log)
        # Current train remainder8000/20 + future train1000 + two val phases100*2.
        self.assertEqual(result["total_seconds"], 1600)

    def test_partial_validation_train_already_done_this_epoch(self):
        log = self.log(["train batch=500 samples=10000 samples_per_s=20.00",
                        "val batch=100 samples=400 samples_per_s=20.00"])
        result = forecast({"epochs": 3}, [epoch(0)], log)
        self.assertEqual(result["total_seconds"], 1000 + 600/20 + 100)

    def test_no_validation_measurement_does_not_invent_complete_eta(self):
        log = self.log(["train batch=100 samples=2000 samples_per_s=20.00"])
        log["completed"] = 0
        result = forecast({"epochs": 3}, [], log)
        self.assertIsNone(result["total_seconds"])
        self.assertIsNotNone(result["remaining_seconds"]["train"])

    def test_epoch_boundaries_reset_counts_and_saved_history_can_be_newer(self):
        log = self.log(["train batch=500 samples=10000 samples_per_s=20.00",
                        json.dumps(epoch(1)), "train batch=100 samples=2000 samples_per_s=20.00"])
        self.assertEqual(log["completed"], 2)
        result = forecast({"epochs": 3}, [epoch(0), epoch(1)], log)
        self.assertEqual(result["total_seconds"], 8000/20 + 100)
        newer = forecast({"epochs": 3}, [epoch(0), epoch(1), epoch(2)], log)
        self.assertEqual(newer["total_seconds"], 0)

    def test_remaining_walltime_formats(self):
        self.assertEqual(slurm_seconds("2-01:02:03"), 2*86400+3723)
        self.assertEqual(slurm_seconds("05:30"), 330)
        self.assertIsNone(slurm_seconds("UNLIMITED"))


if __name__ == "__main__":
    unittest.main()
