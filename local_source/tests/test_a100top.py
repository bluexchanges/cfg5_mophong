"""Dashboard parsing and GPU/process selection without a live Slurm server."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.common import ROOT

spec = importlib.util.spec_from_file_location("a100top_cfg5", ROOT / "scripts/slurm/a100top_cfg5.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"epochs": 40, "batch_size": 4, "num_workers": 0, "_cpus": 4}
        self.history = [{"epoch": 15, "train": {"samples": 58313, "loss": .11},
                         "val": {"samples": 6214, "loss": .20}}]

    def test_batch_four_progress_is_samples_not_batch_count(self):
        text = "Đã lưu 16 epoch; chạy tới tổng 40\ntrain batch=4300 samples=17200 valid_steps=128579"
        p = monitor.progress(text, self.history, self.cfg)
        self.assertEqual((p["epoch"], p["saved"], p["batches"]), (17, 16, 14579))
        lines = monitor.render([], "13923", p, self.cfg, self.history, {"gpu": None}, width=120)
        self.assertIn("29.5%", "\n".join(lines))
        self.assertTrue(all(len(line) <= 119 for line in lines))

    def test_validation_and_checkpoint_are_separate_phases(self):
        text = "train batch=14500 samples=58000\nval batch=100 samples=400"
        p = monitor.progress(text, self.history, self.cfg)
        self.assertEqual((p["phase"], p["epoch"], p["total"], p["batches"]), ("val", 17, 6214, 1554))
        record = dict(self.history[0], epoch=16)
        record["val"] = dict(record["val"], batches=1554)
        p = monitor.progress(text + "\n" + json.dumps(record), self.history, self.cfg)
        self.assertEqual((p["phase"], p["epoch"], p["saved"]), ("lưu checkpoint", 17, 16))
        p = monitor.progress(text + "\n" + json.dumps(record) +
                             "\ntrain batch=100 samples=400", self.history, self.cfg)
        self.assertEqual((p["phase"], p["epoch"]), ("train", 18))

    def test_timing_fields_and_phase_eta(self):
        text = "train batch=100 samples=400 data_wait_ms=2.5 step_ms=200.0 samples_per_s=19.75"
        p = monitor.progress(text, self.history, self.cfg)
        self.assertEqual((p["wait"], p["step"], p["rate"]), (2.5, 200.0, 19.75))
        self.assertIn("ETA riêng phase train", "\n".join(monitor.render(
            [], "13923", p, self.cfg, self.history, {"gpu": None})))

    def test_locate_runtime_config_and_output_from_batch_script(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo / "configs").mkdir()
            (repo / "configs/run.json").write_text(json.dumps(dict(self.cfg, output_dir="outputs/child")))
            script = repo / "job.sbatch"
            script.write_text('CONFIG="configs/run.json"\npython scripts/train.py --config "$CONFIG"\n')
            detail = f"Command={script} StdOut={repo}/job-%j.out NumCPUs=4"
            log, history, cfg = monitor.locate(repo, "cfg5-act-bs4-w2", detail, "456")
            self.assertEqual(log, repo / "job-456.out")
            self.assertEqual(history, repo / "outputs/child/history.json")
            self.assertEqual(cfg["batch_size"], 4)
            self.assertEqual(cfg["_cpus"], 4)

    def test_gpu_uuid_belongs_to_train_pid_and_cpu_is_delta(self):
        probe = monitor.Probe("13923")
        probe.pid, probe.next_find = 123, float("inf")
        commands = []
        def run(args):
            commands.append(args)
            if "--query-compute-apps=pid,gpu_uuid" in args:
                return "999, GPU-other\n123, GPU-own\n"
            return "A100, 75, 10437, 81920, 57, 133\n"
        counters = iter([100, 50, 300, 150])
        def stat(pid):
            return {"pid": pid, "state": "R", "ticks": next(counters), "start": pid, "rss": 1024**3}
        with patch.object(monitor.Path, "exists", return_value=True), \
             patch.object(monitor.Path, "read_text", return_value="124"), \
             patch.object(monitor, "process_stat", side_effect=stat), \
             patch.object(monitor.os, "sysconf", return_value=100, create=True), \
             patch.object(monitor.time, "monotonic", side_effect=[0, 2]), \
             patch.object(monitor, "command", side_effect=run):
            first, second = probe.sample(), probe.sample()
        self.assertIsNone(first["processes"][0]["cpu"])
        self.assertEqual([row["cpu"] for row in second["processes"]], [100, 50])
        gpu_queries = [args for args in commands if "-i" in args]
        self.assertTrue(all(args[args.index("-i") + 1] == "GPU-own" for args in gpu_queries))

    def test_queue_is_user_scoped_and_preserves_pending_reason(self):
        text = "13923|cfg5-act-bs4-to40|R|36:12|1-23:23:48|dgx01\n13924|cfg5-act-bs4-to40|PD|0:00|2-00:00:00|(AssocMaxJobsLimit)\n"
        with patch.object(monitor, "command", return_value=text) as run:
            rows = monitor.queue("user08")
        self.assertEqual(rows[1]["node"], "(AssocMaxJobsLimit)")
        self.assertIn("user08", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
