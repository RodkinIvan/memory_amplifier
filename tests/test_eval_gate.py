"""Decision logic of src/eval_gate.py (CPU only): at most `limit` runs per GPU in their final evaluation at the same time.

    python -m unittest discover -s tests -v
"""
import unittest

from tests import _paths  # noqa: F401
import eval_gate as G

P = lambda pid, in_eval, stopped=False: {"pid": pid, "in_eval": in_eval, "stopped": stopped}  # noqa: E731


class Decide(unittest.TestCase):
    def test_runs_reaching_the_evaluation_together_are_serialised(self):
        procs = [P(1, True), P(2, True), P(3, True), P(4, False)]
        self.assertEqual(G.decide(procs, 1), ([2, 3], []))  # the first keeps running, runs still training are left alone

    def test_a_waiting_run_resumes_when_the_running_one_is_gone(self):
        self.assertEqual(G.decide([P(2, True, True), P(3, True, True)], 1), ([], [2]))

    def test_a_running_evaluation_is_not_interrupted_for_a_waiting_one(self):
        self.assertEqual(G.decide([P(2, True, False), P(3, True, True)], 1), ([], []))
        self.assertEqual(G.decide([P(2, True, True), P(3, True, False)], 1), ([], []))  # the later one runs; the earlier one waits

    def test_limit_two(self):
        self.assertEqual(G.decide([P(1, True), P(2, True), P(3, True)], 2), ([3], []))
        self.assertEqual(G.decide([P(1, True), P(2, True, True), P(3, True, True)], 2), ([], [2]))

    def test_nothing_to_do_while_everyone_trains(self):
        self.assertEqual(G.decide([P(1, False), P(2, False)], 1), ([], []))

    def test_run_name_from_the_sweep_overrides(self):
        cmd = ["python", "src/train.py", "--config", "configs/swa_fb.yaml", "--set", "model.nonlin=none", "model.segment=16", "seed=2"]
        self.assertEqual(G.run_name(cmd), "swa_fb_nlnone_seg16_s2")
        self.assertIsNone(G.run_name(["python", "src/train.py", "--config", "x.yaml"]))


if __name__ == "__main__":
    unittest.main()
