"""The overnight queue parser and the sweep-log parser behind the feedback-loop report (CPU only).

    python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path

from tests import _paths  # noqa: F401
import report as R
import run_queue as Q


class ParseQueue(unittest.TestCase):
    def test_stages_comments_and_quoting(self):
        text = """
        # a comment line
        one | --config configs/swa_fb.yaml --set experiment=x model.segment=8   # trailing comment
        two | --sweep model.nonlin=tanh,none --set 'eval.ns=[32, 64]'
        """
        stages = Q.parse_queue(text)
        self.assertEqual([s for s, _ in stages], ["one", "two"])
        self.assertEqual(stages[0][1], ["--config", "configs/swa_fb.yaml", "--set", "experiment=x", "model.segment=8"])
        self.assertEqual(stages[1][1][-1], "eval.ns=[32, 64]")  # shell quoting is honoured

    def test_rejects_a_line_without_session_and_duplicate_sessions(self):
        with self.assertRaises(ValueError):
            Q.parse_queue("--config a.yaml\n")
        with self.assertRaises(ValueError):
            Q.parse_queue("a | --x\na | --y\n")

    def test_the_committed_queue_parses_and_expands(self):
        """Every stage of configs/queue.txt must be a valid launch.py sweep (names are unique, nothing is skipped silently)."""
        import launch as L
        import argparse
        stages = Q.parse_queue((Q.ROOT / "configs" / "queue.txt").read_text())
        self.assertGreaterEqual(len(stages), 1)
        for session, args in stages:
            ap = argparse.ArgumentParser()
            ap.add_argument("--config"); ap.add_argument("--sweep", nargs="*", default=[]); ap.add_argument("--set", nargs="*", default=[])
            ap.add_argument("--gpus", nargs="+"); ap.add_argument("--procs-per-gpu"); ap.add_argument("--no-wandb", action="store_true")
            a = ap.parse_args(args)
            jobs = L.build_jobs(a.config, a.set, a.sweep)
            self.assertEqual(len({j["name"] for j in jobs}), len(jobs), session)


class SweepProgress(unittest.TestCase):
    LINE = ("[swa_fb_nltanh_seg16_s2] step 4000 loss 0.1660 | n32: pk 0.96 v 0.92 e 0.99 n256: pk 0.10 v 0.86 e 0.98 n1024: pk 0.05 v 0.85 e 0.97")

    def test_parses_validation_lines_and_ignores_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fbsweep").mkdir()
            (Path(d) / "fbsweep" / "swa_fb_nltanh_seg16_s2.log").write_text("noise\n" + self.LINE + "\nwandb: something\n")
            (Path(d) / "fbsweep" / "swa_fb_nlnone_seg64_s0.log").write_text(self.LINE.replace("nltanh_seg16_s2", "nlnone_seg64_s0").replace("step 4000", "step 2000"))
            (Path(d) / "fbsweep" / "swa_fb_seg8_nltanh_s0.log").write_text(self.LINE)  # not part of the main grid: other name order
            prog = R.sweep_progress(Path(d))
        self.assertEqual(sorted(prog), [(16, "tanh"), (64, "none")])
        m = prog[(16, "tanh")][2][4000]
        self.assertEqual((m["pk256"], m["v256"], m["pk1024"], m["loss"]), (0.10, 0.86, 0.05, 0.166))
        self.assertIn(2000, prog[(64, "none")][0])

    def test_a_restarted_run_replaces_its_stopped_log(self):
        """The same run name in two session folders: the newest log is the restart, the older one is the stopped attempt."""
        import os
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d) / "fbsweep" / "swa_fb_nltanh_seg16_s1.log", Path(d) / "fbre_s1a" / "swa_fb_nltanh_seg16_s1.log"
            old.parent.mkdir(); new.parent.mkdir()
            old.write_text(self.LINE.replace("s2", "s1").replace("step 4000", "step 6000"))
            new.write_text(self.LINE.replace("s2", "s1").replace("step 4000", "step 2000"))
            os.utime(old, (1000, 1000)); os.utime(new, (2000, 2000))
            prog = R.sweep_progress(Path(d))
        self.assertEqual(list(prog[(16, "tanh")][1]), [2000])


if __name__ == "__main__":
    unittest.main()
