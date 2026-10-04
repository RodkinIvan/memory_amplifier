"""A training run that crashes in the middle continues from its checkpoint and ends where an uninterrupted run ends (needs a GPU).

    python -m unittest tests.test_resume -v
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from tests import _paths
import config as C
import make_passkey_swar as G
import train as T

REAL_ROOT = _paths.ROOT


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class ResumeTraining(unittest.TestCase):
    STEPS = 40
    OVERRIDES = ["model.d_model=64", "model.num_heads=2", "model.window=16", "model.segment=8", "model.noise=0.1", "model.gain_init=0.1",
                 f"train.steps={STEPS}", "train.batch_size=4", "train.train_ns=[32]", "train.curriculum_stage_steps=0",
                 "train.log_every=10", "train.eval_every=15", "train.eval_ns=[32]", "train.eval_examples=16", "train.warmup_steps=5",
                 "eval.ns=[32]", "eval.splits=[validation]", "eval.tokens_per_batch=65536", "--no-wandb"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        quiet(G.main, ["--out-dir", str(cls.tmp / "data"), "--ns", "32", "--train-ns", "32", "--train-size", "200", "--eval-size", "32",
                       "--workers", "2", "--skip-sanity"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def train(self, root: Path, extra=()):
        """One train.py invocation with `root` as the repository root (runs/ and results/ go there)."""
        (root / "configs").mkdir(parents=True, exist_ok=True)
        if not (root / "configs" / "ref").exists():
            shutil.copytree(REAL_ROOT / "configs" / "ref", root / "configs" / "ref")
        flag = [o for o in self.OVERRIDES if o.startswith("--")]
        sets = [o for o in self.OVERRIDES if not o.startswith("--")]
        cfg = yaml.safe_load((REAL_ROOT / "configs" / "swa_fb.yaml").read_text())
        cfg["data_dir"] = str(self.tmp / "data")  # in the file, not a --set override: overrides become part of the run name
        (root / "configs" / "test.yaml").write_text(yaml.safe_dump(cfg))
        with mock.patch.object(C, "ROOT", root):
            return quiet(T.main, ["--config", str(root / "configs" / "test.yaml"), *flag, "--set", *sets, *extra])

    def result(self, root: Path) -> dict:
        (f,) = (root / "results" / "raw").glob("*.json")
        return json.loads(f.read_text())

    def test_a_crashed_run_continues_from_its_checkpoint_and_matches_an_uninterrupted_run(self):
        ref_root, crash_root = self.tmp / "ref", self.tmp / "crash"
        self.train(ref_root)

        real_pad, calls = T.D.pad_batch, []

        def crash_at_step_27(*a, **k):
            calls.append(1)
            if len(calls) == 27:
                raise RuntimeError("simulated crash")
            return real_pad(*a, **k)

        with mock.patch.object(T.D, "pad_batch", crash_at_step_27), self.assertRaises(RuntimeError):
            self.train(crash_root)
        (name,) = [p.name for p in (crash_root / "runs").iterdir()]
        saved = torch.load(crash_root / "runs" / name / "resume.pt", weights_only=True)
        self.assertEqual((saved["step"], saved["n_acc"]), (15, 5))  # checkpoint of the validation at step 15, in the middle of a loss-logging interval
        self.assertFalse((crash_root / "results" / "raw").exists() and list((crash_root / "results" / "raw").glob("*.json")))

        out = self.train(crash_root)
        self.assertIn("resuming from step 15", out)
        self.assertFalse((crash_root / "runs" / name / "resume.pt").exists(), "the checkpoint is removed once the result is written")

        a, b = self.result(ref_root), self.result(crash_root)
        self.assertEqual([h["step"] for h in b["val_history"]], [15, 30, 40])  # the validation points before the crash survive in the checkpoint
        self.assertEqual([h["step"] for h in a["val_history"]], [15, 30, 40])
        self.assertIn("val/n32/passkey_acc", b["val_history"][0])
        self.assertEqual(b["resumed_at_steps"], [15])
        self.assertEqual(a["resumed_at_steps"], [])
        self.assertAlmostEqual(a["final_train_loss"], b["final_train_loss"], places=3)
        for ra, rb in zip(a["rows"], b["rows"]):
            for k in ("passkey_acc", "ar_valid_acc", "ar_expired_acc"):
                self.assertAlmostEqual(ra[k], rb[k], places=6)
        wa = torch.load(ref_root / "runs" / name / "model.pt", weights_only=True)
        wb = torch.load(crash_root / "runs" / name / "model.pt", weights_only=True)
        worst = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa)
        self.assertLess(worst, 1e-4, "weights after a resume differ from an uninterrupted run")

    def test_a_graphed_run_resumes_like_an_uninterrupted_graphed_run(self):
        extra = ["train.cuda_graph=true", "model.noise=0"]  # the graph's noise stream is the graph-safe generator's, so compare without noise
        ref_root, crash_root = self.tmp / "gref", self.tmp / "gcrash"
        self.train(ref_root, extra=extra)
        real_pad, calls = T.D.pad_batch, []

        def crash_at_step_27(*a, **k):
            calls.append(1)
            if len(calls) == 27:
                raise RuntimeError("simulated crash")
            return real_pad(*a, **k)

        with mock.patch.object(T.D, "pad_batch", crash_at_step_27), self.assertRaises(RuntimeError):
            self.train(crash_root, extra=extra)
        out = self.train(crash_root, extra=extra)
        self.assertIn("resuming from step 15", out)
        (name,) = [p.name for p in (ref_root / "runs").iterdir()]
        self.assertIn("cgTrue", name)  # the switch is part of the run name (and of the diff against the reference)
        a, b = self.result(ref_root), self.result(crash_root)
        self.assertEqual(a["diff"]["train.cuda_graph"], [False, True])
        self.assertAlmostEqual(a["final_train_loss"], b["final_train_loss"], places=3)
        wa = torch.load(ref_root / "runs" / name / "model.pt", weights_only=True)
        wb = torch.load(crash_root / "runs" / name / "model.pt", weights_only=True)
        self.assertLess(max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa), 1e-4)

    def test_a_checkpoint_of_another_config_stops_the_run(self):
        root = self.tmp / "mismatch"
        real_pad, calls = T.D.pad_batch, []

        def crash_at_step_20(*a, **k):
            calls.append(1)
            if len(calls) == 20:
                raise RuntimeError("simulated crash")
            return real_pad(*a, **k)

        with mock.patch.object(T.D, "pad_batch", crash_at_step_20), self.assertRaises(RuntimeError):
            self.train(root)
        (name,) = [p.name for p in (root / "runs").iterdir()]
        # same run name (evaluation settings are not part of the name), different config: the old state must not be mixed into the new run
        with self.assertRaises(SystemExit) as cm:
            self.train(root, extra=["train.eval_examples=8"])
        self.assertIn("different config", str(cm.exception))
        self.assertTrue((root / "runs" / name / "resume.pt").exists())
        out = self.train(root, extra=["train.eval_examples=8", "--force"])  # --force discards the old state and starts over
        self.assertNotIn("resuming", out)
        self.assertEqual(self.result(root)["resumed_at_steps"], [])


class EvalSubsets(unittest.TestCase):
    def test_per_n_subsets_override_the_default_count(self):
        tc = {"eval_examples": 200, "eval_examples_by_n": [[4096, 100], [16384, 32]]}
        self.assertEqual([T.eval_examples_for(tc, n) for n in (32, 1024, 4096, 16384)], [200, 200, 100, 32])
        self.assertEqual(T.eval_examples_for({"eval_examples": 200}, 4096), 200)  # key absent in older configs


class StoredDiff(unittest.TestCase):
    def test_a_run_description_names_the_changes_and_the_schedule(self):
        cfg, _ = C.load_config(REAL_ROOT / "configs" / "swa_fb.yaml", ["model.gain_init=0", "train.steps=50000", "model.segment=32"])
        text = C.run_description(cfg)
        for part in ("swa_fb_v1", "gain_init 0.1 → 0", "steps 20000 → 50000", "segment 0 → 32", "LAST step (50000)", "curriculum: one stage per 2000 steps"):
            self.assertIn(part, text)


if __name__ == "__main__":
    unittest.main()
