"""Tests for config/naming, sweep expansion, data packing and the evaluation bookkeeping (CPU only).

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from tests import _paths
import config as C
import data as D
import evaluate as E
import launch as L
import make_passkey_swar as G
import sanity_checks as S
import train as T

EXP1 = _paths.ROOT / "configs" / "exp1.yaml"


class ConfigAndNaming(unittest.TestCase):
    def name(self, overrides):
        cfg, keys = C.load_config(EXP1, overrides)
        return C.run_name(cfg, keys)

    def test_names_contain_model_and_changed_hyperparameters(self):
        self.assertEqual(self.name(["model.alpha=0.95", "seed=1"]), "gdn_fixed_a0.95_s1")
        self.assertEqual(self.name(["model.alpha=1.0", "seed=0"]), "gdn_fixed_a1_s0")
        self.assertEqual(self.name(["model.alpha=0.95", "model.norm_eps=1e-2", "model.head_dim=64", "seed=2"]),
                         "gdn_fixed_a0.95_eps0.01_hd64_s2")
        self.assertEqual(self.name([]), "gdn_fixed_s0")
        # evaluation settings and the wandb group are not part of the model's name
        self.assertEqual(self.name(["model.alpha=0.9", "train.eval_every=100", "eval.ns=[32,64]", "experiment=explore", "seed=1"]), "gdn_fixed_a0.9_s1")

    def test_swa_run_names(self):
        cfg, keys = C.load_config(_paths.ROOT / "configs" / "swa.yaml", ["model.window=64", "model.n_layers=4", "seed=2"])
        self.assertEqual(C.run_name(cfg, keys), "swa_w64_L4_s2")
        self.assertEqual(cfg["train"]["precision"], "bf16")
        self.assertEqual(self.name(["model.kind=gdn", "model.allow_neg_eigval=true"]), "gdn_negTrue_s0")

    def test_value_parsing(self):
        self.assertEqual(C.parse_value("3"), 3)
        self.assertEqual(C.parse_value("1e-5"), 1e-5)  # plain YAML 1.1 would give the string '1e-5'
        self.assertIsInstance(C.parse_value("1e-5"), float)
        self.assertEqual(C.parse_value("[1, 2]"), [1, 2])
        self.assertIs(C.parse_value("true"), True)

    def test_override_requires_value(self):
        with self.assertRaises(ValueError):
            C.load_config(EXP1, ["model.alpha"])


class Sweep(unittest.TestCase):
    def test_first_axis_varies_fastest(self):
        jobs = L.build_jobs(str(EXP1), [], ["model.alpha=1.0,0.9", "seed=0,1"])
        self.assertEqual([j["name"] for j in jobs],
                         ["gdn_fixed_a1_s0", "gdn_fixed_a0.9_s0", "gdn_fixed_a1_s1", "gdn_fixed_a0.9_s1"])

    def test_base_overrides_apply_to_every_run(self):
        jobs = L.build_jobs(str(EXP1), ["train.steps=100"], ["model.alpha=1.0,0.9"])
        self.assertTrue(all(j["config"]["train"]["steps"] == 100 for j in jobs))
        self.assertTrue(all("steps100" in j["name"] for j in jobs))  # every changed hyperparameter is in the name


class TrainHelpers(unittest.TestCase):
    def test_lr_schedule(self):
        tc = {"lr": 1.0, "warmup_steps": 10, "steps": 110, "min_lr_ratio": 0.1}
        self.assertAlmostEqual(T.lr_at(5, tc), 0.5)
        self.assertAlmostEqual(T.lr_at(10, tc), 1.0)
        self.assertAlmostEqual(T.lr_at(110, tc), 0.1)
        self.assertTrue(all(T.lr_at(s, tc) >= T.lr_at(s + 1, tc) for s in range(10, 110)))


    def test_weighted_ce_default_equals_the_plain_mean_and_weights_only_the_passkey(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 20, 40)
        labels = torch.full((3, 20), -100)
        labels[:, 4], labels[:, 7] = 9, 11          # two AR answers
        labels[:, 15:19] = torch.tensor([8, 9, 10, 11])  # four passkey tokens, then EOS (unsupervised)
        plain = F.cross_entropy(logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1), ignore_index=-100)
        self.assertAlmostEqual(float(T.weighted_ce(logits, labels)), float(plain), places=6)
        w = T.weighted_ce(logits, labels, passkey_weight=10.0)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none").view(3, 19)
        sup = labels[:, 1:] != -100
        pk = torch.zeros_like(sup)
        pk[:, 14:18] = True  # shifted by one: targets of positions 15..18
        expect = (ce * (sup.float() * (1 + 9 * pk.float()))).sum() / (sup.float() * (1 + 9 * pk.float())).sum()
        self.assertAlmostEqual(float(w), float(expect), places=5)

    def test_passkey_weight_anneals_log_linearly_to_one(self):
        tc = {"passkey_weight": 100.0, "passkey_weight_anneal_steps": 1000}
        self.assertAlmostEqual(T.passkey_weight_at(0, tc), 100.0)
        self.assertAlmostEqual(T.passkey_weight_at(500, tc), 10.0)  # geometric midpoint
        self.assertAlmostEqual(T.passkey_weight_at(1000, tc), 1.0)
        self.assertAlmostEqual(T.passkey_weight_at(5000, tc), 1.0)  # stays at 1 afterwards
        self.assertEqual(T.passkey_weight_at(123, {"passkey_weight": 7.0}), 7.0)  # no annealing configured: constant
        self.assertEqual(T.passkey_weight_at(123, {}), 1.0)

    def test_curriculum_adds_one_config_per_stage(self):
        tc = {"train_ns": [32, 64, 128, 256], "curriculum_stage_steps": 1000}
        self.assertEqual(T.active_train_ns(1, tc), [32])
        self.assertEqual(T.active_train_ns(1000, tc), [32])
        self.assertEqual(T.active_train_ns(1001, tc), [32, 64])
        self.assertEqual(T.active_train_ns(3001, tc), [32, 64, 128, 256])
        self.assertEqual(T.active_train_ns(99999, tc), [32, 64, 128, 256])
        self.assertEqual(T.active_train_ns(5, {"train_ns": [32, 64]}), [32, 64])  # no curriculum: everything

    def test_curriculum_samples_only_active_configs(self):
        bounds = {32: (0, 100), 64: (100, 200), 128: (200, 300)}
        tc = {"train_ns": [32, 64, 128], "curriculum_stage_steps": 10}
        rng = np.random.default_rng(0)
        for step, hi in ((1, 100), (11, 200), (21, 300)):
            idx = T.sample_train_indices(rng, tc, bounds, step, 300, 4000)
            self.assertTrue(idx.min() >= 0 and idx.max() < hi, (step, idx.max()))
        idx = T.sample_train_indices(rng, tc, bounds, 11, 300, 4000)
        self.assertEqual(sorted(set((idx // 100).tolist())), [0, 1])  # both active configs are drawn, about equally
        self.assertAlmostEqual(float(np.mean(idx >= 100)), 0.5, delta=0.05)
        plain = T.sample_train_indices(np.random.default_rng(1), {"train_ns": [32]}, bounds, 5, 300, 8)
        np.testing.assert_array_equal(plain, np.random.default_rng(1).integers(0, 300, size=8))  # off: unchanged random stream

    def test_optimizer_gives_loops_their_own_lr_multiplier(self):
        from models import build_swa_fb
        m = build_swa_fb(vocab_size=40, d_model=64, n_layers=2, num_heads=2, window=8)
        opt = T.make_optimizer(m, {"lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.01, "loop_lr_mult": 10.0})
        mult = {id(p): g["lr_mult"] for g in opt.param_groups for p in g["params"]}
        loop_ids = {id(p) for l in m.loops for p in l.parameters()}
        self.assertTrue(all(mult[i] == 10.0 for i in loop_ids))
        self.assertTrue(all(v == 1.0 for i, v in mult.items() if i not in loop_ids))
        self.assertEqual(sum(len(g["params"]) for g in opt.param_groups), len(list(m.parameters())))  # nothing lost or doubled


class PeekingModel(torch.nn.Module):
    """Predicts the true next token (reads it from the input): 100% under teacher forcing if alignment is right."""

    def __init__(self, vocab=40):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[:, :-1] = 10.0 * F.one_hot(input_ids[:, 1:], self.vocab).float()
        return SimpleNamespace(logits=logits)


class NullModel(torch.nn.Module):
    """Always predicts NULL."""

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, 40)
        logits[..., G.NULL] = 10.0
        return SimpleNamespace(logits=logits)


class DataAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            G.main(["--out-dir", str(cls.tmp), "--ns", "32", "128", "--train-ns", "32", "--train-size", "50",
                    "--eval-size", "40", "--workers", "2", "--skip-sanity"])
            meta = json.loads((cls.tmp / "gen_meta.json").read_text())
            D.prepare_kinds(cls.tmp, meta["params"], workers=2)
        cls.params = meta["params"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def load(self, n, split="validation"):
        return D.load_packed(self.tmp / f"n{n}" / f"{split}.parquet", kinds=True)

    def test_kinds_are_cached_next_to_files_and_consistent_with_oracle_stats(self):
        p = self.load(128)
        self.assertTrue(D.kinds_path(self.tmp / "n128" / "validation.parquet").exists())
        stale = never = 0
        for i in range(len(p)):
            st = S.oracle(p.ids[p.offs[i]:p.offs[i + 1]], self.params["w"], self.params["n_keys"], self.params["n_values"])[3]
            stale, never = stale + st["expired_stale"], never + st["expired_never"]
        self.assertEqual(int((p.kinds == 2).sum()), stale)
        self.assertEqual(int((p.kinds == 3).sum()), never)
        self.assertGreater(stale, 0)  # n128 has stale-expired queries
        self.assertEqual(int((self.load(32).kinds == 2).sum()), 0)  # n32 cannot

    def test_packing_helpers(self):
        a, b = self.load(32), self.load(128)
        both = D.concat([a, b])
        self.assertEqual(len(both), len(a) + len(b))
        np.testing.assert_array_equal(both.ids[both.offs[len(a)]:both.offs[len(a) + 1]], b.ids[b.offs[0]:b.offs[1]])
        self.assertEqual(both.toffs[-1], len(a.types) + len(b.types))
        sub = b.subset(5)
        self.assertEqual(len(sub), 5)
        self.assertEqual(len(sub.kinds), sub.toffs[-1])

    def test_pad_batch(self):
        p = self.load(128)
        idx = np.array([3, 0, 7])
        ids, labs = D.pad_batch(p, idx, 256)
        self.assertEqual(ids.shape[1] % 256, 0)
        for j, i in enumerate(idx):
            n = p.offs[i + 1] - p.offs[i]
            np.testing.assert_array_equal(ids[j, :n], p.ids[p.offs[i]:p.offs[i + 1]])
            self.assertTrue((ids[j, n:] == G.PAD).all() and (labs[j, n:] == D.IGNORE).all())

    def test_eval_plan_shapes_and_budget(self):
        L, B = E.eval_plan(np.array([3800, 3900, 4000]), 262144)
        self.assertEqual((L, B), (4096, 64))
        L, B = E.eval_plan(np.array([61000, 61500]), 262144)
        self.assertEqual((L, B), (65536, 4))
        L, B = E.eval_plan(np.array([130, 174]), 262144)
        self.assertEqual((L, B), (256, 1024))
        self.assertEqual(E.eval_plan(np.array([61000]), 10)[1], 1)  # never below one example per batch

    def test_peeking_model_scores_perfectly(self):
        for n in (32, 128):
            m = E.evaluate_packed(PeekingModel(), self.load(n), torch.device("cpu"), tokens_per_batch=4096)
            for k in ("passkey_acc", "ar_valid_acc", "ar_expired_acc"):
                self.assertEqual(m[k], 1.0, (n, k))

    def test_null_model_scores_only_on_expired(self):
        m = E.evaluate_packed(NullModel(), self.load(128), torch.device("cpu"), tokens_per_batch=4096)
        self.assertEqual(m["ar_expired_acc"], 1.0)
        self.assertEqual(m["ar_expired_stale_acc"], 1.0)
        self.assertEqual(m["ar_expired_never_acc"], 1.0)
        self.assertEqual(m["ar_valid_acc"], 0.0)
        self.assertEqual(m["passkey_acc"], 0.0)

    def test_batching_and_filler_rows_do_not_change_metrics(self):
        p = self.load(128)  # 40 examples
        L, _ = E.eval_plan(p.lengths(), 1)
        ref = E.evaluate_packed(PeekingModel(), p, torch.device("cpu"), tokens_per_batch=1)  # B = 1
        for B in (7, 40, 1000):  # 7 leaves a partial last batch (filler rows); 1000 is one big batch
            m = E.evaluate_packed(PeekingModel(), p, torch.device("cpu"), tokens_per_batch=B * L)
            self.assertEqual(m, ref, B)
        ref = E.evaluate_packed(NullModel(), p, torch.device("cpu"), tokens_per_batch=1)
        m = E.evaluate_packed(NullModel(), p, torch.device("cpu"), tokens_per_batch=7 * L)
        self.assertEqual(m, ref)  # counts (n_valid, ...) must not include the filler rows


class RunLines(unittest.TestCase):
    """The retry wrapper of the worker scripts, run for real in bash with fake training commands."""

    def run_script(self, body: str):
        import os, subprocess
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / "fake.sh"
            fake.write_text(body.replace("DIR", str(d)))
            log, failed = d / "run.log", d / "failed.txt"
            lines = L.run_lines(["bash", str(fake)], log, "myrun", failed)
            subprocess.run(["bash", "-c", "\n".join(lines)], check=True, env={**os.environ, "RETRY_SLEEP": "0"})
            attempts = int((d / "count").read_text()) if (d / "count").exists() else 0
            return attempts, failed.read_text() if failed.exists() else ""

    COUNT = 'n=$(cat DIR/count 2>/dev/null || echo 0); echo $((n+1)) > DIR/count\n'

    def test_out_of_memory_at_start_is_retried_until_it_works(self):
        attempts, failed = self.run_script(self.COUNT + 'if [ $n -lt 2 ]; then echo "torch.OutOfMemoryError: CUDA out of memory"; exit 1; fi\necho trained')
        self.assertEqual((attempts, failed), (3, ""))

    def test_another_failure_is_not_retried_and_is_listed(self):
        attempts, failed = self.run_script(self.COUNT + 'echo "ValueError: bad config"; exit 1')
        self.assertEqual(attempts, 1)
        self.assertIn("FAILED myrun", failed)

    def test_out_of_memory_after_training_started_is_not_retried(self):
        attempts, failed = self.run_script(self.COUNT + 'echo "[myrun] step 2000 loss 0.3"; echo "torch.OutOfMemoryError: CUDA out of memory"; exit 1')
        self.assertEqual(attempts, 1)
        self.assertIn("FAILED myrun", failed)

    def test_gives_up_after_three_attempts(self):
        attempts, failed = self.run_script(self.COUNT + 'echo "torch.OutOfMemoryError: CUDA out of memory"; exit 1')
        self.assertEqual(attempts, 3)
        self.assertIn("FAILED myrun", failed)


class WorkerScript(unittest.TestCase):
    def test_header_pins_the_gpu_and_enables_expandable_segments(self):
        head = L.worker_header(3)
        self.assertIn("export CUDA_VISIBLE_DEVICES=3", head)
        self.assertIn("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", head)
        self.assertNotIn("wandb_token", " ".join(h for h in head if "export" in h and "cat" not in h))  # the key is read from the file, never written out


class NameKeys(unittest.TestCase):
    def test_list_values_do_not_put_brackets_or_spaces_in_names(self):
        cfg, keys = C.load_config(Path(__file__).resolve().parents[1] / "configs" / "swa_fb.yaml", ["train.train_ns=[32, 64]", "seed=1"])
        name = C.run_name(cfg, keys)
        self.assertIn("train_ns32-64", name)
        self.assertTrue(all(ch not in name for ch in "[], "))

    def test_only_model_describing_overrides_name_a_run(self):
        keys = ["seed", "experiment", "model.segment", "eval.ns", "eval.splits", "train.eval_ns", "train.eval_every", "model.nonlin"]
        self.assertEqual(C.name_keys(keys), ["model.segment", "model.nonlin"])


class EvaluateWithBackoff(unittest.TestCase):
    """A shared GPU can run out of memory on a long-stream batch; the final evaluation must retry smaller, not die."""

    def run_with(self, fits_up_to):
        calls = []

        def fake(model, packed, dev, tokens_per_batch, amp_dtype=None):
            calls.append(tokens_per_batch)
            if tokens_per_batch > fits_up_to:
                raise torch.cuda.OutOfMemoryError("fake")
            return {"passkey_acc": 1.0}

        real, T.E.evaluate_packed = T.E.evaluate_packed, fake
        try:
            return T.evaluate_with_backoff(None, None, None, 1 << 21, None), calls
        finally:
            T.E.evaluate_packed = real

    def test_halves_until_it_fits(self):
        out, calls = self.run_with(fits_up_to=1 << 19)
        self.assertEqual(out, {"passkey_acc": 1.0})
        self.assertEqual(calls, [1 << 21, 1 << 20, 1 << 19])

    def test_no_retry_when_it_fits(self):
        _, calls = self.run_with(fits_up_to=1 << 30)
        self.assertEqual(calls, [1 << 21])

    def test_gives_up_at_the_floor(self):
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.run_with(fits_up_to=1)


if __name__ == "__main__":
    unittest.main()
