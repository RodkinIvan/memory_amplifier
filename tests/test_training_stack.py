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


if __name__ == "__main__":
    unittest.main()
