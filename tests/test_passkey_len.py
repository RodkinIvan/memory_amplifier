"""Passkeys longer than 4 tokens (the capacity experiments): generator, oracle, run names, data directories, evaluation, fine-tuning from a checkpoint.

    python -m unittest tests.test_passkey_len -v
"""
import contextlib
import hashlib
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
import make_passkey_swar as G
import sanity_checks as S

ROOT = _paths.ROOT
GOLDEN = json.loads((ROOT / "tests" / "golden_passkey_len4.json").read_text())


class DefaultDatasetIsUnchanged(unittest.TestCase):
    """The option must not move a single token of the default (4-token) dataset: examples recorded with the generator as it was before the option existed."""

    def test_default_examples_are_the_recorded_ones(self):
        for key, g in GOLDEN.items():
            n, idx = map(int, key.split("_"))
            seed = G.example_seed(2, n, idx)
            self.assertEqual(seed, g["seed"])
            for p in (G.GenParams(), G.GenParams(passkey_len=4)):
                t, l, ty = G.make_example(seed, n, p)
                self.assertEqual(len(t), g["len"], key)
                self.assertEqual((t[:14], t[-8:]), (g["head"], g["tail"]), key)
                self.assertEqual(hashlib.sha256(json.dumps([t, l, ty]).encode()).hexdigest(), g["tokens_sha"], key)

    def test_metadata_of_a_default_dataset_does_not_mention_the_option(self):
        self.assertNotIn("passkey_len", G.params_dict(G.GenParams()))
        self.assertEqual(G.params_dict(G.GenParams(passkey_len=16))["passkey_len"], 16)
        for p in (G.GenParams(), G.GenParams(passkey_len=32)):
            self.assertEqual(G.GenParams(**G.params_dict(p)), p)  # what gen_meta.json stores rebuilds the parameters

    @unittest.skipUnless((ROOT / "data_cache" / "passkey_swar" / "n32" / "validation.parquet").exists(), "needs the generated dataset")
    def test_the_stored_default_dataset_still_decodes_to_what_the_generator_makes(self):
        p = D.load_packed(ROOT / "data_cache" / "passkey_swar" / "n32" / "validation.parquet")
        for i in (0, 5, 17):
            t, l, _ = G.make_example(G.example_seed(2, 32, i), 32, G.GenParams())
            self.assertEqual(p.ids[p.offs[i]:p.offs[i + 1]].tolist(), t)
            self.assertEqual(p.labels[p.offs[i]:p.offs[i + 1]].tolist(), l)


class LongPasskeys(unittest.TestCase):
    def test_layout_labels_and_oracle_for_longer_passkeys(self):
        for P in (1, 8, 16, 64):
            p = G.GenParams(passkey_len=P)
            t, l, ty = G.make_example(G.example_seed(2, 64, 3), 64, p)
            self.assertEqual((t[0], t[1], t[-1]), (G.BOS, G.PK, G.EOS))
            pk = t[2:2 + P]
            self.assertTrue(all(p.val0 <= x < p.val0 + p.n_values for x in pk))
            self.assertEqual(t[-(P + 2)], G.PKQ)
            self.assertEqual(t[-(P + 1):-1], pk)                       # the passkey is repeated after PKQ
            self.assertEqual(l[-(P + 1):-1], pk)                       # ... and is what the model is asked to produce
            self.assertEqual(ty[-P:], [G.PASSKEY] * P)
            self.assertEqual(sum(x != G.IGNORE for x in l), len(ty))   # one answer type per supervised position
            lab, oty, kind, st = S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values, P)
            self.assertEqual(lab.tolist(), l)
            self.assertEqual(oty.tolist(), ty)
            self.assertEqual(int(st["writes"]), 64)
            self.assertEqual(kind[-P:].tolist(), [0] * P)

    def test_the_oracle_rejects_a_stream_read_with_the_wrong_passkey_length(self):
        t, _, _ = G.make_example(G.example_seed(2, 32, 1), 32, G.GenParams(passkey_len=8))
        with self.assertRaises(S.OracleError):
            S.oracle(np.asarray(t, np.int16), 16, 16, 16, 4)
        with self.assertRaises(S.OracleError):
            S.oracle(np.asarray(t, np.int16), 16, 16, 16, 16)

    def test_streams_are_longer_by_exactly_the_extra_passkey_tokens(self):
        # same seed, same stream length: the two passkeys consume different random numbers, so only the lengths are comparable on average
        n = [np.mean([len(G.make_example(G.example_seed(2, 64, i), 64, G.GenParams(passkey_len=P))[0]) for i in range(200)]) for P in (4, 36)]
        self.assertAlmostEqual(n[1] - n[0], 2 * 32, delta=12)


class DatasetsAndEvaluation(unittest.TestCase):
    P = 8

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.out = cls.tmp / "passkey_swar_pk8"
        with contextlib.redirect_stdout(io.StringIO()):
            G.main(["--out-dir", str(cls.out), "--ns", "32", "64", "--train-ns", "32", "--train-size", "50", "--eval-size", "40", "--workers", "2", "--skip-sanity",
                    "--passkey-len", str(cls.P)])
            cls.meta = json.loads((cls.out / "gen_meta.json").read_text())
            D.prepare_kinds(cls.out, cls.meta["params"], workers=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_the_metadata_carries_the_length_and_the_sanity_checks_pass(self):
        self.assertEqual(self.meta["params"]["passkey_len"], self.P)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = S.run(self.out, self.tmp / "reports", workers=2)
        self.assertTrue(ok)  # oracle, mutation test, determinism, leakage on a passkey of 8 tokens
        rep = json.loads((self.tmp / "reports" / "passkey_swar_stats.json").read_text())
        self.assertEqual(rep["params"]["passkey_len"], self.P)
        self.assertEqual(sum(rep["oracle_errors"].values()), 0)
        self.assertGreater(sum(rep["oracle_labels_checked"].values()), 0)
        self.assertEqual(rep["leakage"], {"train&validation": 0, "train&test": 0, "validation&test": 0})

    def test_kinds_have_one_passkey_entry_per_passkey_token(self):
        p = D.load_packed(self.out / "n32" / "validation.parquet", kinds=True)
        for i in range(len(p)):
            self.assertEqual(p.kinds[p.toffs[i]:p.toffs[i + 1]][-self.P:].tolist(), [0] * self.P)
            self.assertEqual(p.types[p.toffs[i]:p.toffs[i + 1]][-self.P:].tolist(), [0] * self.P)

    def test_exact_match_needs_all_tokens_and_the_token_accuracy_counts_each(self):
        p = D.load_packed(self.out / "n32" / "validation.parquet", kinds=True)
        peek = E.evaluate_packed(Peek(), p, torch.device("cpu"), tokens_per_batch=4096)
        self.assertEqual((peek["passkey_acc"], peek["passkey_token_acc"]), (1.0, 1.0))
        # wrong whenever the current token is a value token: breaks passkey tokens 2..P, keeps the first one and every AR answer (the current token there is a key)
        part = E.evaluate_packed(Peek(break_after_value=True), p, torch.device("cpu"), tokens_per_batch=4096)
        self.assertEqual(part["passkey_acc"], 0.0)
        self.assertAlmostEqual(part["passkey_token_acc"], 1 / self.P)
        self.assertEqual(part["ar_valid_acc"], 1.0)


class PerPositionAndLongPasskeys(DatasetsAndEvaluation):
    """Review C1 and C4 on the P = 8 dataset of the class above."""

    def test_per_position_accuracy_is_one_number_per_passkey_token_and_means_the_token_accuracy(self):
        p = D.load_packed(self.out / "n32" / "validation.parquet", kinds=True)
        full = E.evaluate_packed(Peek(), p, torch.device("cpu"), tokens_per_batch=4096)
        self.assertEqual(full["passkey_pos_acc"], [1.0] * self.P)
        part = E.evaluate_packed(Peek(break_after_value=True), p, torch.device("cpu"), tokens_per_batch=4096)
        self.assertEqual(part["passkey_pos_acc"], [1.0] + [0.0] * (self.P - 1))   # only the first passkey token follows PKQ, the rest follow a value token
        self.assertAlmostEqual(np.mean(part["passkey_pos_acc"]), part["passkey_token_acc"])

    def test_batching_does_not_change_the_per_position_accuracy(self):
        p = D.load_packed(self.out / "n32" / "validation.parquet", kinds=True)
        L, _ = E.eval_plan(p.lengths(), 1)
        ref = E.evaluate_packed(Peek(break_after_value=True), p, torch.device("cpu"), tokens_per_batch=1)
        for B in (7, 40):
            self.assertEqual(E.evaluate_packed(Peek(break_after_value=True), p, torch.device("cpu"), tokens_per_batch=B * L)["passkey_pos_acc"], ref["passkey_pos_acc"])

    def test_probes_scan_the_stream_after_a_passkey_of_any_length(self):
        import probes as PR
        from probes import make_packed
        for P in (4, 8):
            gp = G.GenParams(passkey_len=P)
            packed = make_packed(gp, 100, 6)
            valid, stale = PR.query_distances(packed, P)
            self.assertGreater(len(valid), 0)
            n_q = int(sum((packed.kinds[packed.toffs[i]:packed.toffs[i + 1]][:-P] > 0).sum() for i in range(len(packed))))
            self.assertEqual(len(valid) + len(stale) + sum(int((packed.kinds[packed.toffs[i]:packed.toffs[i + 1]][:-P] == 3).sum()) for i in range(len(packed))), n_q)  # every query is told apart
        v4, _ = PR.query_distances(make_packed(G.GenParams(passkey_len=4), 100, 6), 4)
        with self.assertRaises((KeyError, IndexError, ValueError)):
            PR.query_distances(make_packed(G.GenParams(passkey_len=8), 100, 6), 4)   # reading an 8-token dataset as 4 tokens does not give silent nonsense


class Peek(torch.nn.Module):
    """Predicts the true next token; with break_after_value it predicts NULL whenever the current token is a value token."""

    def __init__(self, break_after_value=False):
        super().__init__()
        self.brk = break_after_value

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, 40)
        logits[:, :-1] = 10.0 * F.one_hot(input_ids[:, 1:], 40).float()
        if self.brk:
            is_value = input_ids >= G.GenParams().val0
            logits[is_value] = 0.0
            logits[..., G.NULL] = torch.where(is_value, torch.full_like(logits[..., G.NULL], 10.0), logits[..., G.NULL])
        return SimpleNamespace(logits=logits)


class ConfigAndNames(unittest.TestCase):
    def cfg(self, *overrides):
        return C.load_config(ROOT / "configs" / "swa_fb.yaml", list(overrides))

    def test_the_data_directory_follows_the_passkey_length(self):
        cfg, _ = self.cfg()
        self.assertEqual(C.data_dir_for(cfg), ROOT / "data_cache" / "passkey_swar")
        self.assertEqual(C.data_dir_for({**cfg, "passkey_len": 4}), ROOT / "data_cache" / "passkey_swar")
        self.assertEqual(C.data_dir_for({**cfg, "passkey_len": 16}), ROOT / "data_cache" / "passkey_swar_pk16")
        self.assertEqual(C.data_dir_for({**cfg, "data_dir": "/abs/x", "passkey_len": 8}), Path("/abs/x_pk8"))

    def test_the_name_tells_the_passkey_length_but_not_the_checkpoint_path_and_the_diff_tells_both(self):
        cfg, keys = self.cfg("model.segment=8", "passkey_len=16", "train.init_from=runs/some_run/model.pt", "train.steps=10000")
        name = C.run_name(cfg, keys)
        self.assertIn("pk16", name)
        self.assertNotIn("/", name)
        self.assertNotIn("init", name)
        d = C.reference_diff(cfg)
        self.assertEqual(d["passkey_len"], [4, 16])
        self.assertEqual(d["train.init_from"], [None, "runs/some_run/model.pt"])
        self.assertIn("passkey_len 4 → 16", C.describe_diff(d))

    def test_old_results_without_the_key_are_passkey_len_4(self):
        cfg, _ = self.cfg()
        self.assertNotIn("passkey_len", cfg)
        self.assertEqual(C.reference_diff(cfg), {"model.kv_cache": [False, True]})  # the one difference of the working config (E7); the passkey length is not one: a missing key is 4


if __name__ == "__main__":
    unittest.main()
