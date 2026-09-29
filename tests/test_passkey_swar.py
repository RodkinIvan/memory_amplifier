"""Tests for the passkey_swar generator and its oracle (CPU only, a few seconds).

    python -m unittest discover -s tests -v
"""
import json
import unittest

import numpy as np

from tests import _paths  # noqa: F401
import make_passkey_swar as G
import sanity_checks as S

SMALL = G.GenParams(w=2, n_keys=4, n_values=4, active_min=2, active_max=3)
K0, V0 = G.KEY0, SMALL.val0


def seq(events, pk=(0, 1, 2, 3), p=SMALL):
    """Build a sequence from ('W', key, value) / ('Q', key, answer-or-None) events."""
    t = [G.BOS, G.PK, *[p.val0 + x for x in pk]]
    for kind, k, x in events:
        if kind == "W":
            t += [G.W, G.KEY0 + k, p.val0 + x]
        else:
            t += [G.Q, G.KEY0 + k, G.NULL if x is None else p.val0 + x]
    return np.array(t + [G.PKQ, *[p.val0 + x for x in pk], G.EOS], np.int16)


def run_oracle(events, p=SMALL):
    return S.oracle(seq(events, p=p), p.w, p.n_keys, p.n_values)


W = lambda k, v: ("W", k, v)  # noqa: E731


class OracleSemantics(unittest.TestCase):
    """w = 2: valid iff age <= 2, gray zone 3..4, expired iff age > 4 or never written (age 1 = previous write)."""

    def kinds(self, events):
        return run_oracle(events)[2].tolist()[:-4]  # drop the 4 passkey positions

    def test_valid_ages(self):
        self.assertEqual(self.kinds([W(0, 1), ("Q", 0, 1)]), [1])
        self.assertEqual(self.kinds([W(0, 1), W(1, 2), ("Q", 0, 1)]), [1])

    def test_gray_zone_is_rejected(self):
        for n_between in (2, 3):  # ages 3 and 4
            ev = [W(0, 1)] + [W(1, 0)] * n_between + [("Q", 0, 1)]
            with self.assertRaises(S.OracleError):
                run_oracle(ev)

    def test_expired_stale_and_never(self):
        stale = [W(0, 1)] + [W(1, 0)] * 4 + [("Q", 0, None)]  # age 5 > 2w
        self.assertEqual(self.kinds(stale), [2])
        self.assertEqual(self.kinds([W(0, 1), ("Q", 3, None)]), [3])

    def test_overwrite_uses_latest_write(self):
        self.assertEqual(self.kinds([W(0, 1), W(1, 1), W(0, 3), ("Q", 0, 3)]), [1])
        with self.assertRaises(S.OracleError):  # answering with the overwritten value
            run_oracle([W(0, 1), W(1, 1), W(0, 3), ("Q", 0, 1)])

    def test_latest_write_resets_age(self):
        # first write of key 0 is 6 writes old, but it was rewritten 1 write ago -> valid
        ev = [W(0, 1)] + [W(1, 0)] * 5 + [W(0, 2), ("Q", 0, 2)]
        self.assertEqual(self.kinds(ev), [1])

    def test_wrong_answers_are_rejected(self):
        with self.assertRaises(S.OracleError):  # valid key answered NULL
            run_oracle([W(0, 1), ("Q", 0, None)])
        with self.assertRaises(S.OracleError):  # expired key answered with its stale value
            run_oracle([W(0, 1)] + [W(1, 0)] * 4 + [("Q", 0, 1)])

    def test_structure_errors(self):
        good = seq([W(0, 1), ("Q", 0, 1)])
        for mutate in (lambda x: x[:-1], lambda x: np.concatenate([x[:6], [G.EOS], x[6:]]),
                       lambda x: np.where(np.arange(len(x)) == len(x) - 2, x[2], x)):  # tail != passkey
            with self.assertRaises(S.OracleError):
                S.oracle(mutate(good.copy()), SMALL.w, SMALL.n_keys, SMALL.n_values)

    def test_labels_supervise_only_answers_and_passkey(self):
        x = seq([W(0, 1), ("Q", 0, 1), W(1, 2)])
        lab, ty, kind, _ = S.oracle(x, SMALL.w, SMALL.n_keys, SMALL.n_values)
        pos = np.flatnonzero(lab != G.IGNORE)
        self.assertEqual(len(pos), len(ty))
        np.testing.assert_array_equal(lab[pos], x[pos])  # HF convention: label == token at that position
        self.assertEqual(ty.tolist(), [G.AR_VALID] + [G.PASSKEY] * 4)


class GeneratorInvariants(unittest.TestCase):
    PARAMS = [G.GenParams(), SMALL, G.GenParams(drift_rate=0.5), G.GenParams(drift_rate=0.0, w=4, query_rate=1.0),
              G.GenParams(n_keys=8, n_values=4, active_min=3, active_max=8, valid_frac=0.8)]

    def test_oracle_reproduces_every_label(self):
        for p in self.PARAMS:
            for n in (1, 5, 32, 100, 300):
                for i in range(15):
                    t, l, ty = G.make_example(G.example_seed(7, n, i), n, p)
                    lab, oty, _, st = S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values)
                    self.assertEqual(l, lab.tolist())
                    self.assertEqual(ty, oty.tolist())
                    self.assertEqual(st["writes"], n)

    def test_deterministic_and_seed_sensitive(self):
        p = G.GenParams()
        a = G.make_example(123, 64, p)
        self.assertEqual(a, G.make_example(123, 64, p))
        self.assertNotEqual(a[0], G.make_example(124, 64, p)[0])

    def test_valid_expired_balance(self):
        p = G.GenParams()
        for n in (32, 256):
            st = S._zero_stats(p.w)
            for i in range(300):
                t, _, _ = G.make_example(G.example_seed(9, n, i), n, p)
                st = S.merge_stats(st, S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values)[3])
            self.assertAlmostEqual(st["valid"] / st["queries"], p.valid_frac, delta=0.03)
            self.assertAlmostEqual(st["queries"] / st["writes"], p.query_rate, delta=0.03)

    def test_stale_expired_impossible_up_to_2w(self):
        p = G.GenParams()
        for i in range(50):  # n = 2w: no key can be older than 2w writes
            t, _, _ = G.make_example(G.example_seed(11, 2 * p.w, i), 2 * p.w, p)
            st = S.oracle(np.asarray(t, np.int16), p.w, p.n_keys, p.n_values)[3]
            self.assertEqual(st["expired_stale"], 0)

    def test_seed_decoding(self):
        s = G.example_seed(3, 16384, 999)
        self.assertEqual((s // 10**12, (s // 10**6) % 10**6, s % 10**6), (3, 16384, 999))
        with self.assertRaises(AssertionError):
            G.example_seed(1, 32, 10**6)

    def test_vocab_json_matches_generator(self):
        committed = json.loads((_paths.ROOT / "data" / "vocab.json").read_text())
        self.assertEqual(committed, G.vocab_dict(G.GenParams()))
        self.assertEqual(committed["size"], 40)


if __name__ == "__main__":
    unittest.main()
