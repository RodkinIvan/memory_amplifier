"""Probe helpers: on-the-fly streams, token distances and reach ceilings (CPU only).

    python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from tests import _paths  # noqa: F401
import make_passkey_swar as G
import probes as P
import sanity_checks as S


class Probes(unittest.TestCase):
    def test_make_packed_is_consistent_with_the_oracle(self):
        p = P.make_packed(G.GenParams(), 40, 8)
        self.assertEqual(len(p), 8)
        self.assertEqual(len(p.kinds), len(p.types))
        for i in range(len(p)):
            ids = p.ids[p.offs[i]:p.offs[i + 1]]
            lab, ty, kind, _ = S.oracle(ids, 16, 16, 16)
            np.testing.assert_array_equal(lab, p.labels[p.offs[i]:p.offs[i + 1]])
            np.testing.assert_array_equal(kind, p.kinds[p.toffs[i]:p.toffs[i + 1]])

    def test_probe_streams_use_their_own_seed(self):
        a = P.make_packed(G.GenParams(), 32, 4)
        for split_seed in (1, 2, 3):  # train / validation / test seeds of the dataset
            self.assertFalse(np.array_equal(a.ids[:a.offs[1]], P.make_packed(G.GenParams(), 32, 1, split_seed).ids))

    def test_distances_match_the_answer_rule(self):
        p = P.make_packed(G.GenParams(), 200, 30)
        valid, stale = P.query_distances(p)
        self.assertGreater(len(valid), 0)
        self.assertGreater(len(stale), 0)
        self.assertGreaterEqual(valid.min(), 2)  # a write is at least `v` then `Q k`: distance >= 2
        self.assertLess(valid.max(), stale.min())  # valid writes are always more recent than stale ones
        self.assertGreaterEqual(stale.min(), 3 * 32)  # more than 2w = 32 writes of 3 tokens each

    def test_reach_ceilings_are_monotone_and_bounded(self):
        c = P.reach_ceilings(P.make_packed(G.GenParams(), 200, 30), [30, 60, 100, 200, 500])
        vr = [c[r]["valid_reachable"] for r in sorted(c)]
        sv = [c[r]["stale_visible"] for r in sorted(c)]
        self.assertEqual(vr, sorted(vr))
        self.assertEqual(sv, sorted(sv))
        self.assertEqual(c[500]["valid_reachable"], 1.0)
        self.assertEqual(c[30]["stale_visible"], 0.0)
        self.assertTrue(all(0 <= v <= 1 for v in vr + sv))


if __name__ == "__main__":
    unittest.main()
