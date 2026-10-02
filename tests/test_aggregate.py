"""aggregate.py on synthetic per-run results (no GPU, no real data).

    python -m unittest discover -s tests -v
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests import _paths
import aggregate as A
import config as C

EXP1 = _paths.ROOT / "configs" / "exp1.yaml"
NS = [32, 256, 1024, 4096, 16384]


def fake_run(raw: Path, overrides, passkey=0.5):
    cfg, keys = C.load_config(EXP1, overrides)
    rows = [{"split": sp, "N": n, "passkey_acc": passkey, "ar_valid_acc": 0.9, "ar_expired_acc": 0.7,
             "ar_expired_stale_acc": 0.6, "ar_expired_never_acc": 0.8, "n_examples": 10, "n_valid": 5,
             "n_expired_stale": 3, "n_expired_never": 2} for sp in ("validation", "test") for n in NS]
    name = C.run_name(cfg, keys)
    (raw / f"{name}.json").write_text(json.dumps({"name": name, "config": cfg, "explicit_keys": keys, "n_params": 1,
                                                  "final_train_loss": 0.1, "train_time_s": 1.0, "rows": rows}))


class Aggregate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.raw = cls.tmp / "raw"
        cls.raw.mkdir()
        for a in (1.0, 0.95, 0.8):
            for s in (0, 1):
                fake_run(cls.raw, [f"model.alpha={a}", f"seed={s}"], passkey=a)
        fake_run(cls.raw, ["model.alpha=0.95", "model.norm_eps=1e-2", "seed=0"])  # a variant model
        cls.df = A.to_frame(A.load_runs(cls.raw))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_summary_csv_has_exactly_the_requested_columns(self):
        A.write_tables(self.df, self.tmp / "out")
        for f in ("summary.csv", "summary_val.csv"):
            t = pd.read_csv(self.tmp / "out" / f)
            self.assertEqual(list(t.columns), ["model", "alpha", "seed", "N", "passkey_acc", "ar_valid_acc", "ar_expired_acc"])
        test = pd.read_csv(self.tmp / "out" / "summary.csv")
        self.assertEqual(len(test), 7 * len(NS))  # 6 base runs + 1 variant, test split only
        self.assertEqual(set(test.model), {"gdn_fixed", "gdn_fixed_eps0.01"})  # variants are not mixed into the base model
        extra = pd.read_csv(self.tmp / "out" / "summary_extra.csv")
        self.assertIn("ar_expired_stale_acc", extra.columns)
        self.assertEqual(set(extra.split), {"validation", "test"})

    def test_rows_are_sorted_alpha_descending(self):
        t = self.df[(self.df.split == "test") & (self.df.model == "gdn_fixed") & (self.df.seed == 0) & (self.df.N == 256)]
        self.assertEqual(t.alpha.tolist(), [1.0, 0.95, 0.8])

    def test_alpha_colors_are_ordered_dark_to_light(self):
        colors = A.alpha_colors([0.8, 1.0, 0.9])
        self.assertEqual(colors[1.0], A.BLUE_RAMP[-1])  # alpha closest to 1 is the darkest
        self.assertEqual(colors[0.8], A.BLUE_RAMP[0])
        self.assertEqual(len(set(A.alpha_colors([1, .999, .995, .99, .98, .95, .9, .8]).values())), 8)

    def test_plots_are_written(self):
        out = self.tmp / "plots" / "fig"
        out.parent.mkdir()
        A.plot_tradeoff(self.df, out, model="gdn_fixed", ns=(256, 1024, 4096, 16384))
        A.plot_tradeoff(self.df, Path(str(out) + "_stale"), model="gdn_fixed", x="ar_expired_stale_acc")
        for suffix in ("", "_stale"):
            for ext in ("png", "pdf"):
                self.assertGreater(Path(f"{out}{suffix}.{ext}").stat().st_size, 1000)


    def test_cluster_points_labels_every_alpha_once(self):
        pts = {1.0: (1.0, 1.0), 0.999: (1.0, 0.995), 0.95: (1.0, 0.0), 0.9: (1.0, 0.0), 0.8: (1.0, 0.01)}
        groups = A.cluster_points(pts)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(a for g in groups for a in g[3]), sorted(pts))  # nothing lost, nothing duplicated
        labels = {g[2] for g in groups}
        self.assertIn("\u03b1 1, 0.999", labels)
        self.assertIn("\u03b1 0.95, 0.9, 0.8", labels)
        many = {a: (1.0, 0.0) for a in (1.0, 0.999, 0.995, 0.99, 0.98, 0.95)}
        self.assertEqual(A.cluster_points(many)[0][2], "\u03b1 1 \u2026 0.95 (6)")


if __name__ == "__main__":
    unittest.main()
