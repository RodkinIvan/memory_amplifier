"""End-to-end test of generate -> sanity checks on a tiny dataset, including the failure paths.

    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests import _paths  # noqa: F401
import make_passkey_swar as G
import sanity_checks as S

ARGS = ["--ns", "32", "64", "--train-ns", "32", "--train-size", "300", "--eval-size", "60", "--workers", "2",
        "--skip-sanity"]


def quiet(fn, *a, **k):
    """Call fn with stdout captured; returns (result, captured text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k), buf.getvalue()


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        quiet(G.main, ["--out-dir", str(cls.tmp / "clean"), *ARGS])  # returns None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def copy(self, name):
        dst = self.tmp / name
        shutil.copytree(self.tmp / "clean", dst)
        return dst

    def run_sanity(self, data_dir):
        return quiet(S.run, data_dir, data_dir / "reports", workers=2)

    @staticmethod
    def refresh_sha(d, rel):
        """Record the current checksum so that only the content problem (not a stale sha) can fail the run."""
        meta = json.loads((d / "gen_meta.json").read_text())
        meta["files"][rel]["sha256"] = G.sha256_file(d / rel)
        (d / "gen_meta.json").write_text(json.dumps(meta))

    def test_clean_dataset_passes(self):
        d = self.copy("ok")
        ok, out = self.run_sanity(d)
        self.assertTrue(ok, out)
        rep = json.loads((d / "reports" / "passkey_swar_stats.json").read_text())
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["leakage"], {"train&validation": 0, "train&test": 0, "validation&test": 0})
        self.assertTrue((d / "sanity_report.json").exists())
        self.assertIn("expired_stale_share", rep["summary"]["n64/validation"])

    def test_corrupted_label_fails(self):
        d = self.copy("bad_label")
        path = d / "n32" / "validation.parquet"
        t = pq.read_table(path)
        labels = t["labels"].to_pylist()
        pos = next(i for i, l in enumerate(labels[0]) if l != G.IGNORE)
        labels[0][pos] += 1
        t = t.set_column(t.schema.get_field_index("labels"), "labels", pa.array(labels, t.schema.field("labels").type))
        pq.write_table(t, path)
        self.refresh_sha(d, "n32/validation.parquet")
        ok, out = self.run_sanity(d)
        self.assertFalse(ok)
        self.assertIn("labels differ from oracle", out)

    def test_leaked_example_fails(self):
        d = self.copy("leak")
        shutil.copy(d / "n32" / "validation.parquet", d / "n32" / "test.parquet")  # same examples in two splits
        self.refresh_sha(d, "n32/test.parquet")
        ok, out = self.run_sanity(d)
        self.assertFalse(ok)
        self.assertIn("'validation&test': 60", out)

    def test_upload_is_refused_when_checks_did_not_pass(self):
        import upload_passkey_swar as U
        d = self.copy("noupload")
        ok, out = self.run_sanity(d)
        self.assertTrue(ok, out)
        rep = json.loads((d / "sanity_report.json").read_text())
        rep["passed"] = False
        (d / "sanity_report.json").write_text(json.dumps(rep))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            U.preflight(d, False)
        self.assertIn("sanity checks did not pass", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
