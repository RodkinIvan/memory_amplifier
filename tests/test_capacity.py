"""The passkey-capacity chain (scripts/passkey_capacity.py): what it trains, how a stage is summarised, when it stops.

    python -m unittest tests.test_capacity -v
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import _paths
import config as C

spec = importlib.util.spec_from_file_location("capacity", _paths.ROOT / "scripts" / "passkey_capacity.py")
CAP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CAP)

INIT = "checkpoints/swa_fb_gi0_nlnone_seg8_steps50000_s2.pt"


def fake_result(exact256: float, token256: float = None) -> dict:
    rows = []
    for split in ("validation", "test"):
        for n in CAP.NS:
            e = exact256 if n == 256 else 1.0
            rows.append({"split": split, "N": n, "passkey_acc": e, "passkey_token_acc": token256 if (n == 256 and token256 is not None) else e, "ar_valid_acc": 0.99})
    return {"rows": rows, "train_time_s": 3600.4, "config": {"train": {"steps": 10000}}}


class StageSetup(unittest.TestCase):
    def test_a_stage_fine_tunes_the_segment_8_model_on_the_training_range_with_a_longer_passkey(self):
        cfg, explicit = C.load_config(_paths.ROOT / "configs" / "swa_fb.yaml", CAP.stage_overrides(16, INIT, 10000, 3e-4))
        self.assertEqual((cfg["model"]["segment"], cfg["model"]["nonlin"], cfg["model"]["gain_init"], cfg["passkey_len"]), (8, "none", 0, 16))
        self.assertEqual(cfg["train"]["init_from"], INIT)
        self.assertEqual(cfg["train"]["train_ns"], [32, 64, 128, 256])          # the training range of the checkpoint
        self.assertEqual(cfg["eval"]["ns"], [32, 64, 128, 256])                  # nothing longer is evaluated
        self.assertEqual(cfg["train"]["curriculum_stage_steps"], 0)              # the loop is already on: all lengths from the start
        self.assertEqual(C.data_dir_for(cfg).name, "passkey_swar_pk16")
        d = C.reference_diff(cfg)
        for key in ("passkey_len", "train.init_from", "train.steps", "train.lr", "model.segment", "model.nonlin", "model.gain_init"):
            self.assertIn(key, d)

    def test_names_tell_the_passkey_length_and_are_unique_per_stage(self):
        names = [CAP.stage_name(P, INIT, 10000, 3e-4) for P in (8, 16, 32)]
        self.assertEqual(len(set(names)), 3)
        for P, n in zip((8, 16, 32), names):
            self.assertIn(f"pk{P}_", n)
            self.assertTrue(n.startswith("swa_fb_seg8_nlnone_gi0_pk"), n)
            self.assertNotIn("/", n)


class Summary(unittest.TestCase):
    def test_the_entry_has_exact_match_and_token_accuracy_per_n_and_the_pass_flag(self):
        e = CAP.summarize(8, "run8", INIT, fake_result(0.95, 0.99))
        self.assertEqual((e["passkey_len"], e["bits"], e["train_time_s"], e["steps"]), (8, 32, 3600, 10000))
        self.assertEqual(e["validation"]["exact"], {"32": 1.0, "64": 1.0, "128": 1.0, "256": 0.95})
        self.assertEqual(e["test"]["token"]["256"], 0.99)
        self.assertTrue(e["passed"])
        self.assertFalse(CAP.summarize(8, "run8", INIT, fake_result(0.8999))["passed"])
        self.assertTrue(CAP.summarize(8, "run8", INIT, fake_result(0.9))["passed"])  # at the threshold counts as passed ("drops below 0.9" stops the chain)

    def test_the_decision_uses_the_validation_split_not_the_test_split(self):
        r = fake_result(0.95)
        for row in r["rows"]:
            if row["split"] == "test" and row["N"] == 256:
                row["passkey_acc"] = 0.1
        self.assertTrue(CAP.summarize(8, "x", INIT, r)["passed"])


class Chain(unittest.TestCase):
    def run_chain(self, exact_by_len, first=8, max_len=512):
        seen = []

        def run_stage(P, init):
            seen.append((P, init))
            return CAP.summarize(P, f"run{P}", init, fake_result(exact_by_len[P])), f"runs/run{P}/model.pt"

        with tempfile.TemporaryDirectory() as d, mock.patch.object(CAP, "OUT", Path(d) / "capacity.json"), mock.patch("builtins.print"):
            stages = CAP.chain(run_stage, first, max_len, INIT)
            saved = json.loads((Path(d) / "capacity.json").read_text())
        return stages, seen, saved

    def test_it_doubles_the_passkey_and_starts_every_stage_from_the_weights_of_the_one_before(self):
        stages, seen, saved = self.run_chain({8: 0.99, 16: 0.95, 32: 0.92, 64: 0.5, 128: 0.99})
        self.assertEqual([s["passkey_len"] for s in stages], [8, 16, 32, 64])  # 128 is never reached: 64 failed
        self.assertEqual([p for p, _ in seen], [8, 16, 32, 64])
        self.assertEqual([i for _, i in seen], [INIT, "runs/run8/model.pt", "runs/run16/model.pt", "runs/run32/model.pt"])
        self.assertEqual([s["passed"] for s in stages], [True, True, True, False])
        self.assertEqual(saved["stages"], stages)
        self.assertEqual(saved["threshold"], 0.9)

    def test_it_stops_at_once_when_the_first_stage_fails_and_at_the_maximum_length(self):
        stages, _, _ = self.run_chain({8: 0.2})
        self.assertEqual(len(stages), 1)
        stages, _, _ = self.run_chain({8: 1.0, 16: 1.0, 32: 1.0}, max_len=32)
        self.assertEqual([s["passkey_len"] for s in stages], [8, 16, 32])


if __name__ == "__main__":
    unittest.main()
