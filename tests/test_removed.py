"""E7: the options that did not work are gone from the code; stored configs keep their meaning, building or training a removed setting stops with a message that names the commit.

    python -m unittest tests.test_removed -v
"""
import argparse
import unittest
from unittest import mock

from tests import _paths
import config as C
import launch as L
import models as M
import run_queue as Q
import train as T

ROOT = _paths.ROOT
FB = {"kind": "swa_fb", "d_model": 64, "n_layers": 2, "num_heads": 2, "window": 16, "kv_cache": True}
P2 = {"kind": "swa_2pass", "d_model": 64, "n_layers": 2, "num_heads": 2, "window": 16, "pass_output": 1}


class RemovedModelSettings(unittest.TestCase):
    def test_what_survives_builds(self):
        self.assertIsInstance(M.build_from_config(FB, 40), M.FeedbackSWA)
        self.assertIsInstance(M.build_from_config({**FB, "noise": 0.0}, 40), M.FeedbackSWA)  # noise 0 is no noise
        self.assertIsInstance(M.build_from_config(P2, 40), M.TwoPassSWA)
        self.assertIsInstance(M.build_from_config({**P2, "kv_cache": True}, 40), M.TwoPassSWA)

    def test_every_removed_setting_stops_the_build_with_the_commit_in_the_message(self):
        cases = {
            "value adaptation": {"kind": "swa_va", "d_model": 64, "n_layers": 2, "num_heads": 2, "window": 16},
            "noise": {**FB, "noise": 0.1},
            "tail path, stated": {**FB, "kv_cache": False},
            "pass 2 as the output": {**P2, "pass_output": 2},
            "two-pass tail path": {**P2, "kv_cache": False},
        }
        for what, mc in cases.items():
            with self.assertRaises(ValueError, msg=what) as cm:
                M.build_from_config(mc, 40)
            self.assertIn(C.REMOVED_IN, str(cm.exception), what)
            self.assertIn("removed in", str(cm.exception), what)

    def test_a_stored_config_without_the_key_means_what_the_code_did_when_it_was_written(self):
        """No kv_cache key = the tail path (false), no pass_output key = the output of pass 2: both are removed, so a config of that age must state the surviving value to be built."""
        fb_old = {k: v for k, v in FB.items() if k != "kv_cache"}
        p2_old = {k: v for k, v in P2.items() if k != "pass_output"}
        for mc in (fb_old, p2_old):
            with self.assertRaises(ValueError):
                M.build_from_config(mc, 40)
        self.assertEqual(C.CODE_DEFAULTS["model.kv_cache"], False)      # the defaults of the stored configs are unchanged ...
        self.assertEqual(C.CODE_DEFAULTS["model.pass_output"], 2)
        self.assertEqual(C.CODE_DEFAULTS["model.noise"], 0.0)
        self.assertEqual(C.CODE_DEFAULTS["train.passkey_weight"], 1.0)
        self.assertEqual(C.CODE_DEFAULTS["train.passkey_weight_anneal_steps"], 0)
        M.build_from_config({**fb_old, "kv_cache": True}, 40)           # ... and stating the surviving value loads old weights on the path that is left
        M.build_from_config({**p2_old, "pass_output": 1}, 40)

    def test_the_kinds_the_removed_settings_do_not_belong_to_are_not_touched(self):
        for mc in ({"kind": "swa", "d_model": 64, "n_layers": 2, "num_heads": 2, "window": 16, "noise": 0.5, "pass_output": 2, "kv_cache": False},):
            self.assertEqual(C.removed_model_settings(mc), [])  # plain SWA has none of these options
        self.assertEqual(C.removed_model_settings({"kind": "gdn2"}), [])

    def test_the_removed_classes_are_gone(self):
        for name in ("ValueAdapter", "ValueAdaptSWA", "build_swa_va"):
            self.assertFalse(hasattr(M, name), name)
        import inspect
        self.assertNotIn("noise", inspect.signature(M.build_swa_fb).parameters)
        self.assertNotIn("kv_cache", inspect.signature(M.build_swa_fb).parameters)
        self.assertNotIn("pass_output", inspect.signature(M.build_swa_2pass).parameters)


class RemovedTrainSettings(unittest.TestCase):
    def test_a_passkey_weight_or_its_annealing_is_refused(self):
        self.assertEqual(C.removed_train_settings({"passkey_weight": 1.0, "passkey_weight_anneal_steps": 0}), [])
        self.assertEqual(C.removed_train_settings({}), [])
        self.assertEqual(len(C.removed_train_settings({"passkey_weight": 100.0})), 1)
        self.assertEqual(len(C.removed_train_settings({"passkey_weight": 100.0, "passkey_weight_anneal_steps": 1000})), 2)
        self.assertEqual(len(C.removed_train_settings({"passkey_weight_anneal_steps": 5})), 1)
        with self.assertRaises(ValueError) as cm:
            C.check_removed({**{"model": FB}, "train": {"passkey_weight": 10.0}})
        self.assertIn(C.REMOVED_IN, str(cm.exception))

    def test_train_py_stops_before_it_builds_anything(self):
        with self.assertRaises(SystemExit) as cm:
            T.main(["--config", str(ROOT / "configs" / "swa_fb.yaml"), "--no-wandb", "--set", "train.passkey_weight=7", "model.window=16"])
        self.assertIn("removed in", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            T.main(["--config", str(ROOT / "configs" / "swa_2pass.yaml"), "--no-wandb", "--set", "model.pass_output=2", "model.window=16"])
        self.assertIn("pass_output = 2", str(cm.exception))

    def test_launch_py_refuses_a_sweep_over_a_removed_setting_before_it_starts_a_worker(self):
        for config, sweep in (("swa_fb.yaml", "model.noise=0.1,0.3"), ("swa_2pass.yaml", "model.pass_output=2,1"), ("swa_fb.yaml", "train.passkey_weight=10")):
            argv = ["launch.py", "--config", str(ROOT / "configs" / config), "--session", "never_started", "--dry-run", "--sweep", sweep]
            with mock.patch("sys.argv", argv), self.assertRaises(SystemExit) as cm, mock.patch("builtins.print"):
                L.main()
            self.assertIn("removed in", str(cm.exception), sweep)


class WorkingConfigsAndTheQueue(unittest.TestCase):
    def test_the_working_configs_build_and_state_the_surviving_values(self):
        for name in ("swa_fb.yaml", "swa_2pass.yaml"):
            cfg = C.load_config(ROOT / "configs" / name, [])[0]
            C.check_removed(cfg)
            M.build_from_config(cfg["model"], 40)
        self.assertIs(C.load_config(ROOT / "configs" / "swa_fb.yaml", [])[0]["model"]["kv_cache"], True)
        self.assertEqual(C.load_config(ROOT / "configs" / "swa_2pass.yaml", [])[0]["model"]["pass_output"], 1)
        self.assertFalse((ROOT / "configs" / "swa_va.yaml").exists())

    def test_no_active_stage_of_the_committed_queue_launches_a_removed_setting(self):
        stages = Q.parse_queue((Q.ROOT / "configs" / "queue.txt").read_text())
        self.assertGreaterEqual(len(stages), 10)
        for session, args in stages:
            ap = argparse.ArgumentParser()
            ap.add_argument("--config"); ap.add_argument("--sweep", nargs="*", default=[]); ap.add_argument("--set", nargs="*", default=[])
            ap.add_argument("--gpus", nargs="+"); ap.add_argument("--procs-per-gpu"); ap.add_argument("--no-wandb", action="store_true")
            a = ap.parse_args(args)
            for job in L.build_jobs(a.config, a.set, a.sweep):
                C.check_removed(job["config"])   # raises for a stage that would be refused
        text = (Q.ROOT / "configs" / "queue.txt").read_text()
        for session in ("fbnoise", "va_32", "va_8", "p2_32", "p2_64", "p2_pilot_lr1", "p2_pilot_gi"):
            self.assertIn(f"# [archive: launches a setting that E7 removed, a re-run is refused] {session} |", text)

    def test_the_frozen_references_still_describe_the_removed_settings(self):
        """The references are never edited: their `noise`, `passkey_weight`, `pass_output`, `kind` keys stay, so old runs keep their names, change lists and report groups."""
        self.assertEqual(C.load_reference({"model": {"kind": "swa_fb"}})["model"]["noise"], 0.0)
        self.assertEqual(C.load_reference({"model": {"kind": "swa_fb"}})["train"]["passkey_weight"], 1.0)
        self.assertEqual(C.load_reference({"model": {"kind": "swa_2pass"}})["model"]["pass_output"], 2)
        self.assertEqual(C.load_reference({"model": {"kind": "swa_va"}})["model"]["kind"], "swa_va")


if __name__ == "__main__":
    unittest.main()
