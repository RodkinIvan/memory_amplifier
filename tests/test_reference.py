"""Frozen reference configs and the per-run diff against them (review H7).

    python -m unittest tests.test_reference -v
"""
import hashlib
import json
import unittest
from pathlib import Path

from tests import _paths
import config as C

ROOT = _paths.ROOT
FROZEN = {  # sha256 of the reference files: they must never be edited (a new recipe gets a new file and a new entry here)
    "gdn_fixed_v1": "6d8251d434a7765cbeddd5dd0fd803c8a16e7242d028790394c62193b039694b",
    "gdn2_v1": "57e8b32e19d2ce05fae3bb74512b67842fd5a604fae762a9ce1b3a5f92e10219",
    "swa_v1": "d9f7bbfcb1baf8db1969270e45c35f1a3800467756b85a0d9912e4a9a5823058",
    "swa_fb_v1": "bdaae668865cd05cc6d507b7303d5921cbdd94d6286b251f1e0679fe440fe6b5",
}


def cfg_of(config: str, *overrides):
    return C.load_config(ROOT / "configs" / config, list(overrides))[0]


class FrozenReferences(unittest.TestCase):
    def test_reference_files_are_unchanged(self):
        for name, sha in FROZEN.items():
            self.assertEqual(hashlib.sha256((ROOT / "configs" / "ref" / f"{name}.yaml").read_bytes()).hexdigest(), sha, f"{name} was edited")

    def test_every_model_kind_has_a_reference(self):
        self.assertEqual(set(C.REFERENCES.values()), set(FROZEN))
        for kind, name in C.REFERENCES.items():
            self.assertEqual(C.load_reference({"model": {"kind": kind}})["model"]["kind"], kind)
        with self.assertRaises(KeyError):
            C.reference_name({"model": {"kind": "no_such_model"}})


class Diff(unittest.TestCase):
    def test_the_reference_recipe_has_an_empty_diff(self):
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml")), {})
        self.assertEqual(C.describe_diff({}), "reference recipe")

    def test_changed_settings_are_listed_with_both_values(self):
        cfg = cfg_of("swa_fb.yaml", "model.gain_init=0", "train.steps=50000", "model.segment=32", "model.nonlin=none")
        self.assertEqual(C.reference_diff(cfg), {"model.gain_init": [0.1, 0], "model.nonlin": ["tanh", "none"], "model.segment": [64, 32], "train.steps": [20000, 50000]})
        self.assertEqual(C.describe_diff(C.reference_diff(cfg)), "gain_init 0.1 → 0, nonlin tanh → none, segment 64 → 32, steps 20000 → 50000")

    def test_segment_zero_means_one_per_window_so_an_explicit_window_sized_segment_is_no_difference(self):
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml", "model.segment=64")), {})  # review L10
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml", "model.segment=0")), {})
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml", "model.segment=8")), {"model.segment": [64, 8]})
        # a different window with segment left at 0 moves the segment with it: both entries are told
        d = C.reference_diff(cfg_of("swa_fb.yaml", "model.window=128"))
        self.assertEqual(d, {"model.window": [64, 128], "model.segment": [64, 128]})
        # plain SWA has no segment: nothing is added to its diff
        self.assertEqual(C.reference_diff(cfg_of("swa.yaml", "model.window=128")), {"model.window": [64, 128]})

    def test_seed_logging_and_evaluation_settings_are_not_differences(self):
        cfg = cfg_of("swa_fb.yaml", "seed=2", "experiment=other", "train.eval_every=100", "train.eval_ns=[32]", "train.eval_examples=5",
                     "train.eval_examples_by_n=[[4096,10]]", "train.log_every=7", "eval.tokens_per_batch=1", "eval.ns=[32]", "project=x")
        self.assertEqual(C.reference_diff(cfg), {})

    def test_settings_that_the_old_report_key_ignored_are_differences(self):
        for override, key in [("model.proj_std=0.5", "model.proj_std"), ("train.loop_lr_mult=1", "train.loop_lr_mult"), ("train.curriculum_stage_steps=0", "train.curriculum_stage_steps"),
                              ("model.kv_cache=true", "model.kv_cache"), ("train.lr=0.002", "train.lr"), ("train.batch_size=64", "train.batch_size"), ("data_dir=elsewhere", "data_dir")]:
            self.assertEqual(list(C.reference_diff(cfg_of("swa_fb.yaml", override))), [key], override)

    def test_a_key_the_stored_config_lacks_counts_as_the_code_default(self):
        old = cfg_of("swa.yaml")  # configs written before the curriculum existed have no curriculum key
        self.assertNotIn("curriculum_stage_steps", old["train"])
        self.assertEqual(C.reference_diff(old), {})
        new = cfg_of("swa.yaml", "train.curriculum_stage_steps=2000")
        self.assertEqual(C.reference_diff(new), {"train.curriculum_stage_steps": [0, 2000]})
        fb = cfg_of("swa_fb.yaml")  # no kv_cache key either: stored as the code default False
        self.assertNotIn("kv_cache", fb["model"])
        self.assertEqual(C.reference_diff(fb), {})
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml", "model.kv_cache=false")), {})


class StoredResults(unittest.TestCase):
    """Every results file carries the diff of its stored config, and it is the diff the code computes now."""

    def files(self):
        return sorted((ROOT / "results" / "raw").glob("*.json")) + sorted((ROOT / "results" / "explore").glob("*.json"))

    def test_every_result_file_is_stamped_and_current(self):
        stale = []
        for f in self.files():
            r = json.loads(f.read_text())
            if r.get("reference") != C.reference_name(r["config"]) or r.get("diff") != C.reference_diff(r["config"]):
                stale.append(f.name)
        self.assertEqual(stale, [], "run `python src/config.py results/raw/*.json results/explore/*.json` to stamp them")

    def test_runs_with_the_same_diff_differ_only_in_ignored_settings(self):
        groups: dict = {}
        for f in self.files():
            r = json.loads(f.read_text())
            groups.setdefault((r["reference"], json.dumps(r["diff"], sort_keys=True)), []).append(r["config"])
        for key, cfgs in groups.items():
            flat = [{k: v for k, v in C.flatten(c).items() if not C.is_ignored(k)} for c in cfgs]
            for k in set().union(*flat):
                self.assertEqual(len({json.dumps(f.get(k, C.CODE_DEFAULTS.get(k))) for f in flat}), 1, (key, k))


if __name__ == "__main__":
    unittest.main()
