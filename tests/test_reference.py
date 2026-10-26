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
    "swa_va_v1": "c8e9425c5d5a4c8012128e1a0bd36f1374ce5cd94f47cadf8feb9c4870d65154",
    "swa_2pass_v1": "510a4dff03154811a4a1926adeeaecbdee07dc86ca03697a0ea71e839ea1ea2c",
}


def cfg_of(config: str, *overrides):
    return C.load_config(ROOT / "configs" / config, list(overrides))[0]


def old_fb(*overrides):
    """A swa_fb config as the runs from before E7 stored it: the working config without its `kv_cache: true` (E7 added that key, so a run of the working config differs from the reference by it)."""
    cfg = C.load_config(ROOT / "configs" / "swa_fb.yaml", [])[0]
    del cfg["model"]["kv_cache"]
    for o in overrides:
        k, _, v = o.partition("=")
        C.set_nested(cfg, k, C.parse_value(v))
    return cfg


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
        self.assertEqual(C.reference_diff(old_fb()), {})
        self.assertEqual(C.describe_diff({}), "reference recipe")

    def test_the_working_loop_config_differs_from_its_reference_by_the_cache_path_only(self):
        """E7: swa_fb.yaml states `kv_cache: true` (the only path), the frozen reference has no such key (= false), so new runs carry `kv_cache False -> True` and their change lists stay explicit."""
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml")), {"model.kv_cache": [False, True]})
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml", "model.segment=8")), {"model.kv_cache": [False, True], "model.segment": [64, 8]})
        for key in ("noise", "passkey_weight"):   # the removed options are gone from the working config (the frozen reference keeps them)
            self.assertNotIn(key, {**cfg_of("swa_fb.yaml")["model"], **cfg_of("swa_fb.yaml")["train"]})
        self.assertEqual(C.load_reference({"model": {"kind": "swa_fb"}})["model"]["noise"], 0.0)

    def test_a_removed_kind_keeps_its_reference_so_stored_results_still_have_a_diff(self):
        """The value-adaptation code is gone (E7) but its frozen reference stays: the stored results of the archive are still stamped and grouped by it."""
        ref = C.load_reference({"model": {"kind": "swa_va"}})
        self.assertEqual(C.reference_diff(ref), {})
        self.assertEqual(C.reference_diff({**ref, "model": {**ref["model"], "segment": 8}}), {"model.segment": [32, 8]})
        self.assertEqual(C.reference_diff({**ref, "model": {**ref["model"], "segment": 0}}), {"model.segment": [32, 64]})  # 0 = one per window
        self.assertFalse((ROOT / "configs" / "swa_va.yaml").exists())

    def test_the_two_pass_recipe_and_its_choices(self):
        """E3: the working config swa_2pass.yaml has pass_output 1; the frozen reference swa_2pass_v1 still has 2, so a run of the working config differs from it in exactly that."""
        self.assertEqual(C.reference_diff(cfg_of("swa_2pass.yaml")), {"model.pass_output": [2, 1]})
        self.assertEqual(C.load_reference({"model": {"kind": "swa_2pass"}})["model"]["pass_output"], 2)   # the reference was not edited
        self.assertEqual(cfg_of("swa_2pass.yaml")["model"]["pass_output"], 1)
        self.assertEqual(C.reference_diff(cfg_of("swa_2pass.yaml", "model.pass_output=1", "seed=2")), {"model.pass_output": [2, 1]})
        self.assertEqual(C.reference_diff(cfg_of("swa_2pass.yaml", "model.pass_output=2")), {})
        self.assertEqual(C.reference_diff(cfg_of("swa_2pass.yaml", "model.segment=64", "model.gain_init=0.1")), {"model.segment": [32, 64], "model.gain_init": [0.0, 0.1], "model.pass_output": [2, 1]})
        self.assertEqual(C.reference_diff(cfg_of("swa_2pass.yaml", "model.segment=0")), {"model.segment": [32, 64], "model.pass_output": [2, 1]})  # 0 = one per window
        cfg, keys = C.load_config(ROOT / "configs" / "swa_2pass.yaml", ["model.pass_output=1", "model.segment=64", "seed=1"])
        self.assertEqual(C.run_name(cfg, keys), "swa_2pass_po1_seg64_s1")
        # the working config and the reference are the same recipe apart from the output pass (a key one lacks counts as the code default)
        a, b = {**C.CODE_DEFAULTS, **C.flatten(cfg_of("swa_2pass.yaml"))}, {**C.CODE_DEFAULTS, **C.flatten(C.load_reference({"model": {"kind": "swa_2pass"}}))}
        self.assertEqual({k for k in set(a) | set(b) if a.get(k) != b.get(k)}, {"model.pass_output"})
        # the first-round runs launched with an explicit pass_output keep their config (and so their finished-result match) when the default changes
        first_round, _ = C.load_config(ROOT / "configs" / "swa_2pass.yaml", ["model.segment=32", "model.pass_output=2", "seed=0"])
        self.assertEqual(first_round["model"]["pass_output"], 2)

    def test_changed_settings_are_listed_with_both_values(self):
        cfg = old_fb("model.gain_init=0", "train.steps=50000", "model.segment=32", "model.nonlin=none")
        self.assertEqual(C.reference_diff(cfg), {"model.gain_init": [0.1, 0], "model.nonlin": ["tanh", "none"], "model.segment": [64, 32], "train.steps": [20000, 50000]})
        self.assertEqual(C.describe_diff(C.reference_diff(cfg)), "gain_init 0.1 → 0, nonlin tanh → none, segment 64 → 32, steps 20000 → 50000")

    def test_segment_zero_means_one_per_window_so_an_explicit_window_sized_segment_is_no_difference(self):
        self.assertEqual(C.reference_diff(old_fb("model.segment=64")), {})  # review L10
        self.assertEqual(C.reference_diff(old_fb("model.segment=0")), {})
        self.assertEqual(C.reference_diff(old_fb("model.segment=8")), {"model.segment": [64, 8]})
        # a different window with segment left at 0 moves the segment with it: both entries are told
        d = C.reference_diff(old_fb("model.window=128"))
        self.assertEqual(d, {"model.window": [64, 128], "model.segment": [64, 128]})
        # plain SWA has no segment: nothing is added to its diff
        self.assertEqual(C.reference_diff(cfg_of("swa.yaml", "model.window=128")), {"model.window": [64, 128]})

    def test_seed_logging_and_evaluation_settings_are_not_differences(self):
        cfg = old_fb("seed=2", "experiment=other", "train.eval_every=100", "train.eval_ns=[32]", "train.eval_examples=5",
                     "train.eval_examples_by_n=[[4096,10]]", "train.log_every=7", "eval.tokens_per_batch=1", "eval.ns=[32]", "project=x")
        self.assertEqual(C.reference_diff(cfg), {})

    def test_settings_that_the_old_report_key_ignored_are_differences(self):
        for override, key in [("model.proj_std=0.5", "model.proj_std"), ("train.loop_lr_mult=1", "train.loop_lr_mult"), ("train.curriculum_stage_steps=0", "train.curriculum_stage_steps"),
                              ("model.kv_cache=true", "model.kv_cache"), ("train.lr=0.002", "train.lr"), ("train.batch_size=64", "train.batch_size"), ("data_dir=elsewhere", "data_dir")]:
            self.assertEqual(list(C.reference_diff(old_fb(override))), [key], override)

    def test_a_key_the_stored_config_lacks_counts_as_the_code_default(self):
        old = cfg_of("swa.yaml")  # configs written before the curriculum existed have no curriculum key
        self.assertNotIn("curriculum_stage_steps", old["train"])
        self.assertEqual(C.reference_diff(old), {})
        new = cfg_of("swa.yaml", "train.curriculum_stage_steps=2000")
        self.assertEqual(C.reference_diff(new), {"train.curriculum_stage_steps": [0, 2000]})
        fb = old_fb()  # a run from before E7 has no kv_cache key either: stored as the code default False
        self.assertNotIn("kv_cache", fb["model"])
        self.assertEqual(C.reference_diff(fb), {})
        self.assertEqual(C.reference_diff(cfg_of("swa_fb.yaml", "model.kv_cache=false")), {})


class SameConfig(unittest.TestCase):
    """Review N1: a stored config that still has a key E7 dropped from the working YAMLs (`train.passkey_weight: 1.0`) is the same config as today's one without it."""

    def test_a_missing_key_equals_its_code_default_and_nothing_else(self):
        a = {"model": {"kind": "swa_2pass"}, "train": {"steps": 5, "passkey_weight": 1.0, "passkey_weight_anneal_steps": 0}}
        b = {"model": {"kind": "swa_2pass"}, "train": {"steps": 5}}
        self.assertTrue(C.same_config(a, b) and C.same_config(b, a))
        self.assertEqual(C.config_difference(a, b), {})
        self.assertFalse(C.same_config({**a, "train": {**a["train"], "passkey_weight": 10.0}}, b))                       # another value is another config
        self.assertEqual(C.config_difference({**a, "train": {**a["train"], "passkey_weight": 10.0}}, b), {"train.passkey_weight": [10.0, 1.0]})
        self.assertFalse(C.same_config({**b, "train": {"steps": 6}}, b))
        self.assertFalse(C.same_config({**b, "extra_key_without_a_default": 1}, b))                                      # a key with no default is a real difference
        # the seed, the logging and the evaluation settings count: a stored result belongs to exactly the config that was run
        self.assertFalse(C.same_config({**b, "seed": 1}, {**b, "seed": 0}))

    def test_an_old_swa_fb_config_without_the_cache_key_still_differs_from_one_that_says_true(self):
        old, new = old_fb(), cfg_of("swa_fb.yaml")
        self.assertEqual(C.config_difference(old, new), {"model.kv_cache": [False, True]})      # a missing key is false: the runs differ in the cache path
        self.assertTrue(C.same_config(old, cfg_of("swa_fb.yaml", "model.kv_cache=false")))

    def test_every_stored_result_of_a_kind_whose_yaml_is_alive_is_the_config_that_launched_it_today(self):
        """Rebuild each stored result's launch config from today's YAML and the keys it overrode: equal for gdn, swa and swa_2pass; for swa_fb the only difference is kv_cache (E7 added it)."""
        by_kind = {"gdn_fixed": "exp1.yaml", "gdn2": "gdn2.yaml", "swa": "swa.yaml", "swa_2pass": "swa_2pass.yaml", "swa_fb": "swa_fb.yaml"}
        checked = {k: 0 for k in by_kind}
        for f in sorted((ROOT / "results" / "raw").glob("*.json")):
            r = json.loads(f.read_text())
            kind = r["config"]["model"]["kind"]
            if kind not in by_kind or "explicit_keys" not in r:
                continue
            overrides = [f"{k}={json.dumps(C.get_nested(r['config'], k))}" for k in r["explicit_keys"] if k != "seed"] + [f"seed={r['config']['seed']}"]
            rebuilt, _ = C.load_config(ROOT / "configs" / by_kind[kind], overrides)
            diff = C.config_difference(r["config"], rebuilt)
            if kind == "swa_fb":   # E7 added kv_cache; the evaluation batch of swa_fb.yaml was lowered after the first sweeps (out-of-memory on the shared GPU)
                diff.pop("model.kv_cache", None)
                diff.pop("eval.tokens_per_batch", None)
            self.assertEqual(diff, {}, (f.name, diff))
            checked[kind] += 1
        self.assertGreater(checked["swa_2pass"], 20)
        self.assertGreater(checked["swa_fb"], 20)


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
