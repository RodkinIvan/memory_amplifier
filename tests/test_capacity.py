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


def fake_result(exact256: float, token256: float = None, pos256=None, history=None, train_size=100_000) -> dict:
    rows = []
    for split in ("validation", "test"):
        for n in CAP.NS:
            e = exact256 if n == 256 else 1.0
            row = {"split": split, "N": n, "passkey_acc": e, "passkey_token_acc": token256 if (n == 256 and token256 is not None) else e, "ar_valid_acc": 0.99}
            if pos256 is not None:
                row["passkey_pos_acc"] = list(pos256) if n == 256 else [1.0] * len(pos256)
            rows.append(row)
    res = {"rows": rows, "train_time_s": 3600.4, "config": {"train": {"steps": 10000, "lr": 0.0003, "warmup_steps": 200, "curriculum_stage_steps": 0}}, "val_history": history or []}
    if train_size is not None:
        res["dataset"] = {"dir": "passkey_swar_pk8", "sizes": {"train": train_size, "validation": 1000, "test": 1000}}
    return res


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


POS = [1.0, 0.04, 1.0, 1.0, 0.03, 1.0, 1.0, 0.06]  # the pattern found in the first P = 8 stage: every third position is at chance


class PerPosition(unittest.TestCase):
    """Review C1: a stage that fails must be readable position by position."""

    def test_the_entry_has_the_accuracy_of_every_position_and_how_many_are_at_0_9(self):
        e = CAP.summarize(8, "run8", INIT, fake_result(0.0, 0.645, POS))
        self.assertEqual(e["validation"]["positions"]["256"], POS)
        self.assertEqual(e["validation"]["positions"]["32"], [1.0] * 8)
        self.assertEqual(e["validation"]["positions_ok"], {"32": 8, "64": 8, "128": 8, "256": 5})
        self.assertEqual(e["test"]["positions_ok"]["256"], 5)
        self.assertFalse(e["passed"])  # the stop rule is still the exact match at N = 256

    def test_a_result_from_before_the_metric_existed_gives_empty_positions_instead_of_failing(self):
        e = CAP.summarize(8, "old", INIT, fake_result(0.95))
        self.assertEqual((e["validation"]["positions"], e["validation"]["positions_ok"]), ({}, {}))

    def test_the_step_at_which_each_position_switched_on_comes_from_the_validation_history(self):
        hist = [{"step": 1000, "val/n256/passkey_pos_acc": [0.9, 0.05, 0.3]}, {"step": 2000, "val/n256/passkey_pos_acc": [1.0, 0.4, 0.95]},
                {"step": 3000, "val/n256/passkey_pos_acc": [1.0, 0.6, 1.0]}]
        self.assertEqual(CAP.position_switch_on(hist), [1000, None, 2000])  # position 2 never reached 0.9
        self.assertEqual(CAP.position_switch_on([{"step": 1, "val/n32/passkey_acc": 1.0}]), [])
        e = CAP.summarize(3, "r", INIT, fake_result(0.5, 0.8, [1.0, 0.6, 1.0], history=hist))
        self.assertEqual(e["switch_on_step"], [1000, None, 2000])


class Records(unittest.TestCase):
    def test_the_training_set_size_of_a_stage_is_in_its_entry_and_the_recipe_too(self):
        self.assertEqual(CAP.summarize(8, "r", INIT, fake_result(0.95, train_size=100_000))["train_size"], 100_000)
        self.assertIsNone(CAP.summarize(8, "r", INIT, fake_result(0.95, train_size=None))["train_size"])
        self.assertEqual(CAP.summarize(8, "r", INIT, fake_result(0.95))["recipe"], {"lr": 0.0003, "warmup_steps": 200, "curriculum_stage_steps": 0})

    def test_a_dataset_with_another_training_size_is_refused_not_used(self):
        """Review C2: the data of a stage has the base dataset's sizes; an old 30,000-stream folder must not be picked up silently."""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "data_cache" / "passkey_swar_pk8"  # where data_dir_for puts it under the (patched) repository root
            base.mkdir(parents=True)
            (base / "gen_meta.json").write_text(json.dumps({"sizes": {"train": 30000, "validation": 1000, "test": 1000}}))
            with mock.patch.object(C, "ROOT", Path(d)), self.assertRaises(SystemExit) as cm:
                CAP.ensure_dataset(8)
            self.assertIn("30000", str(cm.exception))
            (base / "gen_meta.json").write_text(json.dumps({"sizes": {"train": CAP.TRAIN_SIZE, "validation": 1000, "test": 1000}}))
            with mock.patch.object(C, "ROOT", Path(d)):
                self.assertEqual(CAP.ensure_dataset(8), base)
        self.assertEqual(CAP.TRAIN_SIZE, 100_000)

    def test_two_budget_runs_finishing_together_both_end_up_in_the_file(self):
        import threading
        with tempfile.TemporaryDirectory() as d, mock.patch.object(CAP, "OUT", Path(d) / "capacity.json"):
            ts = [threading.Thread(target=CAP.write_capacity, args=({},), kwargs={"budget_run": {"run": f"seed{i}"}}) for i in range(12)]
            [x.start() for x in ts]
            [x.join() for x in ts]
            data = json.loads(CAP.OUT.read_text())
            self.assertEqual(sorted(e["run"] for e in data["budget_runs"]), sorted(f"seed{i}" for i in range(12)))
            CAP.write_capacity({}, budget_run={"run": "seed3", "again": True})  # the same run replaces its entry
            self.assertEqual(len(json.loads(CAP.OUT.read_text())["budget_runs"]), 12)

    def test_the_chain_and_a_budget_run_share_one_file_without_overwriting_each_other(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(CAP, "OUT", Path(d) / "capacity.json"):
            CAP.write_capacity({"stages": [{"passkey_len": 8}]})
            CAP.write_capacity({"budget_runs": [{"run": "b"}]})
            CAP.write_capacity({"stages": [{"passkey_len": 8}, {"passkey_len": 16}]})
            data = json.loads(CAP.OUT.read_text())
        self.assertEqual([s["passkey_len"] for s in data["stages"]], [8, 16])
        self.assertEqual(data["budget_runs"], [{"run": "b"}])
        self.assertEqual(data["threshold"], 0.9)

    def test_a_re_evaluation_keeps_an_entry_that_another_process_wrote_meanwhile(self):
        """Review L12: `reevaluate` reads the file, evaluates for minutes, then writes: an entry added in between (a budget run or a chain stage finishing) must survive."""
        entry = lambda run: {"run": run, "validation": {"exact": {"256": 1.0}}, "test": {"exact": {"256": 1.0}}}  # noqa: E731
        with tempfile.TemporaryDirectory() as d, mock.patch.object(CAP, "OUT", Path(d) / "capacity.json"):
            CAP.write_capacity({"stages": [entry("old")]})
            snapshot = json.loads(CAP.OUT.read_text())  # what reevaluate saw when it started
            CAP.write_capacity({}, budget_run=entry("finished-meanwhile"))  # another process finishes during the evaluation
            found = {"validation": {"positions": {"256": [1.0, 0.1]}, "positions_ok": {"256": 1}}, "test": {"positions": {"256": [1.0, 0.0]}, "positions_ok": {"256": 1}}}
            CAP.merge_positions({"old": found})
            data = json.loads(CAP.OUT.read_text())
        self.assertEqual(snapshot["stages"][0]["run"], "old")
        self.assertEqual([e["run"] for e in data["budget_runs"]], ["finished-meanwhile"])  # not lost
        self.assertEqual(data["stages"][0]["validation"]["positions_ok"], {"256": 1})      # and the re-evaluated fields are there
        self.assertEqual(data["stages"][0]["validation"]["exact"], {"256": 1.0})           # existing fields are kept
        self.assertNotIn("positions", data["budget_runs"][0]["validation"])                # entries without an update stay as they are

    def test_two_drivers_needing_the_same_new_dataset_generate_it_once(self):
        """Review L13: the second driver waits on the folder's lock and then finds the dataset."""
        import sys
        import threading
        import time
        calls = []

        def slow_generate(argv):
            calls.append(argv)
            time.sleep(0.3)  # long enough for the second thread to arrive while this one is generating
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "gen_meta.json").write_text(json.dumps({"sizes": {"train": CAP.TRAIN_SIZE}}))

        fake_G = mock.Mock(main=slow_generate)
        fake_S = mock.Mock(run=lambda *a, **k: True)
        results = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(C, "ROOT", Path(d)), mock.patch.dict(sys.modules, {"make_passkey_swar": fake_G, "sanity_checks": fake_S}):
            ts = [threading.Thread(target=lambda: results.append(CAP.ensure_dataset(16))) for _ in range(2)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])


class ValueAdaptationStage(unittest.TestCase):
    """The same experiment for the value-adaptation model: another config, its own result keys, the default names of the first model unchanged."""

    def test_a_value_adaptation_stage_fine_tunes_the_segment_32_model_with_its_own_config(self):
        ov = CAP.stage_overrides(8, "runs/x/model.pt", 50000, 1e-3, 1, 500, 2000, model="swa_va")
        cfg, explicit = C.load_config(_paths.ROOT / CAP.MODELS["swa_va"]["config"], ov)
        self.assertEqual((cfg["model"]["kind"], cfg["model"]["segment"], cfg["passkey_len"], cfg["train"]["init_from"]), ("swa_va", 32, 8, "runs/x/model.pt"))
        self.assertEqual((cfg["experiment"], cfg["train"]["train_ns"], cfg["eval"]["ns"], cfg["train"]["cuda_graph"]), ("swa_va_capacity", [32, 64, 128, 256], [32, 64, 128, 256], True))
        self.assertNotIn("model.nonlin", ov)  # no loop nonlinearity in this model
        self.assertEqual(CAP.stage_overrides(8, "i", 1, 1e-3, model="swa_va", segment=8)[2], "model.segment=8")
        name = CAP.stage_name(8, "runs/x/model.pt", 50000, 1e-3, 1, 500, 2000, model="swa_va")
        self.assertTrue(name.startswith("swa_va_seg32_pk8_"), name)
        self.assertIn("steps50000", name)

    def test_the_first_models_names_and_keys_are_unchanged(self):
        self.assertTrue(CAP.stage_name(8, INIT, 10000, 3e-4).startswith("swa_fb_seg8_nlnone_gi0_pk8_steps10000"))
        self.assertEqual(CAP.stage_overrides(8, INIT, 10000, 3e-4)[:5], ["experiment=swa_fb_capacity", "seed=0", "model.segment=8", "model.nonlin=none", "model.gain_init=0"])
        self.assertEqual(CAP.result_keys("swa_fb"), ("stages", "budget_runs"))
        self.assertEqual(CAP.result_keys("swa_va"), ("stages_swa_va", "budget_runs_swa_va"))

    def test_results_of_two_models_share_the_file_without_overwriting_each_other(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(CAP, "OUT", Path(d) / "capacity.json"):
            CAP.write_capacity({}, budget_run={"run": "fb-seed0"}, budget_key=CAP.result_keys("swa_fb")[1])
            CAP.write_capacity({}, budget_run={"run": "va-seed0"}, budget_key=CAP.result_keys("swa_va")[1])
            CAP.write_capacity({"stages_swa_va": [{"passkey_len": 8}]})
            data = json.loads(CAP.OUT.read_text())
        self.assertEqual([e["run"] for e in data["budget_runs"]], ["fb-seed0"])
        self.assertEqual([e["run"] for e in data["budget_runs_swa_va"]], ["va-seed0"])
        self.assertEqual(data["stages_swa_va"], [{"passkey_len": 8}])

    def test_the_chain_writes_under_the_models_key_and_a_re_evaluation_reaches_every_group(self):
        entry = lambda run: {"run": run, "validation": {}, "test": {}, "passkey_len": 8, "passed": False}  # noqa: E731
        fake = lambda P, init: (dict(entry("r"), validation={"exact": {"256": 0.1}, "positions_ok": {}}), "w")  # noqa: E731
        with tempfile.TemporaryDirectory() as d, mock.patch.object(CAP, "OUT", Path(d) / "capacity.json"):
            CAP.chain(fake, 8, 64, "init", key="stages_swa_va")
            data = json.loads(CAP.OUT.read_text())
            self.assertIn("stages_swa_va", data)
            self.assertNotIn("stages", data)
            CAP.merge_positions({"r": {"validation": {"positions": {"256": [1.0]}}}})
            self.assertEqual(json.loads(CAP.OUT.read_text())["stages_swa_va"][0]["validation"]["positions"], {"256": [1.0]})


class TwoPassStage(unittest.TestCase):
    """The same experiment for the two-pass layer, from its best checkpoint."""

    def test_a_two_pass_stage_fine_tunes_the_segment_32_output_1_model(self):
        ov = CAP.stage_overrides(8, "checkpoints/swa_2pass_seg32_po1_s5.pt", 50000, 1e-3, 0, 500, 2000, model="swa_2pass")
        cfg, explicit = C.load_config(_paths.ROOT / CAP.MODELS["swa_2pass"]["config"], ov)
        self.assertEqual((cfg["model"]["kind"], cfg["model"]["segment"], cfg["model"]["pass_output"], cfg["passkey_len"]), ("swa_2pass", 32, 1, 8))
        self.assertEqual((cfg["experiment"], cfg["train"]["init_from"], cfg["train"]["train_ns"], cfg["eval"]["ns"], cfg["train"]["cuda_graph"]),
                         ("swa_2pass_capacity", "checkpoints/swa_2pass_seg32_po1_s5.pt", [32, 64, 128, 256], [32, 64, 128, 256], True))
        name = CAP.stage_name(8, "checkpoints/swa_2pass_seg32_po1_s5.pt", 50000, 1e-3, 0, 500, 2000, model="swa_2pass")
        self.assertTrue(name.startswith("swa_2pass_seg32_po1_pk8_"), name)   # the model, its segment and output pass and the passkey length are in the name
        self.assertEqual(C.footprint_gb(cfg), 7.0)
        # the budget recipe (50k steps, lr 1e-3, warm-up 500, curriculum 2000) is the reference recipe, so only the passkey, the start weights and the output pass differ from it
        self.assertEqual(C.reference_diff(cfg), {"model.pass_output": [2, 1], "passkey_len": [4, 8], "train.init_from": [None, "checkpoints/swa_2pass_seg32_po1_s5.pt"]})
        short = C.load_config(_paths.ROOT / "configs" / "swa_2pass.yaml", CAP.stage_overrides(8, "x.pt", 10000, 3e-4, 0, 200, 0, model="swa_2pass"))[0]
        for key in ("train.steps", "train.lr", "train.warmup_steps", "train.curriculum_stage_steps"):
            self.assertIn(key, C.reference_diff(short))  # the short fine-tune of the first chain differs from it in the schedule

    def test_each_model_has_its_best_checkpoint_as_the_default_and_its_own_result_keys(self):
        self.assertEqual(CAP.MODELS["swa_fb"]["init"], "checkpoints/swa_fb_gi0_nlnone_seg8_steps50000_s2.pt")
        self.assertEqual(CAP.MODELS["swa_2pass"]["init"], "checkpoints/swa_2pass_seg32_po1_s5.pt")
        self.assertIsNone(CAP.MODELS["swa_va"]["init"])  # no checkpoint of value adaptation is good enough to start from: --init is required
        for m in CAP.MODELS.values():
            if m["init"]:
                self.assertTrue((_paths.ROOT / m["init"]).exists(), m["init"])
        self.assertEqual(CAP.result_keys("swa_2pass"), ("stages_swa_2pass", "budget_runs_swa_2pass"))
        self.assertEqual(CAP.result_keys("swa_2pass", "budget"), ("stages_swa_2pass_budget", "budget_runs_swa_2pass_budget"))   # two chains of one model with different recipes
        self.assertEqual(CAP.result_keys("swa_fb", "x"), ("stages_x", "budget_runs_x"))
        self.assertEqual(CAP.result_keys("swa_fb"), ("stages", "budget_runs"))

    def test_the_checkpoint_loads_into_the_model_a_stage_builds(self):
        import json
        import torch
        from models import build_from_config
        cfg, _ = C.load_config(_paths.ROOT / "configs" / "swa_2pass.yaml", CAP.stage_overrides(8, "x", 1, 1e-3, model="swa_2pass"))
        vocab = json.loads((C.data_dir_for(cfg) / "vocab.json").read_text())["size"]  # the passkey-8 dataset has the same 40 tokens
        model = build_from_config(cfg["model"], vocab)
        state = torch.load(_paths.ROOT / CAP.MODELS["swa_2pass"]["init"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)  # raises on any missing or unexpected key
        self.assertEqual(sum(p.numel() for p in model.parameters()), 530304)


class Defaults(unittest.TestCase):
    def test_the_chain_runs_on_gpu_0_and_waits_for_free_memory_before_a_stage(self):
        """Review C3."""
        import io, contextlib
        with mock.patch("sys.argv", ["x", "--help"]), contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            CAP.main()
        text = out.getvalue()
        self.assertIn("--gpu GPU", text)
        self.assertRegex(" ".join(text.split()), r"--gpu GPU.*?(?=--start)")
        self.assertEqual(CAP.argparse.ArgumentParser.__name__, "ArgumentParser")
        self.assertNotIn("--gpu 1", CAP.__doc__)
        waits = []
        with mock.patch.object(CAP, "free_gb", side_effect=[3.0, 5.0, 9.0]), mock.patch.object(CAP.time, "sleep", side_effect=lambda s: waits.append(s)), mock.patch("builtins.print"):
            CAP.wait_for_memory(0, 8.0, poll=7)
        self.assertEqual(waits, [7, 7])  # two readings below 8 GB, then it starts

    def test_the_budget_stage_is_the_original_recipe(self):
        cfg, _ = C.load_config(_paths.ROOT / "configs" / "swa_fb.yaml", CAP.stage_overrides(8, INIT, 50000, 1e-3, 0, 500, 2000))
        self.assertEqual((cfg["train"]["steps"], cfg["train"]["lr"], cfg["train"]["warmup_steps"], cfg["train"]["curriculum_stage_steps"]), (50000, 1e-3, 500, 2000))
        self.assertEqual(cfg["train"]["eval_every"], 1000)
        self.assertEqual(cfg["train"]["eval_ns"], [32, 64, 128, 256])
        self.assertIn("steps50000_lr0.001_warmup_steps500_cur2000", CAP.stage_name(8, INIT, 50000, 1e-3, 0, 500, 2000))


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
