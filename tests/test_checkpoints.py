"""The backed-up checkpoints in checkpoints/ are intact and are the models the results files describe (review L11).

The weights are local only (review K2: `checkpoints/*.pt` is git-ignored, the manifest is tracked), so on a fresh clone the manifest lists files that are not there:
those checkpoints are skipped, the checks run on the ones that exist.

    python -m unittest tests.test_checkpoints -v
"""
import hashlib
import json
import unittest

import torch

from tests import _paths
import data as D
import evaluate as E
from models import build_from_config

ROOT = _paths.ROOT
MANIFEST = json.loads((ROOT / "checkpoints" / "manifest.json").read_text())
DATA = ROOT / "data_cache" / "passkey_swar"
PRESENT = {n: m for n, m in MANIFEST.items() if (ROOT / m["file"]).exists()}   # the checkpoints this checkout has


def load(name):
    cfg = json.loads((ROOT / "results" / "raw" / f"{name}.json").read_text())["config"]
    model = build_from_config(cfg["model"], 40)
    model.load_state_dict(torch.load(ROOT / MANIFEST[name]["file"], map_location="cpu", weights_only=True))
    return model.eval()


class Checkpoints(unittest.TestCase):
    def test_the_manifest_lists_the_best_checkpoints_and_their_files_match_the_hashes(self):
        self.assertIn("swa_fb_gi0_nlnone_seg8_steps50000_s2", MANIFEST)
        self.assertIn("swa_fb_gi0_seg8_nlnone_s0", MANIFEST)
        self.assertIn("swa_2pass_seg32_po1_s5", MANIFEST)
        for name, m in PRESENT.items():
            f = ROOT / m["file"]
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), m["sha256"], name)
            self.assertEqual(f.stat().st_size, m["bytes"], name)

    def test_each_checkpoint_loads_into_the_model_of_its_stored_config_and_runs(self):
        for name, m in PRESENT.items():
            model = load(name)
            self.assertEqual(sum(p.numel() for p in model.parameters()), m["n_params"], name)
            if torch.cuda.is_available():  # the attention and rotary kernels are Triton kernels: no forward pass on the CPU
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model.cuda()(input_ids=torch.randint(0, 40, (2, 96), device="cuda")).logits
                self.assertEqual(tuple(logits.shape), (2, 96, 40))
                self.assertTrue(torch.isfinite(logits.float()).all(), name)

    def test_the_manifest_numbers_are_the_results_file_numbers(self):
        for name, m in MANIFEST.items():   # needs the results files, not the weights
            res = json.loads((ROOT / "results" / "raw" / f"{name}.json").read_text())
            test = {r["N"]: r for r in res["rows"] if r["split"] == "test"}
            ns = [n for n in (256, 1024, 4096, 16384) if n in test]
            self.assertEqual(m["test_passkey"], {str(n): round(test[n]["passkey_acc"], 3) for n in ns}, name)
            self.assertEqual(m["test_ar_valid"], {str(n): round(test[n]["ar_valid_acc"], 3) for n in ns}, name)

    def test_the_capacity_stages_are_backed_up(self):
        """The P = 8 stages of the six budget-chain seeds and the P = 16 stage of seed 1 (all 16 positions) are in the manifest (review K2)."""
        stem = "swa_2pass_seg32_po1_pk{}_steps50000_lr0.001_warmup_steps500_cur2000_s{}"
        for s in range(6):
            self.assertIn(stem.format(8, s), MANIFEST)
        self.assertIn(stem.format(16, 1), MANIFEST)

    @unittest.skipUnless(torch.cuda.is_available() and (DATA / "n1024" / "validation.parquet").exists() and "swa_fb_gi0_nlnone_seg8_steps50000_s2" in PRESENT, "needs a GPU, the generated dataset and the checkpoint")
    def test_the_best_checkpoint_reproduces_its_validation_accuracy_on_fresh_evaluation(self):
        name = "swa_fb_gi0_nlnone_seg8_steps50000_s2"
        model = load(name).cuda()
        res = json.loads((ROOT / "results" / "raw" / f"{name}.json").read_text())
        for n in (256, 1024):
            p = D.load_packed(DATA / f"n{n}" / "validation.parquet", kinds=True).subset(60)
            got = E.evaluate_packed(model, p, torch.device("cuda"), amp_dtype=torch.bfloat16)["passkey_acc"]
            stored = next(r["passkey_acc"] for r in res["rows"] if r["split"] == "validation" and r["N"] == n)
            self.assertAlmostEqual(got, stored, delta=0.07, msg=n)  # 60 of the 1000 streams: sampling noise only


if __name__ == "__main__":
    unittest.main()
