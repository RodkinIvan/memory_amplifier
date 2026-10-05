#!/usr/bin/env python
"""Copy the weights of chosen runs from runs/<name>/model.pt (git-ignored) to checkpoints/<name>.pt (tracked) and record them in checkpoints/manifest.json.

    python scripts/backup_checkpoints.py swa_fb_gi0_nlnone_seg8_steps50000_s2 swa_fb_gi0_seg8_nlnone_s0

The manifest has, per checkpoint, the sha256 of the file, the parameter count, the run's config diff against its frozen reference and its test passkey at
N = 256 / 1,024 / 4,096 / 16,384, so a checkpoint can be checked and reloaded without the runs/ folder (review L11):

    import json, torch; from models import build_from_config
    cfg = json.load(open("results/raw/<name>.json"))["config"]; model = build_from_config(cfg["model"], 40)
    model.load_state_dict(torch.load("checkpoints/<name>.pt", map_location="cpu", weights_only=True))
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup(names: list[str], root: Path = ROOT) -> dict:
    manifest_file = root / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    (root / "checkpoints").mkdir(exist_ok=True)
    for name in names:
        src, res = root / "runs" / name / "model.pt", json.loads((root / "results" / "raw" / f"{name}.json").read_text())
        dst = root / "checkpoints" / f"{name}.pt"
        shutil.copy2(src, dst)
        test = {r["N"]: r for r in res["rows"] if r["split"] == "test"}
        manifest[name] = {"file": f"checkpoints/{name}.pt", "sha256": sha256(dst), "bytes": dst.stat().st_size, "n_params": res["n_params"], "diff_vs_reference": res.get("diff"),
                          "test_passkey": {str(n): round(test[n]["passkey_acc"], 3) for n in (256, 1024, 4096, 16384)},
                          "test_ar_valid_at_16384": round(test[16384]["ar_valid_acc"], 3)}
    manifest_file.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for k, v in backup(sys.argv[1:]).items():
        print(k, v["sha256"][:12], v["test_passkey"])
