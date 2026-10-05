"""Evaluation on passkey_swar: passkey exact match, AR valid / expired accuracy (expired split stale vs never-written).

Teacher forcing: the inputs contain the true answers, the model predicts each supervised token from its prefix.
For the passkey, "all 4 tokens correct under teacher forcing" is equivalent to greedy free-running generation
being correct (by induction on the first wrong token), so it is the exact-match accuracy.
"""
from __future__ import annotations

import numpy as np
import torch

import data as D

PASSKEY, VALID, STALE, NEVER = 0, 1, 2, 3  # indices into D.KIND_NAMES


def eval_plan(lengths: np.ndarray, tokens_per_batch: int) -> tuple[int, int]:
    """(padded length L, batch size B) used for *every* batch of a set of examples.

    Triton compiles / autotunes once per input shape (10-25 s each), so all batches of a config get the identical shape:
    lengths within a config differ by a few percent, so padding everything to the longest one costs almost nothing.
    Long sequences are padded to a multiple of 4096, short ones to 256, which keeps the set of shapes small across configs.
    """
    longest = int(lengths.max())
    mult = 4096 if longest > 4096 else 256
    L = -(-longest // mult) * mult
    return L, max(1, tokens_per_batch // L)


@torch.no_grad()
def evaluate_packed(model, p: D.Packed, device, tokens_per_batch: int = 262144, amp_dtype=None) -> dict:
    """Returns accuracies plus counts for one set of examples (needs p.kinds). `amp_dtype` e.g. torch.bfloat16 enables autocast."""
    assert p.kinds is not None, "load with kinds=True"
    was_training = model.training
    model.eval()
    n_ok = np.zeros(4, np.int64)  # correct positions per kind
    n_tot = np.zeros(4, np.int64)
    passkey_ok = 0
    L, B = eval_plan(p.lengths(), tokens_per_batch)
    for start in range(0, len(p), B):
        idx = np.arange(start, min(start + B, len(p)))
        n_real = len(idx)
        if n_real < B <= len(p):  # filler rows (repeats of the last example) keep the batch shape; their results are ignored
            idx = np.concatenate([idx, np.full(B - n_real, idx[-1])])
        ids, labs = D.pad_batch(p, idx, L)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = model(input_ids=torch.from_numpy(ids).to(device)).logits
        tgt = torch.from_numpy(labs[:, 1:]).to(device)
        mask = tgt != D.IGNORE
        correct = ((logits[:, :-1].argmax(-1) == tgt) & mask).cpu().numpy()
        mask = mask.cpu().numpy()
        for j in range(n_real):
            i = idx[j]
            c = correct[j][mask[j]]  # correctness per supervised position, in the order of answer_type / kinds
            k = p.kinds[p.toffs[i]:p.toffs[i + 1]]
            if len(c) != len(k):
                raise RuntimeError(f"example {i}: {len(c)} supervised positions but {len(k)} kinds")
            n_ok += np.bincount(k[c], minlength=4)
            n_tot += np.bincount(k, minlength=4)
            passkey_ok += bool(c[k == PASSKEY].all())
    model.train(was_training)

    def acc(ok, tot):
        return float(ok / tot) if tot else float("nan")

    return {
        "passkey_acc": passkey_ok / len(p),
        "passkey_token_acc": acc(n_ok[PASSKEY], n_tot[PASSKEY]),  # share of single passkey tokens right (teacher forcing); exact match needs all of them
        "ar_valid_acc": acc(n_ok[VALID], n_tot[VALID]),
        "ar_expired_acc": acc(n_ok[STALE] + n_ok[NEVER], n_tot[STALE] + n_tot[NEVER]),
        "ar_expired_stale_acc": acc(n_ok[STALE], n_tot[STALE]),
        "ar_expired_never_acc": acc(n_ok[NEVER], n_tot[NEVER]),
        "n_examples": len(p), "n_valid": int(n_tot[VALID]), "n_expired_stale": int(n_tot[STALE]),
        "n_expired_never": int(n_tot[NEVER]),
    }
