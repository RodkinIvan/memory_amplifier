"""passkey_swar parquet files -> flat numpy arrays, padded batches, and per-position query kinds.

`kinds` refine `answer_type` by splitting expired queries into *stale* (key written earlier, idle > 2w writes) and
*never-written*; they are derived with the verified oracle from data/sanity_checks.py and cached next to each file.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import make_passkey_swar as G  # noqa: E402
import sanity_checks as S  # noqa: E402

IGNORE = G.IGNORE
KIND_NAMES = S.KIND_NAMES  # passkey, ar_valid, ar_expired_stale, ar_expired_never


def _flat_list(col):
    """ChunkedArray<list<T>> -> (offsets starting at 0, flat values as an Arrow array)."""
    arr = col.combine_chunks()
    offs = arr.offsets.to_numpy()
    return (offs - offs[0]).astype(np.int64), arr.values.slice(int(offs[0]))


@dataclass
class Packed:
    """A set of examples stored as flat arrays plus offsets."""
    ids: np.ndarray  # int16, all tokens
    labels: np.ndarray  # int16, -100 except supervised positions
    offs: np.ndarray  # int64 [n+1], token offsets
    types: np.ndarray  # int8, answer_type code per supervised position (G.PASSKEY / AR_VALID / AR_EXPIRED)
    toffs: np.ndarray  # int64 [n+1], offsets into `types` / `kinds`
    kinds: np.ndarray | None = None  # int8, code per supervised position indexing KIND_NAMES

    def __len__(self):
        return len(self.offs) - 1

    def lengths(self) -> np.ndarray:
        return np.diff(self.offs)

    def subset(self, n: int) -> "Packed":
        """First n examples."""
        return Packed(self.ids[:self.offs[n]], self.labels[:self.offs[n]], self.offs[:n + 1],
                      self.types[:self.toffs[n]], self.toffs[:n + 1],
                      None if self.kinds is None else self.kinds[:self.toffs[n]])


def concat(parts: list[Packed]) -> Packed:
    offs, toffs, o, t = [np.zeros(1, np.int64)], [np.zeros(1, np.int64)], 0, 0
    for p in parts:
        offs.append(p.offs[1:] + o)
        toffs.append(p.toffs[1:] + t)
        o, t = o + p.offs[-1], t + p.toffs[-1]
    return Packed(np.concatenate([p.ids for p in parts]), np.concatenate([p.labels for p in parts]),
                  np.concatenate(offs), np.concatenate([p.types for p in parts]), np.concatenate(toffs))


def load_packed(path: Path, kinds: bool = False) -> Packed:
    tbl = pq.read_table(path, columns=["input_ids", "labels", "answer_type"])
    offs, ids = _flat_list(tbl["input_ids"])
    _, labs = _flat_list(tbl["labels"])
    toffs, tvals = _flat_list(tbl["answer_type"])
    types = pc.index_in(tvals, value_set=pa.array(G.TYPE_NAMES))
    if types.null_count:
        raise ValueError(f"{path}: unknown answer_type string")
    p = Packed(ids.to_numpy(), labs.to_numpy(), offs, types.to_numpy(zero_copy_only=False).astype(np.int8), toffs)
    if kinds:
        p.kinds = np.load(kinds_path(path))
        if len(p.kinds) != len(p.types):
            raise ValueError(f"{kinds_path(path)} does not match {path}; delete it and re-run prepare")
    return p


def kinds_path(path: Path) -> Path:
    return Path(path).with_suffix(".kinds.npy")


def _kinds_group(job):
    path, rg, params = job
    tbl = pq.ParquetFile(path).read_row_group(rg)
    offs, ids = _flat_list(tbl["input_ids"])
    toffs, tvals = _flat_list(tbl["answer_type"])
    types = pc.index_in(tvals, value_set=pa.array(G.TYPE_NAMES)).to_numpy(zero_copy_only=False)
    ids = ids.to_numpy()
    out = []
    for r in range(len(offs) - 1):
        _, ty, kind, _ = S.oracle(ids[offs[r]:offs[r + 1]], params["w"], params["n_keys"], params["n_values"])
        if not np.array_equal(ty, types[toffs[r]:toffs[r + 1]]):
            raise ValueError(f"{path}: oracle disagrees with stored answer_type (row group {rg}, row {r})")
        out.append(kind)
    return np.concatenate(out)


def prepare_kinds(data_dir: Path, params: dict, splits=("validation", "test"), workers: int = 16) -> None:
    """Compute (once) the per-position kind arrays for every eval file under data_dir."""
    todo = [f for s in splits for f in sorted(Path(data_dir).glob(f"n*/{s}.parquet")) if not kinds_path(f).exists()]
    if not todo:
        return
    with ProcessPoolExecutor(workers) as pool:
        for f in todo:
            nrg = pq.ParquetFile(f).num_row_groups
            kinds = np.concatenate(list(pool.map(_kinds_group, [(str(f), g, params) for g in range(nrg)])))
            tmp = kinds_path(f).with_name(kinds_path(f).name + ".tmp.npy")
            np.save(tmp, kinds.astype(np.int8))
            tmp.rename(kinds_path(f))  # atomic: concurrent readers never see a partial file
            print(f"kinds cached: {kinds_path(f)} ({len(kinds)} positions)", flush=True)


def pad_batch(p: Packed, idx: np.ndarray, multiple: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad the selected examples (PAD=0, label -100) to a common length that is a multiple of `multiple`."""
    lens = p.offs[idx + 1] - p.offs[idx]
    L = int(-(-lens.max() // multiple) * multiple)
    ids = np.full((len(idx), L), G.PAD, np.int64)
    labs = np.full((len(idx), L), IGNORE, np.int64)
    for j, i in enumerate(idx):
        ids[j, :lens[j]] = p.ids[p.offs[i]:p.offs[i + 1]]
        labs[j, :lens[j]] = p.labels[p.offs[i]:p.offs[i + 1]]
    return ids, labs


def load_train(data_dir: Path, ns: list[int]) -> Packed:
    return concat([load_packed(Path(data_dir) / f"n{n}" / "train.parquet") for n in ns])
