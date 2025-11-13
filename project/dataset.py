# -*- coding: utf-8 -*-
"""
Dataset utilities for Pirate Pain challenge.

Features
--------
- Supports both "long" and "wide" table layouts:
  * Long table: total rows = n_samples * seq_len  (each sample occupies a contiguous
    block of seq_len rows, or rows are grouped by 'sample_index')
  * Wide table: one row per sample (requires time already expanded in columns;
    we cover a minimal path but long-table is recommended)
- Group-aware alignment (by 'sample_index' when available)
- Per-feature normalization using stats.json
- Clean English logs; robust error messages

Expected columns
----------------
- train_csv (long table typical):
    ['sample_index' (optional), 't' (optional), <feature1>, <feature2>, ...]
- label_csv (train mode):
    ['sample_index' (optional), 'label']
- test_csv (test mode):
    same feature columns as train, plus ['sample_index' (optional)]

Notes
-----
- Label mapping defaults to: {'high_pain':0, 'low_pain':1, 'no_pain':2}.
  If unexpected strings appear, it falls back to an alphabetical map and warns.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Optional: allow manual override of sequence length for long-tables
_FORCE_SEQ_LEN = os.environ.get("FORCE_SEQ_LEN")
try:
    _FORCE_SEQ_LEN = int(_FORCE_SEQ_LEN) if _FORCE_SEQ_LEN else None
except Exception:
    _FORCE_SEQ_LEN = None


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

def read_features_txt(path: str | Path) -> List[str]:
    """
    Read feature names from a text file (one feature per line).
    Empty lines or lines starting with '#' are ignored.
    """
    feats: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
    if not feats:
        raise ValueError(f"No features found in {path}")
    return feats


def _load_stats(stats_json: str | Path, n_feats: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mean/std arrays from stats.json, and validate length.
    Returns zero/one arrays if not available.
    """
    mean = np.zeros(n_feats, dtype=np.float32)
    std  = np.ones(n_feats, dtype=np.float32)
    try:
        with open(stats_json, "r", encoding="utf-8") as f:
            s = json.load(f)
        if "mean" in s and "std" in s:
            m = np.asarray(s["mean"], dtype=np.float32)
            v = np.asarray(s["std"],  dtype=np.float32)
            if len(m) == n_feats and len(v) == n_feats:
                mean, std = m, v
            else:
                print(f"[warn] stats.json mean/std length mismatch (got {len(m)}/{len(v)}, expect {n_feats}). Using identity.")
        else:
            print("[warn] stats.json missing mean/std; using identity.")
    except Exception:
        print("[warn] Cannot read stats.json; using identity.")
    return mean, std


def _label_map_from_strings(series: pd.Series) -> Tuple[np.ndarray, dict]:
    """
    Map string labels to integer ids with a stable, expected mapping.
    Default preference: {'high_pain':0, 'low_pain':1, 'no_pain':2}.
    If unseen labels appear, fall back to alphabetical order and warn.
    """
    preferred = {"high_pain": 0, "low_pain": 1, "no_pain": 2}
    uniq = sorted(set([str(x) for x in series.tolist()]))

    if set(uniq).issubset(set(preferred.keys())):
        mp = preferred
    else:
        print(f"[warn] Unexpected label set {uniq}; falling back to alphabetical mapping.")
        mp = {name: i for i, name in enumerate(sorted(uniq))}

    y = series.map(lambda s: mp[str(s)]).to_numpy(dtype=np.int64)
    # Pretty print mapping
    print("[info] label mapping (string -> id):")
    for k, v in mp.items():
        print(f"    {k} -> {v}")
    return y, mp


# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #

class TimeSeriesDataset(Dataset):
    """
    Multivariate time-series dataset that supports both long and wide tables.

    Parameters
    ----------
    train_csv : str
        Path to training (or test) CSV with features.
    label_csv : str | None
        Path to label CSV (ignored in test mode if None).
    features_txt : str
        Path to features.txt (one feature per line).
    stats_json : str | None
        Path to stats.json containing per-feature mean/std.
    mode : str
        'train' or 'test'. In 'test' mode, labels are not required/used.
    normalize : bool
        If True and stats are available, apply (x - mean) / std per feature.
    """

    def __init__(self,
                 train_csv: str,
                 label_csv: Optional[str],
                 features_txt: str,
                 stats_json: Optional[str],
                 mode: str = "train",
                 normalize: bool = True) -> None:

        super().__init__()
        self.mode = str(mode).lower()
        assert self.mode in ("train", "test"), "mode must be 'train' or 'test'"

        self.features: List[str] = read_features_txt(features_txt)
        self.n_feats: int = len(self.features)

        # Load frames
        self.train_df: pd.DataFrame = pd.read_csv(train_csv)
        self.label_df: Optional[pd.DataFrame] = None
        if self.mode == "train":
            if label_csv is None:
                raise ValueError("label_csv is required in train mode.")
            self.label_df = pd.read_csv(label_csv)

        # Stats
        self.normalize = bool(normalize)
        self.mean, self.std = (np.zeros(self.n_feats, dtype=np.float32),
                               np.ones(self.n_feats, dtype=np.float32))
        if self.normalize and stats_json:
            self.mean, self.std = _load_stats(stats_json, self.n_feats)

        # Validate feature columns
        missing = [c for c in self.features if c not in self.train_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in train_csv: {missing[:5]}{' ...' if len(missing)>5 else ''}")

        # Figure out layout and build index
        if self.mode == "train":
            self._build_index_train()
            # Build integer labels aligned to sample index order
            self.labels, self.label_map = _label_map_from_strings(self.label_df["label"])
        else:
            self._build_index_test()

    # --------------------------- index builders --------------------------- #

    def _build_index_train(self) -> None:
        """
        Build sample indexing for training mode:
        - If 'sample_index' exists in both train and label CSVs, align by it.
        - Else, try to infer long-table structure (rows divisible by n_labels).
        - Else, if rows == labels, treat as wide table (1 row per sample).
        - Otherwise raise a descriptive error.
        """
        n_rows = len(self.train_df)
        n_lbls = len(self.label_df)
        has_sid_train = 'sample_index' in self.train_df.columns
        has_sid_label = 'sample_index' in self.label_df.columns

        # Path 1: align by sample_index (recommended)
        if has_sid_train and has_sid_label:
            # Reindex label order by sample_index, and group train by sample_index
            self.label_df = self.label_df.copy()
            # Sort by sample_index to have a stable order
            self.label_df["sample_index"] = self.label_df["sample_index"].astype(str)
            self.label_df.sort_values("sample_index", inplace=True, kind="mergesort")
            # group train
            g = self.train_df.copy()
            g["sample_index"] = g["sample_index"].astype(str)
            grouped = [df for _, df in g.groupby("sample_index", sort=True)]
            if len(grouped) != n_lbls:
                raise ValueError(f"Group count mismatch: grouped={len(grouped)} vs labels={n_lbls}.")

            # Infer seq_len from the first group and validate
            seq_len = len(grouped[0])
            if not all(len(df) == seq_len for df in grouped):
                raise ValueError("Each group must have identical seq_len. Found mismatch.")
            self._is_long = True
            self._use_groups = True
            self._seq_len = int(seq_len)
            self._groups = grouped
            self._chunks = None
            print(f"[info] Train index (grouped by sample_index): n={n_lbls}, seq_len={self._seq_len}")
            return

        # Path 2: auto-detect long table by divisibility
        if n_lbls > 0 and n_rows % n_lbls == 0:
            seq_len = _FORCE_SEQ_LEN if _FORCE_SEQ_LEN is not None else (n_rows // n_lbls)
            self._is_long = True
            self._use_groups = False
            self._seq_len = int(seq_len)
            self._groups = None
            self._chunks = [(i * self._seq_len, (i + 1) * self._seq_len) for i in range(n_lbls)]
            print(f"[info] Train index (auto long table): n={n_lbls}, seq_len={self._seq_len} (= {n_rows} / {n_lbls})")
            return

        # Path 3: rows == labels -> treat as wide table
        if n_rows == n_lbls:
            self._is_long = False
            self._use_groups = False
            self._seq_len = 1
            self._groups = None
            self._chunks = None
            print(f"[info] Train index (wide table): n={n_lbls} rows == labels.")
            return

        # Otherwise: cannot align
        raise ValueError(
            f"Alignment failed: rows(train)={n_rows}, rows(labels)={n_lbls}. "
            f"Not equal and not divisible; no 'sample_index' to group. "
            f"Provide 'sample_index' columns or set FORCE_SEQ_LEN."
        )

    def _build_index_test(self) -> None:
        """
        Build sample indexing for test mode, mirroring training logic
        (but without labels).
        """
        n_rows = len(self.train_df)
        has_sid_train = 'sample_index' in self.train_df.columns

        if has_sid_train:
            g = self.train_df.copy()
            g["sample_index"] = g["sample_index"].astype(str)
            grouped = [df for _, df in g.groupby("sample_index", sort=True)]
            seq_len = len(grouped[0])
            if not all(len(df) == seq_len for df in grouped):
                raise ValueError("Each test group must have identical seq_len. Found mismatch.")
            self._is_long = True
            self._use_groups = True
            self._seq_len = int(seq_len)
            self._groups = grouped
            self._chunks = None
            self._test_sample_index = [str(s) for s in sorted(g["sample_index"].unique())]
            print(f"[info] Test index (grouped by sample_index): n={len(self._test_sample_index)}, seq_len={self._seq_len}")
            return

        # Try environment override or auto long-table
        if _FORCE_SEQ_LEN is not None:
            seq_len = _FORCE_SEQ_LEN
            if n_rows % seq_len != 0:
                raise ValueError(f"FORCE_SEQ_LEN={seq_len} does not divide test rows={n_rows}.")
            n_samples = n_rows // seq_len
            self._is_long = True
            self._use_groups = False
            self._seq_len = int(seq_len)
            self._groups = None
            self._chunks = [(i * self._seq_len, (i + 1) * self._seq_len) for i in range(n_samples)]
            self._test_sample_index = [str(i).zfill(3) for i in range(n_samples)]
            print(f"[info] Test index (forced long table): n={n_samples}, seq_len={self._seq_len}")
            return

        # Auto-divisible long table
        # (If you have a separate test_labels to infer n_samples, customize here)
        # As a fallback, assume contiguous chunks by an inferred seq_len if present in metadata.
        raise ValueError(
            "Test mode without 'sample_index' is ambiguous. "
            "Please add 'sample_index' to test CSV or set FORCE_SEQ_LEN."
        )

    # --------------------------- Dataset protocol ------------------------- #

    def __len__(self) -> int:
        if self.mode == "train":
            if getattr(self, "_is_long", False):
                # Number of samples equals number of labels
                return len(self.label_df)
            else:
                # Wide table: one row per sample
                return len(self.train_df)
        else:
            # Test mode mirrors train indexing
            if getattr(self, "_is_long", False):
                if getattr(self, "_use_groups", False):
                    return len(self._groups)
                return len(self._chunks)
            else:
                return len(self.train_df)

    def _slice_long_chunk(self, idx: int) -> pd.DataFrame:
        """Return the (seq_len, *) chunk DataFrame for sample idx."""
        if getattr(self, "_use_groups", False):
            return self._groups[idx]
        a, b = self._chunks[idx]
        return self.train_df.iloc[a:b]

    def _row_to_tensor(self, row_df: pd.DataFrame) -> np.ndarray:
        """
        Convert a chunk (seq_len, features) to (F, T) numpy float32.
        Applies normalization if enabled.
        """
        X = row_df[self.features].to_numpy(dtype=np.float32)  # (T, F)
        if self.normalize:
            X = (X - self.mean[np.newaxis, :]) / (self.std[np.newaxis, :] + 1e-8)
        return X.T  # (F, T)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            if getattr(self, "_is_long", False):
                # Long-table: slice contiguous rows (or grouped) for one sample
                chunk = self._slice_long_chunk(idx)
                # Optional temporal sort if 't' exists
                if 't' in chunk.columns:
                    chunk = chunk.sort_values('t', kind="mergesort")
                X = self._row_to_tensor(chunk)  # (F, T)
                y = int(self.labels[idx])
                return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)

            # Wide-table (minimal path): expect per-row features already represent a single timestep
            # If your wide format stores time-expanded columns, customize this part accordingly.
            row = self.train_df.iloc[idx]
            X = row[self.features].to_numpy(dtype=np.float32)  # (F,)
            if self.normalize:
                X = (X - self.mean) / (self.std + 1e-8)
            X = X[:, None]  # (F, 1) to keep (F, T) contract
            y = int(self.labels[idx])
            return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)

        # ---- test mode ----
        if getattr(self, "_is_long", False):
            chunk = self._slice_long_chunk(idx)
            if 't' in chunk.columns:
                chunk = chunk.sort_values('t', kind="mergesort")
            X = self._row_to_tensor(chunk)  # (F, T)
            sid = None
            if getattr(self, "_use_groups", False):
                # If grouped by sample_index, we stored the order
                sid = chunk["sample_index"].iloc[0] if "sample_index" in chunk.columns else self._test_sample_index[idx]
            else:
                sid = self._test_sample_index[idx] if hasattr(self, "_test_sample_index") else str(idx).zfill(3)
            return torch.from_numpy(X), str(sid)

        # Wide-table test (fallback)
        row = self.train_df.iloc[idx]
        X = row[self.features].to_numpy(dtype=np.float32)
        if self.normalize:
            X = (X - self.mean) / (self.std + 1e-8)
        X = X[:, None]  # (F, 1)
        sid = row["sample_index"] if "sample_index" in self.train_df.columns else str(idx).zfill(3)
        return torch.from_numpy(X), str(sid)
