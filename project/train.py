# project/train.py
# -*- coding: utf-8 -*-
"""
Train + validate, save best.pt and logs (robust split, relative paths)

- All paths are relative to the repository layout:
    repo_root/
      ├─ data/
      │   ├─ pirate_pain_train.csv
      │   ├─ pirate_pain_train_labels.csv
      │   ├─ features.txt
      │   └─ stats.json
      ├─ outputs/run1/
      └─ project/
          ├─ train.py
          ├─ evaluate.py
          ├─ infer.py
          ├─ dataset.py
          └─ model/pain_cnn.py

- Grouped split priority: sample_index (derived), then fallback.
- Save: split_indices.json, train_curve.csv, val_report.json, best.pt
"""

import os
import sys
import csv
import re
import json
import time
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------- Path bootstrap (RELATIVE) ----------------
PROJ_DIR = Path(__file__).resolve().parent               # .../project
ROOT_DIR = PROJ_DIR.parent                               # repo root
DATA_DIR = ROOT_DIR / "data"
OUT_DIR_DEFAULT = ROOT_DIR / "outputs" / "run1"

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))
importlib.invalidate_caches()


# ---------------- Safe imports ----------------
def safe_import():
    from model.pain_cnn import PainCNN
    from dataset import TimeSeriesDataset, read_features_txt
    return PainCNN, TimeSeriesDataset, read_features_txt


# ---------------- Args ----------------
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv",    type=str, default=str(DATA_DIR / "pirate_pain_train.csv"))
    p.add_argument("--label_csv",    type=str, default=str(DATA_DIR / "pirate_pain_train_labels.csv"))
    p.add_argument("--features_txt", type=str, default=str(DATA_DIR / "features.txt"))
    p.add_argument("--stats_json",   type=str, default=str(DATA_DIR / "stats.json"))
    p.add_argument("--out_dir",      type=str, default=str(OUT_DIR_DEFAULT))

    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--val_ratio",  type=float, default=0.2)

    # Support PyCharm run configuration (no CLI)
    args = p.parse_args(args=[]) if os.environ.get("PYCHARM_HOSTED") else p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return args


# ---------------- Utils ----------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_load_stats(path):
    """Return True if stats.json has mean/std and can be used for normalization."""
    try:
        s = json.load(open(path, "r", encoding="utf-8"))
        if "mean" in s and "std" in s:
            print(f"[info] Normalization enabled: mean/std length = {len(s['mean'])} {len(s['std'])}")
            return True
        print("[warn] stats.json has no 'mean'/'std'; normalization disabled.")
        return False
    except Exception:
        print("[warn] Cannot read/parse stats.json; normalization disabled.")
        return False


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    inv = 1.0 / (counts + 1e-8)
    inv *= (len(inv) / (inv.sum() + 1e-8))
    return inv


def compute_f1_per_class(y_true, y_pred, num_classes):
    eps = 1e-12
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1s.append(float(2 * prec * rec / (prec + rec + eps)))
    return f1s, float(np.mean(f1s)) if f1s else 0.0, cm


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None, num_feats=None):
    """Evaluate one dataloader; accept (B,F,T) or (B,T,F) and sparse inputs."""
    model.eval()
    total, hit, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()

        if x.is_sparse:
            x = x.to_dense()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected input shape {tuple(x.shape)}; expect 3D (B,F,T).")

        # (B,T,F) -> (B,F,T) if necessary
        if num_feats is not None and x.size(1) != num_feats and x.size(2) == num_feats:
            x = x.permute(0, 2, 1).contiguous()

        pred = model(x)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)

        n = min(pred.size(0), y.size(0))
        pred = pred[:n]
        y = y[:n]

        if criterion is not None:
            loss_sum += criterion(pred, y).item() * n

        total += n
        hit += (pred.argmax(1) == y).sum().item()
        all_pred.append(pred.argmax(1).cpu().numpy())
        all_true.append(y.cpu().numpy())

    acc = hit / max(1, total)
    loss = loss_sum / max(1, total) if criterion is not None else 0.0
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], int)
    y_true = np.concatenate(all_true) if all_true else np.array([], int)
    return loss, acc, y_true, y_pred


def derive_group_ids(labels_df: pd.DataFrame) -> np.ndarray:
    """
    Build grouping IDs for leakage-safe split:
      1) If 'sample_index' exists, use it directly (recommended).
      2) Else if 'id' exists, use it.
      3) Else if 'filename' exists, strip extension and use prefix (split by '_' or '-').
      4) Else fallback to per-row index (string).
    """
    if "sample_index" in labels_df.columns:
        gids = labels_df["sample_index"].astype(str).to_numpy()
        print(f"[info] Train index (grouped by sample_index): n={len(gids)}, seq_len≈unknown")
        return gids

    if "id" in labels_df.columns:
        gids = labels_df["id"].astype(str).to_numpy()
        print("[info] Group by label_csv['id'].")
        return gids

    if "filename" in labels_df.columns:
        def to_group(s):
            base = str(s)
            base = re.sub(r"\.[A-Za-z0-9]+$", "", base)
            token = base.split("_")[0]
            token = token.split("-")[0]
            return token

        gids = labels_df["filename"].map(to_group).astype(str).to_numpy()
        print("[info] Group by filename prefix (no 'id').")
        return gids

    print("[warn] No 'sample_index'/'id'/'filename'; group by row index.")
    return np.arange(len(labels_df)).astype(str)


# ---------------- Main ----------------
def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PainCNN, TimeSeriesDataset, read_features_txt = safe_import()

    feats = read_features_txt(args.features_txt)
    num_feats = len(feats)
    print(f"[info] features: {num_feats} dims")

    do_norm = try_load_stats(args.stats_json)

    # Build dataset
    labels_df = pd.read_csv(args.label_csv)
    ds_full = TimeSeriesDataset(
        args.train_csv, args.label_csv, args.features_txt, args.stats_json,
        mode="train", normalize=do_norm
    )

    # ---- Build or reuse split
    out_dir = Path(args.out_dir)
    split_path = out_dir / "split_indices.json"

    if split_path.exists():
        saved = json.load(open(split_path, "r", encoding="utf-8"))
        train_idx = list(map(int, saved["train_idx"]))
        val_idx = list(map(int, saved["val_idx"]))
        print(f"[info] Reusing existing split: {split_path}")
    else:
        group_ids = derive_group_ids(labels_df)
        uniq = np.unique(group_ids)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(uniq)

        k = int(round((1.0 - args.val_ratio) * len(uniq)))
        train_set = set(uniq[:max(1, k)])
        val_set = set(uniq[max(1, k):])

        all_idx = np.arange(len(labels_df))
        train_idx = [int(i) for i in all_idx if group_ids[i] in train_set]
        val_idx = [int(i) for i in all_idx if group_ids[i] in val_set]

        json.dump({"train_idx": train_idx, "val_idx": val_idx},
                  open(split_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[ok] Saved split to: {split_path}")

    ds_train = Subset(ds_full, train_idx)
    ds_val = Subset(ds_full, val_idx)

    # ---- Class stats (on train subset)
    labels_np = np.asarray([ds_full.labels[i] for i in train_idx], dtype=np.int64)
    num_classes = int(labels_np.max()) + 1
    counts = np.bincount(labels_np, minlength=num_classes)
    print(f"Class counts: {{ {', '.join(f'{i}: {int(c)}' for i, c in enumerate(counts))} }}")
    weights = class_weights_from_counts(counts)
    print(f"Class weights (0..{num_classes-1}): {weights.tolist()}")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- Model / optimizer
    model = PainCNN(in_channels=num_feats, num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = Adam(model.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)

    best_acc = -1.0
    best_path = out_dir / "best.pt"
    curve_path = out_dir / "train_curve.csv"
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # ---- Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_sample, hit = 0.0, 0, 0

        for x, y in dl_train:
            x = x.to(device)
            y = y.to(device).long()

            if x.is_sparse:
                x = x.to_dense()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.dim() != 3:
                raise RuntimeError(f"Unexpected input {tuple(x.shape)}; expect (B,F,T)")
            if x.size(1) != num_feats and x.size(2) == num_feats:
                x = x.permute(0, 2, 1).contiguous()

            opt.zero_grad()
            pred = model(x)
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)

            n = min(pred.size(0), y.size(0))
            pred = pred[:n]
            y = y[:n]

            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * n
            n_sample += n
            hit += (pred.argmax(1) == y).sum().item()

        train_loss = loss_sum / max(1, n_sample)
        train_acc = hit / max(1, n_sample)

        val_loss, val_acc, _, _ = evaluate_loader(model, dl_val, device, criterion, num_feats=num_feats)
        sched.step(val_acc)

        dt = time.time() - t0
        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | time={dt:.1f}s")

        with open(curve_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch,
                                    f"{train_loss:.6f}", f"{train_acc:.6f}",
                                    f"{val_loss:.6f}",  f"{val_acc:.6f}"])

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "val_acc": float(best_acc)}, best_path)
            print(f"[save] new best val_acc={best_acc:.4f} -> {best_path}")

    # ---- Final validation report (use best.pt)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    v_loss, v_acc, y_true, y_pred = evaluate_loader(model, dl_val, device, criterion, num_feats=num_feats)
    f1s, macro_f1, cm = compute_f1_per_class(y_true, y_pred, num_classes)
    rep = {
        "val_loss": float(v_loss),
        "val_acc": float(v_acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {str(i): float(f1s[i]) for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
        "best_ckpt": str(best_path),
        "num_classes": int(num_classes),
    }
    rep_path = out_dir / "val_report.json"
    json.dump(rep, open(rep_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[done] best_acc(val)={best_acc:.4f} | best={best_path}")
    print(f"[done] curves: {curve_path}")
    print(f"[done] report: {rep_path}")
    print(f"[done] split : {split_path}")


if __name__ == "__main__":
    main()
