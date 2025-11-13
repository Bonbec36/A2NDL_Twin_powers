# project/evaluate.py
# -*- coding: utf-8 -*-
"""
Independent evaluation script (relative path version)

- Loads outputs/run1/split_indices.json if available
- Uses data/stats.json for normalization (if mean/std exist)
- Accepts input tensors in (B,F,T) or (B,T,F) or sparse format
- Saves evaluation metrics to outputs/run1/val_report.json

Project layout:
repo_root/
  ├─ data/
  │   ├─ pirate_pain_train.csv
  │   ├─ pirate_pain_train_labels.csv
  │   ├─ features.txt
  │   └─ stats.json
  ├─ outputs/run1/
  └─ project/
      ├─ evaluate.py
      ├─ train.py
      ├─ infer.py
      ├─ dataset.py
      └─ model/pain_cnn.py
"""

import os
import sys
import json
import argparse
import importlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ---------------- Path setup ----------------
PROJ_DIR = Path(__file__).resolve().parent     # .../project
ROOT_DIR = PROJ_DIR.parent                     # repo root
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "outputs" / "run1"

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))
importlib.invalidate_caches()

from model.pain_cnn import PainCNN
from dataset import TimeSeriesDataset, read_features_txt


# ---------------- Argument parser ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default=str(DATA_DIR / "pirate_pain_train.csv"))
    p.add_argument("--label_csv", type=str, default=str(DATA_DIR / "pirate_pain_train_labels.csv"))
    p.add_argument("--features_txt", type=str, default=str(DATA_DIR / "features.txt"))
    p.add_argument("--stats_json", type=str, default=str(DATA_DIR / "stats.json"))
    p.add_argument("--split_json", type=str, default=str(OUT_DIR / "split_indices.json"))
    p.add_argument("--model", type=str, default=str(OUT_DIR / "best.pt"))
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--batch_size", type=int, default=128)

    # allow no-CLI mode for PyCharm
    args = p.parse_args(args=[]) if os.environ.get("PYCHARM_HOSTED") else p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return args


# ---------------- Main ----------------
@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = read_features_txt(args.features_txt)
    num_feats = len(feats)

    # Build full dataset (for slicing validation subset)
    ds = TimeSeriesDataset(
        args.train_csv, args.label_csv, args.features_txt, args.stats_json,
        mode="train", normalize=True
    )
    labels_np = np.asarray(ds.labels, dtype=np.int64)
    num_classes = int(labels_np.max()) + 1

    # Load split
    val_idx = None
    try:
        with open(args.split_json, "r", encoding="utf-8") as f:
            sp = json.load(f)
        if "val_idx" in sp and isinstance(sp["val_idx"], list) and len(sp["val_idx"]) > 0:
            val_idx = sp["val_idx"]
            print(f"[info] Using val_idx from split_indices.json (n={len(val_idx)})")
    except Exception:
        print("[warn] split_indices.json not found; will evaluate full dataset.")

    if val_idx is None:
        val_idx = list(range(len(ds)))
        print(f"[warn] Evaluating full dataset (n={len(val_idx)}).")

    val_ds = Subset(ds, val_idx)
    dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt.get("model", ckpt)
    model = PainCNN(in_channels=num_feats, num_classes=num_classes, dropout=0.2).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_pred, all_true = [], []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device).long()

        if x.is_sparse:
            x = x.to_dense()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected input {tuple(x.shape)}; expect (B,F,T)")
        if x.size(1) == num_feats:
            pass
        elif x.size(2) == num_feats:
            x = x.permute(0, 2, 1).contiguous()
        else:
            x = x.permute(0, 2, 1).contiguous()

        p = model(x).argmax(1)
        all_pred.append(p.cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Compute metrics
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        per_class_f1 = f1_score(y_true, y_pred, average=None).tolist()
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "classification_report": report,
        }
    except Exception as e:
        acc = float((y_true == y_pred).mean())
        metrics = {"accuracy": acc, "note": f"sklearn unavailable: {e!s}"}

    # Save output
    out = {
        "metrics": metrics,
        "num_classes": int(num_classes),
        "val_size": int(len(val_idx)),
        "model_path": str(args.model),
        "split_json": str(args.split_json) if Path(args.split_json).exists() else None,
        "features": feats,
    }

    out_path = Path(args.out_dir) / "val_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] val_report saved to: {out_path}")

    if "accuracy" in metrics:
        msg = f"[summary] acc={metrics['accuracy']:.4f}"
        if "macro_f1" in metrics:
            msg += f", macro_f1={metrics['macro_f1']:.4f}"
        print(msg)


if __name__ == "__main__":
    main()
