# project/infer.py
# -*- coding: utf-8 -*-
"""
Inference script (relative-path, robust)

- Uses normalization if data/stats.json has mean/std
- Accepts (B, F, T) / (B, T, F) / sparse tensors
- Ensures minimum time length T by repeating along time dim (with safety cap)
- Infers num_classes from checkpoint when possible
- Saves predictions to outputs/run1/preds.csv
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib
from pathlib import Path
import pandas as pd
import torch

# ---------------- Path setup ----------------
PROJ_DIR = Path(__file__).resolve().parent     # .../project
ROOT_DIR = PROJ_DIR.parent                     # repo root
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "outputs" / "run1"

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))
importlib.invalidate_caches()

from dataset import TimeSeriesDataset, read_features_txt
from model.pain_cnn import PainCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Utils ----------------
def guess_num_classes_from_state_dict(state_dict: dict, fallback: int = 3) -> int:
    """Try to deduce number of classes from common classifier heads."""
    for key in ("classifier.weight", "fc.weight", "head.3.weight", "head.weight"):
        w = state_dict.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return int(w.shape[0])
    for k, v in state_dict.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            out_dim = int(v.shape[0])
            if 2 <= out_dim <= 100:
                return out_dim
    return fallback


def ensure_min_T(xb: torch.Tensor, need_T: int) -> torch.Tensor:
    """Repeat along time dim to reach at least need_T. xb: (B, F, T)."""
    T = xb.size(-1)
    if T >= need_T:
        return xb
    rep = (need_T + T - 1) // T
    xb = xb.repeat(1, 1, rep)  # (B, F, T * rep)
    return xb[:, :, :need_T].contiguous()


# ---------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(OUT_DIR / "best.pt"))
    p.add_argument("--test_csv",   type=str, default=str(DATA_DIR / "pirate_pain_test.csv"))
    p.add_argument("--features",   type=str, default=str(DATA_DIR / "features.txt"))
    p.add_argument("--stats",      type=str, default=str(DATA_DIR / "stats.json"))
    p.add_argument("--out_csv",    type=str, default=str(OUT_DIR / "preds.csv"))
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--min_T",      type=int, default=8)     # starting T; will grow if needed
    p.add_argument("--max_T_cap",  type=int, default=512)   # safety upper bound
    args = p.parse_args(args=[]) if os.environ.get("PYCHARM_HOSTED") else p.parse_args()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    return args


# ---------------- Main ----------------
@torch.no_grad()
def main():
    args = get_args()

    # Load feature order
    feats = read_features_txt(args.features)
    in_channels = len(feats)

    # Build test dataset (IMPORTANT: pass test_csv as the FIRST positional arg)
    ds = TimeSeriesDataset(
        args.test_csv,        # <- goes into 'train_csv' position
        None,                 # label_csv=None for test
        args.features,
        args.stats,
        mode="test",
        normalize=True,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    num_classes = guess_num_classes_from_state_dict(state, fallback=3)

    model = PainCNN(in_channels=in_channels, num_classes=num_classes, dropout=0.0).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_ids, all_preds = [], []

    for batch in dl:
        # Be compatible with both dataset outputs: (x) or (x, sid)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            xb, sid = batch
        else:
            xb, sid = batch, None

        xb = xb.to(DEVICE, dtype=torch.float32)

        # Sparse -> dense
        if xb.is_sparse:
            xb = xb.to_dense()

        # Ensure 3D and (B, F, T)
        if xb.dim() == 2:
            xb = xb.unsqueeze(0)
        if xb.dim() != 3:
            raise RuntimeError(f"Unexpected input shape {tuple(xb.shape)}; expect 3D (B,F,T).")
        if xb.size(1) != in_channels and xb.size(2) == in_channels:
            xb = xb.permute(0, 2, 1).contiguous()

        # Try min_T first; double on pool/conv size errors
        target_T = int(args.min_T)
        while True:
            try:
                xb_pad = ensure_min_T(xb, target_T)
                logits = model(xb_pad)
                pred = logits.argmax(dim=1).detach().cpu().tolist()
                all_preds.extend(pred)

                # IDs if available, else running index fallback
                if sid is not None:
                    if isinstance(sid, (list, tuple)):
                        all_ids.extend([str(s) for s in sid])
                    else:
                        all_ids.extend([str(s) for s in sid])
                else:
                    start = len(all_ids)
                    all_ids.extend([str(i) for i in range(start, start + len(pred))])
                break
            except RuntimeError as e:
                msg = str(e)
                if "output size" in msg or "Invalid computed output size" in msg:
                    target_T *= 2
                    if target_T > int(args.max_T_cap):
                        raise RuntimeError(
                            f"Could not infer even with T expanded to {target_T}. "
                            f"The model likely needs longer sequences."
                        ) from e
                else:
                    raise

    # Save CSV (include sample_index if we have it)
    df = pd.DataFrame({"sample_index": all_ids, "pred": all_preds}) if all_ids else pd.DataFrame({"pred": all_preds})
    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)
    print(f"[done] predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
