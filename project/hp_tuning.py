# project/hp_tune.py
# -*- coding: utf-8 -*-
"""
Hyperparameter tuning script for PainCNN.

For each hyper-parameter configuration:
 - train model (reproducible seed)
 - save best checkpoint, train curve (csv), plots, metrics (json + txt)
 - compute confusion matrix and multiclass AUC (one-vs-rest)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from itertools import product
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import scipy.special
from torchinfo import summary

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

# Project relative import bootstrap (same as train.py)
PROJ_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJ_DIR.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# Safe imports from project
def safe_import():
    from model.pain_cnn import PainCNN
    from dataset import TimeSeriesDataset, read_features_txt
    return PainCNN, TimeSeriesDataset, read_features_txt

# Re-implement utility functions needed
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic flags (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    inv = 1.0 / (counts + 1e-8)
    inv *= (len(inv) / (inv.sum() + 1e-8))
    return inv

@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None, num_feats=None):
    model.eval()
    total, hit, loss_sum = 0, 0, 0.0
    all_pred, all_true, all_logits = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        if x.is_sparse:
            x = x.to_dense()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected input shape {tuple(x.shape)}; expect 3D (B,F,T).")
        # adapt (B,T,F) -> (B,F,T) if needed
        if num_feats is not None and x.size(1) != num_feats and x.size(2) == num_feats:
            x = x.permute(0, 2, 1).contiguous()
        logits = model(x)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        n = min(logits.size(0), y.size(0))
        logits = logits[:n]
        y = y[:n]
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * n
        total += n
        pred = logits.argmax(1)
        hit += (pred == y).sum().item()
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    acc = hit / max(1, total)
    loss = loss_sum / max(1, total) if criterion is not None else 0.0
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], int)
    y_true = np.concatenate(all_true) if all_true else np.array([], int)
    logits_all = np.concatenate(all_logits) if all_logits else np.array([], float)
    return loss, acc, y_true, y_pred, logits_all

def save_txt(path:Path, text:str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def plot_curves(csv_path:Path, out_png:Path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['train_acc'], label='train_acc')
    plt.plot(df['epoch'], df['val_acc'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

def plot_confusion(cm, labels, out_png:Path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

def plot_multiclass_auc(y_true, logits, num_classes, out_png:Path):
    # compute softmax probabilities
    probs = scipy.special.softmax(logits, axis=1)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(6,5))
    for c in range(num_classes):
        try:
            auc = roc_auc_score(y_bin[:, c], probs[:, c])
        except Exception:
            auc = float('nan')
        fpr = None
        # For plotting ROC curve we need fpr/tpr; but to keep dependencies low
        # we will only plot AUC bar chart
        plt.bar(c, auc)
    plt.xticks(range(num_classes), [str(i) for i in range(num_classes)])
    plt.ylabel('AUC (one-vs-rest)')
    plt.title('Per-class AUC')
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

# ----------------- Main routine -----------------
def run_experiment(cfg, args, PainCNN, TimeSeriesDataset, num_feats, labels_df, train_idx, val_idx):
    # cfg: dict with hyperparams
    run_name = f"cnn_lr{cfg['lr']}_bs{cfg['batch_size']}_drop{cfg['dropout']}_seed{cfg['seed']}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"{timestamp}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # reproducibility
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets & loaders
    ds_full = TimeSeriesDataset(args.train_csv, args.label_csv, args.features_txt, args.stats_json, mode="train", normalize=args.normalize)
    ds_train = Subset(ds_full, train_idx)
    ds_val = Subset(ds_full, val_idx)
    dl_train = DataLoader(ds_train, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    # class weights
    labels_np = np.asarray([ds_full.labels[i] for i in train_idx], dtype=np.int64)
    num_classes = int(labels_np.max()) + 1
    counts = np.bincount(labels_np, minlength=num_classes)
    weights = class_weights_from_counts(counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    # model, optimizer, scheduler
    model = PainCNN(in_channels=num_feats, num_classes=num_classes, dropout=cfg['dropout']).to(device)
    opt = Adam(model.parameters(), lr=cfg['lr'])
    sched = None  # keep simple


    # Infer temporal length T from dataset
    sample_x, _ = ds_full[0]

    if sample_x.dim() == 2:  # (T, F)
        T = sample_x.shape[0]
    else:
        raise RuntimeError(f"Unexpected sample shape: {sample_x.shape}")

    # CNN expects (B, F, T)
    try:
        summary_str = str(summary(model, input_size=(cfg['batch_size'], num_feats, T)))
    except Exception as e:
        summary_str = f"torchinfo summary failed: {e}"

    # training
    best_val_acc = -1.0
    best_ckpt = out_dir / "best.pt"
    curve_csv = out_dir / "train_curve.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0; n_sample = 0; hit = 0
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
            pred = pred[:n]; y = y[:n]
            loss = criterion(pred, y)
            loss.backward(); opt.step()

            loss_sum += loss.item() * n
            n_sample += n
            hit += (pred.argmax(1) == y).sum().item()

        train_loss = loss_sum / max(1, n_sample)
        train_acc = hit / max(1, n_sample)

        val_loss, val_acc, _, _, _ = evaluate_loader(model, dl_val, device, criterion, num_feats=num_feats)

        # scheduler step if used
        if sched is not None:
            sched.step(val_acc)

        # log
        with open(curve_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "val_acc": float(val_acc)}, best_ckpt)

    # final eval (load best)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    v_loss, v_acc, y_true, y_pred, logits = evaluate_loader(model, dl_val, device, criterion, num_feats=num_feats)
    f1_macro = float(f1_score(y_true, y_pred, average='macro'))
    f1_per_class = dict(zip(map(str, range(int(np.max(y_true)+1))), f1_score(y_true, y_pred, average=None).tolist()))
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # save report json
    rep = {
        "val_loss": float(v_loss),
        "val_acc": float(v_acc),
        "macro_f1": float(f1_macro),
        "per_class_f1": f1_per_class,
        "confusion_matrix": cm.tolist(),
        "best_ckpt": str(best_ckpt),
        "params": cfg
    }
    json.dump(rep, open(out_dir / "val_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # save txt summary
    txt = []
    txt.append("Model: PainCNN")
    txt.append(f"Experiment folder: {out_dir}")
    txt.append("Hyperparameters:")
    for k, v in cfg.items():
        txt.append(f"  {k}: {v}")
    txt.append("\nTorchinfo summary:\n")
    txt.append(summary_str)
    txt.append("\nFinal results:")
    txt.append(json.dumps(rep, indent=2))
    save_txt(out_dir / "meta.txt", "\n".join(txt))

    # save csv of curves (already exists)
    # plot curves
    plot_curves(curve_csv, out_dir / "loss_acc.png")
    # confusion matrix
    plot_confusion(cm, labels=[str(i) for i in range(cm.shape[0])], out_png=out_dir / "confusion.png")
    # multiclass AUC
    try:
        plot_multiclass_auc(y_true, logits, num_classes, out_dir / "auc.png")
    except Exception as e:
        save_txt(out_dir / "auc_error.txt", str(e))

    return {
        "out_dir": str(out_dir),
        "val_acc": float(v_acc),
        "macro_f1": float(f1_macro),
        "report": rep
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=str(ROOT_DIR / "data" / "pirate_pain_train.csv"))
    parser.add_argument("--label_csv", type=str, default=str(ROOT_DIR / "data" / "pirate_pain_train_labels.csv"))
    parser.add_argument("--features_txt", type=str, default=str(ROOT_DIR / "data" / "features.txt"))
    parser.add_argument("--stats_json", type=str, default=str(ROOT_DIR / "data" / "stats.json"))
    parser.add_argument("--out_root", type=str, default=str(ROOT_DIR / "outputs" / "hp_runs_01"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    # imports
    PainCNN, TimeSeriesDataset, read_features_txt = safe_import()

    # prepare dataset & split indices (reuse train.py logic)
    labels_df = pd.read_csv(args.label_csv)
    group_ids = None
    if "sample_index" in labels_df.columns:
        group_ids = labels_df["sample_index"].astype(str).to_numpy()
    else:
        # fallback to index grouping
        group_ids = np.arange(len(labels_df)).astype(str)
    uniq = np.unique(group_ids)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq)
    k = int(round(0.8 * len(uniq)))
    train_set = set(uniq[:max(1, k)])
    val_set = set(uniq[max(1, k):])
    all_idx = np.arange(len(labels_df))
    train_idx = [int(i) for i in all_idx if group_ids[i] in train_set]
    val_idx = [int(i) for i in all_idx if group_ids[i] in val_set]

    feats = read_features_txt(args.features_txt)
    num_feats = len(feats)

    # hyperparameter grid (you can edit)
    lrs = [1e-3, 3e-4, 1e-4]
    batch_sizes = [32, 64]
    dropouts = [0.2, 0.3, 0.4]
    seeds = [args.seed, args.seed+1]

    grid = list(product(lrs, batch_sizes, dropouts, seeds))
    print(f"[info] Running {len(grid)} experiments")

    results = []
    for lr, bs, drop, sd in grid:
        cfg = {"lr": float(lr), "batch_size": int(bs), "dropout": float(drop), "seed": int(sd)}
        print(f"[exp] starting {cfg}")
        res = run_experiment(cfg, args, PainCNN, TimeSeriesDataset, num_feats, labels_df, train_idx, val_idx)
        print(f"[exp] done -> out: {res['out_dir']} | val_acc={res['val_acc']:.4f} macro_f1={res['macro_f1']:.4f}")
        results.append(res)

    # save global summary
    summary_path = Path(args.out_root) / "hp_summary.json"
    json.dump(results, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] all experiments finished. Summary -> {summary_path}")

if __name__ == "__main__":
    main()
