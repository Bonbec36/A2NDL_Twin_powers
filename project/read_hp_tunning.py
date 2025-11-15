import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# output folder
out_dir = Path("hyperparameter_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------- Load JSON ----------------
with open("outputs\model_runs_hp_light_A\hp_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------- Flatten JSON ----------------
rows = []
for entry in data:
    cfg = entry.get("report", {}).get("params", {})
    macro_f1 = entry.get("report", {}).get("macro_f1", None)
    if macro_f1 is not None:
        rows.append({
            "lr": cfg.get("lr"),
            "dropout": cfg.get("dropout"),
            "batch_size": cfg.get("batch_size"),
            "seed": cfg.get("seed"),
            "macro_f1": macro_f1
        })

df = pd.DataFrame(rows)

# save full dataframe
df.to_csv(out_dir / "all_results.csv", index=False)

sns.set(style="whitegrid", palette="tab10")

# ---------------- Macro F1 vs Learning Rate ----------------
df_lr = df.groupby("lr")["macro_f1"].mean().reset_index()
df_lr.to_csv(out_dir / "macroF1_vs_lr.csv", index=False)
plt.figure(figsize=(6,4))
plt.plot(df_lr["lr"], df_lr["macro_f1"], marker='o')
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Macro F1 (mean over seeds, batch size, dropout)")
plt.title("Effect of Learning Rate on Macro F1")
plt.tight_layout()
plt.savefig(out_dir / "macroF1_vs_lr.png")
plt.close()

# ---------------- Macro F1 vs Batch Size ----------------
df_bs = df.groupby("batch_size")["macro_f1"].mean().reset_index()
df_bs.to_csv(out_dir / "macroF1_vs_batchsize.csv", index=False)
plt.figure(figsize=(6,4))
plt.plot(df_bs["batch_size"], df_bs["macro_f1"], marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Macro F1 (mean over seeds, lr, dropout)")
plt.title("Effect of Batch Size on Macro F1")
plt.tight_layout()
plt.savefig(out_dir / "macroF1_vs_batchsize.png")
plt.close()

# ---------------- Macro F1 vs Dropout ----------------
df_dropout = df.groupby("dropout")["macro_f1"].mean().reset_index()
df_dropout.to_csv(out_dir / "macroF1_vs_dropout.csv", index=False)
plt.figure(figsize=(6,4))
plt.plot(df_dropout["dropout"], df_dropout["macro_f1"], marker='o')
plt.xlabel("Dropout")
plt.ylabel("Macro F1 (mean over seeds, lr, batch size)")
plt.title("Effect of Dropout on Macro F1")
plt.tight_layout()
plt.savefig(out_dir / "macroF1_vs_dropout.png")
plt.close()

print(f"Saved all CSVs and plots in {out_dir}")
