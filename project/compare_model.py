import os
import json
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Provide paths to folders for each model size

model_dirs = {
    "Ultra light": "outputs\model_runs_ultra_light_B",
    "Light": "outputs\model_runs_light_C",
    "Base": "outputs\model_runs_base_D"#,"Complex": "outputs\model_runs_Complex_B"
    
}
"""
model_dirs = {
    "Hp 01": "outputs\hp_runs_01",
    "Hp 02": "outputs\hp_runs_02",
    "Base": "outputs\model_runs_base_D"#,"Complex": "outputs\model_runs_Complex_B"
    
}"""

metrics = ["val_loss", "val_acc", "macro_f1"]
per_class_metric = "per_class_f1"  # This is a dictionary in the JSON

# --- Read JSON files ---
all_data = {metric: {size: [] for size in model_dirs} for metric in metrics}
per_class_data = {size: {0: [], 1: [], 2: []} for size in model_dirs}  # classes 0,1,2

for size, base_dir in model_dirs.items():
    # List seed subfolders
    seed_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    for seed_dir in seed_dirs:
        # Look for final report JSON
        json_files = [f for f in os.listdir(seed_dir) if f.endswith(".json")]
        for json_file in json_files:
            json_path = os.path.join(seed_dir, json_file)
            try:
                with open(json_path, "r") as f:
                    report = json.load(f)
                # Standard metrics
                for metric in metrics:
                    if metric in report:
                        all_data[metric][size].append(report[metric])
                # Per-class F1
                if per_class_metric in report:
                    for cls_str, value in report[per_class_metric].items():
                        cls_idx = int(cls_str)
                        per_class_data[size][cls_idx].append(value)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

# --- Plot boxplots for standard metrics ---
for metric in metrics:
    plt.figure(figsize=(8,5))
    data_to_plot = [all_data[metric][size] for size in model_dirs]
    plt.boxplot(data_to_plot, labels=list(model_dirs.keys()))
    plt.title(f"Distribution of {metric} over 10 seeds per model size")
    plt.ylabel(metric)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# --- Plot boxplots for per-class F1 ---
for cls in [0, 1, 2]:
    plt.figure(figsize=(8,5))
    data_to_plot = [per_class_data[size][cls] for size in model_dirs]
    plt.boxplot(data_to_plot, labels=list(model_dirs.keys()))
    plt.title(f"Per-class F1 (class {cls}) over 10 seeds per model size")
    plt.ylabel("F1 score")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# --- Compute mean and std per model size ---
print("Mean and standard deviation per model size:")
for metric in metrics:
    print(f"\n{metric}:")
    for size in model_dirs:
        values = all_data[metric][size]
        if values:
            print(f"  {size}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        else:
            print(f"  {size}: no data")

# --- Compute mean and std per class ---
print("\nPer-class F1 mean and std per model size:")
for cls in [0, 1, 2]:
    print(f"\nClass {cls}:")
    for size in model_dirs:
        values = per_class_data[size][cls]
        if values:
            print(f"  {size}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        else:
            print(f"  {size}: no data")