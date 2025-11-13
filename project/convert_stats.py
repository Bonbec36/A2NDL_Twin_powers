# -*- coding: utf-8 -*-
"""
自动将 A 组格式的 stats.json 转换为 B/C 组通用格式：
  从 { "joint_00": {"mean":..., "std":...}, ... }
  变为 { "mean": [...], "std": [...] }

生成文件：data/stats_converted.json
"""

import json
import numpy as np
import os

# ================= 路径自动定位 =================
# 当前脚本在 project/ 目录，目标数据在上一级 data/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # .../AN2DL_Project
DATA_DIR = os.path.join(BASE_DIR, "data")

stats_in = os.path.join(DATA_DIR, "stats.json")
features_txt = os.path.join(DATA_DIR, "features.txt")
stats_out = os.path.join(DATA_DIR, "stats_converted.json")

# ================= 文件加载 =================
if not os.path.exists(stats_in):
    raise FileNotFoundError(f"[error] 找不到 stats.json：{stats_in}")
if not os.path.exists(features_txt):
    raise FileNotFoundError(f"[error] 找不到 features.txt：{features_txt}")

feat_order = [l.strip() for l in open(features_txt, "r", encoding="utf-8") if l.strip()]
stats = json.load(open(stats_in, "r", encoding="utf-8"))

mean_list, std_list = [], []

# ================= 格式转换 =================
for feat in feat_order:
    entry = stats.get(feat, {})
    m = entry.get("mean", 0.0)
    s = entry.get("std", 1.0)
    if s == 0:
        s = 1e-6
    mean_list.append(float(m))
    std_list.append(float(s))

# ================= 写入新文件 =================
json.dump(
    {"mean": mean_list, "std": std_list},
    open(stats_out, "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2
)

print(f"[ok] 已转换为数组格式并保存至：{stats_out}")
print(f"[info] mean/std 长度：{len(mean_list)} / {len(std_list)}")

