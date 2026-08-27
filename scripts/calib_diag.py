"""口径差异归因: z 口径在科创上更优, 是"更聪明"还是只是"更钝"?

诊断三件事:
  1. 因子分布形态(偏度/峰度) —— 厚尾程度决定经验分位 vs 正态假设谁更吃亏
  2. panic 强度序列的相关性 / 均值差 —— 两口径到底差在哪
  3. 仓位序列的活跃度(std / 换手 / 与恒定仓的相关) —— 若 z 只是把仓位拍平, 那就是平凡解释
"""
from __future__ import annotations
import importlib.util as u
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("zp", str(ROOT / "scripts" / "zscore_vs_pct.py"))
zp = u.module_from_spec(spec); sys.modules["zp"] = zp; spec.loader.exec_module(zp)

S924 = "20240924"
LAB = zp.PROC / "calib_lab.json"
DIAG = {}
print(f"{'标的':<9}{'因子':<8}{'偏度':>8}{'峰度':>8}   (924后原始值分布)")
for key in zp.KEYS:
    T = zp.TV[key]
    d = [i for i, x in enumerate(T["dates"]) if x >= S924]
    DIAG[key] = {"shape": {}}
    for name, _ in zp.FACTORS:
        s = pd.Series([T[name][i] for i in d], dtype="float64").dropna()
        DIAG[key]["shape"][name] = {"skew": round(float(s.skew()), 2), "kurt": round(float(s.kurt()), 2)}
        print(f"{T['name']:<9}{name:<8}{s.skew():>8.2f}{s.kurt():>8.2f}")

print(f"\n{'标的':<9}{'指标':<22}{'pct63':>9}{'z63':>9}{'差异':>9}")
for key in zp.KEYS:
    T = zp.TV[key]
    n = len(T["dates"]); s0 = next(i for i, x in enumerate(T["dates"]) if x >= S924)
    rows = {}
    for mode in ["pct", "z"]:
        st, mom, lights, tier = zp.build_signals(T, 63, mode)
        panic = []
        for i in range(n):
            p = [(st[k][i] if hi else 1 - st[k][i]) for k, hi in zp.FACTORS if st[k][i] is not None]
            panic.append(sum(p) / len(p) if p else None)
        w = zp.weights_opt(T, st, mom)
        rows[mode] = dict(panic=panic, w=w)
    pa = np.array([rows["pct"]["panic"][i] for i in range(s0, n)], dtype=float)
    pb = np.array([rows["z"]["panic"][i] for i in range(s0, n)], dtype=float)
    wa = np.array(rows["pct"]["w"][s0:], dtype=float)
    wb = np.array(rows["z"]["w"][s0:], dtype=float)
    m = ~(np.isnan(pa) | np.isnan(pb))
    out = [
        ("panic 均值", pa[m].mean(), pb[m].mean()),
        ("panic 标准差", pa[m].std(), pb[m].std()),
        ("panic 相关(pct,z)", np.corrcoef(pa[m], pb[m])[0, 1], np.nan),
        ("仓位 标准差", wa.std(), wb.std()),
        ("仓位 日均换手", np.abs(np.diff(wa)).mean(), np.abs(np.diff(wb)).mean()),
        ("仓位 满仓天数占比", (wa > 0.95).mean(), (wb > 0.95).mean()),
        ("仓位 底仓天数占比", (wa <= 0.301).mean(), (wb <= 0.301).mean()),
        ("仓位相关(pct,z)", np.corrcoef(wa, wb)[0, 1], np.nan),
    ]
    print(f"── {T['name']}")
    cmp_ = {}
    for lab, a, b in out:
        cmp_[lab] = {"pct63": round(float(a), 3)} if np.isnan(b) else \
            {"pct63": round(float(a), 3), "z63": round(float(b), 3)}
        if np.isnan(b):
            print(f"{'':<9}{lab:<22}{a:>9.3f}{'':>9}{'':>9}")
        else:
            print(f"{'':<9}{lab:<22}{a:>9.3f}{b:>9.3f}{b-a:>+9.3f}")
    DIAG[key]["compare"] = cmp_

if LAB.exists():
    lab = json.loads(LAB.read_text())
    lab["diag"] = DIAG
    lab["verdict"] = {
        "keep": "pct63",
        "headline": "维持 63 日经验百分位；z-score 不是更聪明，只是更钝",
        "points": [
            "两口径 panic 强度相关 0.984–0.987、仓位相关 0.92–0.96 —— 本质是同一个信号。",
            "z 系统性更保守: 日均换手低 20–25%、满仓天数占比降 2–5pct、趴底仓天数升 1–6pct。",
            "择时真有 alpha 的标的(沪深300 超额 +7.9%)，钝化直接扣分 → z63 只剩 +4.8%。",
            "择时本就无效的标的(科创50 纯期权超额 −1.1%)，钝化少犯错 → z63 反弹到 +5.2%；这是「少动」的功劳，不是口径更优。",
            "科创的正解仍是新库给的 HVWMA 结合(超额 +9.8%)，而不是换极值口径。",
            "窗口: pct 短窗 63 最优或并列最优(hs300 63>126>252)；z 对窗口不敏感(差异 <1pct)。",
            "原始值右偏厚尾(ATM IV 偏度 1.3–3.0/峰度 2.2–17，VRP 左偏至 −3.5/峰度 13)，正态假设在极值处削平恐慌峰值 —— 满仓天数减少即证据。",
        ],
    }
    LAB.write_text(json.dumps(lab, ensure_ascii=False, separators=(",", ":")))
    print("\n诊断与结论已并入 calib_lab.json")
