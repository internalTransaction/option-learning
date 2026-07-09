"""解决"趋势淹没择时": 趋势提供有上限的基准参与, 期权择时在其上叠加(加减)。
  DUAL     = max(dip, trend_full)          问题结构: 趋势顶满, 择时被淹没
  OVERLAY  = base(=CAP×trend) + (1-base)×dip - 过热cut   趋势留空间, 择时可见叠加
核心指标 = "期权因子边际贡献" = 含择时 − 纯趋势(同结构)。DUAL 该边际小(淹没), OVERLAY 应恢复。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

TV = json.load(open(Path(__file__).resolve().parents[1] / "data/processed/timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, GK, GLO, GHI, FEE = 0.15, 0.45, 0.4, 0.4, 1.5, 0.0005
TRLO, TRHI, CAP = 0.55, 0.92, 0.60      # 趋势基准上限 60%(留 40% 给择时上探)


def clamp(x, a, b): return max(a, min(b, x))


def sig(T, i):
    p = []
    for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = T[k][i]
        if v is not None: p.append(v if hi else 1-v)
    pa = sum(p)/len(p) if p else None
    mp, iv, vp, gz = T["mom_pct"][i], T["iv_pct"][i], T["vrp_pct"][i], (T.get("gex_z") or [None]*len(T["dates"]))[i]
    dip = 0
    if pa is not None and mp is not None:
        b = pa*clamp((0.30-mp)/0.30, 0, 1)
        if gz is not None: b *= clamp(1-GK*gz, GLO, GHI)
        dip = clamp((b-TH)/(FULL-TH), 0, 1)
    tr = clamp((mp-TRLO)/(TRHI-TRLO), 0, 1) if mp is not None else 0
    hot = mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75
    return dip, tr, hot


def build(T, mode):
    n = len(T["dates"]); w = []
    for i in range(n):
        dip, tr, hot = sig(T, i)
        if mode == "trend_full": wi = tr
        elif mode == "dual":     wi = max(dip, tr)
        elif mode == "base":     wi = CAP*tr
        elif mode == "overlay":
            base = CAP*tr
            wi = base + (1-base)*dip - (base*0.4 if hot else 0)
        w.append(clamp(wi, 0, 1))
    return w


def ann_of(T, w, since):
    px = T["price"]; d = T["dates"]; n = len(px); s = 0
    while s < n and d[s] < since: s += 1
    rs, ws = [], []
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        rs.append(w[i-1]*(px[i]/px[i-1]-1)-turn*FEE); ws.append(w[i-1])
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1; shp = rs.mean()/rs.std()*np.sqrt(252)
    nav = np.cumprod(1+rs); mdd = (nav/np.maximum.accumulate(nav)-1).min()
    return ann*100, shp, mdd*100, np.mean(ws)*100


print("=== 924后 · 期权因子边际贡献(含择时年化 − 纯趋势年化) ===")
print(f"  {'标的':<9}{'DUAL vs 纯趋势':>16}{'OVERLAY vs 基准':>18}")
for key in KEYS:
    T = TV[key]
    a_tf = ann_of(T, build(T, "trend_full"), "20240924")[0]
    a_du = ann_of(T, build(T, "dual"), "20240924")[0]
    a_ba = ann_of(T, build(T, "base"), "20240924")[0]
    a_ov = ann_of(T, build(T, "overlay"), "20240924")[0]
    print(f"  {T['name']:<9}{a_du-a_tf:>+14.1f}%{a_ov-a_ba:>+16.1f}%")

print("\n=== 924后 · OVERLAY(趋势基准60%+择时叠加) 完整表现 vs DUAL ===")
print(f"  {'标的':<9}{'':>6}{'均仓':>6}{'年化':>8}{'夏普':>7}{'回撤':>8}")
for key in KEYS:
    T = TV[key]
    for m in ["dual", "overlay"]:
        a, s, mdd, mw = ann_of(T, build(T, m), "20240924")
        print(f"  {T['name'] if m=='dual' else '':<9}{m:>7}{mw:>6.0f}%{a:>7.1f}%{s:>7.2f}{mdd:>7.1f}%")

# 科创下半年 OVERLAY 仓位
T = TV["kc50"]; w = build(T, "overlay")
idx = [i for i in range(len(T["dates"])) if "20250601" <= T["dates"][i] <= "20251031"]
print(f"\n科创 2025-06~10 OVERLAY 均仓 {np.mean([w[i] for i in idx])*100:.0f}% (DUAL 是 47%; 既参与又留择时空间)")
