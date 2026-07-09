"""把 GEX 当 gate 的连续调节接进仓位, 回测是否提升。
buy' = buy × mult,  mult = clamp(1 - K·gex_z, 0.4, 1.4)  (低GEX放大/高GEX压制)
其余(底仓30% + V2棘轮出场)不变。对比 BASE(无GEX) vs +GEX。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
FLOOR, TH, FULL, DECAY, K = 0.30, 0.15, 0.45, 0.04, 0.35


def clamp(x, a, b): return max(a, min(b, x))


def feats(T):
    n = len(T["dates"]); buy = [0.0]*n; hot = [0]*n
    for i in range(n):
        parts = []
        for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
            v = T[k][i]
            if v is not None: parts.append(v if hi else 1-v)
        mp, iv, vp = T["mom_pct"][i], T["iv_pct"][i], T["vrp_pct"][i]
        if parts and mp is not None:
            buy[i] = (sum(parts)/len(parts))*clamp((0.30-mp)/0.30, 0, 1)
        if mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75:
            hot[i] = 1
    return buy, hot


def weights(T, gz, use_gex):
    n = len(T["dates"]); buy, hot = feats(T); mp = T["mom_pct"]
    w = [FLOOR]*n; prev = FLOOR
    for i in range(n):
        b = buy[i]
        if use_gex:
            z = gz.get(T["dates"][i])
            if z is not None: b *= clamp(1 - K*z, 0.4, 1.4)
        tgt = FLOOR + (1-FLOOR)*clamp((b-TH)/(FULL-TH), 0, 1)
        if hot[i]: tgt = min(tgt, FLOOR*0.5)
        fast = hot[i] or (mp[i] is not None and mp[i] >= .85)
        wi = tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY)
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def bt(T, w, since):
    px = T["price"]; n = len(px); s = 0
    while s < n and T["dates"][s] < since: s += 1
    rs, ws = [], []
    for i in range(s+1, n):
        if px[i] and px[i-1]: rs.append(w[i-1]*(px[i]/px[i-1]-1)); ws.append(w[i-1])
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1
    shp = rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0
    eq = np.cumprod(1+rs); mdd = (eq/np.maximum.accumulate(eq)-1).min()
    return ann*100, shp, mdd*100, np.mean(ws)*100


for since, lab in [("20240924", "924后"), ("20000101", "全样本")]:
    print(f"===== {lab} · 年化/夏普/MaxDD/均仓 =====")
    for key in KEYS:
        T = TV[key]; G = json.load(open(PROC/f"gex_{key}.json"))
        gz = dict(zip(G["date"], G["gex_z"]))
        b = bt(T, weights(T, gz, False), since)
        g = bt(T, weights(T, gz, True), since)
        print(f" · {T['name']}")
        print(f"    BASE   {b[0]:>7.1f}{b[1]:>7.2f}{b[2]:>8.1f}{b[3]:>7.0f}%")
        print(f"    +GEX   {g[0]:>7.1f}{g[1]:>7.2f}{g[2]:>8.1f}{g[3]:>7.0f}%   "
              f"Δ夏普 {g[1]-b[1]:+.2f}  Δ回撤 {g[2]-b[2]:+.1f}")
