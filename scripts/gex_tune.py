"""GEX 调节强度 K 的参数扫描: 找稳健平台(非尖峰), 避免过拟合。
mult = clamp(1 - K·gex_z, 0.4, 1.5)。打印各标的 924后 / 全样本 夏普随 K 变化。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
FLOOR, TH, FULL, DECAY = 0.30, 0.15, 0.45, 0.04
LO, HI = 0.4, 1.5
KS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


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


def weights(T, gz, K):
    n = len(T["dates"]); buy, hot = feats(T); mp = T["mom_pct"]
    w = [FLOOR]*n; prev = FLOOR
    for i in range(n):
        b = buy[i]
        if K > 0:
            z = gz.get(T["dates"][i])
            if z is not None: b *= clamp(1 - K*z, LO, HI)
        tgt = FLOOR + (1-FLOOR)*clamp((b-TH)/(FULL-TH), 0, 1)
        if hot[i]: tgt = min(tgt, FLOOR*0.5)
        fast = hot[i] or (mp[i] is not None and mp[i] >= .85)
        wi = tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY)
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def sharpe(T, w, since):
    px = T["price"]; n = len(px); s = 0
    while s < n and T["dates"][s] < since: s += 1
    rs = [w[i-1]*(px[i]/px[i-1]-1) for i in range(s+1, n) if px[i] and px[i-1]]
    rs = np.array(rs)
    return rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0


for since, lab in [("20240924", "924后"), ("20000101", "全样本")]:
    print(f"===== {lab} · 夏普 vs K =====")
    print(f"  {'标的':<9}" + "".join(f"K={k:<5}" for k in KS))
    for key in KEYS:
        T = TV[key]; G = json.load(open(PROC/f"gex_{key}.json"))
        gz = dict(zip(G["date"], G["gex_z"]))
        row = f"  {T['name']:<9}"
        for K in KS:
            row += f"{sharpe(T, weights(T, gz, K), since):<7.2f}"
        print(row)
