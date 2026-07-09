"""趋势持有的止损幅度 TS 扫描(floor=0): 趋势型该松、均值回归型(中证1000)该紧?
看每标的不同 TS 的 均仓/年化/夏普/回撤/择时超额, 定分标的参数。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, GEXK, GLO, GHI, FEE = 0.15, 0.45, 0.4, 0.4, 1.5, 0.0005
TSS = [0.03, 0.05, 0.08, 0.12]


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


def w_trail(T, TS):
    n = len(T["dates"]); buy, hot = feats(T); px = T["price"]; gz = T.get("gex_z")
    w = [0.0]*n; prev = 0.0; pk = None
    for i in range(n):
        b = buy[i]
        if gz and gz[i] is not None: b *= clamp(1-GEXK*gz[i], GLO, GHI)
        t = clamp((b-TH)/(FULL-TH), 0, 1)
        if hot[i]: t = min(t, 0.15)
        if prev > 1e-6 and px[i] is not None:
            pk = px[i] if pk is None else max(pk, px[i])
            wi = max(0.0, t) if px[i] < pk*(1-TS) else max(prev, t)
            if px[i] < pk*(1-TS): pk = None
        else:
            wi = t; pk = px[i] if (wi > 1e-6 and px[i] is not None) else None
        if hot[i]: wi = min(wi, 0.15)
        w[i] = clamp(wi, 0, 1); prev = wi
    return w


def perf(T, w, since="20240924"):
    px = T["price"]; dates = T["dates"]; n = len(px); s = 0
    while s < n and dates[s] < since: s += 1
    sr, cr = [], []; mw = np.mean([w[i] for i in range(s, n)])
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        r = px[i]/px[i-1]-1; turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        sr.append(w[i-1]*r-turn*FEE); cr.append(mw*r)
    sr = np.array(sr); cr = np.array(cr); nn = len(sr)
    ann = (np.prod(1+sr))**(252/nn)-1; shp = sr.mean()/sr.std()*np.sqrt(252)
    nav = np.cumprod(1+sr); mdd = (nav/np.maximum.accumulate(nav)-1).min()
    exc = ann-((np.prod(1+cr))**(252/nn)-1)
    return mw*100, ann*100, shp, mdd*100, exc*100


for key in KEYS:
    T = TV[key]
    print(f"\n=== {T['name']} · floor0 趋势持有 · TS扫描 (均仓/年化/夏普/回撤/超额) ===")
    for ts in TSS:
        mw, ann, shp, mdd, exc = perf(T, w_trail(T, ts))
        print(f"  TS={ts*100:>2.0f}%   {mw:>5.0f}%{ann:>8.1f}%{shp:>7.2f}{mdd:>8.1f}%{exc:>+8.1f}%")
