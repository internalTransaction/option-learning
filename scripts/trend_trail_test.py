"""双引擎入场(抄底+趋势) + trailing出场, 扫趋势入场参数。修科创牛市踏空。
当前净值逻辑=只有抄底入场(t=ramp(buy))，故上涨趋势仓位=0。这里 t=max(抄底, 趋势)。
去掉 melt-up 压制(强趋势里帮倒忙)。trailing TS=5% 保留(控回撤)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, TS, GEXK, GLO, GHI, FEE = 0.15, 0.45, 0.05, 0.4, 0.4, 1.5, 0.0005


def clamp(x, a, b): return max(a, min(b, x))


def dip_buy(T, i):
    parts = []
    for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = T[k][i]
        if v is not None: parts.append(v if hi else 1-v)
    mp = T["mom_pct"][i]
    if not parts or mp is None: return 0.0
    b = (sum(parts)/len(parts))*clamp((0.30-mp)/0.30, 0, 1)
    gz = T.get("gex_z")
    if gz and gz[i] is not None: b *= clamp(1-GEXK*gz[i], GLO, GHI)
    return clamp((b-TH)/(FULL-TH), 0, 1)


def weights(T, TRLO, TRHI):
    n = len(T["dates"]); px = T["price"]; mp = T["mom_pct"]
    w = [0.0]*n; prev = 0.0; pk = None
    for i in range(n):
        dip = dip_buy(T, i)
        trend = clamp((mp[i]-TRLO)/(TRHI-TRLO), 0, 1) if (TRLO is not None and mp[i] is not None) else 0
        t = max(dip, trend)
        if prev > 1e-6 and px[i] is not None:
            pk = px[i] if pk is None else max(pk, px[i])
            if px[i] < pk*(1-TS): wi = max(0.0, t); pk = None    # 跌破峰值-5% → 减
            else: wi = max(prev, t)                              # 骑趋势
        else:
            wi = t; pk = px[i] if (wi > 1e-6 and px[i] is not None) else None
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def perf(T, w, since):
    px = T["price"]; dates = T["dates"]; n = len(px); s = 0
    while s < n and dates[s] < since: s += 1
    sr, cr = [], []; ws = [w[i] for i in range(s, n)]; mw = np.mean(ws)
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        r = px[i]/px[i-1]-1; turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        sr.append(w[i-1]*r-turn*FEE); cr.append(mw*r)
    sr = np.array(sr); cr = np.array(cr); nn = len(sr)
    ann = (np.prod(1+sr))**(252/nn)-1; shp = sr.mean()/sr.std()*np.sqrt(252)
    nav = np.cumprod(1+sr); mdd = (nav/np.maximum.accumulate(nav)-1).min()
    exc = ann-((np.prod(1+cr))**(252/nn)-1)
    return mw*100, ann*100, shp, mdd*100, exc*100


def kc_h2(w):
    T = TV["kc50"]; idx = [i for i in range(len(T["dates"])) if '20250601' <= T["dates"][i] <= '20251031']
    return np.mean([w[i] for i in idx])*100


print("科创 2025-06~10 均仓(那波 +105%):")
print(f"  当前(仅抄底):     {kc_h2(weights(TV['kc50'], None, None)):.0f}%")
for TRLO, TRHI in [(0.50, 0.85), (0.55, 0.90), (0.60, 0.92)]:
    print(f"  双引擎 TRLO={TRLO}:   {kc_h2(weights(TV['kc50'], TRLO, TRHI)):.0f}%")

for TRLO, TRHI in [(None, None), (0.50, 0.85), (0.55, 0.90)]:
    lab = "当前(仅抄底)" if TRLO is None else f"双引擎 {TRLO}/{TRHI}"
    print(f"\n=== {lab} · 924后 · 均仓/年化/夏普/回撤/超额 ===")
    for key in KEYS:
        T = TV[key]; mw, ann, shp, mdd, exc = perf(T, weights(T, TRLO, TRHI), "20240924")
        print(f"  {T['name']:<9}{mw:>5.0f}%{ann:>8.1f}%{shp:>7.2f}{mdd:>8.1f}%{exc:>+8.1f}%")
