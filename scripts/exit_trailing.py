"""出场逻辑: 线性棘轮 vs 趋势持有(trailing stop)。
用户洞察: 线性匀速减仓不合理; 应骑趋势, 涨着就不减, 破了才走 —— 顺便自然提高均仓。

  A 线性棘轮 floor30 (现状)  : 减仓每天-4%
  B 线性棘轮 floor0
  C 趋势持有 floor30          : 抄底进场后, 未跌破峰值-TS% 就保持仓位(不减); 破了回底仓; 过热降
  D 趋势持有 floor0           : 纯信号驱动 + 骑趋势
看 均仓(资金使用率) / 年化 / 夏普 / 回撤 / 择时超额(vs等均仓)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, DECAY, GEXK, GLO, GHI, FEE, TS = 0.15, 0.45, 0.04, 0.4, 0.4, 1.5, 0.0005, 0.08


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


def tgt_of(T, buy, hot, i, floor):
    b = buy[i]; gz = T.get("gex_z")
    if gz and gz[i] is not None: b *= clamp(1-GEXK*gz[i], GLO, GHI)
    t = floor + (1-floor)*clamp((b-TH)/(FULL-TH), 0, 1)
    if hot[i]: t = min(t, floor*0.5 if floor > 0 else 0.15)
    return t


def w_ratchet(T, floor):
    n = len(T["dates"]); buy, hot = feats(T); mp = T["mom_pct"]
    w = [floor]*n; prev = floor
    for i in range(n):
        t = tgt_of(T, buy, hot, i, floor)
        fast = hot[i] or (mp[i] is not None and mp[i] >= .85)
        wi = t if (t >= prev or fast) else max(t, prev-DECAY)
        w[i] = clamp(wi, 0, 1); prev = wi
    return w


def w_trail(T, floor):
    n = len(T["dates"]); buy, hot = feats(T); px = T["price"]
    w = [floor]*n; prev = floor; pk = None
    for i in range(n):
        t = tgt_of(T, buy, hot, i, floor)
        if prev > floor+1e-6 and px[i] is not None:      # 持有超额仓, 骑趋势
            pk = px[i] if pk is None else max(pk, px[i])
            if px[i] < pk*(1-TS):                        # 跌破峰值-TS% → 趋势破, 回底仓
                wi = max(floor, t); pk = None
            else:
                wi = max(prev, t)                        # 未破: 保持(或新信号加仓), 不线性减
        else:
            wi = t
            if wi > floor+1e-6 and px[i] is not None: pk = px[i]
        if hot[i]: wi = min(wi, floor if floor > 0 else 0.15)   # 过热降, 不骑
        w[i] = clamp(wi, 0, 1); prev = wi
    return w


def perf(T, w, since):
    px = T["price"]; dates = T["dates"]; n = len(px); s = 0
    while s < n and dates[s] < since: s += 1
    sr, cr = [], []; ws = [w[i] for i in range(s, n)]; mw = np.mean(ws)
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        r = px[i]/px[i-1]-1
        turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        sr.append(w[i-1]*r-turn*FEE); cr.append(mw*r)
    sr = np.array(sr); cr = np.array(cr); nn = len(sr)
    ann = (np.prod(1+sr))**(252/nn)-1; shp = sr.mean()/sr.std()*np.sqrt(252)
    nav = np.cumprod(1+sr); mdd = (nav/np.maximum.accumulate(nav)-1).min()
    exc = ann-((np.prod(1+cr))**(252/nn)-1)
    return mw*100, ann*100, shp, mdd*100, exc*100


for key in KEYS:
    T = TV[key]
    print(f"\n=== {T['name']} · 924后 · 均仓/年化/夏普/回撤/择时超额 ===")
    rows = [("A 线性棘轮 floor30", w_ratchet(T, 0.30)), ("B 线性棘轮 floor0", w_ratchet(T, 0.0)),
            ("C 趋势持有 floor30", w_trail(T, 0.30)), ("D 趋势持有 floor0", w_trail(T, 0.0))]
    for lab, w in rows:
        mw, ann, shp, mdd, exc = perf(T, w, "20240924")
        print(f"  {lab:<20}{mw:>6.0f}%{ann:>8.1f}%{shp:>7.2f}{mdd:>8.1f}%{exc:>+8.1f}%")
