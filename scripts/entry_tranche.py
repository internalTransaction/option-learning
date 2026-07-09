"""入场方式对比: 连续爬坡 vs 离散分档 vs 纯跌幅分批。都套 V2 棘轮出场。
验证"分段式建仓"是否比现在的连续映射有实质差异(先验: 差异很小)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DATA = json.load(open(Path(__file__).resolve().parents[1] / "data/processed/timing_viz.json"))
FLOOR, DECAY = 0.30, 0.04


def clamp(x, a, b): return max(a, min(b, x))


def feats(d):
    n = len(d["dates"]); buy = [0.0]*n; panic = [0.0]*n
    for i in range(n):
        parts = []
        for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
            v = d[k][i]
            if v is not None: parts.append(v if hi else 1-v)
        mp = d["mom_pct"][i]
        if parts and mp is not None:
            pa = sum(parts)/len(parts); panic[i] = pa
            buy[i] = pa*clamp((0.30-mp)/0.30, 0, 1)
    return buy, panic


def target(mode, buy, mp):
    if mode == "CONT":                       # 连续(现状)
        return FLOOR + (1-FLOOR)*clamp((buy-0.15)/0.30, 0, 1)
    if mode == "STEP3":                      # 3档
        return 1.0 if buy >= .40 else .75 if buy >= .26 else .50 if buy >= .13 else FLOOR
    if mode == "STEP5":                      # 5档
        for th, w in [(.50, 1.0), (.38, .85), (.27, .68), (.18, .50), (.10, .38)]:
            if buy >= th: return w
        return FLOOR
    if mode == "PYRAMID":                    # 纯跌幅分批(不乘panic)
        if mp is None: return FLOOR
        return 1.0 if mp <= .04 else .75 if mp <= .08 else .52 if mp <= .15 else FLOOR
    return FLOOR


def weights(d, mode):
    n = len(d["dates"]); buy, _ = feats(d); mp = d["mom_pct"]
    w = [FLOOR]*n; prev = FLOOR
    for i in range(n):
        tgt = target(mode, buy[i], mp[i])
        fast = mp[i] is not None and mp[i] >= .85
        wi = tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY)
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def bt(d, w, since):
    px = d["price"]; n = len(px); s = 0
    while s < n and d["dates"][s] < since: s += 1
    rs = []
    for i in range(s+1, n):
        if px[i] and px[i-1]: rs.append(w[i-1]*(px[i]/px[i-1]-1))
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1
    shp = rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0
    eq = np.cumprod(1+rs); mdd = (eq/np.maximum.accumulate(eq)-1).min()
    return ann*100, shp, mdd*100


print("===== 入场方式对比 · 924后 · 年化/夏普/MaxDD (均套V2棘轮出场) =====")
for k, d in DATA.items():
    print(f" · {d['name']}")
    for m in ["CONT", "STEP3", "STEP5", "PYRAMID"]:
        a, s, mdd = bt(d, weights(d, m), "20240924")
        print(f"    {m:<9}{a:>7.1f}{s:>7.2f}{mdd:>8.1f}")
