"""底仓 floor 权衡曲线: 资金使用率(均仓) vs 收益/回撤/择时超额。
floor↑ → 均仓↑(资金使用率↑)、收益趋近满仓、回撤↑、择时超额↓。给基金管理人选平衡点。
其余逻辑不变(极值加满 + 棘轮出场 + melt-up降到floor*0.5 + GEX调节)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, DECAY, GEXK, GLO, GHI, FEE = 0.15, 0.45, 0.04, 0.4, 0.4, 1.5, 0.0005
FLOORS = [0.30, 0.50, 0.65, 0.80, 1.00]


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


def weights(T, floor):
    n = len(T["dates"]); buy, hot = feats(T); mp = T["mom_pct"]; gz = T.get("gex_z")
    w = [floor]*n; prev = floor
    for i in range(n):
        b = buy[i]
        if gz and gz[i] is not None: b *= clamp(1-GEXK*gz[i], GLO, GHI)
        tgt = floor + (1-floor)*clamp((b-TH)/(FULL-TH), 0, 1)
        if hot[i]: tgt = min(tgt, floor*0.5)
        fast = hot[i] or (mp[i] is not None and mp[i] >= .85)
        wi = tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY)
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def perf(T, w, since):
    px = T["price"]; dates = T["dates"]; n = len(px); s = 0
    while s < n and dates[s] < since: s += 1
    sr, cr = [], []
    ws = [w[i] for i in range(s, n)]; mw = np.mean(ws)
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        r = px[i]/px[i-1]-1
        turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        sr.append(w[i-1]*r - turn*FEE); cr.append(mw*r)
    sr = np.array(sr); cr = np.array(cr); nn = len(sr)
    ann = (np.prod(1+sr))**(252/nn)-1
    shp = sr.mean()/sr.std()*np.sqrt(252)
    nav = np.cumprod(1+sr); mdd = (nav/np.maximum.accumulate(nav)-1).min()
    cann = (np.prod(1+cr))**(252/nn)-1
    d = sr-cr; ir = d.mean()/d.std()*np.sqrt(252) if d.std() else 0
    return mw*100, ann*100, shp, mdd*100, (ann-cann)*100, ir


for key in KEYS:
    T = TV[key]
    print(f"\n=== {T['name']} · 924后 · 底仓权衡 ===")
    print(f"  {'底仓':>5}{'均仓':>7}{'年化':>8}{'夏普':>7}{'回撤':>8}{'择时超额':>9}{'IR':>6}")
    for f in FLOORS:
        mw, ann, shp, mdd, exc, ir = perf(T, weights(T, f), "20240924")
        tag = "  ←满仓" if f == 1.0 else ("  ←当前" if f == 0.30 else "")
        print(f"  {f*100:>4.0f}%{mw:>6.0f}%{ann:>7.1f}%{shp:>7.2f}{mdd:>7.1f}%{exc:>+8.1f}%{ir:>6.2f}{tag}")
