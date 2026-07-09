"""双入场引擎: 恐慌抄底(跌) + 趋势跟随(涨)。解决科创牛市踏空。
  w_dip   = ramp(panic × 跌幅门)          均值回归: 跌+恐慌才进
  w_trend = ramp(动量分位)                 趋势跟随: 涨势中持仓, 动量回落自然减
  目标 = max(w_dip, w_trend)               两引擎接力(抄底→反弹→动量接棒骑趋势→趋势结束减)
去掉 melt-up 降仓(强趋势里它是元凶)。对比当前 BASE。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, DECAY, GK, GLO, GHI, FEE = 0.15, 0.45, 0.04, 0.4, 0.4, 1.5, 0.0005
TRLO, TRHI = 0.55, 0.92    # 趋势入场: 动量分位 0.55 起加, 0.92 满仓


def clamp(x, a, b): return max(a, min(b, x))


def parts_panic(T, i):
    p = []
    for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = T[k][i]
        if v is not None: p.append(v if hi else 1-v)
    return sum(p)/len(p) if p else None


def w_base(T):        # 当前系统: 抄底 + melt-up + 棘轮, floor30
    n = len(T["dates"]); mp = T["mom_pct"]; gz = T.get("gex_z"); F = 0.30
    w = [F]*n; prev = F
    for i in range(n):
        pa = parts_panic(T, i)
        b = 0
        if pa is not None and mp[i] is not None: b = pa*clamp((0.30-mp[i])/0.30, 0, 1)
        if gz and gz[i] is not None: b *= clamp(1-GK*gz[i], GLO, GHI)
        tgt = F + (1-F)*clamp((b-TH)/(FULL-TH), 0, 1)
        iv, vp = T["iv_pct"][i], T["vrp_pct"][i]
        hot = mp[i] is not None and iv is not None and vp is not None and mp[i] >= .85 and iv >= .75 and vp >= .75
        if hot: tgt = min(tgt, F*0.5)
        fast = hot or (mp[i] is not None and mp[i] >= .85)
        wi = tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY)
        w[i] = clamp(wi, 0, 1); prev = wi
    return w


def w_dual(T):        # 双引擎: max(抄底, 趋势), floor0, 无melt-up
    n = len(T["dates"]); mp = T["mom_pct"]; gz = T.get("gex_z")
    w = [0.0]*n
    for i in range(n):
        pa = parts_panic(T, i); dip = 0
        if pa is not None and mp[i] is not None:
            b = pa*clamp((0.30-mp[i])/0.30, 0, 1)
            if gz and gz[i] is not None: b *= clamp(1-GK*gz[i], GLO, GHI)
            dip = clamp((b-TH)/(FULL-TH), 0, 1)
        trend = clamp((mp[i]-TRLO)/(TRHI-TRLO), 0, 1) if mp[i] is not None else 0
        w[i] = clamp(max(dip, trend), 0, 1)
    return w


def perf(T, w, since, tag=""):
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


print("=== 924后 · BASE(当前) vs DUAL(双引擎) · 均仓/年化/夏普/回撤/超额 ===")
for key in KEYS:
    T = TV[key]
    b = perf(T, w_base(T), "20240924"); d = perf(T, w_dual(T), "20240924")
    print(f" {T['name']}")
    print(f"   BASE  {b[0]:>5.0f}%{b[1]:>8.1f}%{b[2]:>7.2f}{b[3]:>8.1f}%{b[4]:>+8.1f}%")
    print(f"   DUAL  {d[0]:>5.0f}%{d[1]:>8.1f}%{d[2]:>7.2f}{d[3]:>8.1f}%{d[4]:>+8.1f}%")

# 科创 2025下半年起 均仓对比
T = TV["kc50"]; wb = w_base(T); wd = w_dual(T)
idx = [i for i in range(len(T["dates"])) if T["dates"][i] >= "20250601"]
print(f"\n科创50 2025-06起(涨{T['price'][idx[-1]]/T['price'][idx[0]]-1:+.0%}): "
      f"BASE均仓 {np.mean([wb[i] for i in idx])*100:.0f}% → DUAL均仓 {np.mean([wd[i] for i in idx])*100:.0f}%")
