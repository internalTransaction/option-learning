"""解决"趋势架空择时": 趋势只当保底参与度(封顶), 期权择时仍主导高信念仓位。
  trend_base = TCAP × ramp(动量分位)     趋势保底(封顶TCAP, 不满仓; 无趋势=0, 非固定floor)
  dip        = ramp(panic × 跌幅门 × GEX)  期权择时(0~1, 可独立驱动到满仓)
  w = max(trend_base, dip)               趋势设地板, 期权定高信念仓

消融三档, 关键看"期权层的边际贡献" = BLEND超额 − 趋势only超额:
  OPT_ONLY   纯期权择时(踏空趋势)
  TREND_ONLY 纯趋势保底(择时关掉)
  BLEND      两者
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, GK, GLO, GHI, FEE = 0.15, 0.45, 0.4, 0.4, 1.5, 0.0005
TRLO, TRHI, TCAP = 0.50, 0.90, 0.50    # 趋势保底: 动量.5起, .9到封顶; 封顶=50%(留一半给期权)


def clamp(x, a, b): return max(a, min(b, x))


def dip_of(T, i):
    p = []
    for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = T[k][i]
        if v is not None: p.append(v if hi else 1-v)
    mp = T["mom_pct"][i]
    if not p or mp is None: return 0.0
    b = (sum(p)/len(p))*clamp((0.30-mp)/0.30, 0, 1)
    gz = T.get("gex_z")
    if gz and gz[i] is not None: b *= clamp(1-GK*gz[i], GLO, GHI)
    return clamp((b-TH)/(FULL-TH), 0, 1)


def trend_of(T, i):
    mp = T["mom_pct"][i]
    return TCAP*clamp((mp-TRLO)/(TRHI-TRLO), 0, 1) if mp is not None else 0.0


def weights(T, mode):
    n = len(T["dates"]); w = [0.0]*n
    for i in range(n):
        d = dip_of(T, i); tr = trend_of(T, i)
        w[i] = d if mode == "opt" else tr if mode == "trend" else max(d, tr)
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


for since, lab in [("20240924", "924后"), ("20000101", "全样本")]:
    print(f"\n===== {lab} · 均仓/年化/夏普/回撤/超额  (期权贡献=BLEND超额−TREND超额) =====")
    for key in KEYS:
        T = TV[key]
        o = perf(T, weights(T, "opt"), since)
        t = perf(T, weights(T, "trend"), since)
        b = perf(T, weights(T, "blend"), since)
        print(f" {T['name']}")
        print(f"   OPT_ONLY   {o[0]:>5.0f}%{o[1]:>8.1f}%{o[2]:>7.2f}{o[3]:>8.1f}%{o[4]:>+8.1f}%")
        print(f"   TREND_ONLY {t[0]:>5.0f}%{t[1]:>8.1f}%{t[2]:>7.2f}{t[3]:>8.1f}%{t[4]:>+8.1f}%")
        print(f"   BLEND      {b[0]:>5.0f}%{b[1]:>8.1f}%{b[2]:>7.2f}{b[3]:>8.1f}%{b[4]:>+8.1f}%"
              f"   → 期权贡献 {b[4]-t[4]:+.1f}%")

# 科创下半年均仓
T = TV["kc50"]; wb = weights(T, "blend")
idx = [i for i in range(len(T["dates"])) if T["dates"][i] >= "20250601"]
print(f"\n科创50 2025-06起(涨{T['price'][idx[-1]]/T['price'][idx[0]]-1:+.0%}): BLEND均仓 {np.mean([wb[i] for i in idx])*100:.0f}% (原BASE 46%)")
