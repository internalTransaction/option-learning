"""HVWMA 趋势 + 期权抄底「结合」(非二选一)。
注: 用户实盘用 3小时K线×21; 此处只有日线, 用 len≈16 近似(21根3H≈16交易日), 盘中信息丢失、更滞后。
方案对比:
  当前         纯期权择时(抄底+GEX+棘轮, floor30)
  HVWMA二值    绿→满仓1.0 / 红→期权抄底           (上一版, 太粗)
  HVWMA结合    绿→0.65基准 + 期权抄底overlay / 红→期权抄底   (打底+加强)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

TV = json.load(open(Path(__file__).resolve().parents[1] / "data/processed/timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, DECAY, GK, GLO, GHI, FEE = 0.15, 0.45, 0.04, 0.4, 0.4, 1.5, 0.0005
HLEN, SMOOTH, BASE_TREND = 16, 5, 0.65     # 日线近似3H×21; RMA5; 趋势打底仓位


def clamp(x, a, b): return max(a, min(b, x))
def wma(s, n):
    n = max(1, int(n)); w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)


def hvwma_dir(price):
    s = pd.Series(np.log(price))
    half, sq = max(1, HLEN//2), max(1, round(math.sqrt(HLEN)))
    hma = wma(2*wma(s, half) - wma(s, HLEN), sq)
    out = np.exp(hma.ewm(alpha=1/SMOOTH, adjust=False).mean())
    return np.sign(out.diff()).fillna(0).to_numpy()


def w_dip(T):
    n = len(T["dates"]); mp = T["mom_pct"]; gz = T.get("gex_z"); dip = [0.0]*n
    for i in range(n):
        p = [(v if hi else 1-v) for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)] for v in [T[k][i]] if v is not None]
        if p and mp[i] is not None:
            b = (sum(p)/len(p))*clamp((0.30-mp[i])/0.30, 0, 1)
            if gz and gz[i] is not None: b *= clamp(1-GK*gz[i], GLO, GHI)
            dip[i] = clamp((b-TH)/(FULL-TH), 0, 1)
    return dip


def w_current(T):        # 当前系统: 抄底+melt-up+棘轮, floor30
    n = len(T["dates"]); mp = T["mom_pct"]; dip0 = w_dip(T); F = 0.30
    w = [F]*n; prev = F
    for i in range(n):
        tgt = F + (1-F)*dip0[i]
        iv, vp = T["iv_pct"][i], T["vrp_pct"][i]
        hot = mp[i] is not None and iv is not None and vp is not None and mp[i] >= .85 and iv >= .75 and vp >= .75
        if hot: tgt = min(tgt, F*0.5)
        fast = hot or (mp[i] is not None and mp[i] >= .85)
        w[i] = clamp(tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY), 0, 1); prev = w[i]
    return w


def w_binary(T):
    d = hvwma_dir(T["price"]); dip = w_dip(T)
    return [1.0 if d[i] > 0 else dip[i] for i in range(len(d))]


def w_combine(T):        # 绿→0.65打底 + 抄底overlay; 红→抄底
    d = hvwma_dir(T["price"]); dip = w_dip(T)
    return [max(BASE_TREND if d[i] > 0 else 0.0, dip[i]) for i in range(len(d))]


def perf(T, w, since):
    px = T["price"]; dt = T["dates"]; n = len(px); s = 0
    while s < n and dt[s] < since: s += 1
    rs, ws = [], []
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        rs.append(w[i-1]*(px[i]/px[i-1]-1)-turn*FEE); ws.append(w[i-1])
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1; shp = rs.mean()/rs.std()*np.sqrt(252)
    nav = np.cumprod(1+rs); mdd = (nav/np.maximum.accumulate(nav)-1).min()
    return ann*100, shp, mdd*100, np.mean(ws)*100


for since, lab in [("20240924", "924后"), ("20000101", "全样本")]:
    print(f"\n===== {lab} · 均仓/年化/夏普/回撤 =====")
    for key in KEYS:
        T = TV[key]
        rows = [("当前(纯期权)", w_current(T)), ("HVWMA二值", w_binary(T)), ("HVWMA结合", w_combine(T))]
        print(f" {T['name']}")
        for nm, w in rows:
            a, s, m, mw = perf(T, w, since)
            print(f"   {nm:<14}均仓{mw:>4.0f}%  年化{a:>6.1f}%  夏普{s:>5.2f}  回撤{m:>6.1f}%")

T = TV["kc50"]
idx = [i for i in range(len(T["dates"])) if "20250601" <= T["dates"][i] <= "20251031"]
for nm, w in [("当前", w_current(T)), ("HVWMA结合", w_combine(T))]:
    print(f"科创 2025-06~10 {nm} 均仓 {np.mean([w[i] for i in idx])*100:.0f}%") if nm else None
