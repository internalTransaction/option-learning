"""60min K线 → 完整 VWMA 版 HVWMA(你的参数, 时间尺度等价3H×21)→ 日频趋势方向。
在60min上用 len=63(=21根3H×每根3根60min)、RMA平滑30(=10×3)、几何、成交量加权,
避开A股午休导致的3H边界歧义。取每日最后一根bar的方向作当日趋势状态, 整合回测对比。
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TV = json.load(open(ROOT/"data/processed/timing_viz.json"))
N60, SMOOTH60, BASE_TREND = 63, 30, 0.65
TH, FULL, GK, GLO, GHI, FEE = 0.15, 0.45, 0.4, 0.4, 1.5, 0.0005


def clamp(x, a, b): return max(a, min(b, x))
def wma(s, n):
    n = max(1, int(n)); w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
def vwma(s, cv, n):
    n = max(1, int(n))
    return wma(s*cv, n)/wma(cv, n)
def vol_hma(s, cv, n):
    half, sq = max(1, n//2), max(1, round(math.sqrt(n)))
    return wma(2*vwma(s, cv, half) - vwma(s, cv, n), sq)


def hvwma_daily_dir(key):
    """返回 {YYYYMMDD: +1绿/-1红}"""
    m = pd.read_parquet(ROOT/f"data/raw/min60_{key}.parquet").copy()
    m["dt"] = pd.to_datetime(m["trade_time"])
    m["t"] = m["dt"].dt.strftime("%H:%M")
    m = m[m["t"].isin(["10:30", "11:30", "14:00", "15:00"])]        # 标准4根/天, 丢开盘竞价
    m = m.sort_values("dt").reset_index(drop=True)
    cv = m["vol"].fillna(0)+1.0
    base = np.log(m["close"])
    hma = vol_hma(base, cv, N60)
    out = np.exp(hma.ewm(alpha=1/SMOOTH60, adjust=False).mean())
    m["dir"] = np.sign(out.diff()).fillna(0)
    m["date"] = m["dt"].dt.strftime("%Y%m%d")
    return m.groupby("date")["dir"].last().to_dict()


def w_dip(T):
    n = len(T["dates"]); mp = T["mom_pct"]; gz = T.get("gex_z"); dip = [0.0]*n
    for i in range(n):
        p = [(v if hi else 1-v) for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)] for v in [T[k][i]] if v is not None]
        if p and mp[i] is not None:
            b = (sum(p)/len(p))*clamp((0.30-mp[i])/0.30, 0, 1)
            if gz and gz[i] is not None: b *= clamp(1-GK*gz[i], GLO, GHI)
            dip[i] = clamp((b-TH)/(FULL-TH), 0, 1)
    return dip


def w_current(T):
    n = len(T["dates"]); mp = T["mom_pct"]; dip0 = w_dip(T); F = 0.30; w = [F]*n; prev = F
    for i in range(n):
        tgt = F+(1-F)*dip0[i]; iv, vp = T["iv_pct"][i], T["vrp_pct"][i]
        hot = mp[i] is not None and iv is not None and vp is not None and mp[i] >= .85 and iv >= .75 and vp >= .75
        if hot: tgt = min(tgt, F*0.5)
        fast = hot or (mp[i] is not None and mp[i] >= .85)
        w[i] = clamp(tgt if (tgt >= prev or fast) else max(tgt, prev-0.04), 0, 1); prev = w[i]
    return w


def w_combine(T, dirmap):
    dip = w_dip(T)
    return [max(BASE_TREND if dirmap.get(T["dates"][i], 0) > 0 else 0.0, dip[i]) for i in range(len(T["dates"]))]


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


if __name__ == "__main__":
    key = "kc50"; T = TV[key]
    dm = hvwma_daily_dir(key)
    cov = [d for d in T["dates"] if d in dm]
    print(f"{key}: 60min覆盖 {len(dm)}天, 与曲面重叠 {len(cov)}天  ({cov[0] if cov else '-'}~{cov[-1] if cov else '-'})")
    for since, lab in [("20240924", "924后"), ("20000101", "全样本")]:
        print(f"\n== {lab} · 均仓/年化/夏普/回撤 ==")
        for nm, w in [("当前(纯期权)", w_current(T)), ("HVWMA-60min结合", w_combine(T, dm))]:
            a, s, m, mw = perf(T, w, since)
            print(f"  {nm:<16}均仓{mw:>4.0f}% 年化{a:>6.1f}% 夏普{s:>5.2f} 回撤{m:>6.1f}%")
    idx = [i for i in range(len(T["dates"])) if "20250601" <= T["dates"][i] <= "20251031" and T["dates"][i] in dm]
    wc = w_combine(T, dm)
    print(f"\n科创 2025-06~10: 60min-HVWMA绿占比 {np.mean([dm[T['dates'][i]]>0 for i in idx])*100:.0f}%  结合均仓 {np.mean([wc[i] for i in idx])*100:.0f}%")
