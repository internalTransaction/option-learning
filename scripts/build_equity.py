"""净值数据: 两套策略(纯期权择时 / HVWMA结合) + 满仓 + 等均仓恒定。
纯期权 = 恐慌灯抄底×动态跌幅 + GEX + melt-up + 棘轮, floor30。
HVWMA结合 = HVWMA(日线近似len16)绿→0.65打底, 红→期权抄底overlay。
(HVWMA分钟精确版待限频恢复; 此处日线近似。)含5bps换手成本。
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
FLOOR, TH, FULL, DECAY, GK, GLO, GHI, FEE = 0.30, 0.15, 0.45, 0.04, 0.4, 0.4, 1.5, 0.0005
BASE_TREND, HLEN, SMOOTH = 0.65, 16, 5
RANGES = [("924", "20240924"), ("all", "20000101")]


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


def dip_arr(T):
    n = len(T["dates"]); mp = T["mom_pct"]; gz = T.get("gex_z"); dip = [0.0]*n
    for i in range(n):
        p = [(v if hi else 1-v) for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)] for v in [T[k][i]] if v is not None]
        if p and mp[i] is not None:
            b = (sum(p)/len(p))*clamp((0.30-mp[i])/0.30, 0, 1)
            if gz and gz[i] is not None: b *= clamp(1-GK*gz[i], GLO, GHI)
            dip[i] = clamp((b-TH)/(FULL-TH), 0, 1)
    return dip


def weights_opt(T):
    n = len(T["dates"]); mp = T["mom_pct"]; dip = dip_arr(T); w = [FLOOR]*n; prev = FLOOR
    for i in range(n):
        tgt = FLOOR+(1-FLOOR)*dip[i]; iv, vp = T["iv_pct"][i], T["vrp_pct"][i]
        hot = mp[i] is not None and iv is not None and vp is not None and mp[i] >= .85 and iv >= .75 and vp >= .75
        if hot: tgt = min(tgt, FLOOR*0.5)
        fast = hot or (mp[i] is not None and mp[i] >= .85)
        w[i] = clamp(tgt if (tgt >= prev or fast) else max(tgt, prev-DECAY), 0, 1); prev = w[i]
    return w


def weights_hv(T):
    d = hvwma_dir(T["price"]); dip = dip_arr(T)
    return [clamp(max(BASE_TREND if d[i] > 0 else 0.0, dip[i]), 0, 1) for i in range(len(d))]


def stat(rets):
    rets = np.array(rets); n = len(rets); nav = np.cumprod(1+rets)
    ann = nav[-1]**(252/n)-1; shp = rets.mean()/rets.std()*np.sqrt(252) if rets.std() else 0
    mdd = (nav/np.maximum.accumulate(nav)-1).min()
    return dict(ann=round(ann*100, 1), sharpe=round(shp, 2), mdd=round(mdd*100, 1),
                calmar=round(ann/abs(mdd), 2) if mdd else 0)


def strat_series(T, s, w):
    px = T["dates"]; pr = T["price"]; n = len(pr)
    sr, cr, wser = [], [], [round(w[s]*100, 1)]
    mw = np.mean([w[i] for i in range(s, n)])
    for i in range(s+1, n):
        if pr[i] is None or pr[i-1] is None: continue
        r = pr[i]/pr[i-1]-1; turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        sr.append(w[i-1]*r-turn*FEE); cr.append(mw*r); wser.append(round(w[i]*100, 1))
    snav = np.cumprod([1]+[1+x for x in sr]); cnav = np.cumprod([1]+[1+x for x in cr])
    sdd = snav/np.maximum.accumulate(snav)-1
    d = np.array(sr)-np.array(cr); ir = d.mean()/d.std()*np.sqrt(252) if d.std() else 0
    st = stat(sr); st.update(meanw=round(mw*100), ir=round(ir, 2),
                             excess_ann=round(st["ann"]-stat(cr)["ann"], 1))
    return {"strat": [round(x, 4) for x in snav], "dd": [round(x*100, 2) for x in sdd],
            "excess": [round(x, 4) for x in (snav/cnav)], "weight": wser, "stats": st}


def run(T, since):
    dates = T["dates"]; pr = T["price"]; n = len(pr); s = 0
    while s < n and dates[s] < since: s += 1
    ds = [dates[s]]; br = []
    for i in range(s+1, n):
        if pr[i] is None or pr[i-1] is None: continue
        br.append(pr[i]/pr[i-1]-1); ds.append(dates[i])
    bnav = np.cumprod([1]+[1+x for x in br]); bdd = bnav/np.maximum.accumulate(bnav)-1
    return {"dates": ds, "bench": [round(x, 4) for x in bnav], "bdd": [round(x*100, 2) for x in bdd],
            "bench_stat": stat(br),
            "opt": strat_series(T, s, weights_opt(T)), "hv": strat_series(T, s, weights_hv(T))}


out = {}
for key in KEYS:
    T = TV[key]
    out[key] = {"name": T["name"], "ranges": {r: run(T, st) for r, st in RANGES}}
    s = out[key]["ranges"]["924"]
    print(f"{T['name']:<8} 924后  纯期权 年化{s['opt']['stats']['ann']}%/夏普{s['opt']['stats']['sharpe']}/超额{s['opt']['stats']['excess_ann']:+}%/均仓{s['opt']['stats']['meanw']}%"
          f"  |  HVWMA 年化{s['hv']['stats']['ann']}%/夏普{s['hv']['stats']['sharpe']}/超额{s['hv']['stats']['excess_ann']:+}%/均仓{s['hv']['stats']['meanw']}%")

(PROC/"equity.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print("写出 equity.json")
