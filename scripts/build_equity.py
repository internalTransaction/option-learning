"""净值曲线数据: 策略 vs 满仓买入持有 vs 等均仓恒定持仓(择时alpha基准)。
策略仓位 = 最终系统(恐慌灯×动态跌幅 + 底仓阈值 + 棘轮出场 + melt-up降仓 + GEX调节)。
含交易成本 FEE=5bps×换手。输出 data/processed/equity.json 供净值HTML用。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
TH, FULL, TS = 0.15, 0.45, 0.05                # 无底仓 + 趋势持有(trailing 5%)
TRLO, TRHI = 0.55, 0.90                         # 趋势入场: 20日动量分位 0.55 起加, 0.90 满仓
GEXK, GLO, GHI = 0.4, 0.4, 1.5
FEE = 0.0005          # 单位换手成本(5bps)
RANGES = [("924", "20240924"), ("all", "20000101")]


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


def weights(T):
    # 双引擎入场: t = max(恐慌抄底, 趋势跟随); trailing 出场; 无底仓; 无 melt-up 压制。
    n = len(T["dates"]); buy, _ = feats(T); px = T["price"]; gz = T.get("gex_z"); mp = T["mom_pct"]
    w = [0.0]*n; prev = 0.0; pk = None
    for i in range(n):
        b = buy[i]
        if gz and gz[i] is not None: b *= clamp(1-GEXK*gz[i], GLO, GHI)
        dip = clamp((b-TH)/(FULL-TH), 0, 1)                          # 均值回归: 跌+恐慌
        trend = clamp((mp[i]-TRLO)/(TRHI-TRLO), 0, 1) if mp[i] is not None else 0  # 趋势跟随: 涨势
        t = max(dip, trend)
        if prev > 1e-6 and px[i] is not None:            # 骑趋势: 未破位则保持, 不线性减
            pk = px[i] if pk is None else max(pk, px[i])
            if px[i] < pk*(1-TS): wi = max(0.0, t); pk = None        # 跌破峰值-5% → 减
            else: wi = max(prev, t)
        else:
            wi = t; pk = px[i] if (wi > 1e-6 and px[i] is not None) else None
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def stats(rets):
    rets = np.array(rets); n = len(rets)
    nav = np.cumprod(1+rets)
    ann = nav[-1]**(252/n)-1
    shp = rets.mean()/rets.std()*np.sqrt(252) if rets.std() else 0
    mdd = (nav/np.maximum.accumulate(nav)-1).min()
    return dict(ann=round(ann*100, 1), sharpe=round(shp, 2), mdd=round(mdd*100, 1),
                calmar=round(ann/abs(mdd), 2) if mdd else 0)


def run(T, since):
    px = T["price"]; dates = T["dates"]; n = len(px)
    w = weights(T); s = 0
    while s < n and dates[s] < since: s += 1
    ds, sr, br, cr, wser = [dates[s]], [], [], [], [round(w[s]*100, 1)]
    mw = np.mean([w[i] for i in range(s, n)])       # 均仓 → 恒定持仓基准
    idx = [s]
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        r = px[i]/px[i-1]-1
        turn = abs(w[i-1]-(w[i-2] if i-2 >= s else w[s]))
        sr.append(w[i-1]*r - turn*FEE)              # 策略(含成本)
        br.append(r)                                 # 满仓买入持有
        cr.append(mw*r)                              # 等均仓恒定
        ds.append(dates[i]); idx.append(i); wser.append(round(w[i]*100, 1))
    snav = np.cumprod([1]+[1+x for x in sr])
    bnav = np.cumprod([1]+[1+x for x in br])
    cnav = np.cumprod([1]+[1+x for x in cr])
    sdd = (snav/np.maximum.accumulate(snav)-1)
    bdd = (bnav/np.maximum.accumulate(bnav)-1)
    exc = snav/cnav                                  # 相对等均仓的择时alpha
    # 信息比率: 策略 vs 恒定持仓
    d = np.array(sr)-np.array(cr)
    ir = d.mean()/d.std()*np.sqrt(252) if d.std() else 0
    st = {"strat": stats(sr), "bench": stats(br), "const": stats(cr),
          "meanw": round(mw*100), "ir": round(ir, 2),
          "excess_ann": round(stats(sr)["ann"]-stats(cr)["ann"], 1)}
    return {"dates": ds,
            "strat": [round(x, 4) for x in snav], "bench": [round(x, 4) for x in bnav],
            "const": [round(x, 4) for x in cnav], "dd": [round(x*100, 2) for x in sdd],
            "bdd": [round(x*100, 2) for x in bdd], "excess": [round(x, 4) for x in exc],
            "weight": wser, "stats": st}


out = {}
for key in KEYS:
    T = TV[key]
    out[key] = {"name": T["name"], "ranges": {r: run(T, st) for r, st in RANGES}}
    s9 = out[key]["ranges"]["924"]["stats"]
    print(f"{T['name']:<8} 924后: 策略 年化{s9['strat']['ann']}% 夏普{s9['strat']['sharpe']} "
          f"回撤{s9['strat']['mdd']}% | 满仓 {s9['bench']['ann']}%/{s9['bench']['sharpe']} | "
          f"等均仓{s9['meanw']}% {s9['const']['ann']}% → 择时超额 {s9['excess_ann']:+}% (IR {s9['ir']})")

(PROC/"equity.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print("写出 equity.json")
