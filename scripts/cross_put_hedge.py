"""科创过热时, 买"其他品种ETF put"做对冲(期权买方风险可控)。
先看科创过热日各品种的IV分位(谁的put贵), 再回测各品种近月ATM put在科创过热日买入持有H日的
胜率/均值/赔率——作为科创的"保险"哪个品种最划算。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TV = json.load(open(ROOT/"data/processed/timing_viz.json"))
DAILY_F = {"kc50": "ts_optdaily_kc50_20230601_20260707.parquet",
           "hs300": "ts_optdaily_300etf_20200101_20260707.parquet",
           "zz1000": "ts_optdaily_zz1000_20220801_20260707.parquet",
           "cyb": "ts_optdaily_cyb_20220901_20260707.parquet"}
BASIC_F = {"kc50": "ts_optbasic_kc50.parquet", "hs300": "ts_optbasic_300etf.parquet",
           "zz1000": "ts_optbasic_zz1000.parquet", "cyb": "ts_optbasic_cyb.parquet"}
H = 20


def load(key):
    b = pd.read_parquet(ROOT/"data/raw"/BASIC_F[key])
    b = b[b["call_put"] == "P"][["ts_code", "exercise_price", "maturity_date"]].copy()
    b["maturity_date"] = b["maturity_date"].astype(str)
    d = pd.read_parquet(ROOT/"data/raw"/DAILY_F[key])[["ts_code", "trade_date", "close"]].copy()
    d["trade_date"] = d["trade_date"].astype(str)
    px = {(r.ts_code, r.trade_date): r.close for r in d.itertuples()}
    T = TV[key]
    return dict(puts=b, px=px, dates=T["dates"], spot=T["price"],
                di={x: i for i, x in enumerate(T["dates"])})


def daysb(a, b): return (pd.Timestamp(b)-pd.Timestamp(a)).days


def put_pnl(L, date, S, ei):
    c = L["puts"].copy(); c["dte"] = c["maturity_date"].apply(lambda m: daysb(date, m))
    c = c[(c["dte"] >= H+5) & (c["dte"] <= 60)]
    if c.empty: return None
    near = c["maturity_date"].min(); cc = c[c["maturity_date"] == near].copy()
    cc["dist"] = (cc["exercise_price"]-S).abs(); r = cc.sort_values("dist").iloc[0]
    code, K = r["ts_code"], float(r["exercise_price"])
    entry = L["px"].get((code, date))
    if not entry or entry <= 0: return None
    exd = L["dates"][ei+H]; exit_px = L["px"].get((code, exd))
    if exit_px is None: exit_px = max(K-(L["spot"][ei+H] or S), 0.0)
    return exit_px/entry-1.0


def stats(a):
    a = np.array([x for x in a if x is not None])
    if len(a) == 0: return None
    win, loss = a[a > 0], a[a <= 0]
    return dict(n=len(a), w=round((a > 0).mean()*100, 1), avg=round(a.mean()*100, 1),
                payoff=round(win.mean()/abs(loss.mean()), 2) if len(win) and len(loss) else np.nan)


kc = TV["kc50"]
def hot(j):
    mp, iv, vp = kc["mom_pct"][j], kc["iv_pct"][j], kc["vrp_pct"][j]
    return mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75
hot_dates = [kc["dates"][j] for j in range(len(kc["dates"])) if hot(j)]

print("== 科创过热日, 各品种 ATM-IV 分位均值(越低=put越便宜) ==")
for key in ["kc50", "hs300", "zz1000", "cyb"]:
    di = {d: i for i, d in enumerate(TV[key]["dates"])}
    ivs = [TV[key]["iv_pct"][di[d]] for d in hot_dates if d in di and TV[key]["iv_pct"][di[d]] is not None]
    print(f"  {TV[key]['name']:<9} IV分位 {np.mean(ivs):.2f}   (n={len(ivs)})")

print(f"\n== 科创过热日买各品种近月ATM put · 持有{H}日 · 作科创保险 ==")
L = {k: load(k) for k in DAILY_F}
for key in ["kc50", "hs300", "zz1000", "cyb"]:
    l = L[key]
    pnls = []
    for d in hot_dates:
        if d not in l["di"]: continue
        i = l["di"][d]
        if i+H >= len(l["dates"]) or l["spot"][i] is None: continue
        pnls.append(put_pnl(l, d, l["spot"][i], i))
    s = stats(pnls)
    if s: print(f"  买 {TV[key]['name']:<9} put  n={s['n']:>3}  胜率{s['w']:>5}%  均值{s['avg']:>7}%  赔率{s['payoff']}")
