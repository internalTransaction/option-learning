"""在"低IV(平静期)"买put保险 vs "高IV(过热)"买put — 分品种。
假设: 低IV时put便宜, 之后若IV上升+标的跌则双赢, 形成低胜率高赔率的尾部保险。
科创melt-up可能仍差; 沪深300/中证1000/创业板回调更真实, 可能更划算。
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
H, LO, HI = 10, 0.25, 0.75      # 2周交易日


def load(key):
    b = pd.read_parquet(ROOT/"data/raw"/BASIC_F[key])
    b = b[b["call_put"] == "P"][["ts_code", "exercise_price", "maturity_date"]].copy()
    b["maturity_date"] = b["maturity_date"].astype(str)
    d = pd.read_parquet(ROOT/"data/raw"/DAILY_F[key])[["ts_code", "trade_date", "close"]].copy()
    d["trade_date"] = d["trade_date"].astype(str)
    px = {(r.ts_code, r.trade_date): r.close for r in d.itertuples()}
    T = TV[key]
    return dict(puts=b, px=px, dates=T["dates"], spot=T["price"], iv=T["iv_pct"],
                di={x: i for i, x in enumerate(T["dates"])})


def daysb(a, b): return (pd.Timestamp(b)-pd.Timestamp(a)).days


def put_pnl(L, i):
    date = L["dates"][i]; S = L["spot"][i]
    if S is None or i+H >= len(L["dates"]): return None
    c = L["puts"].copy(); c["dte"] = c["maturity_date"].apply(lambda m: daysb(date, m))
    c = c[(c["dte"] >= H+5) & (c["dte"] <= 60)]
    if c.empty: return None
    near = c["maturity_date"].min(); cc = c[c["maturity_date"] == near].copy()
    cc["dist"] = (cc["exercise_price"]-S).abs(); r = cc.sort_values("dist").iloc[0]
    code, K = r["ts_code"], float(r["exercise_price"])
    entry = L["px"].get((code, date))
    if not entry or entry <= 0: return None
    exd = L["dates"][i+H]; ex = L["px"].get((code, exd))
    if ex is None: ex = max(K-(L["spot"][i+H] or S), 0.0)
    return ex/entry-1.0


def stats(a):
    a = np.array([x for x in a if x is not None])
    if len(a) == 0: return None
    win, loss = a[a > 0], a[a <= 0]
    be = None
    w = (a > 0).mean()
    need = (1-w)/w if w > 0 else np.inf
    return dict(n=len(a), w=round(w*100, 1), avg=round(a.mean()*100, 1),
                payoff=round(win.mean()/abs(loss.mean()), 2) if len(win) and len(loss) else np.nan,
                mx=round(a.max()*100, 0), need=round(need, 1))


print(f"== 各品种 低IV(≤{LO}) vs 高IV(≥{HI}) 买近月ATM put · 持有{H}日 ==")
print(f"  {'品种/时机':<16}{'n':>4}{'胜率':>7}{'均值':>8}{'赔率':>7}{'平衡赔率':>9}{'最大赔付':>9}")
for key in ["hs300", "zz1000", "kc50", "cyb"]:
    L = load(key)
    lo_idx = [i for i in range(len(L["dates"])) if L["iv"][i] is not None and L["iv"][i] <= LO]
    hi_idx = [i for i in range(len(L["dates"])) if L["iv"][i] is not None and L["iv"][i] >= HI]
    for tag, idxs in [("低IV买put", lo_idx), ("高IV买put", hi_idx)]:
        s = stats([put_pnl(L, i) for i in idxs])
        if s:
            print(f"  {TV[key]['name']+' '+tag:<16}{s['n']:>4}{s['w']:>6}%{s['avg']:>7}%{s['payoff']:>7}{s['need']:>9}{s['mx']:>8}%")
    print()
