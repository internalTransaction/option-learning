"""科创过热时买中证1000不同虚值程度的put — 看赔率能否随"更虚"而变凸, 匹配低胜率。
ATM(1.0) / 95% / 90% / 85% 行权。对冲要的是凸性: 平时小损, 暴跌爆发赔付。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TV = json.load(open(ROOT/"data/processed/timing_viz.json"))
H = 20
b = pd.read_parquet(ROOT/"data/raw/ts_optbasic_zz1000.parquet")
PUTS = b[b["call_put"] == "P"][["ts_code", "exercise_price", "maturity_date"]].copy()
PUTS["maturity_date"] = PUTS["maturity_date"].astype(str)
d = pd.read_parquet(ROOT/"data/raw/ts_optdaily_zz1000_20220801_20260707.parquet")[["ts_code", "trade_date", "close"]].copy()
d["trade_date"] = d["trade_date"].astype(str)
PX = {(r.ts_code, r.trade_date): r.close for r in d.itertuples()}
Z = TV["zz1000"]; ZD = {x: i for i, x in enumerate(Z["dates"])}
kc = TV["kc50"]


def hot(j):
    mp, iv, vp = kc["mom_pct"][j], kc["iv_pct"][j], kc["vrp_pct"][j]
    return mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75
HOT = [kc["dates"][j] for j in range(len(kc["dates"])) if hot(j)]


def daysb(a, b): return (pd.Timestamp(b)-pd.Timestamp(a)).days


def pnl(date, m):
    if date not in ZD: return None
    i = ZD[date]; S = Z["price"][i]
    if S is None or i+H >= len(Z["dates"]): return None
    tgt = m*S
    c = PUTS.copy(); c["dte"] = c["maturity_date"].apply(lambda x: daysb(date, x))
    c = c[(c["dte"] >= H+5) & (c["dte"] <= 60)]
    if c.empty: return None
    near = c["maturity_date"].min(); cc = c[c["maturity_date"] == near].copy()
    cc["dist"] = (cc["exercise_price"]-tgt).abs(); r = cc.sort_values("dist").iloc[0]
    code, K = r["ts_code"], float(r["exercise_price"])
    if abs(K-tgt)/S > 0.05: return None            # 没有接近该虚值的合约
    entry = PX.get((code, date))
    if not entry or entry <= 0: return None
    exd = Z["dates"][i+H]; ex = PX.get((code, exd))
    if ex is None: ex = max(K-(Z["price"][i+H] or S), 0.0)
    return ex/entry-1.0, K/S


print(f"== 科创过热日买中证1000 put · 不同虚值 · 持有{H}日 ==")
print(f"  {'目标虚值':<8}{'实际K/S':>8}{'n':>4}{'胜率':>7}{'均值':>8}{'赔率':>7}{'最大赔付':>9}{'全损率':>8}")
for m in [1.00, 0.97, 0.95, 0.92, 0.90]:
    rows = [pnl(dt, m) for dt in HOT]
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"  {int(m*100)}%      (无合约)"); continue
    a = np.array([x[0] for x in rows]); ks = np.mean([x[1] for x in rows])
    win, loss = a[a > 0], a[a <= 0]
    payoff = win.mean()/abs(loss.mean()) if len(win) and len(loss) else np.nan
    full = (a <= -0.9).mean()
    print(f"  {int(m*100)}%       {ks:>7.2f}{len(a):>4}{(a>0).mean()*100:>6.0f}%{a.mean()*100:>7.1f}%{payoff:>7.2f}{a.max()*100:>8.0f}%{full*100:>7.0f}%")
print("\n盈亏平衡所需赔率 = (1-胜率)/胜率;  对冲看'最大赔付'(暴跌凸性)与'均值'(保险费)权衡")
