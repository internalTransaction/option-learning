"""科创50 短期过热时买入看跌期权(put)的胜率与赔率(用真实期权价格)。
过热定义(与信号台 melt-up 一致): 动量分位≥.85 且 IV分位≥.75 且 VRP分位≥.75。
每个过热日买入近月 ATM put(strike最接近spot, 剩余DTE∈[H+5,60]), 持有 H 个交易日,
到期则按内在价值 max(K−spot,0) 结算。统计胜率、平均盈亏、赔率(均值盈利/|均值亏损|)。
对照: 全样本随机日买同类 put 的基准。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TV = json.load(open(ROOT/"data/processed/timing_viz.json"))["kc50"]
BASIC = pd.read_parquet(ROOT/"data/raw/ts_optbasic_kc50.parquet")
DAILY = pd.read_parquet(ROOT/"data/raw/ts_optdaily_kc50_20230601_20260707.parquet")
DAILY["trade_date"] = DAILY["trade_date"].astype(str)
BASIC["maturity_date"] = BASIC["maturity_date"].astype(str)
PUTS = BASIC[BASIC["call_put"] == "P"][["ts_code", "exercise_price", "maturity_date"]]
# 价格查询: (ts_code, date) -> close
PX = {(r.ts_code, r.trade_date): r.close for r in DAILY.itertuples()}

dates = TV["dates"]; spot = TV["price"]; n = len(dates)
di = {d: i for i, d in enumerate(dates)}


def days_between(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days


def pick_put(t_date, S, H):
    """近月(剩余DTE∈[H+5,60]) ATM put 的 ts_code, strike, maturity"""
    cand = PUTS.copy()
    cand["dte"] = cand["maturity_date"].apply(lambda m: days_between(t_date, m))
    cand = cand[(cand["dte"] >= H+5) & (cand["dte"] <= 60)]
    if cand.empty: return None
    near = cand["maturity_date"].min()
    c = cand[cand["maturity_date"] == near].copy()
    c["dist"] = (c["exercise_price"] - S).abs()
    r = c.sort_values("dist").iloc[0]
    return r["ts_code"], float(r["exercise_price"]), r["maturity_date"]


def put_pnl(i, H):
    t = dates[i]; S = spot[i]
    if i+H >= n or S is None: return None
    p = pick_put(t, S, H)
    if p is None: return None
    code, K, mat = p
    entry = PX.get((code, t))
    if not entry or entry <= 0: return None
    ex_date = dates[i+H]
    exit_px = PX.get((code, ex_date))
    if exit_px is None:                      # 已到期/停牌 → 内在价值结算
        exit_px = max(K - (spot[i+H] or S), 0.0)
    return exit_px/entry - 1.0


def stats(pnls):
    a = np.array([x for x in pnls if x is not None])
    if len(a) == 0: return None
    wins, losses = a[a > 0], a[a <= 0]
    payoff = (wins.mean()/abs(losses.mean())) if len(wins) and len(losses) else np.nan
    return dict(n=len(a), win=round((a > 0).mean()*100, 1), avg=round(a.mean()*100, 1),
                med=round(np.median(a)*100, 1), payoff=round(payoff, 2),
                avg_win=round(wins.mean()*100, 1) if len(wins) else 0,
                avg_loss=round(losses.mean()*100, 1) if len(losses) else 0)


def is_hot(i):
    mp, iv, vp = TV["mom_pct"][i], TV["iv_pct"][i], TV["vrp_pct"][i]
    return mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75


for H in [5, 10, 20]:
    hot_idx = [i for i in range(n) if is_hot(i)]
    all_idx = list(range(n))
    sh = stats([put_pnl(i, H) for i in hot_idx])
    sa = stats([put_pnl(i, H) for i in all_idx])
    print(f"\n===== 买入近月ATM put · 持有{H}日 =====")
    if sh:
        print(f"  过热日买put  n={sh['n']:>3}  胜率{sh['win']:>5}%  均值{sh['avg']:>6}%  中位{sh['med']:>6}%  赔率{sh['payoff']:>5}  (赢均{sh['avg_win']}% / 亏均{sh['avg_loss']}%)")
    if sa:
        print(f"  全样本基准   n={sa['n']:>3}  胜率{sa['win']:>5}%  均值{sa['avg']:>6}%  中位{sa['med']:>6}%  赔率{sa['payoff']:>5}")
