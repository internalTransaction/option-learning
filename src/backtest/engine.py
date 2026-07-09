"""向量化回测引擎(择时/仓位型)。

给定仓位信号(-1/0/+1)与标的收益, 计算策略净值与常用绩效。
信号在 T 日收盘生成, T+1 日按仓位吃收益, 避免未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(
    signal: pd.Series,
    underlying: pd.DataFrame,
    fee_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """
    参数
    ----
    signal     : index=date 的 {-1,0,+1} 仓位
    underlying : fetch_underlying 返回值 (date, close, ...)
    返回 : DataFrame(index=date), 含 ret / strat_ret / equity / bench_equity
    """
    px = underlying.set_index("date")["close"].astype(float).sort_index()
    ret = px.pct_change().rename("ret")

    pos = signal.reindex(ret.index).ffill().fillna(0)
    pos_lag = pos.shift(1).fillna(0)  # T+1 生效, 防未来函数

    cost_rate = (fee_bps + slippage_bps) / 1e4
    turnover = pos.diff().abs().fillna(0)
    cost = turnover * cost_rate

    strat_ret = (pos_lag * ret - cost).rename("strat_ret")
    df = pd.concat([ret, pos.rename("pos"), strat_ret], axis=1).dropna()
    df["equity"] = (1 + df["strat_ret"]).cumprod()
    df["bench_equity"] = (1 + df["ret"]).cumprod()
    return df


def performance(bt: pd.DataFrame, col: str = "strat_ret") -> dict:
    """常用绩效指标。"""
    r = bt[col].dropna()
    if r.empty:
        return {}
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    win = (r > 0).mean()
    return {
        "annual_return": round(float(ann), 4),
        "annual_vol": round(float(vol), 4),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(dd), 4),
        "win_rate": round(float(win), 3),
        "n_days": int(len(r)),
    }
