"""极值反转盈亏比分析（面向主观/择时视角, 非量化因子）。

不同于 IC(衡量全分布的平均单调预测力), 本模块只问:
  当信号走到极端(超卖/恐慌)时, 之后是否倾向反转, 且盈亏比是否可观?

对每个触发日, 看未来 horizon 天的路径, 用最大有利/不利波动刻画机会的不对称性:
  MFE  Max Favorable Excursion: 窗口内相对入场的最大涨幅(多头有利)
  MAE  Max Adverse Excursion:   窗口内相对入场的最大跌幅(多头不利)
  payoff = 均值MFE / |均值MAE|  盈亏比
  hit    = 未来 horizon 日收益为正的比例  胜率

与"全样本基准"对比, 判断极值是否提供了普通日子没有的不对称机会。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _path_stats(px: np.ndarray, i: int, h: int) -> tuple[float, float, float]:
    """入场 i, 看 i+1..i+h 的路径, 返回 (ret_h, MFE, MAE)(相对入场价)。"""
    if i + h >= len(px):
        return np.nan, np.nan, np.nan
    base = px[i]
    fwd = px[i + 1:i + 1 + h] / base - 1.0
    return float(px[i + h] / base - 1.0), float(fwd.max()), float(fwd.min())


def extreme_reversal(df: pd.DataFrame, pct_col: str, direction: str = "high",
                     thresh: float = 0.9, h: int = 20, price_col: str = "spot") -> dict:
    """
    pct_col   : 信号的滚动分位列(0~1)
    direction : 'high'=分位>=thresh 触发(如恐慌类); 'low'=分位<=1-thresh 触发
    返回极值组与全样本基准的 胜率/均值收益/MFE/MAE/盈亏比。
    """
    d = df.dropna(subset=[pct_col, price_col]).reset_index(drop=True)
    px = d[price_col].to_numpy(dtype=float)
    trig = (d[pct_col] >= thresh) if direction == "high" else (d[pct_col] <= (1 - thresh))

    def agg(mask):
        rows = [_path_stats(px, i, h) for i in np.where(mask)[0]]
        rows = [r for r in rows if np.isfinite(r[0])]
        if not rows:
            return None
        arr = np.array(rows)
        ret, mfe, mae = arr[:, 0], arr[:, 1], arr[:, 2]
        avg_mfe, avg_mae = mfe.mean(), mae.mean()
        return {
            "n": len(rows),
            "hit": round(float((ret > 0).mean()), 3),
            "avg_ret": round(float(ret.mean()) * 100, 2),
            "med_ret": round(float(np.median(ret)) * 100, 2),
            "avg_mfe": round(float(avg_mfe) * 100, 2),
            "avg_mae": round(float(avg_mae) * 100, 2),
            "payoff": round(float(avg_mfe / abs(avg_mae)), 2) if avg_mae != 0 else np.nan,
        }

    return {"extreme": agg(trig.to_numpy()), "baseline": agg(np.ones(len(d), bool)),
            "thresh": thresh, "h": h, "direction": direction}
