"""信号有效性检验（事件研究/分位数前瞻收益/IC）。

研究导向, 不做交易策略。回答: 某曲面指标(如 IV 情绪比)进入某分位后,
未来 N 日标的收益/波动是否有系统性差异。

核心函数:
  add_forward(df)            为每行加未来 N 日收益与已实现波动
  rolling_pct(s, window)     滚动分位数(中金用 63 日≈3个月)
  quantile_forward(...)      按信号分位分组, 看各组未来收益均值(反转/动量证据)
  information_coef(...)      信号与未来收益的秩相关(IC)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward(df: pd.DataFrame, price_col: str = "spot",
                horizons=(5, 10, 20)) -> pd.DataFrame:
    """加未来 N 日收益 fwd_ret_N 与未来 N 日已实现波动 fwd_rv_N(年化)。"""
    out = df.copy()
    px = out[price_col].astype(float)
    logret = np.log(px / px.shift(1))
    for h in horizons:
        out[f"fwd_ret_{h}"] = px.shift(-h) / px - 1.0
        out[f"fwd_rv_{h}"] = (
            logret.shift(-h).rolling(h).std().reindex(out.index) * np.sqrt(252)
        )
        # 未来 h 日已实现波动: 用 t+1..t+h 的收益标准差
        out[f"fwd_rv_{h}"] = [
            logret.iloc[i + 1:i + 1 + h].std() * np.sqrt(252) if i + 1 + h <= len(logret) else np.nan
            for i in range(len(out))
        ]
    return out


def add_vrp(df: pd.DataFrame, iv_col: str = "atm_iv", price_col: str = "spot",
            rv_window: int = 20) -> pd.DataFrame:
    """加已实现波动 rv(过去 rv_window 日年化) 与方差风险溢价 vrp = IV − RV。

    vrp>0(常态): 期权隐含波动高于已实现, 卖方赚风险溢价; vrp 极端可作情绪/风险状态信号。
    """
    out = df.copy()
    logret = np.log(out[price_col].astype(float) / out[price_col].astype(float).shift(1))
    out["rv"] = logret.rolling(rv_window).std() * np.sqrt(252)
    out["vrp"] = out[iv_col] - out["rv"]
    return out


def rolling_pct(s: pd.Series, window: int = 63) -> pd.Series:
    """滚动分位数(当前值在过去 window 期中的分位, 0~1)。"""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: (x[-1] >= x).mean(), raw=True
    )


def quantile_forward(df: pd.DataFrame, signal_col: str, ret_col: str = "fwd_ret_5",
                     n: int = 5) -> pd.DataFrame:
    """按 signal 分 n 组, 统计各组未来收益均值/胜率/样本数。

    若信号是反转型(如 IV 情绪比), 期望最高分位组的未来收益显著更高。
    """
    d = df[[signal_col, ret_col]].dropna()
    if len(d) < n * 5:
        return pd.DataFrame()
    d = d.assign(bucket=pd.qcut(d[signal_col], n, labels=False, duplicates="drop"))
    g = d.groupby("bucket")[ret_col]
    res = pd.DataFrame({
        "mean_fwd_ret": g.mean(),
        "win_rate": g.apply(lambda x: (x > 0).mean()),
        "n": g.size(),
        "signal_lo": d.groupby("bucket")[signal_col].min(),
        "signal_hi": d.groupby("bucket")[signal_col].max(),
    })
    return res.round(4)


def information_coef(df: pd.DataFrame, signal_col: str, ret_col: str = "fwd_ret_5") -> float:
    """信号与未来收益的 Spearman 秩相关(IC)。正=动量, 负=反转。"""
    d = df[[signal_col, ret_col]].dropna()
    if len(d) < 20:
        return float("nan")
    return float(d[signal_col].rank().corr(d[ret_col].rank()))


def summary(df: pd.DataFrame, signal_col: str, horizons=(5, 10, 20), n: int = 5) -> dict:
    """对一个信号做多期限的分位分析 + IC 汇总。"""
    d = add_forward(df, horizons=horizons)
    out = {"signal": signal_col, "n_obs": int(d[signal_col].notna().sum()), "ic": {}, "spread": {}}
    for h in horizons:
        rc = f"fwd_ret_{h}"
        out["ic"][h] = round(information_coef(d, signal_col, rc), 4)
        q = quantile_forward(d, signal_col, rc, n)
        if not q.empty:
            # 最高分位组 − 最低分位组 的未来收益差(反转信号应为正)
            out["spread"][h] = round(float(q["mean_fwd_ret"].iloc[-1] - q["mean_fwd_ret"].iloc[0]), 4)
    return out
