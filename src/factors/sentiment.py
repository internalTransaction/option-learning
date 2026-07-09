"""情绪/持仓类因子。

基于期权链快照(认购 call / 认沽 put 的持仓量、成交量)刻画市场情绪:
  - PCR (Put/Call Ratio): 认沽/认购 比值。
      * 成交量 PCR: 短期情绪, 过高常为恐慌(可能反转向上), 过低为乐观。
      * 持仓量 PCR: 中期仓位结构。
  - 认沽认购成交额比、持仓变化等。

⚠ akshare 的 option_current_em 是**当日快照**, 计算时序 PCR 需按日累积落盘。
本模块提供两种入口:
  - from_snapshot(): 从单日快照算出当日 PCR(用于实盘/当日信号)
  - compute(): 从已累积的历史 PCR 序列算因子(百分位、变化等)
"""
from __future__ import annotations

import pandas as pd

from src.factors.base import Factor, rolling_percentile, zscore


def _classify_side(df: pd.DataFrame) -> pd.Series:
    """从期权名称/字段判断认购(call)或认沽(put)。"""
    # option_current_em 常见列: '名称' 含 '购'/'沽'
    name_col = next((c for c in df.columns if c in ("名称", "期权名称", "name")), None)
    if name_col is not None:
        s = df[name_col].astype(str)
        return pd.Series(["put" if "沽" in x else "call" for x in s], index=df.index)
    raise ValueError("无法从期权链识别认购/认沽, 请检查列名")


def pcr_from_snapshot(chain: pd.DataFrame, underlying_filter: str | None = None) -> dict:
    """从单日期权链快照计算 PCR 指标。

    underlying_filter: 若给定(如 '300ETF'), 只统计名称含该关键字的合约。
    返回 {volume_pcr, oi_pcr, ...}。
    """
    df = chain.copy()
    if underlying_filter:
        name_col = next((c for c in df.columns if c in ("名称", "期权名称", "name")), None)
        df = df[df[name_col].astype(str).str.contains(underlying_filter)]
    df["_side"] = _classify_side(df)

    vol_col = next((c for c in df.columns if c in ("成交量", "volume")), None)
    oi_col = next((c for c in df.columns if c in ("持仓量", "openInterest", "oi")), None)

    res: dict = {}
    if vol_col:
        by = df.groupby("_side")[vol_col].sum()
        res["volume_pcr"] = by.get("put", 0) / max(by.get("call", 0), 1e-9)
    if oi_col:
        by = df.groupby("_side")[oi_col].sum()
        res["oi_pcr"] = by.get("put", 0) / max(by.get("call", 0), 1e-9)
    return res


class SentimentFactor(Factor):
    name = "sentiment"

    def compute(
        self,
        pcr_history: pd.DataFrame,
        window: int = 252,
        **_,
    ) -> pd.DataFrame:
        """
        参数
        ----
        pcr_history : index=date, 至少含列 volume_pcr / oi_pcr (由快照按日累积得到)
        """
        out = pcr_history.copy().sort_index()
        for col in ("volume_pcr", "oi_pcr"):
            if col in out.columns:
                out[f"{col}_pct"] = rolling_percentile(out[col], window)
                out[f"{col}_z"] = zscore(out[col], window)
        return out
