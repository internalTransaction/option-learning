"""偏度/波动率微笑类因子。

期权在不同行权价上的 IV 并不相同, 构成"波动率微笑/偏斜":
  - Skew(偏斜): 虚值认沽 IV 高于虚值认购 IV 的程度, 反映市场为下跌尾部风险付的溢价。
      偏斜陡峭 -> 市场担忧下跌(防御情绪浓); 偏斜平坦/倒挂 -> 追涨情绪。
  - 25-delta Risk Reversal: IV(25Δ put) - IV(25Δ call), 经典的偏斜度量。

需要按行权价展开的期权链(含每个合约的 IV 与 delta/行权价), 由 option_current_em 快照提供。
与情绪因子类似, 时序化需按日累积。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import Factor, rolling_percentile


def skew_from_snapshot(
    chain: pd.DataFrame,
    underlying_filter: str,
    spot: float | None = None,
) -> dict:
    """从单日快照近似计算偏斜。

    做法(近似): 取虚值认沽与虚值认购各一档(距离平值最近的一档 OTM), 比较其 IV。
    需要期权链含: 名称(判断购/沽)、行权价、隐含波动率。
    返回 {skew, otm_put_iv, otm_call_iv}。
    """
    df = chain.copy()
    name_col = next((c for c in df.columns if c in ("名称", "期权名称", "name")), None)
    df = df[df[name_col].astype(str).str.contains(underlying_filter)].copy()

    strike_col = next((c for c in df.columns if c in ("行权价", "执行价", "strike")), None)
    iv_col = next((c for c in df.columns if c in ("隐含波动率", "impl_vol", "iv")), None)
    if strike_col is None or iv_col is None:
        raise ValueError("快照缺少 行权价/隐含波动率 列, 无法计算偏斜")

    df["_side"] = ["put" if "沽" in str(x) else "call" for x in df[name_col]]
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[iv_col] = pd.to_numeric(df[iv_col], errors="coerce")

    if spot is None:  # 用行权价中位数近似平值
        spot = df[strike_col].median()

    puts = df[(df["_side"] == "put") & (df[strike_col] < spot)]
    calls = df[(df["_side"] == "call") & (df[strike_col] > spot)]
    if puts.empty or calls.empty:
        return {}

    otm_put_iv = puts.loc[puts[strike_col].idxmax(), iv_col]   # 最接近平值的虚值认沽
    otm_call_iv = calls.loc[calls[strike_col].idxmin(), iv_col]  # 最接近平值的虚值认购
    return {
        "skew": float(otm_put_iv - otm_call_iv),
        "otm_put_iv": float(otm_put_iv),
        "otm_call_iv": float(otm_call_iv),
        "spot": float(spot),
    }


class SkewFactor(Factor):
    name = "skew"

    def compute(self, skew_history: pd.DataFrame, window: int = 252, **_) -> pd.DataFrame:
        """从已累积的 skew 序列算因子(百分位等)。"""
        out = skew_history.copy().sort_index()
        if "skew" in out.columns:
            out["skew_pct"] = rolling_percentile(out["skew"], window)
        return out
