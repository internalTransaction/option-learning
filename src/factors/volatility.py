"""波动率类因子。

核心思想: 期权隐含波动率(IV)蕴含市场对未来风险的定价。
  - IV 绝对水平 / 百分位: 高位 -> 风险溢价高、常伴随下跌或恐慌(择时看空/防守);
                          低位 -> 情绪平静、隐含风险低(择时可布局)。
  - IV - HV 价差(方差风险溢价 VRP): IV 显著高于已实现波动 -> 期权偏贵, 情绪偏防御。
  - 波动率期限结构: 近月 IV 高于远月(backwardation) 常见于恐慌; 正常为 contango。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import Factor, rolling_percentile, zscore


class VolatilityFactor(Factor):
    name = "volatility"

    def compute(
        self,
        qvix: pd.DataFrame,
        underlying: pd.DataFrame,
        hv_window: int = 20,
        iv_percentile_window: int = 252,
        **_,
    ) -> pd.DataFrame:
        """
        参数
        ----
        qvix       : fetch_qvix 返回值 (date, open/high/low/close), close 为 IV 指数(%)
        underlying : fetch_underlying 返回值 (date, close, ...), 用于算 HV
        """
        iv = qvix.set_index("date")["close"].rename("iv").astype(float)

        # 历史(已实现)波动率: 对数收益年化标准差, 与 IV 同为百分数
        px = underlying.set_index("date")["close"].astype(float)
        logret = np.log(px / px.shift(1))
        hv = logret.rolling(hv_window).std() * np.sqrt(252) * 100
        hv = hv.rename("hv")

        out = pd.concat([iv, hv], axis=1).sort_index()
        out["iv"] = out["iv"].ffill()
        out["hv"] = out["hv"].ffill()

        # 方差风险溢价: IV - HV
        out["vrp"] = out["iv"] - out["hv"]
        # IV 百分位(过去一年), 0~1
        out["iv_percentile"] = rolling_percentile(out["iv"], iv_percentile_window)
        # IV z-score(标准化水平)
        out["iv_zscore"] = zscore(out["iv"], iv_percentile_window)
        # IV 日变化
        out["iv_chg"] = out["iv"].diff()

        return out.dropna(subset=["iv"])
