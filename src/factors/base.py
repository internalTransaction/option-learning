"""因子基类。

一个因子 = 输入若干原始数据, 输出一列/多列按日期对齐的因子值 (index=date)。
所有因子实现 compute() 并返回 pd.DataFrame(index=DatetimeIndex)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    name: str = "factor"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def compute(self, **data) -> pd.DataFrame:
        """计算因子, 返回以 date 为索引的 DataFrame。"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Factor {self.name} {self.params}>"


def zscore(s: pd.Series, window: int) -> pd.Series:
    """滚动 z-score, 用于把不同量纲因子标准化到可比区间。"""
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    return (s - mean) / std


def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """滚动百分位 (当前值在过去 window 期中的分位, 0~1)。"""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: (x[-1] >= x).mean(), raw=True
    )
