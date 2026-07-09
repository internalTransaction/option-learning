"""择时信号生成。

把因子组合成 -1/0/+1 的仓位信号 (看空/空仓/看多)。
起步版实现最经典、可直接跑通的**波动率择时**逻辑, 其余因子留好接口待接入。

波动率择时直觉(逆向为主):
  - IV 处于历史极高位(恐慌) -> 往往对应阶段性底部 -> 倾向看多(+1)
  - IV 处于历史极低位(自满) -> 隐含风险累积 -> 倾向谨慎(-1 或 0)
这是一个可被回测证伪的假设, 阈值需用 backtest 校准, 不要当作既定结论。
"""
from __future__ import annotations

import pandas as pd


def volatility_signal(
    vol_factor: pd.DataFrame,
    iv_pct_high: float = 0.80,
    iv_pct_low: float = 0.20,
    contrarian: bool = True,
) -> pd.Series:
    """基于 IV 百分位的择时信号。

    contrarian=True: 高 IV 看多、低 IV 看空(逆向)。
    contrarian=False: 高 IV 看空、低 IV 看多(顺势/趋势)。
    返回 index=date 的 {-1, 0, +1} 仓位序列。
    """
    pct = vol_factor["iv_percentile"]
    sig = pd.Series(0, index=pct.index, dtype=int)
    high, low = pct >= iv_pct_high, pct <= iv_pct_low
    if contrarian:
        sig[high] = 1
        sig[low] = -1
    else:
        sig[high] = -1
        sig[low] = 1
    return sig.rename("signal")


def combine_signals(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """多因子信号加权合成 -> 连续分数, 再离散成 {-1,0,+1}。

    signals: {name: Series(index=date, -1/0/+1)}
    """
    df = pd.DataFrame(signals).sort_index().fillna(0)
    w = weights or {k: 1.0 for k in df.columns}
    score = sum(df[k] * w.get(k, 0.0) for k in df.columns)
    total_w = sum(abs(v) for v in w.values()) or 1.0
    score = score / total_w
    out = pd.Series(0, index=df.index, dtype=int)
    out[score >= 0.5] = 1
    out[score <= -0.5] = -1
    return out.rename("signal")
