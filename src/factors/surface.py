"""波动率曲面结构因子（从期权链快照提取）。

研究对象（对应偏度/微笑家族）：
  - atm_iv       : 平值隐含波动率(现价处), 曲面的"水平"
  - rr_25d       : 25-delta 风险反转 = IV(25Δput) − IV(25Δcall), 曲面的"斜率/偏度"
                   正值=认沽更贵=市场偏防御(股票市场常态)。
  - bf_25d       : 25-delta 蝶式 = (IV(25Δput)+IV(25Δcall))/2 − atm_iv, 曲面的"凸度/微笑深度"
  - skew_slope   : IV 对行权价的局部斜率(每 1% moneyness 的 IV 变化)
  - cp_atm_gap   : 同一平值行权价上 put IV − call IV, 反映看跌看涨平价偏离(你关注的"ATM偏离")

输入为 src.data.option_chain.fetch_chain 的快照 DataFrame。
先清洗掉无成交的垃圾 IV(深度实值, IV≈0), 再在各 wing 上按 delta 插值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(chain: pd.DataFrame, iv_floor: float = 0.02) -> pd.DataFrame:
    """剔除无效 IV(深度实值无成交合约 IV≈0)。"""
    df = chain.copy()
    df = df[(df["iv"] > iv_floor) & df["iv"].notna() & df["strike"].notna()]
    return df


def _interp_iv_by_delta(wing: pd.DataFrame, target_abs_delta: float) -> float:
    """在单个 wing(call 或 put)上, 按 |delta| 插值出目标 delta 处的 IV。"""
    w = wing.dropna(subset=["delta", "iv"]).copy()
    if len(w) < 2:
        return float("nan")
    w["ad"] = w["delta"].abs()
    w = w.sort_values("ad")
    if not (w["ad"].min() <= target_abs_delta <= w["ad"].max()):
        return float("nan")  # 不外推
    return float(np.interp(target_abs_delta, w["ad"], w["iv"]))


def _atm_iv(df: pd.DataFrame, spot: float) -> tuple[float, float, float]:
    """平值 IV: 分别用 call、put 在 strike=spot 处插值, 返回 (atm_call, atm_put, atm_avg)。"""
    def at(side):
        s = df[df["side"] == side].dropna(subset=["strike", "iv"]).sort_values("strike")
        if len(s) < 2 or not (s["strike"].min() <= spot <= s["strike"].max()):
            return float("nan")
        return float(np.interp(spot, s["strike"], s["iv"]))
    ac, ap = at("call"), at("put")
    avg = np.nanmean([ac, ap])
    return ac, ap, avg


def surface_metrics(chain: pd.DataFrame, delta: float = 0.25) -> dict:
    """从一张期权链快照计算曲面结构指标。"""
    spot = float(chain["spot"].iloc[0])
    df = _clean(chain)
    calls = df[df["side"] == "call"]
    puts = df[df["side"] == "put"]

    atm_call, atm_put, atm_avg = _atm_iv(df, spot)
    iv_call_25 = _interp_iv_by_delta(calls, delta)   # OTM call, delta≈+0.25
    iv_put_25 = _interp_iv_by_delta(puts, delta)     # OTM put,  delta≈-0.25

    rr = iv_put_25 - iv_call_25 if np.isfinite(iv_put_25) and np.isfinite(iv_call_25) else float("nan")
    bf = (np.nanmean([iv_put_25, iv_call_25]) - atm_avg
          if np.isfinite(atm_avg) else float("nan"))

    # 局部 skew 斜率: 用 90%~110% moneyness 区间的 IV 变化 / moneyness 变化
    slope = _skew_slope(df, spot)

    return {
        "spot": round(spot, 4),
        "expiry": str(chain["expiry"].iloc[0]),
        "atm_iv": _r(atm_avg),
        "atm_call_iv": _r(atm_call),
        "atm_put_iv": _r(atm_put),
        "cp_atm_gap": _r(atm_put - atm_call),      # ATM 看跌-看涨偏离
        f"iv_{int(delta*100)}d_call": _r(iv_call_25),
        f"iv_{int(delta*100)}d_put": _r(iv_put_25),
        "rr_25d": _r(rr),                          # 风险反转(偏度)
        "bf_25d": _r(bf),                          # 蝶式(凸度)
        "skew_slope": _r(slope),
        "n_valid": int(len(df)),
    }


def _skew_slope(df: pd.DataFrame, spot: float) -> float:
    """对全体有效点做 IV~moneyness 线性回归的斜率(每 +1% moneyness 的 IV 变化, 百分点)。"""
    d = df.dropna(subset=["strike", "iv"]).copy()
    if len(d) < 3:
        return float("nan")
    money = d["strike"] / spot - 1.0        # 0=平值, 正=虚值认购方向
    # 只用近月主区间 ±15% 避免深度虚值噪声
    m = money.abs() <= 0.15
    if m.sum() < 3:
        return float("nan")
    coef = np.polyfit(money[m] * 100, d["iv"][m] * 100, 1)[0]
    return float(coef)


def smile_curve(chain: pd.DataFrame) -> pd.DataFrame:
    """返回用于画微笑曲线的整洁表: strike, moneyness, side, iv, delta。"""
    df = _clean(chain)
    spot = float(chain["spot"].iloc[0])
    df["moneyness"] = df["strike"] / spot - 1.0
    return df[["side", "strike", "moneyness", "iv", "delta"]].sort_values(["side", "strike"])


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)
