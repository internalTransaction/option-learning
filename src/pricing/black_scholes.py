"""Black-Scholes 定价、希腊字母与隐含波动率反解。

用于从历史合约价格反解隐含波动率(重建历史波动率曲面), 以及做 delta 换算。
记号沿用 Natenberg《Option Volatility and Pricing》: 标的现价 S、行权价 K、
到期年限 T、无风险利率 r、波动率 sigma。ETF 期权按无连续分红处理(r 可含持有成本)。
"""
from __future__ import annotations

import math

from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_price(S, K, T, r, sigma, is_call=True) -> float:
    """欧式期权 BS 理论价。"""
    if T <= 0 or sigma <= 0:
        # 到期/零波动 -> 内在价值
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, is_call=True) -> float:
    if T <= 0 or sigma <= 0:
        return float((S > K) if is_call else -(S < K))
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def bs_vega(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T)


def implied_vol(
    price, S, K, T, r, is_call=True,
    lo=1e-4, hi=5.0, tol=1e-6, max_iter=100,
) -> float:
    """从期权市场价反解隐含波动率(二分法, 稳健)。

    价格低于内在价值或超界时返回 nan。
    """
    intrinsic = max(0.0, (S - K) if is_call else (K - S))
    if price < intrinsic - 1e-8 or price <= 0 or T <= 0:
        return float("nan")
    f = lambda s: bs_price(S, K, T, r, s, is_call) - price
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:  # 无根(价格越界)
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)
