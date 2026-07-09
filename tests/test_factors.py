"""因子与回测冒烟测试(用合成数据, 不依赖网络)。"""
import numpy as np
import pandas as pd

from src.backtest.engine import backtest, performance
from src.factors.volatility import VolatilityFactor
from src.signals.generator import volatility_signal


def _fake_data(n=400):
    dates = pd.bdate_range("2022-01-01", periods=n)
    rng = np.random.default_rng(0)
    px = 3.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    iv = 20 + 10 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 1, n)
    qvix = pd.DataFrame({"date": dates, "open": iv, "high": iv, "low": iv, "close": iv})
    etf = pd.DataFrame({"date": dates, "open": px, "close": px, "high": px, "low": px,
                        "volume": 1, "amount": 1, "pct_chg": 0})
    return qvix, etf


def test_volatility_factor_columns():
    qvix, etf = _fake_data()
    out = VolatilityFactor().compute(qvix=qvix, underlying=etf, iv_percentile_window=60)
    for col in ("iv", "hv", "vrp", "iv_percentile", "iv_zscore"):
        assert col in out.columns
    assert out["iv_percentile"].dropna().between(0, 1).all()


def test_signal_and_backtest_run():
    qvix, etf = _fake_data()
    vol = VolatilityFactor().compute(qvix=qvix, underlying=etf, iv_percentile_window=60)
    sig = volatility_signal(vol)
    assert set(sig.unique()).issubset({-1, 0, 1})
    bt = backtest(sig, etf)
    perf = performance(bt)
    assert "sharpe" in perf and "max_drawdown" in perf
