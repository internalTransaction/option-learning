"""端到端示例: 拉数据 -> 算波动率因子 -> 生成择时信号 -> 回测。

用法:
    python -m scripts.run_signal                 # 默认沪深300, 逆向波动率择时
    python -m scripts.run_signal --key kc50
    python -m scripts.run_signal --trend         # 改用顺势(高IV看空)
"""
from __future__ import annotations

import argparse

from src.backtest.engine import backtest, performance
from src.data import akshare_loader as loader
from src.factors.volatility import VolatilityFactor
from src.signals.generator import volatility_signal
from src.utils.config import enabled_underlyings, load_config
from src.utils.logger import get_logger

log = get_logger("signal")


def main() -> None:
    cfg = load_config()
    under = enabled_underlyings(cfg)

    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=next(iter(under)), choices=list(under),
                    help="标的 key (config.underlyings)")
    ap.add_argument("--trend", action="store_true", help="顺势(高IV看空); 默认逆向")
    args = ap.parse_args()

    u = under[args.key]
    log.info("标的: %s (%s)", args.key, u["name"])

    # 1. 数据
    qvix = loader.fetch_qvix(u["qvix"])
    etf = loader.fetch_underlying(u["etf_code"], exchange=u["exchange"])

    # 2. 因子
    fp = cfg["factors"]
    vol = VolatilityFactor().compute(
        qvix=qvix, underlying=etf,
        hv_window=fp["hv_window"],
        iv_percentile_window=fp["iv_percentile_window"],
    )
    log.info("波动率因子: %d 行, 最新 IV=%.2f, IV百分位=%.2f, VRP=%.2f",
             len(vol), vol["iv"].iloc[-1], vol["iv_percentile"].iloc[-1], vol["vrp"].iloc[-1])

    # 3. 信号
    sp = cfg["signals"]
    sig = volatility_signal(
        vol, iv_pct_high=sp["iv_pct_high"], iv_pct_low=sp["iv_pct_low"],
        contrarian=not args.trend,
    )
    log.info("最新信号: %d (%s)", sig.iloc[-1],
             {1: "看多", 0: "空仓", -1: "看空"}[int(sig.iloc[-1])])

    # 4. 回测
    bp = cfg["backtest"]
    bt = backtest(sig, etf, fee_bps=bp["fee_bps"], slippage_bps=bp["slippage_bps"])
    perf = performance(bt)
    bench = performance(bt.rename(columns={"ret": "strat_ret", "strat_ret": "_"}), col="strat_ret")

    log.info("---- 策略绩效 (%s) ----", "顺势" if args.trend else "逆向")
    for k, v in perf.items():
        log.info("  %-14s %s", k, v)
    log.info("  基准年化(买入持有): %s", bench.get("annual_return"))


if __name__ == "__main__":
    main()
