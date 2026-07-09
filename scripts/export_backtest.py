"""跑全量回测并导出 JSON, 供可视化面板使用。

对每个 enabled 标的, 分别用逆向/顺势两种波动率择时逻辑回测,
导出净值曲线、IV、IV百分位、信号、价格与绩效指标。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.backtest.engine import backtest, performance
from src.data import akshare_loader as loader
from src.factors.volatility import VolatilityFactor
from src.signals.generator import volatility_signal
from src.utils.config import enabled_underlyings, load_config


def _series(idx, s):
    """转成可 JSON 化的列表, NaN -> None。"""
    return [None if pd.isna(v) else round(float(v), 4) for v in s.reindex(idx)]


def run() -> dict:
    cfg = load_config()
    fp, sp, bp = cfg["factors"], cfg["signals"], cfg["backtest"]
    result = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "underlyings": {}}

    for key, u in enabled_underlyings(cfg).items():
        qvix = loader.fetch_qvix(u["qvix"])
        etf = loader.fetch_underlying(u["etf_code"], exchange=u["exchange"])
        vol = VolatilityFactor().compute(
            qvix=qvix, underlying=etf,
            hv_window=fp["hv_window"], iv_percentile_window=fp["iv_percentile_window"],
        )

        variants = {}
        for name, contrarian in [("contrarian", True), ("trend", False)]:
            sig = volatility_signal(
                vol, iv_pct_high=sp["iv_pct_high"], iv_pct_low=sp["iv_pct_low"],
                contrarian=contrarian,
            )
            bt = backtest(sig, etf, fee_bps=bp["fee_bps"], slippage_bps=bp["slippage_bps"])
            perf = performance(bt)
            # 基准绩效
            bench = performance(bt.assign(_b=bt["ret"]), col="_b")
            variants[name] = {
                "perf": perf,
                "bench_perf": {k: bench[k] for k in ("annual_return", "sharpe", "max_drawdown")},
                "equity": _series(bt.index, bt["equity"]),
                "pos": [int(v) for v in bt["pos"].fillna(0)],
            }

        # 公共时间轴: 从 IV 首个有效日开始(此前信号恒为0, 无意义的平段)
        bt_ref = backtest(volatility_signal(vol), etf)
        iv_start = vol["iv"].dropna().index.min()
        idx = bt_ref.index[bt_ref.index >= iv_start]
        # 净值从裁剪起点重新归一化到 1
        for name in variants:
            eq = pd.Series(variants[name]["equity"], index=bt_ref.index)
            eq = eq.reindex(idx)
            eq = eq / eq.iloc[0]
            variants[name]["equity"] = _series(idx, eq)
            variants[name]["pos"] = [int(v) for v in
                                     pd.Series(variants[name]["pos"], index=bt_ref.index).reindex(idx).fillna(0)]
        bench_eq = bt_ref["bench_equity"].reindex(idx)
        bench_eq = bench_eq / bench_eq.iloc[0]
        result["underlyings"][key] = {
            "name": u["name"],
            "etf_code": u["etf_code"],
            "dates": [d.strftime("%Y-%m-%d") for d in idx],
            "price": _series(idx, etf.set_index("date")["close"]),
            "bench_equity": _series(idx, bench_eq),
            "iv": _series(idx, vol["iv"]),
            "iv_pct": _series(idx, vol["iv_percentile"]),
            "vrp": _series(idx, vol["vrp"]),
            "variants": variants,
            "iv_start": str(vol["iv"].dropna().index.min().date()) if hasattr(vol.index, "date") else None,
        }

    return result


if __name__ == "__main__":
    out = run()
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/backtest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n = len(out["underlyings"])
    print(f"导出 {n} 个标的 -> {path}")
    for k, v in out["underlyings"].items():
        c = v["variants"]["contrarian"]["perf"]
        t = v["variants"]["trend"]["perf"]
        print(f"  {k:10s} 逆向 Sharpe={c['sharpe']:.2f} 年化={c['annual_return']:.2%} | "
              f"顺势 Sharpe={t['sharpe']:.2f} 年化={t['annual_return']:.2%} | "
              f"基准年化={c and v['variants']['contrarian']['bench_perf']['annual_return']:.2%}")
