"""科创50 周度波动幅度: 期权事前定的周区间 vs 实际走出来的幅度。

每周以"上周最后一个交易日"为起点(收盘价 + 当日 ATM IV, 无前视), 推出本周的
sigma_5d = IV * sqrt(5/252), 再看本周实际的高/低/振幅落在多少个 sigma 上。

用于回答: 一次反弹到底是噪音(远不到 1σ), 还是有意义的移动(触到区间上沿)。

用法: python -m scripts.weekly_range_kc50 --start 20260601
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import abspath

# GBM 下 5 日路径的期望振幅(high-low)约 1.6σ, 用作"这周走得算宽还是算窄"的基准
RANGE_MULT = 1.6


def load(start: str) -> pd.DataFrame:
    px = pd.read_parquet(abspath("data/raw") / "ts_fund_kc50_ohlc_20260601_20260811.parquet")
    D = json.load(open(abspath("data/processed") / "timing_viz.json"))["kc50"]
    iv = pd.DataFrame({"trade_date": D["dates"], "iv": D["atm_iv"], "iv_pct": D["iv_pct"]})
    df = px.merge(iv, on="trade_date", how="left").sort_values("trade_date").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["wk"] = df["dt"].dt.to_period("W")
    return df[df["trade_date"] >= start].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260601")
    args = ap.parse_args()
    df = load(args.start)

    rows = []
    weeks = list(df.groupby("wk"))
    for wi, (wk, g) in enumerate(weeks):
        if wi == 0:
            continue
        prev = weeks[wi - 1][1].iloc[-1]          # 上周最后一个交易日 = 本周起点
        base, iv = float(prev["close"]), prev["iv"]
        if pd.isna(iv):
            continue
        sig = float(iv) * np.sqrt(5 / 252)
        hi, lo, close = g["high"].max(), g["low"].min(), g["close"].iloc[-1]
        hi_d = g.loc[g["high"].idxmax(), "trade_date"]
        rng = (hi - lo) / base
        rows.append({
            "周": str(wk)[5:10] + "~" + str(wk)[16:21],
            "起点": prev["trade_date"][4:],
            "起点价": base, "IV": iv, "σ": sig,
            "周高": hi, "高z": (hi / base - 1) / sig, "高发生": hi_d[4:],
            "周低": lo, "低z": (lo / base - 1) / sig,
            "收盘": close, "收z": (close / base - 1) / sig,
            "振幅": rng, "振幅/预期": rng / (RANGE_MULT * sig),
            # 收盘落在本周 [低, 高] 区间的相对位置: 1=收在最高, 0=收在最低
            "收在区间": (close - lo) / (hi - lo) if hi > lo else np.nan,
        })

    t = pd.DataFrame(rows)
    print("=== 科创50 周度: 期权事前区间 vs 实际 (σ = 起点日ATM IV × √(5/252)) ===\n")
    show = t.copy()
    for c in ["起点价", "周高", "周低", "收盘"]:
        show[c] = show[c].map("{:.3f}".format)
    show["IV"] = show["IV"].map("{:.1%}".format)
    show["σ"] = show["σ"].map("{:.1%}".format)
    show["振幅"] = show["振幅"].map("{:.1%}".format)
    for c in ["高z", "低z", "收z"]:
        show[c] = show[c].map("{:+.2f}".format)
    show["振幅/预期"] = show["振幅/预期"].map("{:.2f}".format)
    show["收在区间"] = show["收在区间"].map("{:.0%}".format)
    print(show.to_string(index=False))

    print("\n=== 汇总 ===")
    print(f"周振幅 / IV预期振幅  中位 {t['振幅/预期'].median():.2f}  均值 {t['振幅/预期'].mean():.2f}"
          f"  (>1 = 实际走得比期权定价更宽)")
    print(f"周高 z  中位 {t['高z'].median():+.2f}   周低 z  中位 {t['低z'].median():+.2f}")
    print(f"周高触及 +1σ 的周数: {(t['高z'] >= 1).sum()}/{len(t)}   "
          f"周低跌破 -1σ: {(t['低z'] <= -1).sum()}/{len(t)}")


if __name__ == "__main__":
    main()
