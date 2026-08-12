"""周期重置波动带: 周一(或月初)用当天 IV 定死本期区间, 期末重算。

与 vol_cone_anchored.py 的区别 —— 后者从某个高点起算、sigma 随已跌天数张开,
横轴依赖"这轮跌了多久", 事前不可知; 本脚本是固定期限:

    base  = 上期最后一个交易日收盘        (周一开盘前就已知)
    sigma = IV(base日) * sqrt(N/252)      (N=5 周 / 21 月, 固定)
    z(t)  = (P(t)/base - 1) / sigma       (本期内 base 与 sigma 都不再变)

于是 -1sigma 在期初就是一个确定价位, 可以直接挂单。另给一列 z_scaled,
按已过天数缩放 sigma(第1天就跌1sigma 比第5天才跌1sigma 极端得多)。

用法:
    python -m scripts.period_cone --key kc50 --freq W --start 20260601
    python -m scripts.period_cone --key kc50 --freq M --start 20260101
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import abspath

NDAYS = {"W": 5, "M": 21}


def load(key: str) -> pd.DataFrame:
    D = json.load(open(abspath("data/processed") / "timing_viz.json"))[key]
    df = pd.DataFrame({
        "date": D["dates"], "price": D["price"], "iv": D["atm_iv"],
        "iv_pct": D["iv_pct"], "sent_pct": D["sent_pct"], "rr_pct": D["rr_pct"],
        "slope_pct": D["slope_pct"], "vrp_pct": D["vrp_pct"],
    }).dropna(subset=["price", "iv"]).reset_index(drop=True)
    df["name"] = D["name"]
    df["dt"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["panic"] = pd.concat(
        [df["iv_pct"], df["sent_pct"], 1 - df["rr_pct"],
         1 - df["slope_pct"], 1 - df["vrp_pct"]], axis=1).mean(axis=1)
    return df


def mark(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """给每天打上所属周期, 并算出该周期的 base / sigma / z。"""
    n = NDAYS[freq]
    df = df.copy()
    df["period"] = df["dt"].dt.to_period(freq)
    periods = sorted(df["period"].unique())
    base, sig, day_in = [], [], []
    for p in periods:
        idx = df.index[df["period"] == p]
        pi = periods.index(p)
        if pi == 0:
            base += [np.nan] * len(idx); sig += [np.nan] * len(idx)
            day_in += list(range(1, len(idx) + 1)); continue
        prev = df[df["period"] == periods[pi - 1]].iloc[-1]
        b, s = float(prev["price"]), float(prev["iv"]) * np.sqrt(n / 252)
        base += [b] * len(idx); sig += [s] * len(idx)
        day_in += list(range(1, len(idx) + 1))
    df["base"] = base; df["sigma"] = sig; df["day_in"] = day_in
    df["z"] = (df["price"] / df["base"] - 1) / df["sigma"]
    # 按已过天数缩放: 第1天跌1sigma 远比第5天跌1sigma 极端
    df["z_scaled"] = (df["price"] / df["base"] - 1) / (
        df["sigma"] * np.sqrt(df["day_in"] / n))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="kc50")
    ap.add_argument("--freq", default="W", choices=["W", "M"])
    ap.add_argument("--start", default="20260601")
    args = ap.parse_args()

    df = mark(load(args.key), args.freq)
    d = df[df["date"] >= args.start].dropna(subset=["z"])
    lab = "周" if args.freq == "W" else "月"
    print(f"=== {d['name'].iloc[0]} 每{lab}重置波动带 (N={NDAYS[args.freq]}日, "
          f"base=上{lab}最后收盘, sigma 期初定死) ===\n")

    for p, g in d.groupby("period"):
        b, s = g["base"].iloc[0], g["sigma"].iloc[0]
        lo_i = g["price"].astype(float).idxmin()
        print(f"{lab}: {str(p)}   base {b:.3f}  IV→σ {s:.1%}   "
              f"-1σ={b * (1 - s):.3f}  -2σ={b * (1 - 2 * s):.3f}")
        for _, r in g.iterrows():
            flag = " ←最低" if r.name == lo_i else ""
            print(f"   {r['date'][4:]}  {float(r['price']):.3f}  "
                  f"{r['price'] / b - 1:+6.1%}  z={r['z']:+5.2f}  "
                  f"z_scaled={r['z_scaled']:+5.2f}  恐慌{r['panic']:.2f}{flag}")
        print()


if __name__ == "__main__":
    main()
