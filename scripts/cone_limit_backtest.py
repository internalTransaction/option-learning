"""在前瞻锥的下沿挂买单, 到底能不能成交、成交后是不是底。

每天 t 按 SOXX expected-move 的口径画一把前瞻锥:
    L(m) = P_t * (1 - m * IV_t * sqrt(H/252))
在 L(m) 挂买单, H 个交易日内有效。用<日内最低价>判成交(触及即成交)。

回答三件事:
  1. 成交率      —— 挂了会不会根本不成交(挂太深=永远等不到)
  2. 成交后收益  —— 成交价往后 K 日的收益
  3. 抄到底没有  —— 成交后还要再跌多少(MAE), 以及成交价离窗口真实最低点差多远

用法: python -m scripts.cone_limit_backtest --H 20 --K 20
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

KEYS = ["zz1000", "kc50", "cyb", "hs300"]
LEVELS = [0.5, 1.0, 1.5, 2.0]


def load(key: str) -> pd.DataFrame:
    tv = json.load(open(abspath("data/processed") / "timing_viz.json"))[key]
    iv = pd.DataFrame({
        "trade_date": tv["dates"], "iv": tv["atm_iv"], "iv_pct": tv["iv_pct"],
        "sent_pct": tv["sent_pct"], "rr_pct": tv["rr_pct"],
        "slope_pct": tv["slope_pct"], "vrp_pct": tv["vrp_pct"],
    })
    px = pd.read_parquet(abspath("data/raw") / f"ohlc_{key}.parquet")
    df = px.merge(iv, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    df["panic"] = pd.concat(
        [df["iv_pct"], df["sent_pct"], 1 - df["rr_pct"],
         1 - df["slope_pct"], 1 - df["vrp_pct"]], axis=1).mean(axis=1)
    df["name"] = tv["name"]
    df["key"] = key
    return df.dropna(subset=["iv", "close", "low"]).reset_index(drop=True)


def simulate(df: pd.DataFrame, m: float, H: int, K: int) -> pd.DataFrame:
    """每天挂一单, 返回每单的结果(未成交的也记一行)。"""
    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    iv = df["iv"].to_numpy(float)
    n = len(df)
    rows = []
    for t in range(n - H - K):
        lvl = close[t] * (1 - m * iv[t] * np.sqrt(H / 252))
        win_lo = low[t + 1:t + 1 + H]
        hit = np.nonzero(win_lo <= lvl)[0]
        rec = {"t": t, "date": df["trade_date"].iloc[t], "key": df["key"].iloc[t],
               "panic": df["panic"].iloc[t], "level": lvl, "filled": len(hit) > 0,
               "base_ret": close[t + K] / close[t] - 1}   # 不挂单, 当天买入的对照
        if len(hit):
            f = t + 1 + int(hit[0])
            rec["fill_i"] = f
            rec["wait"] = f - t
            rec["ret"] = close[min(f + K, n - 1)] / lvl - 1
            # 成交后还要再跌多少(最大浮亏)
            rec["mae"] = low[f:min(f + K, n)].min() / lvl - 1
            # 成交价离"这轮真实最低点"差多远(>0 表示买贵了)
            true_lo = low[t + 1:t + 1 + H].min()
            rec["vs_low"] = lvl / true_lo - 1
        rows.append(rec)
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, m: float, K: int, tag: str = "") -> None:
    fill = res["filled"].mean()
    f = res[res["filled"]]
    if len(f) < 20:
        print(f"  -{m}σ {tag:10s} 成交率 {fill:5.1%}   成交样本不足({len(f)})")
        return
    r = f["ret"]
    print(f"  -{m}σ {tag:10s} 成交率 {fill:5.1%}  "
          f"成交后{K}日 {r.mean():+6.2%}/胜率{(r > 0).mean():4.0%}  "
          f"再跌 {f['mae'].mean():+6.2%}(最深 {f['mae'].min():+.1%})  "
          f"离真低 {f['vs_low'].mean():+5.2%}  等待 {f['wait'].mean():4.1f}日")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20, help="挂单有效期(交易日)")
    ap.add_argument("--K", type=int, default=20, help="成交后持有天数")
    args = ap.parse_args()
    H, K = args.H, args.K

    frames = {k: load(k) for k in KEYS}
    print(f"=== 锥下沿挂单回测 (挂单有效 {H} 日, 成交后持有 {K} 日, 日内最低价判成交) ===")
    for k, df in frames.items():
        print(f"\n{df['name'].iloc[0]}  n={len(df)}  {df['trade_date'].iloc[0]}->{df['trade_date'].iloc[-1]}")
        base = df["close"].shift(-K) / df["close"] - 1
        print(f"  对照: 任意一天直接买入, {K}日后 {base.mean():+.2%}/胜率{(base > 0).mean():.0%}")
        for m in LEVELS:
            report(simulate(df, m, H, K), m, K)

    print(f"\n=== 池化四标的 ===")
    allres = {m: pd.concat([simulate(df, m, H, K) for df in frames.values()], ignore_index=True)
              for m in LEVELS}
    pooled_base = pd.concat([f["base_ret"] for f in allres[1.0].groupby("key")["base_ret"]
                             .apply(lambda s: s.to_frame())], ignore_index=True) \
        if False else allres[1.0]["base_ret"]
    print(f"  对照: 任意一天直接买入 {pooled_base.mean():+.2%}/胜率{(pooled_base > 0).mean():.0%}")
    for m in LEVELS:
        report(allres[m], m, K)

    print(f"\n=== 加恐慌门(挂单当天 panic>=0.7 才挂) ===")
    for m in LEVELS:
        report(allres[m][allres[m]["panic"] >= 0.7], m, K, "panic≥.7")

    print(f"\n=== 成交后是不是底? 成交价 vs 窗口真实最低点 ===")
    for m in LEVELS:
        f = allres[m][allres[m]["filled"]]
        if len(f) < 20:
            continue
        print(f"  -{m}σ  成交价平均高于真低 {f['vs_low'].mean():.2%}，"
              f"中位 {f['vs_low'].median():.2%}，"
              f"其中 {(f['mae'] >= -0.01).mean():.0%} 的单子成交后基本没再跌(浮亏<1%)")


if __name__ == "__main__":
    main()
