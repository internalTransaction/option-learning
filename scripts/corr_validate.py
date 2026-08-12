"""隐含相关性指数的有效性检验: 是否对已有恐慌框架有增量。

检验三件事:
  1. 形态  相关性在下跌/恐慌时是否抬升(应该抬升, 否则口径错了)
  2. 独立性 与现有五灯因子的相关性(太高=冗余)
  3. 增量  在恐慌触发日内, 按相关性分组的前瞻收益差异(用户视角: 极值反转盈亏比)

用法: python -m scripts.corr_validate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.config import abspath

PROC = abspath("data/processed")


def pctile(s: pd.Series, w: int = 252) -> pd.Series:
    return s.rolling(w, min_periods=w // 2).rank(pct=True)


def load() -> pd.DataFrame:
    c = pd.read_parquet(PROC / "implied_corr_zz1000_21d.parquet")
    t = json.load(open(PROC / "timing_viz.json"))["zz1000"]
    tv = pd.DataFrame({k: t[k] for k in
                       ["dates", "price", "mom_pct", "iv_pct", "sent_pct", "rr_pct",
                        "slope_pct", "vrp_pct", "rvvol_pct"]})
    tv = tv.rename(columns={"dates": "trade_date"})
    df = c.merge(tv, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)

    # 恐慌指数(与 continuous_panic_index.py 同口径)
    parts = [df["iv_pct"], df["sent_pct"], 1 - df["rr_pct"], 1 - df["slope_pct"], 1 - df["vrp_pct"]]
    df["panic"] = pd.concat(parts, axis=1).mean(axis=1)

    for col in ["corr_rv", "corr_semi", "corr_crp", "dispersion"]:
        df[col + "_pct"] = pctile(df[col])

    px = df["price"].astype(float)
    for h in (5, 10, 20):
        df[f"fwd{h}"] = px.shift(-h) / px - 1
    return df


def main() -> None:
    df = load()
    print(f"样本 {len(df)} 天  {df['trade_date'].iloc[0]} -> {df['trade_date'].iloc[-1]}\n")

    print("=== 1. 水平与形态 ===")
    print(df[["corr_rv", "corr_semi", "corr_crp", "dispersion"]].describe().round(3).to_string())

    print("\n按当日指数涨跌幅分组的平均相关性:")
    ret = df["price"].pct_change()
    bins = pd.cut(ret, [-1, -0.03, -0.01, 0.01, 0.03, 1],
                  labels=["跌>3%", "跌1-3%", "±1%", "涨1-3%", "涨>3%"])
    print(df.groupby(bins, observed=True)[["corr_rv", "corr_semi"]].mean().round(3).to_string())

    print("\n=== 2. 与现有因子的相关性(Spearman) ===")
    cols = ["corr_rv_pct", "corr_semi_pct", "corr_crp_pct", "iv_pct", "sent_pct",
            "rr_pct", "slope_pct", "vrp_pct", "rvvol_pct", "panic", "mom_pct"]
    print(df[cols].corr(method="spearman").round(2).loc[
        ["corr_rv_pct", "corr_semi_pct", "corr_crp_pct"]].to_string())

    print("\n=== 3. 增量: 恐慌触发日内按相关性分组 ===")
    trig = df[(df["panic"] >= 0.70) & (df["mom_pct"] <= 0.30)].dropna(subset=["fwd20"])
    print(f"触发样本 {len(trig)} 天")
    for key in ["corr_semi_pct", "corr_rv_pct", "corr_crp_pct"]:
        sub = trig.dropna(subset=[key])
        if len(sub) < 30:
            print(f"  {key}: 样本不足({len(sub)})")
            continue
        hi = sub[sub[key] >= 0.6]
        lo = sub[sub[key] <= 0.4]
        print(f"\n  分组依据 {key}   (高 {len(hi)} 天 / 低 {len(lo)} 天)")
        for name, g in [("高", hi), ("低", lo)]:
            if len(g) < 10:
                continue
            r = g["fwd20"]
            up, dn = r[r > 0], r[r <= 0]
            pf = (up.mean() / abs(dn.mean())) if len(dn) and dn.mean() != 0 else np.nan
            print(f"    {name}: n={len(g):3d}  20日均收益 {r.mean():+.2%}  "
                  f"胜率 {(r > 0).mean():.0%}  盈亏比 {pf:.2f}")

    print("\n=== 4. 全样本: 相关性分位 vs 前瞻收益 ===")
    for key in ["corr_semi_pct", "corr_rv_pct"]:
        g = df.dropna(subset=[key, "fwd20"])
        q = pd.qcut(g[key], 5, labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"])
        tbl = g.groupby(q, observed=True)["fwd20"].agg(["mean", "count"])
        tbl["胜率"] = g.groupby(q, observed=True)["fwd20"].apply(lambda x: (x > 0).mean())
        print(f"\n  {key}:")
        print(tbl.assign(mean=lambda d: (d["mean"] * 100).round(2)).round(2).to_string())


if __name__ == "__main__":
    main()
