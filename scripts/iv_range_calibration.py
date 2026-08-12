"""IV 隐含波动区间的校准检验: 期权定的价, 到底覆盖了多少实际波动?

问题: 用 t 日 ATM IV 推出的未来 H 日 ±1σ 区间, 实际收益落在里面的比例是多少?
      正态假设下应为 68.3%; 偏离方向告诉我们 IV 是系统性高估还是低估波动范围。

产出:
  1. 全样本覆盖率(1σ/2σ) 与下行尾部命中率
  2. 按 IV 分位分层的覆盖率 —— 低 IV 时是否更容易被击穿(这才是风险所在)
  3. 经验条件分布 —— 不假设正态, 直接给历史上未来 H 日收益的分位数

用法: python -m scripts.iv_range_calibration --h 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import abspath

KEYS = ["zz1000", "kc50", "cyb", "hs300"]


def load(h: int) -> pd.DataFrame:
    D = json.load(open(abspath("data/processed") / "timing_viz.json"))
    frames = []
    for k in KEYS:
        d = D[k]
        df = pd.DataFrame({
            "key": k, "name": d["name"], "date": d["dates"],
            "price": d["price"], "iv": d["atm_iv"], "iv_pct": d["iv_pct"],
            "rv": d["rv"], "vrp": d["vrp"], "mom_pct": d["mom_pct"],
        }).dropna(subset=["price", "iv"])
        px = df["price"].astype(float)
        df["fwd"] = px.shift(-h) / px - 1
        # 未来 H 日实际路径最大跌幅(比终点收益更贴近"能不能扛住")
        df["fwd_mdd"] = [
            px.iloc[i:i + h + 1].min() / px.iloc[i] - 1 if i + h < len(px) else np.nan
            for i in range(len(px))
        ]
        df["sigma"] = df["iv"].astype(float) * np.sqrt(h / 252)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).dropna(subset=["fwd"])


def cover(g: pd.DataFrame) -> dict:
    z = g["fwd"] / g["sigma"]
    return {
        "n": len(g),
        "±1σ内": (z.abs() <= 1).mean(),
        "跌破-1σ": (z <= -1).mean(),
        "跌破-2σ": (z <= -2).mean(),
        "路径破-1σ": (g["fwd_mdd"] / g["sigma"] <= -1).mean(),
        "IV均值": g["iv"].mean(),
        "实际|r|/σ": z.abs().mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20, help="前瞻交易日")
    args = ap.parse_args()
    h = args.h
    df = load(h)

    print(f"=== 1. 全样本校准 (H={h}日, 正态基准: ±1σ内 68.3% / 跌破-1σ 15.9% / 跌破-2σ 2.3%) ===")
    rows = {r["name"]: cover(g) for r, (_, g) in
            zip(df.groupby("key").first().reset_index().to_dict("records"), df.groupby("key"))}
    t = pd.DataFrame(rows).T
    print(t.assign(n=t["n"].astype(int)).round(3).to_string())
    print("\n合计:", {k: round(v, 3) for k, v in cover(df).items()})

    print(f"\n=== 2. 按 IV 分位分层 (低 IV 时区间是否更容易被击穿) ===")
    d = df.dropna(subset=["iv_pct"]).copy()
    d["层"] = pd.cut(d["iv_pct"], [0, 0.25, 0.5, 0.75, 1.0],
                     labels=["IV低(0-25)", "25-50", "50-75", "IV高(75-100)"])
    t2 = pd.DataFrame({str(k): cover(g) for k, g in d.groupby("层", observed=True)}).T
    print(t2.assign(n=t2["n"].astype(int)).round(3).to_string())

    print(f"\n=== 3. 经验条件分布: 未来 {h} 日收益分位数 (不假设正态) ===")
    for lab, sub in [("全样本", d), ("IV低(<25分位)", d[d["iv_pct"] <= 0.25]),
                     ("IV高(>75分位)", d[d["iv_pct"] >= 0.75])]:
        q = sub["fwd"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95])
        mdd = sub["fwd_mdd"].quantile([0.01, 0.05, 0.25])
        print(f"  {lab:16s} n={len(sub):4d}  "
              f"1% {q[0.01]:+.1%} | 5% {q[0.05]:+.1%} | 25% {q[0.25]:+.1%} | "
              f"中位 {q[0.5]:+.1%} | 95% {q[0.95]:+.1%}   路径最大回撤 5%分位 {mdd[0.05]:+.1%}")

    print(f"\n=== 4. 用 σ 归一后的经验分位(可直接乘当前 σ 得区间) ===")
    for lab, sub in [("全样本", d), ("IV低(<25分位)", d[d["iv_pct"] <= 0.25]),
                     ("IV高(>75分位)", d[d["iv_pct"] >= 0.75])]:
        z = (sub["fwd"] / sub["sigma"]).quantile([0.01, 0.05, 0.25, 0.5, 0.95])
        zm = (sub["fwd_mdd"] / sub["sigma"]).quantile([0.05])
        print(f"  {lab:16s} z: 1% {z[0.01]:+.2f} | 5% {z[0.05]:+.2f} | 25% {z[0.25]:+.2f} | "
              f"中位 {z[0.5]:+.2f} | 95% {z[0.95]:+.2f}   路径回撤z 5%分位 {zm[0.05]:+.2f}")


if __name__ == "__main__":
    main()
