"""波动概率锥: 从 N 日前的期权定价看, 今天跌到了第几个 sigma。

    z(t,N) = [P(t)/P(t-N) - 1] / [IV(t-N) * sqrt(N/252)]

用途是回答"这个位置该不该承接" —— 避免在还没跌到 -1σ 时就出手(抄在半山腰)。

本脚本做三件事:
  1. 历史验证: z 跌到各档位后的前瞻收益, 看 -1σ / -2σ 是否真是承接位
  2. 与现有闸门对比: sigma 归一 vs mom_pct(分位归一) vs 绝对跌幅, 谁更会挑位置
  3. 当前读数: 各标的今天在锥的第几层

用法: python -m scripts.vol_cone --n 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import abspath

KEYS = ["zz1000", "kc50", "cyb", "hs300"]


def load(n: int) -> pd.DataFrame:
    D = json.load(open(abspath("data/processed") / "timing_viz.json"))
    out = []
    for k in KEYS:
        d = D[k]
        df = pd.DataFrame({
            "key": k, "name": d["name"], "date": d["dates"], "price": d["price"],
            "iv": d["atm_iv"], "iv_pct": d["iv_pct"], "mom_pct": d["mom_pct"],
            "sent_pct": d["sent_pct"], "rr_pct": d["rr_pct"],
            "slope_pct": d["slope_pct"], "vrp_pct": d["vrp_pct"],
        }).dropna(subset=["price", "iv"]).reset_index(drop=True)
        px = df["price"].astype(float)

        # 锥: 用 N 日前的 IV 定 sigma, 看今天走到了几个 sigma
        sig = df["iv"].astype(float).shift(n) * np.sqrt(n / 252)
        df["ret_n"] = px / px.shift(n) - 1
        df["z"] = df["ret_n"] / sig
        df["sigma_n"] = sig

        df["panic"] = pd.concat(
            [df["iv_pct"], df["sent_pct"], 1 - df["rr_pct"],
             1 - df["slope_pct"], 1 - df["vrp_pct"]], axis=1).mean(axis=1)
        for h in (5, 10, 20):
            df[f"fwd{h}"] = px.shift(-h) / px - 1
        out.append(df)
    return pd.concat(out, ignore_index=True)


def stat(g: pd.DataFrame, h: int = 20) -> str:
    if len(g) < 15:
        return f"n={len(g):4d}  样本不足"
    r = g[f"fwd{h}"].dropna()
    if len(r) < 15:
        return f"n={len(r):4d}  样本不足"
    up, dn = r[r > 0], r[r <= 0]
    pf = up.mean() / abs(dn.mean()) if len(dn) and dn.mean() != 0 else np.nan
    return (f"n={len(r):4d}  均值 {r.mean():+6.2%}  胜率 {(r > 0).mean():4.0%}  "
            f"盈亏比 {pf:4.2f}  最差 {r.min():+6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="锥的回看/展望天数")
    ap.add_argument("--h", type=int, default=20, help="前瞻评估天数")
    args = ap.parse_args()
    df = load(args.n)
    h = args.h
    d = df.dropna(subset=["z", f"fwd{h}"])

    print(f"=== 1. 跌到第几个 σ 之后, 未来 {h} 日如何 (锥 N={args.n}) ===")
    bins = [(-99, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 99)]
    labs = ["≤-2σ", "-2~-1.5σ", "-1.5~-1σ", "-1~-0.5σ", "-0.5~0σ", ">0σ"]
    for (lo, hi), lab in zip(bins, labs):
        print(f"  {lab:10s} {stat(d[(d['z'] > lo) & (d['z'] <= hi)], h)}")

    print(f"\n=== 2. 同样的分档, 分标的 (≤-1σ 是否普适) ===")
    for k in KEYS:
        s = d[d["key"] == k]
        print(f"  {s['name'].iloc[0]:10s} ≤-1σ: {stat(s[s['z'] <= -1], h)}")
        print(f"  {'':10s} -1~0σ: {stat(s[(s['z'] > -1) & (s['z'] <= 0)], h)}")

    print(f"\n=== 3. 闸门对比: 谁更会挑位置(各取最极端的约 15% 样本) ===")
    q = d["z"].quantile(0.15)
    print(f"  σ归一 z≤{q:+.2f}      {stat(d[d['z'] <= q], h)}")
    qm = d["mom_pct"].quantile(0.15)
    print(f"  分位 mom_pct≤{qm:.2f}  {stat(d[d['mom_pct'] <= qm], h)}")
    qr = d["ret_n"].quantile(0.15)
    print(f"  绝对跌幅 ≤{qr:+.1%}    {stat(d[d['ret_n'] <= qr], h)}")

    print(f"\n=== 4. 锥 × 恐慌度 (承接位 + 出清确认) ===")
    for zl, zlab in [(-1, "z≤-1σ"), (-1.5, "z≤-1.5σ")]:
        s = d[d["z"] <= zl]
        print(f"  {zlab}  恐慌≥0.7: {stat(s[s['panic'] >= 0.7], h)}")
        print(f"  {zlab}  恐慌<0.7: {stat(s[s['panic'] < 0.7], h)}")

    print(f"\n=== 5. 当前读数 (锥 N={args.n}) ===")
    print(f"  {'标的':10s} {'现价':>8s} {'IV':>6s} {'σ':>6s} {'N日前':>8s} "
          f"{'已跌':>7s} {'z':>6s} | {'-1σ位':>8s} {'-1.5σ位':>8s} {'-2σ位':>8s}")
    for k in KEYS:
        s = df[df["key"] == k].dropna(subset=["z"])
        r = s.iloc[-1]
        base = float(r["price"]) / (1 + r["ret_n"])
        sg = r["sigma_n"]
        print(f"  {r['name']:10s} {float(r['price']):>8.3f} {r['iv']:>6.1%} {sg:>6.1%} "
              f"{base:>8.3f} {r['ret_n']:>+7.1%} {r['z']:>+6.2f} | "
              f"{base * (1 - sg):>8.3f} {base * (1 - 1.5 * sg):>8.3f} {base * (1 - 2 * sg):>8.3f}")


if __name__ == "__main__":
    main()
