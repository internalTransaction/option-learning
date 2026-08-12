"""锚定式波动概率锥: 从一个参考高点出发, 锥随天数张开, 看今天踩在第几个 sigma。

    sigma(t) = IV_anchor * sqrt(days_since_anchor / 252)
    z(t)     = [P(t)/P_anchor - 1] / sigma(t)

与 vol_cone.py 的滚动 z(固定 N 日回看)不同, 这里锚点固定、锥口随时间张开,
对应"从这轮见顶起算, 现在跌到锥的第几层"的实际用法。

锚点 = 过去 LOOKBACK 日内的最高收盘, 且要求距今至少 3 个交易日(锥要张得开)。

用法: python -m scripts.vol_cone_anchored --lookback 60
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import abspath

KEYS = ["zz1000", "kc50", "cyb", "hs300"]
MIN_DAYS = 3


def build(key: str, lookback: int) -> pd.DataFrame:
    D = json.load(open(abspath("data/processed") / "timing_viz.json"))[key]
    df = pd.DataFrame({
        "date": D["dates"], "price": D["price"], "iv": D["atm_iv"],
        "iv_pct": D["iv_pct"], "sent_pct": D["sent_pct"], "rr_pct": D["rr_pct"],
        "slope_pct": D["slope_pct"], "vrp_pct": D["vrp_pct"],
    }).dropna(subset=["price", "iv"]).reset_index(drop=True)
    df["name"] = D["name"]
    px = df["price"].astype(float)
    iv = df["iv"].astype(float)

    anchor_i, anchor_p, anchor_iv, days = [], [], [], []
    for i in range(len(df)):
        lo = max(0, i - lookback + 1)
        j = int(px.iloc[lo:i + 1].idxmax())
        if i - j < MIN_DAYS:                       # 锚点太近, 锥还没张开
            anchor_i.append(np.nan); anchor_p.append(np.nan)
            anchor_iv.append(np.nan); days.append(np.nan)
            continue
        anchor_i.append(j); anchor_p.append(px.iloc[j])
        anchor_iv.append(iv.iloc[j]); days.append(i - j)

    df["anchor_date"] = [df["date"].iloc[int(x)] if pd.notna(x) else None for x in anchor_i]
    df["anchor_px"] = anchor_p
    df["anchor_iv"] = anchor_iv
    df["days"] = days
    df["sigma"] = df["anchor_iv"] * np.sqrt(df["days"] / 252)
    df["dd"] = px / df["anchor_px"] - 1
    df["z"] = df["dd"] / df["sigma"]

    df["panic"] = pd.concat(
        [df["iv_pct"], df["sent_pct"], 1 - df["rr_pct"],
         1 - df["slope_pct"], 1 - df["vrp_pct"]], axis=1).mean(axis=1)
    for h in (5, 10, 20):
        df[f"fwd{h}"] = px.shift(-h) / px - 1
    return df


def stat(g: pd.DataFrame, h: int) -> str:
    r = g[f"fwd{h}"].dropna()
    if len(r) < 15:
        return f"n={len(r):4d}  样本不足"
    up, dn = r[r > 0], r[r <= 0]
    pf = up.mean() / abs(dn.mean()) if len(dn) and dn.mean() != 0 else np.nan
    return (f"n={len(r):4d}  均值 {r.mean():+6.2%}  胜率 {(r > 0).mean():4.0%}  "
            f"盈亏比 {pf:4.2f}  最差 {r.min():+6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60, help="找锚点高点的回看窗口")
    ap.add_argument("--h", type=int, default=20)
    args = ap.parse_args()
    h = args.h

    frames = [build(k, args.lookback) for k in KEYS]
    df = pd.concat(frames, ignore_index=True)
    d = df.dropna(subset=["z", f"fwd{h}"])

    print(f"=== 1. 从 {args.lookback} 日高点起算, 踩到第几层锥 → 未来 {h} 日 ===")
    for lo, hi, lab in [(-99, -2.5, "≤-2.5σ"), (-2.5, -2, "-2.5~-2σ"), (-2, -1.5, "-2~-1.5σ"),
                        (-1.5, -1, "-1.5~-1σ"), (-1, -0.5, "-1~-0.5σ"), (-0.5, 99, ">-0.5σ")]:
        print(f"  {lab:10s} {stat(d[(d['z'] > lo) & (d['z'] <= hi)], h)}")

    print(f"\n=== 2. 锥 × 恐慌度 ===")
    for zl in (-1, -1.5, -2):
        s = d[d["z"] <= zl]
        print(f"  z≤{zl}σ  恐慌≥0.7: {stat(s[s['panic'] >= 0.7], h)}")
        print(f"  z≤{zl}σ  恐慌<0.7: {stat(s[s['panic'] < 0.7], h)}")

    print(f"\n=== 3. 当前位置 (锚点 = 过去 {args.lookback} 日最高收盘) ===")
    for f in frames:
        r = f.dropna(subset=["z"]).iloc[-1]
        ap_, sg, dys = r["anchor_px"], r["sigma"], int(r["days"])
        print(f"\n  {r['name']}  {r['date']}  现价 {float(r['price']):.3f}")
        print(f"    锚点 {r['anchor_date']} @ {ap_:.3f}  (IV {r['anchor_iv']:.1%}, 已过 {dys} 日 → σ={sg:.1%})")
        print(f"    已跌 {r['dd']:+.1%}  →  z = {r['z']:+.2f}σ   恐慌度 {r['panic']:.2f}")
        line = "    锥位: "
        for m in (1, 1.5, 2, 2.5):
            line += f"-{m}σ={ap_ * (1 - m * sg):.3f}  "
        print(line)


if __name__ == "__main__":
    main()
