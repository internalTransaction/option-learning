"""跨标的对比: IV 情绪比信号在不同标的上的强度。

验证研报假设——中小盘(中证1000/创业板/科创50)的期权情绪择时强于大盘(沪深300)。
对每个已重建的标的, 计算:
  - iv_ratio_25d 的分位前瞻收益 IC(5/10/20日)
  - 高IV日"恐慌/追涨"条件实验的收益差(顶底区分度)
  - iv_ratio 常态水平与波动(小盘偏斜溢价是否更高)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import cache
from src.research.validate import add_forward, information_coef, quantile_forward, rolling_pct

# 已重建的标的与其起止(需与 build_surface 一致)
KEYS = {
    "300etf": ("20200101", "20260707", "沪深300"),
    "kc50":   ("20230601", "20260707", "科创50"),
    "cyb":    ("20220901", "20260707", "创业板"),
    "zz1000": ("20220801", "20260707", "中证1000"),
}


def analyze_key(key, start, end) -> dict | None:
    name = f"surface_{key}_{start}_{end}"
    if not cache.exists(name, "processed"):
        return None
    df = cache.load(name, "processed").sort_values("date").reset_index(drop=True)
    df = add_forward(df, "spot", (5, 10, 20))
    df["iv_pct"] = rolling_pct(df["atm_iv"], 63)
    df["ratio_pct"] = rolling_pct(df["iv_ratio_25d"], 63)

    res = {"n": len(df),
           "ratio_mean": round(float(df["iv_ratio_25d"].mean()), 4),
           "ratio_std": round(float(df["iv_ratio_25d"].std()), 4),
           "atm_iv_mean": round(float(df["atm_iv"].mean()), 4)}
    # IC: iv_ratio 分位 与 裸 IV 分位
    res["ic_ratio"] = {h: round(information_coef(df, "ratio_pct", f"fwd_ret_{h}"), 4) for h in (5, 10, 20)}
    res["ic_ivlevel"] = {h: round(information_coef(df, "iv_pct", f"fwd_ret_{h}"), 4) for h in (5, 10, 20)}
    # 条件实验: 高IV日按情绪比中位数分顶/底
    hi = df[df["iv_pct"] > 0.7].dropna(subset=["iv_ratio_25d", "fwd_ret_20"])
    if len(hi) > 20:
        med = hi["iv_ratio_25d"].median()
        pan = hi[hi["iv_ratio_25d"] >= med]; eu = hi[hi["iv_ratio_25d"] < med]
        res["cond"] = {h: {"panic": round(float(pan[f"fwd_ret_{h}"].mean()) * 100, 2),
                           "eupho": round(float(eu[f"fwd_ret_{h}"].mean()) * 100, 2)}
                       for h in (5, 10, 20)}
    # 高分位组(ratio_pct>0.8) 未来收益 vs 全样本(反转信号强度)
    top = df[df["ratio_pct"] > 0.8]
    res["top_quintile_fwd"] = {h: round(float(top[f"fwd_ret_{h}"].mean()) * 100, 2) for h in (5, 10, 20)}
    res["all_fwd"] = {h: round(float(df[f"fwd_ret_{h}"].mean()) * 100, 2) for h in (5, 10, 20)}
    return res


def main():
    out = {}
    for key, (s, e, name) in KEYS.items():
        r = analyze_key(key, s, e)
        if r is None:
            print(f"[跳过] {key} 未重建")
            continue
        r["name"] = name
        out[key] = r

    print("\n===== 跨标的 IV 情绪比信号强度对比 =====\n")
    print(f"{'标的':<10}{'样本':>6}{'ATM均值':>9}{'比值均值':>9}{'IC10日':>9}{'高IV顶底差(20日)':>16}")
    for key, r in out.items():
        cond20 = r.get("cond", {}).get(20, {})
        spread = round(cond20.get("panic", 0) - cond20.get("eupho", 0), 2) if cond20 else float("nan")
        print(f"{r['name']:<10}{r['n']:>6}{r['atm_iv_mean']:>9.2%}{r['ratio_mean']:>9.3f}"
              f"{r['ic_ratio'][10]:>9.3f}{spread:>14}pp")

    json.dump(out, open("data/processed/compare.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print("\n已导出 data/processed/compare.json")


if __name__ == "__main__":
    main()
