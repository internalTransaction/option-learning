"""分析历史曲面时序, 验证"IV 情绪比区分顶/底"假设, 并导出可视化数据。

核心问题(用户洞察): A股 melt-up 顶部与恐慌底部都可能出现极高 IV, 故裸 IV 水平歧义。
Put/Call IV 情绪比应能区分二者: 底部抢认沽→比值高; 顶部追认购→比值低。

产出:
  1. 时序: date, price, atm_iv, iv_pct, iv_ratio_25d, ratio_pct, skew_25d, pcr_oi
  2. 分位前瞻收益: 分别对 atm_iv 百分位 与 iv_ratio 百分位 做 5 分位×多期限
  3. 条件实验: 高IV日中, 按 iv_ratio 高/低分组的未来收益
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.data import cache
from src.research.validate import add_forward, quantile_forward, information_coef, rolling_pct


def analyze(key: str, start: str, end: str) -> dict:
    df = cache.load(f"surface_{key}_{start}_{end}", "processed")
    df = df.sort_values("date").reset_index(drop=True)
    df = add_forward(df, price_col="spot", horizons=(5, 10, 20))

    # 滚动分位(63日≈3个月, 中金口径)
    df["iv_pct"] = rolling_pct(df["atm_iv"], 63)
    df["ratio_pct"] = rolling_pct(df["iv_ratio_25d"], 63)
    df["skew_pct"] = rolling_pct(df["skew_25d"], 63)

    out = {"key": key, "n": len(df)}

    # 分位前瞻收益(反转信号: 高分位组未来收益应更高)
    out["quantile"] = {}
    for sig in ["iv_pct", "ratio_pct", "skew_pct"]:
        tab = {}
        for h in (5, 10, 20):
            q = quantile_forward(df, sig, f"fwd_ret_{h}", n=5)
            if not q.empty:
                tab[h] = {
                    "mean_ret": q["mean_fwd_ret"].tolist(),
                    "win": q["win_rate"].tolist(),
                    "ic": round(information_coef(df, sig, f"fwd_ret_{h}"), 4),
                }
        out["quantile"][sig] = tab

    # 条件实验: 高IV日(iv_pct>0.7) 里, 按 iv_ratio 中位数分"恐慌底/逼空顶"
    hi = df[df["iv_pct"] > 0.7].dropna(subset=["iv_ratio_25d", "fwd_ret_10"])
    if len(hi) > 20:
        med = hi["iv_ratio_25d"].median()
        panic = hi[hi["iv_ratio_25d"] >= med]   # 高比值=认沽贵=恐慌
        eupho = hi[hi["iv_ratio_25d"] < med]    # 低比值=认购贵=追涨
        out["conditional"] = {
            "median_ratio": round(float(med), 4),
            "panic_high_ratio": {h: round(float(panic[f"fwd_ret_{h}"].mean()), 4) for h in (5, 10, 20)},
            "euphoria_low_ratio": {h: round(float(eupho[f"fwd_ret_{h}"].mean()), 4) for h in (5, 10, 20)},
            "n_panic": len(panic), "n_eupho": len(eupho),
        }

    # 时序(周频抽样导出)
    step = max(1, len(df) // 320)
    idx = list(range(0, len(df), step))
    def ds(col):
        return [None if pd.isna(df[col].iloc[i]) else round(float(df[col].iloc[i]), 4) for i in idx]
    out["series"] = {
        "dates": [df["date"].iloc[i] for i in idx],
        "price": ds("spot"), "atm_iv": ds("atm_iv"), "iv_pct": ds("iv_pct"),
        "iv_ratio": ds("iv_ratio_25d"), "ratio_pct": ds("ratio_pct"),
        "skew": ds("skew_25d"), "pcr_oi": ds("pcr_oi"),
    }
    return out


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "300etf"
    start = sys.argv[2] if len(sys.argv) > 2 else "20200101"
    end = sys.argv[3] if len(sys.argv) > 3 else "20260707"
    res = analyze(key, start, end)

    # 控制台摘要
    print(f"=== {key} 曲面信号分析 ({res['n']} 日) ===")
    print("\n[分位前瞻收益 IC] (负=反转, 正=动量)")
    for sig, tab in res["quantile"].items():
        ics = {h: v["ic"] for h, v in tab.items()}
        print(f"  {sig:12s} IC(5/10/20日): {ics}")
    if "conditional" in res:
        c = res["conditional"]
        print(f"\n[条件实验] 高IV日(n={c['n_panic']+c['n_eupho']}), 按IV情绪比中位数{c['median_ratio']}切分")
        print(f"  恐慌(高比值,n={c['n_panic']}) 未来收益 5/10/20日: {c['panic_high_ratio']}")
        print(f"  追涨(低比值,n={c['n_eupho']}) 未来收益 5/10/20日: {c['euphoria_low_ratio']}")

    out_path = f"data/processed/analysis_{key}.json"
    json.dump(res, open(out_path, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"\n已导出 {out_path}")


if __name__ == "__main__":
    main()
