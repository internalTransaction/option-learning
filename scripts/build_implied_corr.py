"""中证1000 隐含相关性指数(A股版 CBOE COR1M 代理)。

CBOE COR1M 的定义:
    rho = (sigma_idx^2 - sum w_i^2 sigma_i^2) / ((sum w_i sigma_i)^2 - sum w_i^2 sigma_i^2)
分子用指数期权 IV, 分母用成分股期权 IV。A股无个股期权, 分母只能用已实现波动率,
因此本脚本产出三条序列:

    corr_rv    全已实现: 分子分母都用 21日已实现波动率  -> 真实同涨同跌程度
    corr_semi  半隐含  : 分子用 MO 期权 ATM IV, 分母仍用已实现  -> 市场为系统性风险付的价
    corr_crp   相关性风险溢价 = corr_semi - corr_rv

注意 corr_semi 中混入了指数的方差风险溢价(VRP), 不是纯粹的相关性预期;
两条序列必须同时看, 差值才是可解释的部分。

用法:
    python -m scripts.build_implied_corr --window 21
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.data.catalog import latest_ranged_file
from src.utils.config import abspath
from src.utils.logger import get_logger

log = get_logger("corr")

ANN = np.sqrt(252.0)


def load_inputs(window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, proc = abspath("data/raw"), abspath("data/processed")

    wf = sorted(raw.glob("zz1000_weight_*.parquet"))
    rf = sorted(raw.glob("zz1000_consret_*.parquet"))
    if not wf or not rf:
        raise FileNotFoundError("缺成分股数据, 先跑 python -m scripts.fetch_constituents")
    w = pd.read_parquet(wf[-1])
    r = pd.read_parquet(rf[-1])

    idxf = sorted(raw.glob("idx_000852_*.parquet"))
    idx = pd.read_parquet(idxf[-1])

    sf = latest_ranged_file(proc, "surface_zz1000")
    if sf is None:
        raise FileNotFoundError("缺 surface_zz1000_*.parquet")
    surf = pd.read_parquet(sf.path, columns=["date", "atm_iv"])
    surf = surf.rename(columns={"date": "trade_date"})
    surf["trade_date"] = surf["trade_date"].astype(str)
    log.info("权重 %d 行 / 成分收益 %d 行 / 指数 %d 行 / 曲面 %d 行",
             len(w), len(r), len(idx), len(surf))
    return w, r, (idx.merge(surf, on="trade_date", how="left"))


def compute(window: int) -> pd.DataFrame:
    w, r, idx = load_inputs(window)

    # 成分股收益宽表 (日期 x 股票), 单位: 小数
    ret = (r.assign(ret=r["pct_chg"].astype(float) / 100.0)
             .pivot_table(index="trade_date", columns="ts_code", values="ret"))
    ret = ret.sort_index()
    days = ret.index.tolist()

    # 每日适用的权重 = 最近一个不晚于当日的成分截面
    wdates = sorted(w["trade_date"].unique())
    wmap = {d: g.set_index("con_code")["weight"] for d, g in w.groupby("trade_date")}

    idx = idx.set_index("trade_date").sort_index()
    idx_ret = idx["pct_chg"].astype(float) / 100.0

    rows = []
    for i in range(window, len(days)):
        d = days[i]
        wd = [x for x in wdates if x <= d]
        if not wd:
            continue
        wt = wmap[wd[-1]]

        win = ret.iloc[i - window + 1: i + 1]
        # 成分股要求窗口内至少 80% 有效(剔停牌/新上市)
        valid = win.columns[win.notna().sum() >= window * 0.8]
        cols = wt.index.intersection(valid)
        if len(cols) < 300:
            continue

        sig = win[cols].std(ddof=1).values * ANN          # 个股已实现波动率
        wv = wt[cols].values
        wv = wv / wv.sum()                                 # 权重重归一

        sum_w2s2 = float((wv ** 2 * sig ** 2).sum())       # Σ wi² σi²
        sum_ws = float((wv * sig).sum())                   # Σ wi σi
        denom = sum_ws ** 2 - sum_w2s2                     # 交叉项总量
        if denom <= 0:
            continue

        # 指数已实现波动率(用真实指数序列, 而非按权重合成, 避免权重误差)
        iwin = idx_ret.reindex(days[i - window + 1: i + 1])
        if iwin.isna().sum() > 0:
            continue
        idx_rv = float(iwin.std(ddof=1) * ANN)
        iv = idx.at[d, "atm_iv"] if d in idx.index else np.nan

        rows.append({
            "trade_date": d,
            "idx_rv": idx_rv,
            "idx_iv": float(iv) if pd.notna(iv) else np.nan,
            "avg_cons_vol": sum_ws,
            "n_cons": len(cols),
            "corr_rv": (idx_rv ** 2 - sum_w2s2) / denom,
            "corr_semi": ((float(iv) ** 2 - sum_w2s2) / denom) if pd.notna(iv) else np.nan,
        })

    out = pd.DataFrame(rows)
    out["corr_crp"] = out["corr_semi"] - out["corr_rv"]
    # 分散度: 指数波动相对成分股平均波动的折价, 相关性的等价表达
    out["dispersion"] = out["avg_cons_vol"] - out["idx_rv"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=21, help="已实现波动率窗口(交易日)")
    args = ap.parse_args()

    df = compute(args.window)
    proc = abspath("data/processed")
    df.to_parquet(proc / f"implied_corr_zz1000_{args.window}d.parquet", index=False)

    d = df.dropna(subset=["corr_semi"])
    log.info("样本 %d 天 (%s -> %s)", len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1])
    print(df[["corr_rv", "corr_semi", "corr_crp", "idx_iv", "idx_rv", "n_cons"]].describe().round(3).to_string())
    print("\n最近 10 天:")
    print(d.tail(10).round(3).to_string(index=False))

    (proc / f"implied_corr_zz1000_{args.window}d.json").write_text(
        json.dumps({"rows": df.round(4).to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
