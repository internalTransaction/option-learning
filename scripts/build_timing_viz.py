"""构建期权择时信号可视化数据集 (主图 + 副图, 十字光标交互).

从各标的 surface parquet 计算统一指标序列, 输出精简 JSON, 直接内嵌到 HTML。
指标:
  price      标的收盘价 (主图)
  atm_iv     平值隐含波动率
  iv_pct     ATM IV 滚动 1 年百分位  (高IV情绪)
  sent       情绪比 = put25d IV / call25d IV  (券商框架核心信号)
  rr         25d Risk Reversal (call-put 偏斜)
  slope      smile 斜率
  vrp        方差风险溢价 = atm_iv - rv20
  rv         20 日已实现波动率(年化)
  pcr        认沽认购成交量比
  corr_semi  半隐含相关性 = (IV_idx² − Σw²σ²)/((Σwσ)² − Σw²σ²), A股版 COR1M 代理
  corr_rv    同式全用已实现波动率, 真实齐涨齐跌程度
每个副图指标同时给出滚动百分位, 用于标注极值带。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.catalog import latest_ranged_file

PROC = ROOT / "data" / "processed"

SOURCES = {
    "hs300":  ("沪深300ETF", "300etf"),
    "zz1000": ("中证1000",   "zz1000"),
    "zz500":  ("中证500",    "500etf"),
    "sz50":   ("上证50",     "50etf"),
    "kc50":   ("科创50",     "kc50"),
    "cyb":    ("创业板",     "cyb"),
}

WIN = 63           # 滚动百分位窗口 (63交易日≈3个月, 中金/恐慌择时报告口径)
MINP = WIN // 2    # 最小样本 (与 research.validate.rolling_pct 一致)
RV_WIN = 20        # 已实现波动率窗口


def roll_pct(s: pd.Series, win: int = WIN, minp: int = MINP) -> pd.Series:
    """滚动百分位 (当前值在过去 win 天中的分位, 0-1)。"""
    return s.rolling(win, min_periods=minp).apply(
        lambda x: (x <= x[-1]).mean(), raw=True
    )


def realized_vol(price: pd.Series) -> pd.Series:
    logret = np.log(price / price.shift(1))
    return logret.rolling(RV_WIN, min_periods=10).std() * np.sqrt(252)


def clean(series: pd.Series, nd: int = 4):
    """转 list, NaN -> None, 保留 nd 位小数。"""
    out = []
    for v in series:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            out.append(None)
        else:
            out.append(round(float(v), nd))
    return out


def build_one(df: pd.DataFrame) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    df["price"] = df["spot"]
    # 20日回撤 = 收盘 / 过去20日最高 - 1 (旧口径, tooltip 仍展示)
    df["dd20"] = df["price"] / df["price"].rolling(20, min_periods=1).max() - 1.0
    # 20日动量及其 252 日滚动分位 = 动态阈值(替代 hardcode 回撤; 低分位=跌得狠, 高分位=涨得猛)
    df["mom20"] = df["price"] / df["price"].shift(20) - 1.0
    df["rv"] = realized_vol(df["price"])
    df["vrp"] = df["atm_iv"] - df["rv"]
    df["sent"] = df["iv_ratio_25d"]          # put/call IV 比
    df["rr"] = df["rr_25d"]
    df["slope"] = df["smile_slope"]
    df["pcr"] = df["pcr_vol"]
    # vol-of-vol 两个代理(A股无VIX期权, 无法算真VVIX):
    #   bf   蝶式凸度 = (IV_c25+IV_p25)/2 − ATM_IV, 市场为大波动/尾部付的溢价(横截面)
    #   rvvol 已实现 vol-of-vol = RV20 对数变化的20日波动(只用价格, 最正交)
    df["bf"] = df["bf_25d"]
    df["rvvol"] = np.log(df["rv"]).diff().rolling(20, min_periods=10).std() * np.sqrt(252)

    d = {
        "dates": [str(x) for x in df["date"].tolist()],
        "price": clean(df["price"], 4),
        "dd20": clean(df["dd20"], 4),
        "mom20": clean(df["mom20"], 4),
        "mom_pct": clean(roll_pct(df["mom20"], 252, 126), 3),
        "atm_iv": clean(df["atm_iv"], 4),
        "iv_pct": clean(roll_pct(df["atm_iv"]), 3),
        "sent": clean(df["sent"], 4),
        "sent_pct": clean(roll_pct(df["sent"]), 3),
        "rr": clean(df["rr"], 4),
        "rr_pct": clean(roll_pct(df["rr"]), 3),
        "slope": clean(df["slope"], 4),
        "slope_pct": clean(roll_pct(df["slope"]), 3),
        "vrp": clean(df["vrp"], 4),
        "vrp_pct": clean(roll_pct(df["vrp"]), 3),
        "rv": clean(df["rv"], 4),
        "pcr": clean(df["pcr"], 4),
        "pcr_pct": clean(roll_pct(df["pcr"]), 3),
        "bf": clean(df["bf"], 4),
        "bf_pct": clean(roll_pct(df["bf"]), 3),
        "rvvol": clean(df["rvvol"], 4),
        "rvvol_pct": clean(roll_pct(df["rvvol"]), 3),
    }
    return d


def main() -> None:
    out = {}
    for key, (name, surface_key) in SOURCES.items():
        found = latest_ranged_file(PROC, f"surface_{surface_key}")
        if found is None:
            print("跳过(缺失):", f"surface_{surface_key}_*.parquet")
            continue
        df = pd.read_parquet(found.path)
        rec = build_one(df)
        rec["name"] = name
        # 并入 GEX 标准分(按日期对齐, 缺失=None) — 供仓位 GEX 调节 + 副图
        gexf = PROC / f"gex_{key}.json"
        if gexf.exists():
            G = json.load(open(gexf))
            gz = {d: z for d, z in zip(G["date"], G["gex_z"])}
            rec["gex_z"] = [round(gz[d], 3) if isinstance(gz.get(d), (int, float)) and np.isfinite(gz[d]) else None
                            for d in rec["dates"]]
        else:
            rec["gex_z"] = [None] * len(rec["dates"])
        # 并入隐含相关性(A股版 COR1M 代理)。分位在原始连续序列上算完再对齐,
        # 避免 reindex 产生的 NaN 空洞污染滚动窗口。目前只有中证1000有(需成分股数据)。
        corrf = PROC / f"implied_corr_{key}_21d.parquet"
        n = len(rec["dates"])
        if corrf.exists():
            C = pd.read_parquet(corrf).sort_values("trade_date")
            for col in ("corr_semi", "corr_rv"):
                C[col + "_pct"] = roll_pct(C[col])
            C = C.set_index("trade_date")
            for col in ("corr_semi", "corr_rv", "corr_semi_pct", "corr_rv_pct"):
                rec[col] = clean(C[col].reindex(rec["dates"]), 3)
        else:
            for col in ("corr_semi", "corr_rv", "corr_semi_pct", "corr_rv_pct"):
                rec[col] = [None] * n

        out[key] = rec
        print(f"{key:7s} {name:8s} {len(rec['dates'])} 天  "
              f"{rec['dates'][0]}->{rec['dates'][-1]}  ({found.path.name})"
              f"{'  +corr' if corrf.exists() else ''}")

    dest = PROC / "timing_viz.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    kb = dest.stat().st_size / 1024
    print(f"写出 {dest}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
