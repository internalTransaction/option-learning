"""美股指数期权 · 择时信号台数据构建。

对 SPY/QQQ/SOXX 重建历史波动率曲面(Polygon 逐合约 + BS 反解 IV), 复用 A 股口径的
build_one() 计算统一指标+滚动百分位, 输出 us_timing_viz.json(schema 与 timing_viz.json
一致), 供 timing 模板渲染。

用法:
    python -m scripts.build_us_timing_viz            # 默认日线, 近 ~14 个月
    python -m scripts.build_us_timing_viz hour 2026-04-01 2026-07-22
    python -m scripts.build_us_timing_viz day 2025-05-01 2026-07-22 --no-cache
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_timing_viz import build_one, clean, roll_pct   # 日线口径
from src.data import cache
from src.research import us_surface_history as ush
from src.utils.logger import get_logger

BARS_PER_DAY = {"day": 1, "hour": 7}     # RTH 每日 bar 数(小时线 9:00-15:00 ET)

log = get_logger("build_us_viz")
PROC = ROOT / "data" / "processed"

# 标的: key -> (中文名, Polygon ticker)
SOURCES = {
    "spy":  ("标普500 (SPY)",   "SPY"),
    "qqq":  ("纳指100 (QQQ)",   "QQQ"),
    "soxx": ("费城半导体 (SOXX)", "SOXX"),
}


def _rpct(s: pd.Series, win: int, minp: int) -> pd.Series:
    """NaN 稳健的滚动分位(窗口内剔除 NaN)。"""
    def f(x):
        last = x[-1]
        if np.isnan(last):
            return np.nan
        v = x[~np.isnan(x)]
        return (v <= last).mean() if len(v) else np.nan
    return s.rolling(win, min_periods=minp).apply(f, raw=True)


def add_supplementary(rec: dict, surf: pd.DataFrame, bpd: int) -> None:
    """把回测验证过的美股补充因子注入 rec: VVIX(波动的波动) / 凸度 / 下行VRP。"""
    d = surf.sort_values("date").reset_index(drop=True)
    W, MP = 63 * bpd, (63 * bpd) // 2
    var_ann = 252 * bpd

    ivchg = np.log(d["atm_iv"] / d["atm_iv"].shift(1))
    vvix = ivchg.rolling(10 * bpd, min_periods=max(5, 5 * bpd)).std() * np.sqrt(var_ann)
    conv = d["bf_25d"] if "bf_25d" in d.columns else pd.Series(np.nan, index=d.index)
    logret = np.log(d["spot"] / d["spot"].shift(1))
    neg = np.minimum(logret, 0.0)
    drv = np.sqrt((neg ** 2).rolling(20 * bpd, min_periods=10).mean() * var_ann)
    dvrp = d["atm_iv"] - drv

    skew = d["skew_cboe"] if "skew_cboe" in d.columns else pd.Series(np.nan, index=d.index)

    rec["vvix"] = clean(vvix, 4);  rec["vvix_pct"] = clean(_rpct(vvix, W, MP), 3)
    rec["conv"] = clean(conv, 4);  rec["conv_pct"] = clean(_rpct(conv, W, MP), 3)
    rec["dvrp"] = clean(dvrp, 4);  rec["dvrp_pct"] = clean(_rpct(dvrp, W, MP), 3)
    rec["skew"] = clean(skew, 2);  rec["skew_pct"] = clean(_rpct(skew, W, MP), 3)


def build_one_scaled(df: pd.DataFrame, bpd: int) -> dict:
    """盘中口径: 与 build_one 相同, 但所有滚动窗口按 bar/天(bpd)放大。

    恐慌/分位窗口 = 63 交易日 × bpd; 动量回看 = 20 交易日 × bpd;
    动量分位窗口 = 63 交易日 × bpd(盘中数据较短, 用 3 月参照而非 1 年)。
    """
    W, MP = 63 * bpd, (63 * bpd) // 2
    RVW, MOMW = 20 * bpd, 20 * bpd
    MPCT_W, MPCT_MP = 63 * bpd, 20 * bpd
    ann = np.sqrt(252 * bpd)

    df = df.sort_values("date").reset_index(drop=True)
    df["price"] = df["spot"]
    df["dd20"] = df["price"] / df["price"].rolling(20 * bpd, min_periods=1).max() - 1.0
    df["mom20"] = df["price"] / df["price"].shift(MOMW) - 1.0
    logret = np.log(df["price"] / df["price"].shift(1))
    df["rv"] = logret.rolling(RVW, min_periods=max(5, RVW // 2)).std() * ann
    df["vrp"] = df["atm_iv"] - df["rv"]
    df["sent"], df["rr"] = df["iv_ratio_25d"], df["rr_25d"]
    df["slope"], df["pcr"] = df["smile_slope"], df["pcr_vol"]

    def rp(s):
        return roll_pct(s, W, MP)

    return {
        "dates": [str(x) for x in df["date"].tolist()],
        "price": clean(df["price"], 4), "dd20": clean(df["dd20"], 4),
        "mom20": clean(df["mom20"], 4),
        "mom_pct": clean(roll_pct(df["mom20"], MPCT_W, MPCT_MP), 3),
        "atm_iv": clean(df["atm_iv"], 4), "iv_pct": clean(rp(df["atm_iv"]), 3),
        "sent": clean(df["sent"], 4), "sent_pct": clean(rp(df["sent"]), 3),
        "rr": clean(df["rr"], 4), "rr_pct": clean(rp(df["rr"]), 3),
        "slope": clean(df["slope"], 4), "slope_pct": clean(rp(df["slope"]), 3),
        "vrp": clean(df["vrp"], 4), "vrp_pct": clean(rp(df["vrp"]), 3),
        "rv": clean(df["rv"], 4),
        "pcr": clean(df["pcr"], 4), "pcr_pct": clean(rp(df["pcr"]), 3),
    }


def get_surface(key: str, ticker: str, start: str, end: str,
                gran: str, use_cache: bool):
    name = f"us_surface_{key}_{gran}_{start}_{end}"
    if use_cache and cache.exists(name, "processed"):
        log.info("载入缓存 %s", name)
        return cache.load(name, "processed")
    df = ush.reconstruct(ticker, start, end, gran)
    cache.save(df, name, "processed")
    return df


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_cache = "--no-cache" not in sys.argv
    gran = args[0] if len(args) > 0 else "day"
    end = args[2] if len(args) > 2 else datetime.now().strftime("%Y-%m-%d")
    if len(args) > 1:
        start = args[1]
    else:
        back = 430 if gran == "day" else 160     # 日线~14月; 盘中~5月(留 63 日分位预热)
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=back)).strftime("%Y-%m-%d")

    log.info("构建美股信号台: %s  %s~%s  cache=%s", gran, start, end, use_cache)
    out = {}
    for key, (name, ticker) in SOURCES.items():
        try:
            surf = get_surface(key, ticker, start, end, gran, use_cache)
        except Exception as e:
            log.error("%s 失败: %s", key, e)
            continue
        if surf is None or surf.empty:
            log.warning("%s 无数据, 跳过", key)
            continue
        rec = build_one(surf) if gran == "day" else build_one_scaled(surf, BARS_PER_DAY[gran])
        add_supplementary(rec, surf, BARS_PER_DAY[gran])
        rec["name"] = name
        rec["gex_z"] = [None] * len(rec["dates"])   # 美股暂无 GEX
        out[key] = rec
        log.info("%-5s %-14s %d 时点  %s->%s", key, name, len(rec["dates"]),
                 rec["dates"][0], rec["dates"][-1])

    suffix = "" if gran == "day" else f"_{gran}"
    dest = PROC / f"us_timing_viz{suffix}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    log.info("写出 %s (%.0f KB)", dest, dest.stat().st_size / 1024)


if __name__ == "__main__":
    main()
