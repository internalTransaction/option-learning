"""拉取中证1000成分股日收益 + 月度权重, 供隐含相关性(A股版 COR1M 代理)计算。

按 trade_date 逐日拉全市场日线, 只保留成分股并集的 pct_chg, 缓存到 data/raw。

用法:
    python -m scripts.fetch_constituents --start 20220801 --end 20260803
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data.catalog import latest_ranged_file
from src.data.tushare_loader import call
from src.utils.config import abspath
from src.utils.logger import get_logger

log = get_logger("fetch.cons")

INDEX = "000852.SH"


def fetch_weights(start: str, end: str) -> pd.DataFrame:
    """月度成分权重(每个自然月拉一次, tushare 返回该月最后一个调整日)。"""
    months = pd.date_range(start, end, freq="MS").strftime("%Y%m").tolist()
    frames = []
    for ym in months:
        m0 = f"{ym}01"
        m1 = (pd.Timestamp(m0) + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")
        try:
            d = call("index_weight", {"index_code": INDEX, "start_date": m0, "end_date": m1})
        except Exception as e:  # noqa: BLE001
            log.warning("权重 %s 失败: %s", ym, e)
            continue
        if len(d):
            frames.append(d)
            log.info("权重 %s: %d 行", ym, len(d))
    out = pd.concat(frames, ignore_index=True)
    out["weight"] = out["weight"].astype(float) / 100.0
    return out.drop_duplicates(["trade_date", "con_code"])


def fetch_returns(codes: set[str], start: str, end: str) -> pd.DataFrame:
    cal = call("trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end, "is_open": "1"})
    days = sorted(cal["cal_date"].tolist())
    log.info("交易日 %d 天", len(days))
    frames = []
    for i, d in enumerate(days):
        try:
            px = call("daily", {"trade_date": d}, fields="ts_code,trade_date,pct_chg")
        except Exception as e:  # noqa: BLE001
            log.warning("日线 %s 失败: %s", d, e)
            continue
        frames.append(px[px["ts_code"].isin(codes)])
        if i % 50 == 0:
            log.info("  %s (%d/%d)", d, i + 1, len(days))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20220801")
    ap.add_argument("--end", default="20260803")
    args = ap.parse_args()

    raw = abspath("data/raw")

    # 权重: 月度且便宜, 直接按需重拉整段(起点前多取一个月, 保证首日有可用的上月末权重)
    wpath = raw / f"zz1000_weight_{args.start}_{args.end}.parquet"
    if wpath.exists():
        w = pd.read_parquet(wpath)
    else:
        wstart = (pd.Timestamp(args.start) - pd.offsets.MonthBegin(2)).strftime("%Y%m%d")
        w = fetch_weights(wstart, args.end)
        w.to_parquet(wpath, index=False)
    log.info("权重: %d 行, %d 个截面, %d 只股票",
             len(w), w["trade_date"].nunique(), w["con_code"].nunique())

    # 收益: 逐日全市场拉取很贵, 走增量 —— 已有文件只补它之后的交易日
    prev = latest_ranged_file(raw, "zz1000_consret")
    if prev is not None and prev.end >= args.end:
        log.info("收益已是最新: %s~%s", prev.start, prev.end)
        return
    if prev is None:
        r = fetch_returns(set(w["con_code"]), args.start, args.end)
        rpath = raw / f"zz1000_consret_{args.start}_{args.end}.parquet"
    else:
        inc_start = (pd.Timestamp(prev.end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        log.info("补增量: %s~%s (当前 %s~%s)", inc_start, args.end, prev.start, prev.end)
        add = fetch_returns(set(w["con_code"]), inc_start, args.end)
        r = pd.concat([pd.read_parquet(prev.path), add], ignore_index=True)
        r = r.drop_duplicates(["ts_code", "trade_date"], keep="last")
        rpath = raw / f"zz1000_consret_{prev.start}_{args.end}.parquet"
    r.to_parquet(rpath, index=False)
    log.info("收益: %d 行 -> %s", len(r), rpath.name)

    # 指数本身刷新到最新
    idx = call("index_daily", {"ts_code": INDEX, "start_date": args.start, "end_date": args.end},
               fields="trade_date,close,pct_chg")
    idx.to_parquet(raw / f"idx_000852_{args.start}_{args.end}.parquet", index=False)
    log.info("指数: %d 行", len(idx))


if __name__ == "__main__":
    main()
