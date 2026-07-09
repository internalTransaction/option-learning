"""拉 ETF 60min K线(tushare stk_mins, 含成交量) → data/raw/min60_<key>.parquet。
stk_mins 限频 1次/分钟, 故分段拉、段间 sleep 65s。先只跑科创(验证3H HVWMA概念)。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data.tushare_loader import call

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
JOBS = {
    "kc50": ("588000.SH", "2023-06-01", "2026-07-09"),
    # 验证成功后再开:
    # "hs300": ("510300.SH", "2020-01-01", "2026-07-09"),
    # "cyb":   ("159915.SZ", "2022-09-01", "2026-07-09"),
}
FIELDS = "ts_code,trade_time,open,high,low,close,vol,amount"


def seg_ranges(start, end, months=18):
    s = pd.Timestamp(start); e = pd.Timestamp(end); out = []
    while s < e:
        nx = min(s + pd.DateOffset(months=months), e)
        out.append((s.strftime("%Y-%m-%d"), nx.strftime("%Y-%m-%d")))
        s = nx + pd.Timedelta(days=1)
    return out


def fetch(key, ts_code, start, end):
    parts = []
    for i, (a, b) in enumerate(seg_ranges(start, end, months=120)):   # 单段: 全历史一次拉(<8000行)
        print(f"  [{key}] 段{i+1}: {a}~{b}  (先 sleep 70s 避限频…)", flush=True)
        time.sleep(70)
        df = call("stk_mins", {"ts_code": ts_code, "freq": "60min",
                               "start_date": a+" 09:00:00", "end_date": b+" 15:00:00"},
                  FIELDS, tries=4, pause=30)
        print(f"     → {len(df)} 根", flush=True)
        parts.append(df)
    out = pd.concat(parts).drop_duplicates("trade_time").sort_values("trade_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    dest = RAW / f"min60_{key}.parquet"
    out.to_parquet(dest)
    print(f"  [{key}] 写出 {dest}  ({len(out)} 根, {out['trade_time'].min()}~{out['trade_time'].max()})", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(JOBS)
    for k in keys:
        fetch(k, *JOBS[k])
    print("完成。", flush=True)
