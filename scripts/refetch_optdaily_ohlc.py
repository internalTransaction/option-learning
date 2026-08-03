"""重拉带 open/high/low 的期权日线 (之前只存了 close/settle)。
用于 t+1 开盘建仓的真实回测。存为单一权威文件 ts_optdaily_{key}_ohlc_{start}_{end}.parquet。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
from src.data import tushare_loader as t

ROOT = Path(__file__).resolve().parents[1]


def refetch(key: str, start: str, end: str):
    u = t.UNDERLYINGS[key]
    codes = set(t.opt_basic(key)["ts_code"])
    dates = t.trade_dates(start, end)
    frames = []
    for i, dt in enumerate(dates):
        day = t.call("opt_daily", {"trade_date": dt, "exchange": u["exchange"]},
                     "ts_code,trade_date,open,high,low,close,settle,vol,amount,oi")
        frames.append(day[day["ts_code"].isin(codes)])
        if (i + 1) % 50 == 0:
            print(f"  {key}: {i+1}/{len(dates)} 日", flush=True)
        time.sleep(0.13)
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "settle", "vol", "amount", "oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = ROOT / f"data/raw/ts_optdaily_{key}_ohlc_{start}_{end}.parquet"
    df.to_parquet(out)
    print(f"✓ {key}: {len(df)} 行 -> {out.name}  (open覆盖 {df['open'].gt(0).mean()*100:.0f}%)", flush=True)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "zz1000"
    start = sys.argv[2] if len(sys.argv) > 2 else "20220801"
    end = sys.argv[3] if len(sys.argv) > 3 else "20260720"
    print(f"重拉 {key} {start}~{end} ...", flush=True)
    refetch(key, start, end)
