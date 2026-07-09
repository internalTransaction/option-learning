"""Update raw/surface data to the latest available settled trading day.

This script appends only missing dates, then refreshes derived report data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import cache
from src.data import tushare_loader as ts
from src.data.catalog import latest_ranged_file
from src.research.surface_history import reconstruct

PROC = ROOT / "data" / "processed"

SURFACES = {
    "300etf": "20200101",
    "zz1000": "20220801",
    "kc50": "20230601",
    "cyb": "20220901",
}


def latest_available_trade_date() -> str:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    start = (today - timedelta(days=14)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    dates = ts.trade_dates(start, end)
    for dt in reversed(dates):
        day = ts.call(
            "opt_daily",
            {"trade_date": dt, "exchange": "SSE"},
            "ts_code",
            tries=2,
            pause=0.5,
        )
        if len(day):
            return dt
    raise RuntimeError(f"{start}~{end} 未找到可用 opt_daily 数据")


def next_trade_dates(after: str, end: str) -> list[str]:
    start = (pd.to_datetime(after, format="%Y%m%d") + pd.Timedelta(days=1)).strftime("%Y%m%d")
    if start > end:
        return []
    return ts.trade_dates(start, end)


def update_surface(key: str, target_end: str) -> Path | None:
    existing = latest_ranged_file(PROC, f"surface_{key}")
    if existing and existing.end >= target_end:
        print(f"{key:7s} 已是最新: {existing.start}~{existing.end}")
        return existing.path

    ts.opt_basic(key, use_cache=False)

    if existing is None:
        start = SURFACES[key]
        print(f"{key:7s} 初次重建: {start}~{target_end}")
        reconstruct(key, start, target_end, use_cache=False)
        created = latest_ranged_file(PROC, f"surface_{key}")
        return created.path if created else None

    dates = next_trade_dates(existing.end, target_end)
    if not dates:
        print(f"{key:7s} 无新增交易日: {existing.start}~{existing.end}")
        return existing.path

    inc_start = dates[0]
    print(f"{key:7s} 补增量: {inc_start}~{target_end}  (当前 {existing.start}~{existing.end})")
    added = reconstruct(key, inc_start, target_end, use_cache=False)
    if added.empty:
        print(f"{key:7s} 增量为空，保持原文件")
        return existing.path

    base = pd.read_parquet(existing.path)
    merged = pd.concat([base, added], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    out_name = f"surface_{key}_{existing.start}_{target_end}"
    return cache.save(merged, out_name, "processed")


def parse_keys(value: str) -> list[str]:
    keys = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in keys if x not in SURFACES]
    if unknown:
        raise SystemExit(f"未知 key: {', '.join(unknown)}; 可选: {', '.join(SURFACES)}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", help="指定目标交易日 YYYYMMDD；缺省自动找最新已落库日线")
    parser.add_argument("--keys", default=",".join(SURFACES), help="逗号分隔: 300etf,zz1000,kc50,cyb")
    parser.add_argument("--no-reports", action="store_true", help="只更新数据，不刷新报告 JSON/HTML")
    args = parser.parse_args()

    target_end = args.end or latest_available_trade_date()
    keys = parse_keys(args.keys)
    print(f"目标最新交易日: {target_end}")
    for key in keys:
        update_surface(key, target_end)

    if not args.no_reports:
        print("\n刷新报告数据与 HTML")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "refresh_reports.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
