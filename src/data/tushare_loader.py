"""Tushare Pro 数据加载层（历史期权数据主力源）。

提供历史期权全集与日线，用于重建历史波动率曲面、PCR、IV 情绪比等时序。
Token 读取顺序：环境变量 TUSHARE_TOKEN -> config/tushare_token.txt。

关键接口：
  opt_basic  期权合约元数据(行权价/认购认沽/上市退市/标的)
  opt_daily  期权日线(open/high/low/close/settle/vol/amount/oi)
  fund_daily ETF 日线(标的价, 供 BS 用)
  index_daily / trade_cal 指数与交易日历
"""
from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd
import requests

from src.data import cache
from src.utils.config import abspath
from src.utils.logger import get_logger

log = get_logger("data.tushare")
_URL = "http://api.tushare.pro"

# 标的 -> tushare opt_code / 交易所 / 标的价来源(fund=ETF, index=指数)
UNDERLYINGS = {
    "50etf":  {"opt_code": "OP510050.SH", "exchange": "SSE",   "under": "510050.SH", "utype": "fund"},
    "300etf": {"opt_code": "OP510300.SH", "exchange": "SSE",   "under": "510300.SH", "utype": "fund"},
    "500etf": {"opt_code": "OP510500.SH", "exchange": "SSE",   "under": "510500.SH", "utype": "fund"},
    "kc50":   {"opt_code": "OP588000.SH", "exchange": "SSE",   "under": "588000.SH", "utype": "fund"},
    "cyb":    {"opt_code": "OP159915.SZ", "exchange": "SZSE",  "under": "159915.SZ", "utype": "fund"},
    "zz1000": {"opt_code": "OP000852.SH", "exchange": "CFFEX", "under": "000852.SH", "utype": "index"},
    "hs300i": {"opt_code": "OP000300.SH", "exchange": "CFFEX", "under": "000300.SH", "utype": "index"},
}


@lru_cache(maxsize=1)
def _token() -> str:
    import os
    tok = os.environ.get("TUSHARE_TOKEN")
    if tok:
        return tok.strip()
    p = abspath("config/tushare_token.txt")
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError("未找到 tushare token(环境变量 TUSHARE_TOKEN 或 config/tushare_token.txt)")


def call(api_name: str, params: dict | None = None, fields: str = "",
         tries: int = 6, pause: float = 1.5) -> pd.DataFrame:
    """调用 tushare 接口, 返回 DataFrame。含重试与限频退避。"""
    payload = {"api_name": api_name, "token": _token(), "params": params or {}, "fields": fields}
    last = None
    for i in range(tries):
        try:
            r = requests.post(_URL, json=payload, timeout=60)
            j = r.json()
            if j.get("code") == 0:
                d = j["data"]
                return pd.DataFrame(d["items"], columns=d["fields"])
            msg = j.get("msg", "")
            if "每分钟" in msg or "多" in msg or "限" in msg:  # 限频
                time.sleep(5)
            last = RuntimeError(f"{api_name}: {msg}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(pause)
    raise last


def opt_basic(key: str, use_cache: bool = True) -> pd.DataFrame:
    """某标的的全部期权合约元数据。"""
    u = UNDERLYINGS[key]
    name = f"ts_optbasic_{key}"
    if use_cache and cache.exists(name):
        return cache.load(name)
    df = call("opt_basic", {"exchange": u["exchange"]},
              "ts_code,name,call_put,exercise_price,list_date,delist_date,opt_code,maturity_date")
    df = df[df["opt_code"] == u["opt_code"]].copy()
    df["exercise_price"] = pd.to_numeric(df["exercise_price"], errors="coerce")
    cache.save(df, name)
    return df


def trade_dates(start: str, end: str) -> list[str]:
    """SSE 交易日列表(YYYYMMDD)。"""
    df = call("trade_cal", {"exchange": "SSE", "start_date": start, "end_date": end, "is_open": "1"},
              "cal_date")
    return sorted(df["cal_date"].tolist())


def opt_daily_range(key: str, start: str, end: str, use_cache: bool = True,
                    pause: float = 0.15) -> pd.DataFrame:
    """某标的期权在 [start,end] 的全部日线(按交易日循环拉取)。

    返回列: ts_code, trade_date, open, high, low, close, settle, vol, amount, oi。
    """
    u = UNDERLYINGS[key]
    name = f"ts_optdaily_{key}_{start}_{end}"
    if use_cache and cache.exists(name):
        return cache.load(name)

    codes = set(opt_basic(key)["ts_code"])
    dates = trade_dates(start, end)
    frames = []
    for i, dt in enumerate(dates):
        day = call("opt_daily", {"trade_date": dt, "exchange": u["exchange"]},
                   "ts_code,trade_date,open,high,low,close,settle,vol,amount,oi")
        day = day[day["ts_code"].isin(codes)]
        frames.append(day)
        if (i + 1) % 50 == 0:
            log.info("  opt_daily %s: %d/%d 日", key, i + 1, len(dates))
        time.sleep(pause)
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "settle", "vol", "amount", "oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cache.save(df, name)
    log.info("opt_daily %s: %d 行 (%s~%s)", key, len(df), start, end)
    return df


def fund_daily(key: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """标的日线收盘(BS 用)。ETF 走 fund_daily, 指数走 index_daily。返回 trade_date, close。"""
    u = UNDERLYINGS[key]
    name = f"ts_fund_{key}_{start}_{end}"
    if use_cache and cache.exists(name):
        return cache.load(name)
    api = "index_daily" if u.get("utype") == "index" else "fund_daily"
    df = call(api, {"ts_code": u["under"], "start_date": start, "end_date": end},
              "trade_date,close")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    cache.save(df, name)
    return df
