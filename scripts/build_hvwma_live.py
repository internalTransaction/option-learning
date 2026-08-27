"""信号台 HVWMA 实盘对齐 — 用新浪免费近期60min线算「当前」3H HVWMA 趋势态(红/绿)。

目的(用户定): 信号台展示的趋势状态 = 用户实盘用的 3H×21 HVWMA。
数据源选择: 本机(腾讯云IP)被东财 push2his 网络层拒绝(HTTP 000), akshare 东财源不可用;
           改用**新浪** getKLineData(scale=60), ETF+指数皆可, 免费、含成交量、约近80交易日。
口径: N60=63(=21根3H×3)、SMOOTH60=30、成交量加权、几何 —— 与 build_hvwma_3h.py 完全一致。
新浪失败则回退日线近似(exit_machine.hvwma_daily_dir), 信号台永远有值。

输出: data/processed/hvwma_live.json
  { key: {name, dir(当前+1/-1), asof, source(sina/fallback), recent:[{date,dir,close}], line:[平滑值] } }
"""
from __future__ import annotations
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"

# key -> (新浪代码, 名称)  ETF: sh/sz+代码; 指数: sh000852
SYMS = {
    "hs300":  ("sh510300", "沪深300ETF"),
    "zz1000": ("sh000852", "中证1000"),
    "kc50":   ("sh588000", "科创50"),
    "cyb":    ("sz159915", "创业板"),
}
N60, SMOOTH60 = 63, 30
STD_BARS = ["10:30", "11:30", "14:00", "15:00"]
DATALEN = 320       # 60min bar 数(~80交易日, 足够 N60=63 warmup)


def _wma(x, n):
    n = max(1, int(n)); w = np.arange(1, n + 1)
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        seg = x[i - n + 1:i + 1]
        out[i] = np.dot(seg, w) / w.sum()
    return out


def _vwma(x, cv, n):
    return _wma(x * cv, n) / _wma(cv, n)


def _vol_hma(x, cv, n):
    half, sq = max(1, n // 2), max(1, round(math.sqrt(n)))
    return _wma(2 * _vwma(x, cv, half) - _vwma(x, cv, n), sq)


def fetch_sina(sym, tries=3):
    """新浪 60min K线 -> DataFrame[dt, close, vol]。温和重试。"""
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/cb/CN_MarketDataService.getKLineData"
           f"?symbol={sym}&scale=60&ma=no&datalen={DATALEN}")
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
            m = re.search(r"(\[.*\])", r, re.S)
            data = json.loads(m.group(1)) if m else []
            if data:
                df = pd.DataFrame(data)
                df["dt"] = pd.to_datetime(df["day"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["vol"] = pd.to_numeric(df["volume"], errors="coerce")
                return df[["dt", "close", "vol"]]
        except Exception as e:
            last = e
        if i < tries - 1:
            time.sleep(4)
    raise RuntimeError(f"sina {sym} 拉取失败: {last}")


def hvwma_from_min(df):
    """60min DataFrame[dt,close,vol] -> (每日方向dict, 平滑线尾段, 收盘dict)。"""
    df = df.copy()
    df["hm"] = df["dt"].dt.strftime("%H:%M")
    df = df[df["hm"].isin(STD_BARS)].sort_values("dt").reset_index(drop=True)
    cv = df["vol"].fillna(0).to_numpy() + 1.0
    base = np.log(df["close"].to_numpy())
    hma = _vol_hma(base, cv, N60)
    a = 1.0 / SMOOTH60; out = np.copy(hma)
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
        elif not np.isnan(out[i - 1]):
            out[i] = a * out[i] + (1 - a) * out[i - 1]
    line = np.exp(out)
    d = np.sign(np.diff(line, prepend=line[0]))
    df["date"] = df["dt"].dt.strftime("%Y%m%d")
    df["dir"] = d
    df = df[np.isfinite(df["dir"])].reset_index(drop=True)   # 丢弃 warmup NaN 段
    if df.empty:
        raise RuntimeError("有效bar不足(warmup后为空), 需更长 DATALEN")
    per = df.groupby("date").agg(dir=("dir", "last"), close=("close", "last")).reset_index()
    dirmap = {r.date: int(r.dir) for r in per.itertuples()}
    closemap = {r.date: float(r.close) for r in per.itertuples()}
    tail = [round(float(x), 4) for x in line[-60:] if not np.isnan(x)]
    return dirmap, tail, closemap


def fallback_dir(key):
    import importlib.util as u
    spec = u.spec_from_file_location("exit_machine", str(ROOT / "scripts" / "exit_machine.py"))
    em = u.module_from_spec(spec); sys.modules["exit_machine"] = em
    spec.loader.exec_module(em)
    T = em.TV[key]
    d = em.hvwma_daily_dir(T)
    dirmap = {T["dates"][i]: int(d[i]) for i in range(len(d))}
    closemap = {T["dates"][i]: T["price"][i] for i in range(len(d)) if T["price"][i] is not None}
    return dirmap, [], closemap


def main():
    out = {}
    for key, (sym, name) in SYMS.items():
        source = "sina"
        try:
            df = fetch_sina(sym)
            dirmap, line, closemap = hvwma_from_min(df)
        except Exception as e:
            print(f"{key}: 新浪失败({str(e)[:50]}), 回退日线近似", flush=True)
            source = "fallback"
            dirmap, line, closemap = fallback_dir(key)
        dates = sorted(dirmap)
        asof = dates[-1] if dates else None
        recent = [{"date": d, "dir": dirmap[d], "close": round(closemap.get(d, 0), 4)}
                  for d in dates[-40:]]
        out[key] = {"name": name, "dir": dirmap.get(asof, 0), "asof": asof,
                    "source": source, "recent": recent, "line": line}
        print(f"{key:7s} {name:8s} 当前{'GREEN多' if out[key]['dir'] > 0 else 'RED空'} "
              f"asof={asof} src={source} ({len(dates)}天)", flush=True)
        time.sleep(2)

    dest = PROC / "hvwma_live.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"写出 {dest} ({dest.stat().st_size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
