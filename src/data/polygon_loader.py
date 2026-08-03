"""Polygon(massive 第三方中转)行情客户端。

通过代理访问 Polygon 全量股票/期权 API:
  - 把 https://api.polygon.io 换成代理地址
  - Header 带 X-Proxy-Key(代理自动注入 apiKey)

代理限额: 1000 请求/分钟, 2 GiB/日, 超限返回 429。本客户端内置最小请求间隔
节流 + 429 指数退避 + next_url 自动翻页, 供历史曲面重建批量调用。

密钥文件 config/polygon_key.txt(gitignore):
    第 1 行 = 代理 base_url
    第 2 行 = X-Proxy-Key
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.utils.logger import get_logger

log = get_logger("data.polygon")

ROOT = Path(__file__).resolve().parents[2]
KEYFILE = ROOT / "config" / "polygon_key.txt"

# 节流: 目标 ~800 req/min(留 20% 余量), 即最小间隔 ~0.075s。多线程共享。
_MIN_INTERVAL = 0.075
_last_ts = 0.0
_lock = threading.Lock()


def _creds() -> tuple[str, str]:
    lines = KEYFILE.read_text().strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"{KEYFILE} 需两行: base_url / key")
    return lines[0].strip().rstrip("/"), lines[1].strip()


BASE_URL, _KEY = _creds()
_HEADERS = {"X-Proxy-Key": _KEY}
_session = requests.Session()


def _throttle() -> None:
    global _last_ts
    with _lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_ts)
        if wait > 0:
            time.sleep(wait)
        _last_ts = time.monotonic()


def get(path: str, params: dict | None = None, *, tries: int = 5) -> dict:
    """带节流与 429/5xx 退避的 GET, 返回解析后的 JSON。path 以 / 开头。"""
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    backoff = 2.0
    for attempt in range(tries):
        _throttle()
        try:
            r = _session.get(url, headers=_HEADERS, params=params, timeout=(10, 60))
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            log.warning("网络异常 %s, 重试 %d/%d", e, attempt + 1, tries)
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 429:            # 触发限频, 退避
            wait = float(r.headers.get("Retry-After", backoff))
            log.warning("429 限频, 等待 %.1fs", wait)
            time.sleep(wait)
            backoff = min(backoff * 2, 30)
            continue
        if r.status_code >= 500:
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"请求多次失败: {url}")


def _paged(path: str, params: dict, *, max_pages: int = 40) -> list[dict]:
    """跟随 next_url 翻页, 汇总 results。"""
    out: list[dict] = []
    js = get(path, params)
    out.extend(js.get("results") or [])
    pages = 1
    nxt = js.get("next_url")
    while nxt and pages < max_pages:
        # next_url 指向真实 api.polygon.io, 需改写回代理域名(cursor 保留)
        nxt = nxt.replace("https://api.polygon.io", BASE_URL).replace(
            "http://api.polygon.io", BASE_URL)
        js = get(nxt)
        out.extend(js.get("results") or [])
        nxt = js.get("next_url")
        pages += 1
    return out


# ----------------------------- 聚合(K线) -----------------------------

# 代理对盘中(minute/hour)聚合单次响应有行数上限(~实测 <900 行会被截断,
# 忽略 limit 参数)。故分块拉取: 每块天数控制在 bar 数安全区内。
_CHUNK_DAYS = {"minute": 3, "hour": 20}


def _aggs_raw(ticker: str, mult: int, span: str, start: str, end: str,
              adjusted: bool, limit: int) -> pd.DataFrame:
    path = f"/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start}/{end}"
    js = get(path, {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": limit})
    res = js.get("results") or []
    if not res:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw", "n"])
    df = pd.DataFrame(res)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    keep = [c for c in ["t", "o", "h", "l", "c", "v", "vw", "n"] if c in df.columns]
    return df[keep]


def aggs(ticker: str, mult: int, span: str, start: str, end: str,
         *, adjusted: bool = True, limit: int = 50000) -> pd.DataFrame:
    """日/分钟/小时 K 线。span: day|hour|minute。start/end: YYYY-MM-DD。

    盘中长区间自动分块(绕过代理响应行数上限)后拼接去重。
    返回列: t(UTC datetime), o,h,l,c,v,vw,n。空则返回空 DataFrame。
    """
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    chunk = _CHUNK_DAYS.get(span)
    df = _aggs_raw(ticker, mult, span, start, end, adjusted, limit)

    # 截断检测: 盘中且返回行数逼近上限、末尾却远早于 end → 分块重取。
    # 稀疏合约(成交少)行数低, 不会误触发; 只有稠密序列(标的)才被分块。
    if chunk is not None and len(df) >= 800:
        last = df["t"].max().date()
        if last < e - timedelta(days=2):
            parts, cur = [], s
            while cur <= e:
                ce = min(cur + timedelta(days=chunk), e)
                parts.append(_aggs_raw(ticker, mult, span, cur.isoformat(),
                                       ce.isoformat(), adjusted, limit))
                cur = ce + timedelta(days=1)
            df = pd.concat(parts, ignore_index=True) if parts else df

    if df.empty:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw", "n"])
    return df.drop_duplicates("t").sort_values("t").reset_index(drop=True)


# ----------------------------- 期权参考/快照 -----------------------------

def list_contracts(underlying: str, *, as_of: str | None = None,
                   expired: bool | None = None,
                   exp_gte: str | None = None, exp_lte: str | None = None,
                   contract_type: str | None = None) -> pd.DataFrame:
    """期权合约参考(可翻页取全量)。

    列: ticker, contract_type(call/put), strike_price, expiration_date, exercise_style。
    as_of: 以该日为准的合约清单(含当时在市的已到期合约需 expired=True)。
    """
    params: dict = {"underlying_ticker": underlying, "limit": 1000}
    if as_of:
        params["as_of"] = as_of
    if expired is not None:
        params["expired"] = str(expired).lower()
    if exp_gte:
        params["expiration_date.gte"] = exp_gte
    if exp_lte:
        params["expiration_date.lte"] = exp_lte
    if contract_type:
        params["contract_type"] = contract_type
    rows = _paged("/v3/reference/options/contracts", params)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def snapshot_chain(underlying: str) -> pd.DataFrame:
    """当前期权链快照(Polygon 自带 IV/greeks/OI)。

    返回 surface_metrics 期望的整洁列:
        spot, expiry, side(call/put), strike, iv, delta, gamma, vega, theta,
        oi, volume, close。
    """
    rows = _paged(f"/v3/snapshot/options/{underlying}", {"limit": 250})
    out = []
    spot = None
    for r in rows:
        det = r.get("details", {})
        grk = r.get("greeks", {}) or {}
        day = r.get("day", {}) or {}
        ua = r.get("underlying_asset", {}) or {}
        if spot is None and ua.get("price"):
            spot = ua.get("price")
        out.append({
            "expiry": det.get("expiration_date"),
            "side": det.get("contract_type"),
            "strike": det.get("strike_price"),
            "iv": r.get("implied_volatility"),
            "delta": grk.get("delta"),
            "gamma": grk.get("gamma"),
            "vega": grk.get("vega"),
            "theta": grk.get("theta"),
            "oi": r.get("open_interest"),
            "volume": day.get("volume"),
            "close": day.get("close"),
            "ticker": det.get("ticker"),
        })
    df = pd.DataFrame(out)
    if spot is None:                        # 快照未带现价 → 用标的最近日线收盘
        spot = last_spot(underlying)
    df["spot"] = spot
    return df


def last_spot(ticker: str) -> float:
    """标的最近一个交易日收盘价。"""
    js = get(f"/v2/aggs/ticker/{ticker}/prev", {"adjusted": "true"})
    res = js.get("results") or []
    return float(res[0]["c"]) if res else float("nan")


if __name__ == "__main__":
    print("base:", BASE_URL)
    print(aggs("SPY", 1, "day", "2026-07-01", "2026-07-15").tail())
    ch = snapshot_chain("SPY")
    print("snapshot rows:", len(ch), "spot:", ch["spot"].iloc[0] if len(ch) else None)
    print(ch.head())
