"""期权链快照加载层（新浪·上交所）。

用于波动率曲面研究：取某标的某到期月的全行权价合约，含每张合约的
隐含波动率(IV)、Delta、行权价、最新价。覆盖上交所 ETF 期权(510300 沪深300 / 588000 科创50)。

数据性质：**实时快照**。历史曲面需用 Black-Scholes 从合约日线价格反解 IV(见 src/pricing)。
新浪逐合约取希腊字母，一次快照需数十次 HTTP，故默认落盘缓存。
"""
from __future__ import annotations

import time
import warnings

import akshare as ak
import pandas as pd

from src.data import cache
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger("data.chain")


def _retry(fn, *args, tries: int = 4, pause: float = 1.5, **kwargs):
    """新浪接口易瞬断, 简单重试。"""
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(pause)
    raise last


def list_expiries(etf_code: str) -> list[str]:
    """返回可交易到期月, 如 ['202607','202608',...]。"""
    return list(_retry(ak.option_sse_list_sina, symbol=etf_code, exchange="null"))


def underlying_spot(etf_code: str, exchange: str = "sse") -> float:
    """标的 ETF 现价。"""
    sym = ("sh" if exchange == "sse" else "sz") + etf_code
    df = _retry(ak.option_sse_underlying_spot_price_sina, symbol=sym)
    # 返回字段-值两列, 取"最近成交价"
    row = df[df["字段"].astype(str).str.contains("最近|最新|现价", na=False)]
    return float(row["值"].iloc[0]) if not row.empty else float("nan")


def _contract_greeks(code: str) -> dict | None:
    """取单张合约的 IV/Delta/行权价/最新价。"""
    try:
        g = _retry(ak.option_sse_greeks_sina, symbol=str(code).strip(), tries=3, pause=1.0)
    except Exception as e:  # noqa: BLE001
        log.warning("greeks 失败 %s: %s", code, str(e)[:60])
        return None
    kv = dict(zip(g["字段"], g["值"]))
    def num(k):
        try:
            return float(kv.get(k))
        except (TypeError, ValueError):
            return float("nan")
    return {
        "code": kv.get("交易代码", code),
        "strike": num("行权价"),
        "iv": num("隐含波动率"),
        "delta": num("Delta"),
        "gamma": num("Gamma"),
        "vega": num("Vega"),
        "theta": num("Theta"),
        "last": num("最新价"),
        "volume": num("成交量"),
    }


def fetch_chain(
    etf_code: str,
    expiry: str | None = None,
    exchange: str = "sse",
    use_cache: bool = True,
    pause: float = 0.05,
) -> pd.DataFrame:
    """取某到期月全行权价的期权链快照。

    返回列: expiry, side(call/put), strike, iv, delta, gamma, vega, theta, last, volume, spot。
    expiry 缺省取最近月。
    """
    expiries = list_expiries(etf_code)
    expiry = expiry or expiries[0]
    name = f"chain_{etf_code}_{expiry}"
    if use_cache and cache.exists(name):
        return cache.load(name)

    spot = underlying_spot(etf_code, exchange)
    rows = []
    for side, label in [("看涨期权", "call"), ("看跌期权", "put")]:
        codes = _retry(ak.option_sse_codes_sina, symbol=side, trade_date=expiry, underlying=etf_code)
        for code in codes["期权代码"]:
            g = _contract_greeks(code)
            if g is None:
                continue
            g["side"] = label
            g["expiry"] = expiry
            rows.append(g)
            time.sleep(pause)

    df = pd.DataFrame(rows)
    df["spot"] = spot
    df = df.sort_values(["side", "strike"]).reset_index(drop=True)
    log.info("chain %s %s: %d 合约, spot=%.4f", etf_code, expiry, len(df), spot)
    cache.save(df, name)
    return df


if __name__ == "__main__":
    c = fetch_chain("510300", use_cache=False)
    print(c.head(20).to_string())
