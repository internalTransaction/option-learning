"""AkShare 数据加载层。

对外暴露三类数据：
  1. fetch_qvix(key)       -> 隐含波动率指数(QVIX)历史 OHLC, 波动率因子的主干
  2. fetch_underlying(code)-> 标的 ETF 日线, 用于历史波动率(HV)与信号回测
  3. fetch_option_chain()  -> 实时期权链快照(全市场), 用于 PCR / 偏度因子

所有函数带本地缓存与简单重试。QVIX / ETF 为历史序列(可增量), 期权链为当日快照。
"""
from __future__ import annotations

import time
import warnings

import akshare as ak
import pandas as pd

from src.data import cache
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")  # akshare 内部大量 SettingWithCopyWarning
log = get_logger("data.akshare")

# QVIX 接口关键字 -> akshare 函数名后缀。config 里的 qvix 字段对应这里的 key。
_QVIX_MAP = {
    "50etf": "index_option_50etf_qvix",
    "300etf": "index_option_300etf_qvix",
    "500etf": "index_option_500etf_qvix",
    "100etf": "index_option_100etf_qvix",
    "cyb": "index_option_cyb_qvix",     # 创业板
    "kc50": "index_option_kcb_qvix",    # 科创板(akshare 命名为 kcb)
    "kcb": "index_option_kcb_qvix",
    "1000index": "index_option_1000index_qvix",
}


def _retry(fn, *args, tries: int = 3, pause: float = 2.0, **kwargs):
    """对网络接口做简单重试。"""
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("call failed (%d/%d): %s", i + 1, tries, e)
            time.sleep(pause)
    raise last


# ---------------------------------------------------------------------------
# 1. 隐含波动率指数 (QVIX)
# ---------------------------------------------------------------------------
def fetch_qvix(key: str, use_cache: bool = True) -> pd.DataFrame:
    """拉取某标的的隐含波动率指数历史 (date, open, high, low, close)。

    close 即当日 IV 指数水平(百分数, 如 20.29 表示 20.29%)。
    """
    name = f"qvix_{key}"
    if use_cache and cache.exists(name):
        return cache.load(name)

    fn_name = _QVIX_MAP.get(key)
    if fn_name is None:
        raise KeyError(f"未知 QVIX key: {key}. 可选: {list(_QVIX_MAP)}")
    fn = getattr(ak, fn_name)
    df = _retry(fn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    cache.save(df, name)
    return df


# ---------------------------------------------------------------------------
# 2. 标的 ETF 日线
# ---------------------------------------------------------------------------
_ETF_COL_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
    "最低": "low", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg",
}


def _sina_symbol(etf_code: str, exchange: str) -> str:
    """ETF 代码 + 交易所 -> 新浪代码 (如 sh510300 / sz159915)。"""
    prefix = "sh" if exchange == "sse" else "sz"
    return f"{prefix}{etf_code}"


def _fetch_underlying_em(etf_code, start_date, end_date, adjust) -> pd.DataFrame:
    raw = ak.fund_etf_hist_em(
        symbol=etf_code, period="daily",
        start_date=start_date, end_date=end_date, adjust=adjust,
    )
    return raw.rename(columns=_ETF_COL_MAP)[list(_ETF_COL_MAP.values())].copy()


def _fetch_underlying_sina(etf_code, exchange) -> pd.DataFrame:
    raw = ak.fund_etf_hist_sina(symbol=_sina_symbol(etf_code, exchange))
    keep = ["date", "open", "close", "high", "low", "volume", "amount"]
    df = raw[keep].copy()
    df["pct_chg"] = df["close"].astype(float).pct_change() * 100
    return df


def fetch_underlying(
    etf_code: str,
    exchange: str = "sse",
    start_date: str = "20150101",
    end_date: str = "20991231",
    adjust: str = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """拉取标的 ETF 日线 (date, open, high, low, close, volume, ...)。

    优先东方财富(支持前复权), 失败则回退到新浪(host 不同, 更易连通)。
    """
    name = f"etf_{etf_code}"
    if use_cache and cache.exists(name):
        return cache.load(name)

    try:
        df = _retry(_fetch_underlying_em, etf_code, start_date, end_date, adjust, tries=2)
        src = "eastmoney"
    except Exception as e:  # noqa: BLE001
        log.warning("东财源失败, 回退新浪: %s", str(e)[:80])
        df = _retry(_fetch_underlying_sina, etf_code, exchange, tries=3)
        src = "sina"

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("etf %s 数据源=%s", etf_code, src)
    cache.save(df, name)
    return df


# ---------------------------------------------------------------------------
# 3. 实时期权链快照 (用于 PCR / 偏度)
# ---------------------------------------------------------------------------
def fetch_option_chain(use_cache: bool = False) -> pd.DataFrame:
    """全市场期权实时快照。含认购/认沽、持仓量、成交量、隐含波动率、希腊字母。

    注意: 这是**当日实时快照**, 非历史。做时序 PCR/偏度需按交易日定时落盘累积。
    """
    name = "option_chain_snapshot"
    if use_cache and cache.exists(name):
        return cache.load(name)
    df = _retry(ak.option_current_em)
    cache.save(df, name)
    return df


if __name__ == "__main__":
    # 简单自检
    q = fetch_qvix("300etf")
    log.info("qvix 300etf: %d rows, %s -> %s", len(q), q["date"].min().date(), q["date"].max().date())
    e = fetch_underlying("510300")
    log.info("etf 510300: %d rows", len(e))
