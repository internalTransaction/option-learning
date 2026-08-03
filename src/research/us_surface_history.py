"""美股期权历史波动率曲面重建(Polygon 逐合约 K 线 + Black-Scholes 反解 IV)。

与 A 股口径(src.research.surface_history)完全一致, 只是数据源换成 Polygon 代理:
对每个"时点"(日线 or 盘中 bar)重建期权链, 用 BS 从合约收盘价反解 IV, 抽取

  atm_iv        平值隐含波动率(现价插值)
  iv_ratio_25d  IV 情绪比 = IV(25Δput)/IV(25Δcall)
  rr_25d        风险反转 = IV(25Δcall) − IV(25Δput)
  skew_25d      下跌偏斜 = −rr_25d
  smile_slope   微笑斜率
  pcr_vol       近月认沽/认购成交量比

30 日"常数期限": 用包住 30DTE 的两个到期月分别算, 再按 DTE 线性插值。
复用 surface_history 的 _smile_metrics / _pcr(纯数值, 与数据源无关)。

美股 ETF/指数期权多为美式, 但仅用**虚值**合约反解 IV, 美式早行权溢价可忽略;
且用 put-call parity 隐含远期作平值基准, 自动吸收分红/持有成本。

granularity:
  "day"   -> 日线, x 轴 = YYYYMMDD
  "hour"  -> 盘中小时线(仅 RTH 13:30-20:00 UTC), x 轴 = YYYYMMDDHHMM
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data import polygon_loader as pg
from src.research.surface_history import _forward, _pcr, _smile_metrics
from src.utils.logger import get_logger

log = get_logger("research.us_surface")

R = 0.04              # 无风险利率(美债近似, 高于 A 股口径)
TARGET_DTE = 30
MIN_DTE = 5
BAND = 0.25           # 行权价带 ±25% (BKM 需深度虚值尾部; 再深的对 SPY≈0价、贡献可忽略)


def _bkm(g: pd.DataFrame, S: float, T: float, r: float = R) -> dict:
    """Bakshi-Kapadia-Madan(2003) 风险中性矩 → model-free 偏度。

    用整条虚值期权价格(K≥F 用 call, K<F 用 put)离散积分出方差/立方/四次合约
    (V/W/X), 再组合成风险中性偏度; skew_cboe = 100 − 10·偏度(CBOE SKEW 口径)。
    g 需含 strike, call_put(C/P), price。
    """
    d = g.dropna(subset=["strike", "call_put", "price"])
    d = d[d["price"] > 0]
    if len(d) < 8:
        return {}
    F = _forward(d, S, T)
    calls = d[(d["call_put"] == "C") & (d["strike"] >= F)].sort_values("strike")
    puts = d[(d["call_put"] == "P") & (d["strike"] < F)].sort_values("strike")
    if len(calls) < 3 or len(puts) < 3:
        return {}

    def spacing(k: np.ndarray) -> np.ndarray:
        dk = np.empty_like(k, dtype=float)
        if len(k) == 1:
            return np.array([k[0] * 0.05])
        dk[1:-1] = (k[2:] - k[:-2]) / 2
        dk[0] = k[1] - k[0]
        dk[-1] = k[-1] - k[-2]
        return dk

    V = W = X = 0.0
    for side, is_call in ((calls, True), (puts, False)):
        K = side["strike"].to_numpy(dtype=float)
        O = side["price"].to_numpy(dtype=float)
        dk = spacing(K)
        x = np.log(K / F)                      # call 侧 x≥0
        if is_call:
            wV = 2 * (1 - x) / K ** 2
            wW = (6 * x - 3 * x ** 2) / K ** 2
            wX = (12 * x ** 2 - 4 * x ** 3) / K ** 2
        else:
            y = -x                             # put 侧 y=ln(F/K)>0
            wV = 2 * (1 + y) / K ** 2
            wW = -(6 * y + 3 * y ** 2) / K ** 2
            wX = (12 * y ** 2 + 4 * y ** 3) / K ** 2
        V += float(np.sum(wV * O * dk))
        W += float(np.sum(wW * O * dk))
        X += float(np.sum(wX * O * dk))

    er = np.exp(r * T)
    mu = er - 1 - er * V / 2 - er * W / 6 - er * X / 24
    var = er * V - mu ** 2
    if not np.isfinite(var) or var <= 0:
        return {}
    skew = (er * W - 3 * mu * er * V + 2 * mu ** 3) / var ** 1.5
    if not np.isfinite(skew):
        return {}
    return {"mf_skew": round(float(skew), 4), "skew_cboe": round(100 - 10 * float(skew), 2)}

# RTH(常规交易时段)按**纽约本地小时**过滤(跨夏令时稳健):
# 09:00 bar 含 09:30 开盘、...、15:00 bar 为收盘前最后一小时。共 7 根/日。
RTH_ET_HOURS = [9, 10, 11, 12, 13, 14, 15]


def _third_fridays(start: str, end: str) -> list[str]:
    """[start,end] 内每月第三个周五(标准月度到期), YYYY-MM-DD。"""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    y, m = s.year, s.month
    while datetime(y, m, 1) <= e:
        first = datetime(y, m, 1)
        # 第一个周五
        offset = (4 - first.weekday()) % 7
        third = first + timedelta(days=offset + 14)
        if s <= third <= e:
            out.append(third.strftime("%Y-%m-%d"))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _spot_daily(underlying: str, start: str, end: str) -> pd.Series:
    """标的日线收盘, index=YYYY-MM-DD(str)。"""
    df = pg.aggs(underlying, 1, "day", start, end)
    if df.empty:
        return pd.Series(dtype=float)
    df["d"] = df["t"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return df.groupby("d")["c"].last()


def _contract_bars(ticker: str, span: str, mult: int, start: str, end: str) -> pd.DataFrame:
    """单合约 K 线 -> 列 asof(str key), price(close), vol。span=day|hour。"""
    df = pg.aggs(ticker, mult, span, start, end)
    if df.empty:
        return df
    ny = df["t"].dt.tz_convert("America/New_York")
    if span == "day":
        df["asof"] = ny.dt.strftime("%Y%m%d")
    else:
        df = df[ny.dt.hour.isin(RTH_ET_HOURS)].copy()
        ny = df["t"].dt.tz_convert("America/New_York")
        df["asof"] = ny.dt.strftime("%Y%m%d%H%M")
    return df.rename(columns={"c": "price", "v": "vol"})[["asof", "price", "vol"]]


def _gather_chain(underlying: str, start: str, end: str, span: str, mult: int,
                  spot_daily: pd.Series) -> pd.DataFrame:
    """拉取窗口内所有相关合约的逐时点价格, 汇总为长表。

    列: asof, expiry(YYYY-MM-DD), dte, strike, call_put(C/P), price, vol。
    """
    # 覆盖 [start-8d, end+58d] 的月度到期, 保证 30DTE 常有两个月度包夹
    e_lo = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=8)).strftime("%Y-%m-%d")
    e_hi = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=58)).strftime("%Y-%m-%d")
    expiries = _third_fridays(e_lo, e_hi)
    log.info("%s 月度到期 %d 个: %s", underlying, len(expiries), expiries)

    # 先汇总所有 (contract, 取数窗口) 任务, 再多线程并发拉取(共享节流锁保证不超限)
    tasks = []
    for E in expiries:
        Ed = datetime.strptime(E, "%Y-%m-%d")
        # 该到期月被用作 30D 包夹的时间窗(它 DTE∈[8,58] 的日子)
        w_lo = max(Ed - timedelta(days=58), datetime.strptime(start, "%Y-%m-%d"))
        w_hi = min(Ed - timedelta(days=1), datetime.strptime(end, "%Y-%m-%d"))
        if w_lo > w_hi:
            continue
        # 参考现价: 取 ~E-30 附近的日线收盘定行权价带
        ref_day = (Ed - timedelta(days=30)).strftime("%Y-%m-%d")
        sref = _nearest_spot(spot_daily, ref_day)
        if not np.isfinite(sref):
            continue
        lo_k, hi_k = sref * (1 - BAND), sref * (1 + BAND)
        # 已过期到期月需 expired=true, 未到期(近月/实时)需 expired=false
        exp_flag = Ed.date() < datetime.now().date()
        cons = pg.list_contracts(underlying, expired=exp_flag, exp_gte=E, exp_lte=E)
        if cons.empty:      # 偶发空响应会整月丢失, 重试一次
            cons = pg.list_contracts(underlying, expired=exp_flag, exp_gte=E, exp_lte=E)
        if cons.empty:
            log.warning("  %s: 合约清单为空, 跳过", E)
            continue
        cons = cons[(cons["strike_price"] >= lo_k) & (cons["strike_price"] <= hi_k)]
        f_lo, f_hi = w_lo.strftime("%Y-%m-%d"), w_hi.strftime("%Y-%m-%d")
        for c in cons.itertuples():
            tasks.append((c.ticker, E, float(c.strike_price),
                          "C" if c.contract_type == "call" else "P", f_lo, f_hi))
        log.info("  %s: %d 合约", E, len(cons))

    if not tasks:
        return pd.DataFrame()

    def _fetch(t):
        ticker, E, strike, cp, f_lo, f_hi = t
        try:                       # 单合约取数失败(403越界/网络等)不拖垮整体
            bars = _contract_bars(ticker, span, mult, f_lo, f_hi)
        except Exception:
            return None
        if bars.empty:
            return None
        bars["expiry"] = E
        bars["strike"] = strike
        bars["call_put"] = cp
        return bars

    frames = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(_fetch, tasks):
            if res is not None:
                frames.append(res)
    log.info("%s 合约取数完成: %d/%d 有数据", underlying, len(frames), len(tasks))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _nearest_spot(spot_daily: pd.Series, day: str) -> float:
    """取 <=day 的最近一个交易日收盘。"""
    idx = spot_daily.index[spot_daily.index <= day]
    if len(idx) == 0:
        return spot_daily.iloc[0] if len(spot_daily) else float("nan")
    return float(spot_daily.loc[idx[-1]])


def _asof_spot(underlying: str, start: str, end: str, span: str, mult: int) -> pd.Series:
    """时点 -> 标的价 的映射(与合约同粒度)。"""
    df = pg.aggs(underlying, mult, span, start, end)
    if df.empty:
        return pd.Series(dtype=float)
    ny = df["t"].dt.tz_convert("America/New_York")
    if span == "day":
        key = ny.dt.strftime("%Y%m%d")
    else:
        df = df[ny.dt.hour.isin(RTH_ET_HOURS)].copy()
        ny = df["t"].dt.tz_convert("America/New_York")
        key = ny.dt.strftime("%Y%m%d%H%M")
    return pd.Series(df["c"].values, index=key)


def reconstruct(underlying: str, start: str, end: str,
                granularity: str = "day") -> pd.DataFrame:
    """重建 [start,end] 的曲面时序。granularity: day|hour。"""
    span, mult = ("day", 1) if granularity == "day" else ("hour", 1)
    spot_daily = _spot_daily(underlying, start, end)
    if spot_daily.empty:
        raise RuntimeError(f"{underlying} 无标的日线")

    chain = _gather_chain(underlying, start, end, span, mult, spot_daily)
    if chain.empty:
        raise RuntimeError(f"{underlying} 无期权数据")

    asof_spot = _asof_spot(underlying, start, end, span, mult)

    # 到期日期(用于 DTE)。盘中: 以当日日期计 DTE(粗略, 30D 插值对小数不敏感)
    chain["exp_dt"] = pd.to_datetime(chain["expiry"])
    chain["asof_day"] = pd.to_datetime(chain["asof"].str[:8], format="%Y%m%d")
    chain["dte"] = (chain["exp_dt"] - chain["asof_day"]).dt.days

    out = []
    for asof, day in chain.groupby("asof"):
        S = asof_spot.get(asof, np.nan)
        if not np.isfinite(S):
            continue
        day = day[day["dte"] >= MIN_DTE]
        expiries = sorted(day["dte"].unique())
        if not expiries:
            continue
        rec = {"date": asof, "spot": round(float(S), 4), "near_dte": int(expiries[0])}
        near = day[day["dte"] == expiries[0]]
        rec.update(_pcr_vol_only(near))

        lo = max([e for e in expiries if e <= TARGET_DTE], default=expiries[0])
        hi = min([e for e in expiries if e >= TARGET_DTE], default=expiries[-1])
        g_lo = _prep(day[day["dte"] == lo])
        m_lo = {**_smile_metrics(g_lo, S, lo / 365), **_bkm(g_lo, S, lo / 365)}
        if hi != lo:
            g_hi = _prep(day[day["dte"] == hi])
            m_hi = {**_smile_metrics(g_hi, S, hi / 365), **_bkm(g_hi, S, hi / 365)}
        else:
            m_hi = m_lo
        _merge_metrics(rec, m_lo, m_hi, lo, hi)

        # 期限结构(近月/远月 ATM IV 比) = VIX/VIX3M backwardation 的自建代理:
        #   >1 = backwardation(近端更贵·恐慌), <1 = contango(平静, 常态)。
        f_dte, r_dte = expiries[0], expiries[-1]
        m_f = m_lo if f_dte == lo else _smile_metrics(_prep(day[day["dte"] == f_dte]), S, f_dte / 365)
        m_r = m_hi if r_dte == hi else _smile_metrics(_prep(day[day["dte"] == r_dte]), S, r_dte / 365)
        av_f = m_f.get("atm_iv") if m_f else None
        av_r = m_r.get("atm_iv") if m_r else None
        if (av_f and av_r and np.isfinite(av_f) and np.isfinite(av_r) and av_r > 0
                and r_dte > f_dte):
            rec["atm_iv_front"] = round(float(av_f), 4)
            rec["atm_iv_far"] = round(float(av_r), 4)
            rec["ts_ratio"] = round(float(av_f / av_r), 4)
            rec["ts_dtes"] = f"{f_dte}/{r_dte}"
        out.append(rec)

    df = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    log.info("%s[%s]: %d 时点 (%s~%s)", underlying, granularity, len(df),
             df["date"].iloc[0] if len(df) else "-", df["date"].iloc[-1] if len(df) else "-")
    return df


def _prep(g: pd.DataFrame) -> pd.DataFrame:
    """_smile_metrics 需要列: strike, call_put, price。去重(同 strike/cp 取一个)。"""
    return g[["strike", "call_put", "price"]].dropna()


def _pcr_vol_only(g: pd.DataFrame) -> dict:
    """近月 PCR(仅成交量, 日线/盘中均有)。"""
    c = g[g.call_put == "C"]["vol"].sum()
    p = g[g.call_put == "P"]["vol"].sum()
    return {"pcr_vol": round(float(p / c), 4) if c > 0 else None}


def _merge_metrics(rec: dict, m_lo: dict, m_hi: dict, lo: int, hi: int) -> None:
    if m_lo and m_hi:
        w = 0.0 if hi == lo else (TARGET_DTE - lo) / (hi - lo)
        w = min(max(w, 0.0), 1.0)
        for k in m_lo:
            a, b = m_lo.get(k, np.nan), m_hi.get(k, np.nan)
            if np.isfinite(a) and np.isfinite(b):
                rec[k] = round(float(a + w * (b - a)), 4)
            elif np.isfinite(a):
                rec[k] = round(float(a), 4)
    elif m_lo or m_hi:
        for k, v in (m_lo or m_hi).items():
            rec[k] = round(float(v), 4) if np.isfinite(v) else None


if __name__ == "__main__":
    import sys
    u = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    s = sys.argv[2] if len(sys.argv) > 2 else "2026-05-15"
    e = sys.argv[3] if len(sys.argv) > 3 else "2026-07-15"
    g = sys.argv[4] if len(sys.argv) > 4 else "day"
    df = reconstruct(u, s, e, g)
    cols = [c for c in ["date", "spot", "atm_iv", "rr_25d", "smile_slope",
                        "mf_skew", "skew_cboe", "ts_ratio", "near_dte"] if c in df.columns]
    print(df[cols].tail(15).to_string())
