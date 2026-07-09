"""历史波动率曲面重建（tushare 期权日线 + Black-Scholes 反解 IV）。

对每个交易日重建期权链，用 BS 从结算价反解每张合约的隐含波动率，
再抽取研报关注的曲面时序指标：

  atm_iv        平值隐含波动率(现价处插值), 波动率水平
  iv_ratio_25d  中金式 IV 情绪比 = IV(25Δ put) / IV(25Δ call), >1 表示认沽更贵(悲观)
  iv_ratio_15m  中金原口径 = IV(-15% 虚值 put) / IV(+15% 虚值 call)
  skew_25d      IV(25Δ put) − IV(25Δ call), 下跌偏斜(正=防御)
  rr_25d        风险反转 IV(25Δ call) − IV(25Δ put)(标准口径, 与 skew 反号)
  pcr_vol/oi/amount  近月认沽/认购 的 成交量/持仓量/成交额 比

采用 30 日"常数期限": 对包住 30DTE 的两个到期月分别算指标, 再按 DTE 线性插值。
利率取常数 r。价格用结算价(settle), 缺失回退收盘价。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import tushare_loader as ts
from src.pricing.black_scholes import bs_delta, implied_vol
from src.utils.logger import get_logger

log = get_logger("research.surface")

R = 0.02              # 无风险利率(近似)
TARGET_DTE = 30       # 常数期限目标天数
MIN_DTE = 5           # 跳过到期周噪声


def _forward(g: pd.DataFrame, S: float, T: float) -> float:
    """由看跌看涨平价反解隐含远期 F = K + e^{rT}(C−P), 取近 ATM 若干行权价的中位数。

    ETF 期权受分红/持有成本影响, 远期常低于现货; 用 F 而非 S 作平值基准才不会
    把分红效应误读成偏斜。
    """
    piv = g.pivot_table(index="strike", columns="call_put", values="price")
    if "C" not in piv or "P" not in piv:
        return S
    piv = piv.dropna(subset=["C", "P"])
    if piv.empty:
        return S
    fwd = piv.index.to_numpy() + np.exp(R * T) * (piv["C"].to_numpy() - piv["P"].to_numpy())
    dist = np.abs(piv.index.to_numpy() - S)
    near = np.argsort(dist)[:4]
    return float(np.median(fwd[near]))


def _smile_metrics(g: pd.DataFrame, S: float, T: float) -> dict:
    """对单一到期月的合约集合(g)计算曲面指标。g 含 strike, call_put, price。

    用平价隐含远期 F 作平值基准; 只用**虚值**合约构建微笑(K<F 用 put, K>F 用 call),
    剔除实值合约(价≈内在价值, IV 反解不稳)。BS 里用 S_eff=F·e^{-rT} 等效替代现货。
    """
    F = _forward(g, S, T)
    S_eff = F * np.exp(-R * T)      # 使前向价=F 的等效现货, 代入现货式 BS
    rows = []
    for r in g.itertuples():
        is_call = r.call_put == "C"
        if is_call and r.strike < F:      # 只保留虚值(相对远期)
            continue
        if (not is_call) and r.strike > F:
            continue
        iv = implied_vol(r.price, S_eff, r.strike, T, R, is_call)
        if not np.isfinite(iv) or iv <= 0.02 or iv >= 3:
            continue
        delta = bs_delta(S_eff, r.strike, T, R, iv, is_call)
        rows.append((r.call_put, r.strike, iv, delta))
    if len(rows) < 4:
        return {}
    d = pd.DataFrame(rows, columns=["cp", "strike", "iv", "delta"])
    calls = d[d.cp == "C"].sort_values("strike")   # 虚值认购(K>=F)
    puts = d[d.cp == "P"].sort_values("strike")     # 虚值认沽(K<=F)
    if len(calls) < 2 or len(puts) < 2:
        return {}
    S = F   # 以下 ATM/moneyness 均以远期为基准

    # 合并成一条虚值微笑(按行权价), 用于 ATM 与 moneyness 取样
    otm = pd.concat([puts, calls]).sort_values("strike")

    def atm_from_otm():
        s = otm
        if not (s.strike.min() <= S <= s.strike.max()):
            return np.nan
        return float(np.interp(S, s.strike, s.iv))

    def iv_at_delta(side, tgt):  # 按 |delta| 插值(越界夹取最近端点, 保证时序连续)
        s = side.assign(ad=side.delta.abs()).sort_values("ad")
        if len(s) < 2:
            return np.nan
        return float(np.interp(tgt, s.ad, s.iv))

    def iv_at_money(side, m):    # 目标 moneyness 处按行权价插值(越界则夹取最近)
        s = side.sort_values("strike")
        K = S * (1 + m)
        return float(np.interp(K, s.strike, s.iv))  # np.interp 自动夹取端点

    def smile_slope():
        # OTM 微笑上 IV 对 moneyness(K/F−1) 的线性斜率, 仅用 ±12% 主区间。
        # 负=下跌偏斜(低行权价 IV 更高)。单位: 每 +1% moneyness 的 IV 变化(百分点)。
        mny = otm["strike"].to_numpy() / S - 1.0
        mask = np.abs(mny) <= 0.12
        if mask.sum() < 3:
            return np.nan
        return float(np.polyfit(mny[mask] * 100, otm["iv"].to_numpy()[mask] * 100, 1)[0])

    atm_iv = atm_from_otm()
    c25, p25 = iv_at_delta(calls, 0.25), iv_at_delta(puts, 0.25)
    c15, p15 = iv_at_money(calls, 0.15), iv_at_money(puts, -0.15)
    ok25 = np.isfinite(c25) and np.isfinite(p25)
    ok15 = np.isfinite(c15) and np.isfinite(p15) and c15 > 0
    return {
        "atm_iv": atm_iv,
        "iv_call_25d": c25, "iv_put_25d": p25,
        "iv_ratio_25d": (p25 / c25) if (ok25 and c25) else np.nan,
        "skew_25d": (p25 - c25) if ok25 else np.nan,        # 下跌偏斜(正=防御)
        "rr_25d": (c25 - p25) if ok25 else np.nan,          # 风险反转(标准口径, 负=下跌偏斜)
        "bf_25d": ((c25 + p25) / 2 - atm_iv) if (ok25 and np.isfinite(atm_iv)) else np.nan,  # 蝶式/凸度
        "iv_ratio_15m": (p15 / c15) if ok15 else np.nan,    # 固定虚值 情绪比
        "rr_15m": (c15 - p15) if ok15 else np.nan,          # 固定虚值 风险反转
        "smile_slope": smile_slope(),                        # 微笑斜率
    }


def _pcr(g: pd.DataFrame) -> dict:
    """近月 PCR: put/call 的 成交量/持仓量/成交额 比。"""
    c = g[g.call_put == "C"]; p = g[g.call_put == "P"]
    def ratio(col):
        cs, ps = c[col].sum(), p[col].sum()
        return (ps / cs) if cs > 0 else np.nan
    return {"pcr_vol": ratio("vol"), "pcr_oi": ratio("oi"), "pcr_amount": ratio("amount")}


def reconstruct(key: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """重建 [start,end] 的每日曲面指标时序。"""
    name = f"surface_{key}_{start}_{end}"
    if use_cache and __import__("src.data.cache", fromlist=["exists"]).exists(name, "processed"):
        from src.data import cache
        return cache.load(name, "processed")

    ob = ts.opt_basic(key)[["ts_code", "call_put", "exercise_price", "maturity_date"]].rename(
        columns={"exercise_price": "strike"})
    od = ts.opt_daily_range(key, start, end)
    fd = ts.fund_daily(key, start, end).set_index("trade_date")["close"]

    od = od.merge(ob, on="ts_code", how="left")
    od["price"] = od["settle"].where(od["settle"] > 0, od["close"])
    od = od.dropna(subset=["strike", "maturity_date", "price"])
    od["mat"] = pd.to_datetime(od["maturity_date"], format="%Y%m%d")

    out = []
    for dt, day in od.groupby("trade_date"):
        S = fd.get(dt, np.nan)
        if not np.isfinite(S):
            continue
        cur = pd.to_datetime(dt, format="%Y%m%d")
        day = day.assign(dte=(day["mat"] - cur).dt.days)
        expiries = sorted(day.loc[day.dte >= MIN_DTE, "dte"].unique())
        if not expiries:
            continue

        # 近月 PCR(最近到期月)
        near = day[day.dte == expiries[0]]
        rec = {"date": dt, "spot": round(float(S), 4), "near_dte": int(expiries[0])}
        rec.update(_pcr(near))

        # 常数 30D: 找包住 30 的两个到期月, 分别算指标再插值
        lo = max([e for e in expiries if e <= TARGET_DTE], default=expiries[0])
        hi = min([e for e in expiries if e >= TARGET_DTE], default=expiries[-1])
        m_lo = _smile_metrics(day[day.dte == lo], S, lo / 365)
        m_hi = _smile_metrics(day[day.dte == hi], S, hi / 365) if hi != lo else m_lo
        if m_lo and m_hi:
            w = 0.0 if hi == lo else (TARGET_DTE - lo) / (hi - lo)
            w = min(max(w, 0.0), 1.0)
            for k in m_lo:
                a, b = m_lo.get(k, np.nan), m_hi.get(k, np.nan)
                rec[k] = round(float(a + w * (b - a)), 4) if np.isfinite(a) and np.isfinite(b) else (
                    round(float(a), 4) if np.isfinite(a) else None)
        elif m_lo or m_hi:
            for k, v in (m_lo or m_hi).items():
                rec[k] = round(float(v), 4) if np.isfinite(v) else None
        out.append(rec)

    df = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    from src.data import cache
    cache.save(df, name, "processed")
    log.info("surface %s: %d 日 (%s~%s)", key, len(df), start, end)
    return df
