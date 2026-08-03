"""验证 GEX 的 IV 反解口径差异是否影响信号。
口径A(现状 build_gex): close 价 + 直接现货 S
口径B(与曲面统一):     settle(回退close) + 平价隐含远期 S_eff = F·e^{-rT}
比较两者 gex_z 的相关性/符号一致率/极值区判定一致率。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
R, MULT = 0.02, 10000


def bs(S, K, T, s, cp):
    if s <= 0 or T <= 0: return np.nan
    d1 = (np.log(S/K) + (R + .5*s*s)*T)/(s*np.sqrt(T)); d2 = d1 - s*np.sqrt(T)
    if cp == "C": return S*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)
    return K*np.exp(-R*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def iv_solve(p, S, K, T, cp):
    intr = max(0., (S-K) if cp == "C" else (K-S))
    if p <= intr + 1e-6 or T <= 0: return np.nan
    try: return brentq(lambda s: bs(S, K, T, s, cp) - p, 1e-3, 5., maxiter=60, xtol=1e-4)
    except Exception: return np.nan


def gam(S, K, T, s):
    if s <= 0 or T <= 0: return np.nan
    d1 = (np.log(S/K) + (R + .5*s*s)*T)/(s*np.sqrt(T))
    return norm.pdf(d1)/(S*s*np.sqrt(T))


def run(key, basic, daily, surf):
    b = pd.read_parquet(ROOT/"data/raw"/basic)[["ts_code", "call_put", "exercise_price", "maturity_date"]]
    d = pd.read_parquet(ROOT/"data/raw"/daily)[["ts_code", "trade_date", "close", "settle", "oi"]]
    sf = pd.read_parquet(ROOT/"data/processed"/surf)[["date", "spot"]]
    d["trade_date"] = d["trade_date"].astype(str)
    b["maturity_date"] = b["maturity_date"].astype(str)
    sf["date"] = sf["date"].astype(str)
    spot = dict(zip(sf["date"], sf["spot"]))
    df = d.merge(b, on="ts_code", how="left").dropna(subset=["exercise_price", "maturity_date"])
    df["S"] = df["trade_date"].map(spot)
    df = df.dropna(subset=["S"])
    df["T"] = (pd.to_datetime(df["maturity_date"]) - pd.to_datetime(df["trade_date"])).dt.days
    df = df[df["T"] >= 3]

    rows = []
    for dt, g in df.groupby("trade_date"):
        gm = g[g["maturity_date"] == g["maturity_date"].min()]
        S = float(gm["S"].iloc[0]); T = float(gm["T"].iloc[0])/365.
        pb = np.where(gm["settle"].to_numpy() > 0, gm["settle"].to_numpy(), gm["close"].to_numpy())
        piv = gm.assign(p=pb).pivot_table(index="exercise_price", columns="call_put", values="p")
        F = S
        if "C" in piv and "P" in piv:
            pv = piv.dropna(subset=["C", "P"])
            if not pv.empty:
                fwd = pv.index.to_numpy() + np.exp(R*T)*(pv["C"].to_numpy() - pv["P"].to_numpy())
                near4 = np.argsort(np.abs(pv.index.to_numpy() - S))[:4]
                F = float(np.median(fwd[near4]))
        S_eff = F*np.exp(-R*T)
        tA = tB = 0.; okA = okB = 0
        for r in gm.itertuples():
            K, cp, oi = float(r.exercise_price), r.call_put, r.oi
            if oi is None or oi <= 0: continue
            sgn = 1. if cp == "C" else -1.
            if r.close and r.close > 0:
                iv = iv_solve(float(r.close), S, K, T, cp)
                gg = gam(S, K, T, iv) if np.isfinite(iv) else np.nan
                if np.isfinite(gg): tA += gg*oi*MULT*S*S*.01*sgn; okA += 1
            p2 = r.settle if (r.settle and r.settle > 0) else r.close
            if p2 and p2 > 0:
                iv = iv_solve(float(p2), S_eff, K, T, cp)
                gg = gam(S_eff, K, T, iv) if np.isfinite(iv) else np.nan
                if np.isfinite(gg): tB += gg*oi*MULT*S_eff*S_eff*.01*sgn; okB += 1
        if okA >= 3 and okB >= 3: rows.append((dt, tA, tB))

    res = pd.DataFrame(rows, columns=["date", "gexA", "gexB"])
    for c in ["gexA", "gexB"]:
        res[c+"_z"] = (res[c] - res[c].rolling(252, min_periods=60).mean())/res[c].rolling(252, min_periods=60).std()
    v = res.dropna(subset=["gexA_z", "gexB_z"])
    print(f"\n=== {key} (n={len(v)}) ===")
    print(f"  gex_z 相关性              {np.corrcoef(v.gexA_z, v.gexB_z)[0,1]:.4f}")
    print(f"  符号一致率                {(np.sign(v.gexA_z)==np.sign(v.gexB_z)).mean()*100:.1f}%")
    print(f"  低GEX(z≤−1)判定一致率     {((v.gexA_z<=-1)==(v.gexB_z<=-1)).mean()*100:.1f}%")
    print(f"  GEX>0 占比   A(现状) {(res.gexA>0).mean()*100:.0f}%   B(统一) {(res.gexB>0).mean()*100:.0f}%")
    print(f"  |z_A − z_B| 中位数        {(v.gexA_z-v.gexB_z).abs().median():.3f}")


if __name__ == "__main__":
    run("沪深300ETF", "ts_optbasic_300etf.parquet",
        "ts_optdaily_300etf_ohlc_20200102_20260720.parquet", "surface_300etf_20200101_20260720.parquet")
    run("创业板", "ts_optbasic_cyb.parquet",
        "ts_optdaily_cyb_ohlc_20220919_20260720.parquet", "surface_cyb_20220901_20260720.parquet")
