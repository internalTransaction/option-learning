"""构建 Dealer Gamma Exposure (GEX) 时间序列 —— A股ETF期权高阶greeks研究第一步。

数据: ts_optbasic(strike/认沽认购/到期) + ts_optdaily(每日每合约 close/oi) + surface(spot)。
每交易日取最近月(剩余≥3天)全部合约, 用期权收盘价 BS 反解 IV, 算 gamma, 汇总:
  GEX = Σ gamma_i × OI_i × mult × S² × 0.01 × sign_i
  sign: 标准 dealer 口径假设 call=+1(dealer long gamma) / put=-1(short gamma)。
  (A股 dealer 方向存疑, 先按标准口径产出, 再看信号是否成立)
输出 data/processed/gex_<key>.json 并打印 GEX 与次日已实现波动的初步关系。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.catalog import latest_ranged_file, read_ranged_parquets

RAW, PROC = ROOT/"data"/"raw", ROOT/"data"/"processed"
R, MULT = 0.02, 10000

SRC = {
    "hs300":  ("300etf", "300etf"),
    "zz1000": ("zz1000", "zz1000"),
    "kc50":   ("kc50", "kc50"),
    "cyb":    ("cyb", "cyb"),
}


def bs_price(S, K, T, sig, cp):
    if sig <= 0 or T <= 0: return np.nan
    d1 = (np.log(S/K) + (R + .5*sig*sig)*T)/(sig*np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
    if cp == "C": return S*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)
    return K*np.exp(-R*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def implied_vol(price, S, K, T, cp):
    intr = max(0.0, (S-K) if cp == "C" else (K-S))
    if price <= intr + 1e-6 or T <= 0: return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, s, cp) - price, 1e-3, 5.0, maxiter=60, xtol=1e-4)
    except (ValueError, RuntimeError):
        return np.nan


def gamma(S, K, T, sig):
    if sig <= 0 or T <= 0: return np.nan
    d1 = (np.log(S/K) + (R + .5*sig*sig)*T)/(sig*np.sqrt(T))
    return norm.pdf(d1)/(S*sig*np.sqrt(T))


def build(key):
    raw_key, surface_key = SRC[key]
    surface = latest_ranged_file(PROC, f"surface_{surface_key}")
    if surface is None:
        raise FileNotFoundError(f"缺少 surface_{surface_key}_*.parquet")
    b = pd.read_parquet(RAW/f"ts_optbasic_{raw_key}.parquet")[
        ["ts_code", "call_put", "exercise_price", "maturity_date"]
    ]
    d = read_ranged_parquets(
        RAW, f"ts_optdaily_{raw_key}", columns=["ts_code", "trade_date", "close", "oi"]
    )
    sf = pd.read_parquet(surface.path)[["date", "spot"]]
    for c in ("trade_date",): d[c] = d[c].astype(str)
    b["maturity_date"] = b["maturity_date"].astype(str)
    sf["date"] = sf["date"].astype(str)
    spot = dict(zip(sf["date"], sf["spot"]))

    df = d.merge(b, on="ts_code", how="left").dropna(subset=["exercise_price", "maturity_date"])
    df["S"] = df["trade_date"].map(spot)
    df = df.dropna(subset=["S"])
    df["T"] = (pd.to_datetime(df["maturity_date"]) - pd.to_datetime(df["trade_date"])).dt.days
    df = df[df["T"] >= 3]

    out = {"date": [], "spot": [], "gex": [], "near_dte": [], "max_oi_k": []}
    for dt, g in df.groupby("trade_date"):
        near = g["maturity_date"].min()
        gm = g[g["maturity_date"] == near]
        S = gm["S"].iloc[0]; T = gm["T"].iloc[0]/365.0
        tot = 0.0; ok = 0
        for _, row in gm.iterrows():
            K, cp, oi, price = row["exercise_price"], row["call_put"], row["oi"], row["close"]
            if oi is None or oi <= 0 or price is None or price <= 0: continue
            iv = implied_vol(price, S, K, T, cp)
            if not np.isfinite(iv): continue
            gam = gamma(S, K, T, iv)
            if not np.isfinite(gam): continue
            sign = 1.0 if cp == "C" else -1.0
            tot += gam*oi*MULT*S*S*0.01*sign; ok += 1
        if ok < 3: continue
        # 最大持仓行权价(call+put合并) — pin/磁吸位
        koi = gm.dropna(subset=["oi"]).groupby("exercise_price")["oi"].sum()
        max_k = float(koi.idxmax()) if len(koi) else float("nan")
        out["date"].append(dt); out["spot"].append(round(float(S), 4))
        out["gex"].append(round(tot, 1)); out["near_dte"].append(int(gm["T"].iloc[0]))
        out["max_oi_k"].append(round(max_k, 4))

    res = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    # 标准化 + 次日已实现波动关系
    res["gex_z"] = (res["gex"] - res["gex"].rolling(252, min_periods=60).mean()) / \
                   res["gex"].rolling(252, min_periods=60).std()
    res["ret"] = np.log(res["spot"]/res["spot"].shift(1))
    res["fwd_absret"] = res["ret"].shift(-1).abs()
    return res


def main():
    for key in SRC:
        res = build(key)
        rec = {c: (res[c].where(res[c].notna(), None).tolist()) for c in ["date", "spot", "gex", "gex_z", "near_dte", "max_oi_k"]}
        (PROC/f"gex_{key}.json").write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
        print(f"[{key}] {len(res)} 天  {res['date'].iloc[0]}~{res['date'].iloc[-1]}")
        print(f"  GEX 最新: {res['gex'].iloc[-1]:.3e}   近月DTE {res['near_dte'].iloc[-1]}")
        print(f"  GEX>0 占比: {(res['gex']>0).mean():.0%}   (正=dealer多gamma假设下抑波动)")
        # 初步: 低GEX(负/极低) 是否对应次日更大波动?
        v = res.dropna(subset=["gex_z", "fwd_absret"])
        lo = v[v["gex_z"] <= -1]["fwd_absret"].mean()
        hi = v[v["gex_z"] >= 1]["fwd_absret"].mean()
        mid = v["fwd_absret"].mean()
        print(f"  次日|收益| : GEX_z≤-1 {lo*100:.2f}%  | 全样本 {mid*100:.2f}%  | GEX_z≥+1 {hi*100:.2f}%")
        print(f"  (若 负GEX→次日波动更大, 则 ≤-1 那列应显著 > ≥+1 那列)")


if __name__ == "__main__":
    main()
