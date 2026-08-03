"""RR 短put行权价选法对比: 固定% vs σ倍数(RV/IV) vs delta目标。
诊断核心: 固定%让每笔的put delta随vol乱跳(尾部风险不一致); vol标定/delta目标把尾锁成常数。
口径: t+1开盘, panic≥0.70, 持15, 对名义本金收益, 滑点1%/边。买腿固定ATM call。
"""
from __future__ import annotations
import importlib.util as u
from math import log, sqrt, exp
from pathlib import Path
import json
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("rr", ROOT / "scripts/risk_reversal_entry.py")
rr = u.module_from_spec(spec); spec.loader.exec_module(rr)
cs = rr.cs
SLIP = 0.01
HOLD, TDTE = 15, 30


def put_delta(S, K, sigma, T):
    if sigma <= 0 or T <= 0:
        return -1.0 if K > S else 0.0
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    return norm.cdf(d1) - 1.0                      # 认沽 delta ∈ (-1,0)


def choose_put(ed, S, rule, *, rv, iv, z=1.0, tgt_delta=0.15, mp=0.05, dte=30):
    puts = ed[ed.call_put == "P"]
    T = dte / 365.0
    if rule == "fixed":
        target_k = S * (1 - mp)
    elif rule.startswith("sig"):
        sig = rv if rule == "sig_rv" else iv
        target_k = S * exp(-z * sig * sqrt(HOLD / 252.0))   # 持有期内 z 个标准差下方
    elif rule == "delta":
        # 扫描现有put, 取 |delta| 最接近目标的
        i = (puts.exercise_price.map(lambda K: abs(abs(put_delta(S, K, iv, T)) - tgt_delta))).idxmin()
        return puts.loc[i]
    i = (puts.exercise_price - target_k).abs().idxmin()
    return puts.loc[i]


def run(rule, dates, price, buy, panic, rv_a, iv_a, chain, pc, po, **kw):
    by_day = {dt: g for dt, g in chain.groupby("trade_date")}
    rows = []
    i, n = 0, len(dates)
    while i < n - 1:
        b = buy[i]
        if not (b is not None and b >= 0.18 and panic[i] is not None and panic[i] >= 0.70):
            i += 1; continue
        ei = i + 1
        if ei > n - 2: break
        dt, spot = dates[ei], price[i]
        rv, iv = rv_a[i], iv_a[i]
        if dt not in by_day or spot is None or rv is None or iv is None:
            i += 1; continue
        day = by_day[dt]
        exp_ = cs.pick_expiry(day, TDTE, min_dte=HOLD + 5)
        if exp_ is None:
            i += 1; continue
        ed = day[day.maturity_date == exp_]
        dte = int(ed.dte.iloc[0])
        call = rr.pick(ed, "C", spot)
        put = choose_put(ed, spot, rule, rv=rv, iv=iv, dte=dte, **kw)
        c_en, p_en = po.get((call.ts_code, dt)), po.get((put.ts_code, dt))
        if not c_en or not p_en or c_en <= 0 or p_en <= 0:
            i += 1; continue
        j = min(ei + HOLD, n - 1); xd = dates[j]
        c_ex = pc.get((call.ts_code, xd), max(price[j] - call.exercise_price, 0.0))
        p_ex = pc.get((put.ts_code, xd), max(put.exercise_price - price[j], 0.0))
        pnl = (c_ex * (1 - SLIP) - c_en * (1 + SLIP)) + (p_en * (1 - SLIP) - p_ex * (1 + SLIP))
        pdlt = put_delta(spot, put.exercise_price, iv, dte / 365.0)
        rows.append((pnl / spot, price[j] / price[ei] - 1, put.exercise_price / spot - 1, pdlt,
                     (c_en - p_en) / spot))
        i = j + 1
    a = np.array(rows)
    rn, ix, otm, pdl, deb = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    eq = np.cumprod(1 + rn)
    return dict(n=len(rn), win=(rn > 0).mean(), avg=rn.mean(), med=np.median(rn),
                worst=rn.min(), tot=eq[-1] - 1, dd=(eq / np.maximum.accumulate(eq) - 1).min(),
                otm=otm.mean(), otm_sd=otm.std(), pdl=pdl.mean(), pdl_sd=pdl.std(), deb=deb.mean())


def main():
    dates, price, buy, panic = cs.load_signal("zz1000")
    tv = json.load(open(ROOT / "data/processed/timing_viz.json"))["zz1000"]
    rv_a, iv_a = tv["rv"], tv["atm_iv"]
    chain, pc, po = rr.load_chain_both("zz1000")
    print("RR 短put选法对比  zz1000  t+1开盘 panic≥0.70 持15  买腿=ATM call  (对名义本金)\n")
    cfgs = [
        ("固定 -5%",        "fixed", dict(mp=0.05)),
        ("σ-RV ×1",        "sig_rv", dict(z=1.0)),
        ("σ-RV ×2",        "sig_rv", dict(z=2.0)),
        ("σ-IV ×1",        "sig_iv", dict(z=1.0)),
        ("σ-IV ×2",        "sig_iv", dict(z=2.0)),
        ("delta目标 15Δ",   "delta", dict(tgt_delta=0.15)),
        ("delta目标 10Δ",   "delta", dict(tgt_delta=0.10)),
    ]
    h = (f"  {'短put选法':<15}{'笔数':>5}{'胜率':>6}{'均值':>7}{'中位':>7}{'最差':>8}{'终值x':>7}{'回撤':>7}"
         f"{'│平均OTM':>9}{'OTM离散':>8}{'│平均Δ':>8}{'Δ离散':>7}{'净支出':>7}")
    print(h); print("  " + "-" * (len(h) - 2))
    for lab, rule, kw in cfgs:
        s = run(rule, dates, price, buy, panic, rv_a, iv_a, chain, pc, po, **kw)
        print(f"  {lab:<15}{s['n']:>5}{s['win']*100:>5.0f}%{s['avg']*100:>6.1f}%{s['med']*100:>6.1f}%"
              f"{s['worst']*100:>7.1f}%{1+s['tot']:>6.2f}{s['dd']*100:>6.0f}%"
              f"{s['otm']*100:>8.1f}%{s['otm_sd']*100:>7.1f}%{s['pdl']:>8.2f}{s['pdl_sd']:>7.2f}{s['deb']*100:>6.1f}%")
    print("\n看点: 固定% 的 Δ离散(每笔尾部风险不一致)vs vol标定/delta目标把 Δ 锁住; worst 谁更稳。")


if __name__ == "__main__":
    main()
