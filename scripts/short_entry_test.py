"""空头腿两项改进测试:
 (1) 反弹入场信号: 粗动量 vs 期权过热镜像(情绪比低=追call自满) vs 情绪比单因子;
 (2) 中性带防churn: 200线斜率|·|>band才激活regime, 模糊区空仓; 看交易数与净值变化。
下跌regime内, bear call价差, 统一风险口径押15%, 4标的合并。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import math, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("rs", ROOT / "scripts/regime_switched.py")
rs = u.module_from_spec(spec); spec.loader.exec_module(rs); cs = rs.cs
SLIP, FRISK = 0.01, 0.15
TV = json.load(open(ROOT / "data/processed/timing_viz.json"))


def overheat_gauges(key):
    """从 timing_viz 百分位算过热镜像。返回 {name: array对齐dates}。"""
    T = TV[key]; n = len(T["dates"])
    comp = [None] * n; sentlo = [None] * n
    for i in range(n):
        # 过热镜像 = 恐慌的反面: 低IV/低情绪比/高RR/高斜率/高VRP
        cs_ = []
        for k, hi in [("iv_pct", 0), ("sent_pct", 0), ("rr_pct", 1), ("slope_pct", 1), ("vrp_pct", 1)]:
            v = T[k][i]
            if v is not None: cs_.append(v if hi else 1 - v)
        if cs_: comp[i] = sum(cs_) / len(cs_)
        sp = T["sent_pct"][i]
        if sp is not None: sentlo[i] = 1 - sp        # 情绪比低=追call
    return {"comp": comp, "sentlo": sentlo}


def slope_band(idx_code, band):
    """返回 {date:(bull,bear_red,green)} 带中性带; band=斜率占均线比例阈值。"""
    df = pd.read_parquet(ROOT / f"data/raw/idx_{idx_code}.parquet").sort_values("trade_date")
    p = df.close.to_numpy()
    m = pd.Series(p).rolling(200, min_periods=100).mean().to_numpy()
    sl = np.full(len(m), np.nan); sl[20:] = (m[20:] - m[:-20]) / m[20:]   # 相对斜率
    grn = rs.hvwma_green(p)
    out = {}
    for i, d in enumerate(df.trade_date.astype(str).tolist()):
        up = sl[i] > band; dn = sl[i] < -band
        out[d] = (bool(up), bool(dn and not grn[i]), bool(dn and grn[i]))
    return out


def short_leg(dates, price, chain, px, po, reg, gate, *, mc=0.0, width=0.08, hold=15):
    by_day = {dt: g for dt, g in chain.groupby("trade_date")}
    trades = []; i, n = 5, len(dates)
    while i < n - 1:
        st = reg.get(dates[i])
        if not (st and st[1] and gate(i)):     # 下跌red + 入场gate
            i += 1; continue
        ei = i + 1
        if ei > n - 2: break
        dt, spot = dates[ei], price[i]
        if dt not in by_day or spot is None:
            i += 1; continue
        day = by_day[dt]; exp = cs.pick_expiry(day, 30, min_dte=hold + 5)
        if exp is None:
            i += 1; continue
        ed = day[day.maturity_date == exp]
        scc = cs.pick_strike(ed, spot*(1+mc)); lcc = cs.pick_strike(ed, spot*(1+mc+width))
        if scc.ts_code == lcc.ts_code:
            i += 1; continue
        s_en, l_en = po.get((scc.ts_code, dt)), po.get((lcc.ts_code, dt))
        if not s_en or s_en <= 0 or l_en is None:
            i += 1; continue
        j = min(ei + hold, n - 1)
        for k in range(ei + 1, j):
            stk = reg.get(dates[k])
            if stk and stk[2]: j = min(k + 1, n - 1); break
        xd = dates[j]
        s_ex = px.get((scc.ts_code, xd), max(price[j]-scc.exercise_price, 0.0))
        l_ex = px.get((lcc.ts_code, xd), max(price[j]-lcc.exercise_price, 0.0))
        pnl = (s_en*(1-SLIP)-s_ex*(1+SLIP)) + (l_ex*(1-SLIP)-l_en*(1+SLIP))
        ml = (lcc.exercise_price-scc.exercise_price) - (s_en-l_en)
        if ml <= 0: i += 1; continue
        trades.append(pnl/ml); i = j + 1
    return trades


def stat(rr):
    r = np.array(rr)
    if len(r) == 0: return None
    eq = np.cumprod(1 + FRISK*r)
    return len(r), (r > 0).mean(), r.mean(), eq[-1], (eq/np.maximum.accumulate(eq)-1).min()


def main():
    D = {}
    for uk, idxc in rs.IDXMAP.items():
        key, pfx = cs.UNIV[uk]; d, p, b, pa = cs.load_signal(key); chain, px, po = cs.load_chain(pfx)
        D[uk] = (key, idxc, d, p, chain, px, po)

    print("=== (1) 空头反弹入场信号对比 (下跌regime, band=0) ===")
    print(f"  {'入场信号':<22}{'笔数':>5}{'胜率':>6}{'均值':>7}{'终值x':>7}{'回撤':>7}")
    for lab, mk in [("粗5日动量反弹", "mom"), ("期权过热镜像≥0.6", "comp"), ("情绪比低≥0.6(核心)", "sentlo"),
                    ("动量+情绪比低", "both")]:
        pool = []
        for uk, (key, idxc, d, p, chain, px, po) in D.items():
            reg = slope_band(idxc, 0.0); g = overheat_gauges(key)
            def gate(i, mk=mk, g=g, p=p):
                bounce = p[i] is not None and p[i-5] is not None and p[i] > p[i-5]
                if mk == "mom": return bounce
                if mk == "comp": return g["comp"][i] is not None and g["comp"][i] >= 0.6
                if mk == "sentlo": return g["sentlo"][i] is not None and g["sentlo"][i] >= 0.6
                return bounce and (g["sentlo"][i] is not None and g["sentlo"][i] >= 0.5)
            pool += short_leg(d, p, chain, px, po, reg, gate)
        s = stat(pool)
        if s: print(f"  {lab:<22}{s[0]:>5}{s[1]*100:>5.0f}%{s[2]*100:>+6.1f}%{s[3]:>6.2f}{s[4]*100:>6.0f}%")

    print("\n=== (2) 中性带防churn (入场=情绪比低≥0.6) ===")
    print(f"  {'斜率中性带':<16}{'空头笔数':>7}{'胜率':>6}{'终值x':>7}{'回撤':>7}")
    for band in [0.0, 0.005, 0.01, 0.02]:
        pool = []
        for uk, (key, idxc, d, p, chain, px, po) in D.items():
            reg = slope_band(idxc, band); g = overheat_gauges(key)
            def gate(i, g=g): return g["sentlo"][i] is not None and g["sentlo"][i] >= 0.6
            pool += short_leg(d, p, chain, px, po, reg, gate)
        s = stat(pool)
        tag = "无(0)" if band == 0 else f"±{band*100:.1f}%"
        if s: print(f"  {tag:<16}{s[0]:>7}{s[1]*100:>5.0f}%{s[3]:>6.2f}{s[4]*100:>6.0f}%")


if __name__ == "__main__":
    main()
