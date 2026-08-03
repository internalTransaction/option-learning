"""三结构公平对比: 全部用"对名义本金收益"(PnL点/现价)同一分母, 同一入场(t+1开盘, panic≥0.70)。
裸call / call价差 / risk reversal 直接可比。另列"对权利金收益"(仅debit结构)以显式暴露杠杆。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("rr", ROOT / "scripts/risk_reversal_entry.py")
rr = u.module_from_spec(spec); spec.loader.exec_module(rr)
cs = rr.cs
SLIP = 0.01


def run(kind, dates, price, buy, panic, chain, pc, po, *, hold=15, tdte=30,
        mc=0.0, width=0.08, mp=0.05):
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
        if dt not in by_day or spot is None:
            i += 1; continue
        day = by_day[dt]
        exp = cs.pick_expiry(day, tdte, min_dte=hold + 5)
        if exp is None:
            i += 1; continue
        ed = day[day.maturity_date == exp]
        j = min(ei + hold, n - 1); xd = dates[j]
        lc = rr.pick(ed, "C", spot * (1 + mc))
        lc_en = po.get((lc.ts_code, dt));
        if not lc_en or lc_en <= 0:
            i += 1; continue
        lc_ex = pc.get((lc.ts_code, xd), max(price[j] - lc.exercise_price, 0.0))
        idx = price[j] / price[ei] - 1
        if kind == "naked":
            pnl = lc_ex * (1 - SLIP) - lc_en * (1 + SLIP)
            prem = lc_en * (1 + SLIP)
        elif kind == "spread":
            sc = rr.pick(ed, "C", spot * (1 + mc + width))
            sc_en = po.get((sc.ts_code, dt)) or 0.0
            sc_ex = pc.get((sc.ts_code, xd), max(price[j] - sc.exercise_price, 0.0))
            pnl = (lc_ex * (1 - SLIP) - lc_en * (1 + SLIP)) + (sc_en * (1 - SLIP) - sc_ex * (1 + SLIP))
            prem = lc_en * (1 + SLIP) - sc_en * (1 - SLIP)
        else:  # rr
            pt = rr.pick(ed, "P", spot * (1 - mp))
            pt_en = po.get((pt.ts_code, dt))
            if not pt_en or pt_en <= 0:
                i += 1; continue
            pt_ex = pc.get((pt.ts_code, xd), max(pt.exercise_price - price[j], 0.0))
            pnl = (lc_ex * (1 - SLIP) - lc_en * (1 + SLIP)) + (pt_en * (1 - SLIP) - pt_ex * (1 + SLIP))
            prem = lc_en * (1 + SLIP) - pt_en * (1 - SLIP)
        rows.append((pnl / spot, idx, prem / spot))
        i = j + 1
    a = np.array(rows)
    rn, ix, pr = a[:, 0], a[:, 1], a[:, 2]
    # 对名义本金
    eqn = np.cumprod(1 + rn); ddn = (eqn / np.maximum.accumulate(eqn) - 1).min()
    out = dict(n=len(rn), win=(rn > 0).mean(), avgN=rn.mean(), medN=np.median(rn),
               worstN=rn.min(), bestN=rn.max(), totN=eqn[-1] - 1, ddN=ddn,
               prem=pr.mean(), edge=(rn - ix).mean())
    # 对权利金 (debit结构才有意义, prem>0)
    if pr.mean() > 0.002:
        rp = rn / pr    # PnL/premium
        eqp = np.cumprod(1 + 0.15 * rp)
        out["totP15"] = eqp[-1] - 1
    else:
        out["totP15"] = None
    return out


def main():
    dates, price, buy, panic = cs.load_signal("zz1000")
    chain, pc, po = rr.load_chain_both("zz1000")
    idx_eq = None
    print("zz1000  t+1开盘  panic≥0.70  持15日  滑点1%/边   (11笔, 2023-2026)\n")
    print("同一分母=对名义本金收益(PnL点/现价):")
    h = f"  {'结构':<20}{'delta估':>7}{'胜率':>6}{'均值':>7}{'中位':>7}{'最好':>7}{'最差':>8}{'终值x':>7}{'回撤':>7}{'超额idx':>8}{'净支出':>7}"
    print(h); print("  " + "-" * (len(h) - 2))
    cfgs = [("裸ATM call", "naked", "~0.5", {}),
            ("ATM/+8% call价差", "spread", "~0.2", {}),
            ("RR ATM/-5%put", "rr", "~0.8", {})]
    res = {}
    for lab, kind, dlt, kw in cfgs:
        s = run(kind, dates, price, buy, panic, chain, pc, po, **kw)
        res[lab] = s
        print(f"  {lab:<20}{dlt:>7}{s['win']*100:>5.0f}%{s['avgN']*100:>6.1f}%{s['medN']*100:>6.1f}%"
              f"{s['bestN']*100:>6.1f}%{s['worstN']*100:>7.1f}%{1+s['totN']:>6.2f}{s['ddN']*100:>6.0f}%"
              f"{s['edge']*100:>7.1f}%{s['prem']*100:>6.1f}%")
    print("\n显式暴露杠杆 — 对权利金收益(仅debit, 每笔押15%资金):")
    for lab in ["裸ATM call", "ATM/+8% call价差"]:
        t = res[lab]["totP15"]
        print(f"  {lab:<20} 终值x(对权利金) = {1+t:.2f}   (净支出仅现价的 {res[lab]['prem']*100:.1f}% → 杠杆≈{1/res[lab]['prem']:.0f}x)")
    print("\n注: 名义本金口径下三者可比; call价差每笔只投~现价1-2%的权利金, 故同样名义暴露占用资金极少(杠杆来源)。")


if __name__ == "__main__":
    main()
