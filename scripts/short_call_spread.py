"""空头腿探索: 下跌趋势里反弹时卖 bear call 价差(卖低call+买高call, 风险封顶)。
逻辑: 顺势做空 + 收权利金 + 赌下行regime里的反弹失败。裸卖call无限风险故不做。
收益口径=对名义本金(PnL点/现价), 对标"同窗口做空指数"。重点看最差单笔(melt-up尾)。
建仓=t+1开盘, 滑点1%/边。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("rf", ROOT / "scripts/regime_filter.py")
rf = u.module_from_spec(spec); spec.loader.exec_module(rf)
cs = rf.cs
SLIP = 0.01
UNDERS = rf.UNDERS


def simulate_short(dates, price, chain, px, po, mask, *, mc, width, hold,
                   target_dte=30, mom_win=5, need_bounce=True):
    by_day = {dt: g for dt, g in chain.groupby("trade_date")}
    trades = []
    i, n = mom_win, len(dates)
    while i < n - 1:
        # 进场: 下行regime + (可选)近mom_win日反弹
        okreg = mask[i]
        bounce = price[i] is not None and price[i - mom_win] is not None and price[i] > price[i - mom_win]
        if not okreg or (need_bounce and not bounce):
            i += 1; continue
        ei = i + 1
        if ei > n - 2:
            break
        dt, spot = dates[ei], price[i]
        if dt not in by_day or spot is None:
            i += 1; continue
        day = by_day[dt]
        exp = cs.pick_expiry(day, target_dte, min_dte=hold + 5)
        if exp is None:
            i += 1; continue
        ed = day[day.maturity_date == exp]
        sc = cs.pick_strike(ed, spot * (1 + mc))            # 卖的低腿
        lc = cs.pick_strike(ed, spot * (1 + mc + width))    # 买的高腿(保护)
        if sc.ts_code == lc.ts_code:
            i += 1; continue
        s_en, l_en = po.get((sc.ts_code, dt)), po.get((lc.ts_code, dt))
        if not s_en or s_en <= 0 or l_en is None:
            i += 1; continue
        j = min(ei + hold, n - 1); xd = dates[j]
        s_ex = px.get((sc.ts_code, xd), max(price[j] - sc.exercise_price, 0.0))
        l_ex = px.get((lc.ts_code, xd), max(price[j] - lc.exercise_price, 0.0))
        # 卖低腿: 收bid, 买回付ask ; 买高腿: 付ask, 卖出收bid
        pnl_short = s_en * (1 - SLIP) - s_ex * (1 + SLIP)
        pnl_long = l_ex * (1 - SLIP) - l_en * (1 + SLIP)
        pnl = pnl_short + pnl_long
        credit = (s_en - l_en) / spot
        idx = price[j] / price[ei] - 1
        trades.append(dict(ret=pnl / spot, credit=credit, idx=idx, entry=dt))
        i = j + 1
    return trades


def pstat(pool):
    if not pool:
        return None
    r = np.array([t["ret"] for t in pool]); idx = np.array([t["idx"] for t in pool])
    cr = np.array([t["credit"] for t in pool])
    eq = np.cumprod(1 + r)
    return dict(n=len(r), win=(r > 0).mean(), avg=r.mean(), med=np.median(r),
                worst=r.min(), best=r.max(), tot=eq[-1] - 1,
                dd=(eq / np.maximum.accumulate(eq) - 1).min(),
                short_idx=(-idx).mean(), credit=cr.mean())


def run(regime_name, need_bounce, mc, width, hold):
    pool = []
    for ukey, _ in UNDERS:
        key, prefix = cs.UNIV[ukey]
        d, p, b, pa = cs.load_signal(key)
        chain, px, po = cs.load_chain(prefix)
        mask = rf.regime(regime_name, p)
        pool += simulate_short(d, p, chain, px, po, mask, mc=mc, width=width,
                               hold=hold, need_bounce=need_bounce)
    return pstat(pool)


def main():
    print("bear call价差(卖低+买高) 五标的合并  t+1开盘 滑点1%/边  对名义本金收益")
    print(f"  {'场景':<26}{'笔数':>5}{'胜率':>6}{'均值':>7}{'中位':>7}{'最好':>7}{'最差':>8}{'终值x':>7}{'回撤':>7}{'均权利金':>8}")
    print("  " + "-" * 82)
    cfgs = [
        ("下行+反弹 卖ATM/+8宽", "MA200降", True, 0.0, 0.08, 15),
        ("下行+反弹 卖+2%/+10宽", "MA200降", True, 0.02, 0.08, 15),
        ("下行(不限反弹) 卖ATM", "MA200降", False, 0.0, 0.08, 15),
        ("下行+反弹 持20日", "MA200降", True, 0.0, 0.08, 20),
        ("[对照]上行+反弹 卖ATM", "MA200升", True, 0.0, 0.08, 15),
    ]
    for lab, reg, nb, mc, w, h in cfgs:
        s = run(reg, nb, mc, w, h)
        if s:
            print(f"  {lab:<26}{s['n']:>5}{s['win']*100:>5.0f}%{s['avg']*100:>+6.1f}%{s['med']*100:>+6.1f}%"
                  f"{s['best']*100:>+6.1f}%{s['worst']*100:>+7.1f}%{1+s['tot']:>6.2f}{s['dd']*100:>6.0f}%{s['credit']*100:>6.1f}%")
    # 单独打印做空指数基准
    s = run("MA200降", True, 0.0, 0.08, 15)
    print(f"\n  参考: 同批(下行+反弹)窗口内'做空指数'平均收益 = {s['short_idx']*100:+.1f}%/笔")


if __name__ == "__main__":
    main()
