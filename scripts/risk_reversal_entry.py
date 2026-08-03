"""纯期权·加仓腿之 Risk Reversal: 恐慌触发 → 买call + 卖put(卖在最贵的put skew上)。
净空vega(躲IV crush)、常近零成本, 但短put破了有限风险: 底没到会挨打。

收益口径不同于debit结构: 用"对标的名义本金的收益" = PnL(点)/现价。
并对标"同窗口直接持有指数", 分离出 RR 的 skew+结构 alpha。
建仓=t+1开盘(零前视), 门槛与call腿一致(panic≥0.70 + 跌幅门)。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("cs", ROOT / "scripts/call_spread_entry.py")
cs = u.module_from_spec(spec); spec.loader.exec_module(cs)

SLIP = 0.01   # 单边滑点(近ATM实测~0.3-0.5%, put腿稍宽, 取1%偏保守)


def load_chain_both(prefix: str):
    """同时载入 call 与 put, 返回按日分组的链 + open/close 查询表。"""
    basic = pd.read_parquet(ROOT / f"data/raw/ts_optbasic_{prefix}.parquet")
    basic = basic[["ts_code", "call_put", "exercise_price", "maturity_date"]].copy()
    ohlc = sorted((ROOT / "data/raw").glob(f"ts_optdaily_{prefix}_ohlc_*.parquet"))
    daily = pd.concat([pd.read_parquet(f) for f in ohlc], ignore_index=True)
    daily = daily.drop_duplicates(subset=["ts_code", "trade_date"])
    df = daily.merge(basic, on="ts_code", how="inner")
    df["trade_date"] = df["trade_date"].astype(str)
    df["maturity_date"] = df["maturity_date"].astype(str)
    df["dte"] = [(cs.dparse(m) - cs.dparse(t)).days for m, t in zip(df.maturity_date, df.trade_date)]
    pc = {(r.ts_code, r.trade_date): r.close for r in df.itertuples()}
    po = {(r.ts_code, r.trade_date): (r.open if r.open and r.open > 0 else r.close)
          for r in df.itertuples()}
    return df, pc, po


def pick(exp_df, cp, target_k):
    sub = exp_df[exp_df.call_put == cp]
    i = (sub.exercise_price - target_k).abs().idxmin()
    return sub.loc[i]


def simulate_rr(dates, price, buy, chain, pc, po, *, theta, min_panic, panic_arr,
                hold, target_dte, mc, mp, slip=SLIP):
    """mc: call虚值度(0=ATM); mp: put虚值度(0.05=虚5%卖put)。t+1开盘建仓。"""
    by_day = {dt: g for dt, g in chain.groupby("trade_date")}
    trades = []
    i, n = 0, len(dates)
    while i < n - 1:
        b = buy[i]
        ok = b is not None and b >= theta and (min_panic <= 0 or
             (panic_arr[i] is not None and panic_arr[i] >= min_panic))
        if not ok:
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
        call = pick(ed, "C", spot * (1 + mc))
        put = pick(ed, "P", spot * (1 - mp))
        c_en, p_en = po.get((call.ts_code, dt)), po.get((put.ts_code, dt))
        if not c_en or not p_en or c_en <= 0 or p_en <= 0:
            i += 1; continue
        j = min(ei + hold, n - 1)
        xd = dates[j]
        c_ex = pc.get((call.ts_code, xd), max(price[j] - call.exercise_price, 0.0))
        p_ex = pc.get((put.ts_code, xd), max(put.exercise_price - price[j], 0.0))
        # 建仓: 买call付ask, 卖put收bid ; 平仓: 卖call收bid, 买回put付ask
        pnl_call = c_ex * (1 - slip) - c_en * (1 + slip)
        pnl_put = p_en * (1 - slip) - p_ex * (1 + slip)
        pnl = pnl_call + pnl_put                       # 点
        ret = pnl / spot                               # 对名义本金收益
        net_debit = (c_en - p_en) / spot               # 净支出(占现价, 负=净收权利金)
        idx_ret = price[j] / price[ei] - 1             # 同窗口指数收益(基准)
        trades.append(dict(entry=dt, exit=xd, spot=spot, ret=ret, idx=idx_ret,
                           debit=net_debit, kc=call.exercise_price, kp=put.exercise_price))
        i = j + 1
    return trades


def stats(trades, f=1.0):
    if not trades:
        return None
    r = np.array([t["ret"] for t in trades])
    idx = np.array([t["idx"] for t in trades])
    deb = np.array([t["debit"] for t in trades])
    eq = np.cumprod(1 + f * r); dd = (eq / np.maximum.accumulate(eq) - 1).min()
    eqi = np.cumprod(1 + f * idx)
    return dict(n=len(r), win=(r > 0).mean(), avg=r.mean(), med=np.median(r),
                worst=r.min(), best=r.max(), edge=(r - idx).mean(),
                idx_avg=idx.mean(), debit=deb.mean(), tot=eq[-1] - 1, idx_tot=eqi[-1] - 1, mdd=dd)


def main():
    key, prefix = "zz1000", "zz1000"
    dates, price, buy, panic = cs.load_signal(key)
    chain, pc, po = load_chain_both(prefix)
    print(f"标的=zz1000 RiskReversal  建仓=t+1开盘  θ=0.18 panic≥0.70  滑点={SLIP:.1%}/边")
    print("收益口径=对名义本金(PnL点/现价); 资金曲线f=1(满名义); 对标同窗口持有指数\n")
    hdr = (f"{'结构(买call/卖put)':<22}{'笔数':>5}{'胜率':>6}{'均值':>7}{'中位':>7}{'最好':>7}"
           f"{'最差':>8}{'净支出':>7}{'超额idx':>8}{'终值x':>7}{'指数x':>7}{'回撤':>7}")
    print(hdr); print("-" * len(hdr))
    grid = []
    for mc in [0.0, 0.02]:
        for mp in [0.03, 0.05, 0.08]:
            for hold in [15, 20]:
                tr = simulate_rr(dates, price, buy, chain, pc, po, theta=0.18, min_panic=0.70,
                                 panic_arr=panic, hold=hold, target_dte=30, mc=mc, mp=mp)
                s = stats(tr)
                if s:
                    s["lab"] = f"C{'ATM' if mc==0 else f'+{int(mc*100)}%'}/P-{int(mp*100)}% H{hold}"
                    grid.append(s)
    grid.sort(key=lambda s: s["avg"], reverse=True)
    for s in grid:
        print(f"{s['lab']:<22}{s['n']:>5}{s['win']*100:>5.0f}%{s['avg']*100:>6.0f}%{s['med']*100:>6.0f}%"
              f"{s['best']*100:>6.0f}%{s['worst']*100:>7.0f}%{s['debit']*100:>6.1f}%{s['edge']*100:>7.0f}%"
              f"{1+s['tot']:>6.2f}{1+s['idx_tot']:>6.2f}{s['mdd']*100:>6.0f}%")


if __name__ == "__main__":
    main()
