"""纯期权·加仓腿回测: 恐慌灯触发 → 买 call / call-spread, 持有若干日后平。

复用信号台口径 (timing_viz.json 的 63 日滚动百分位五灯 + 跌幅门):
  panic = mean(高IV, 高情绪比, 低RR, 低斜率, 低VRP)  # 与 threshold_entry 完全一致
  gate  = clip((0.30 - mom_pct)/0.30, 0, 1)            # 越跌越大
  buy   = panic * gate ; buy>=θ 触发

用真实期权收盘价 (ts_optdaily) + 合约要素 (ts_optbasic) 建/平仓, 非重叠持仓。
返回口径: 每笔 = (平仓价-开仓价)/开仓价 (debit spread 为净支出), 扣双边滑点。
本脚本只做"买方"腿, 风险有限 (最大亏=权利金)。目的: 网格优化 strike / expiry / 持有期。
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# 标的 -> (timing_viz key, optbasic/optdaily 文件前缀)
UNIV = {
    "zz1000": ("zz1000", "zz1000"),
    "zz500": ("zz500", "500etf"),
    "sz50": ("sz50", "50etf"),
    "hs300": ("hs300", "300etf"),
    "kc50": ("kc50", "kc50"),
    "cyb": ("cyb", "cyb"),
}
SLIP = 0.015          # 单边滑点 (占权利金比例, 近似买卖价差的一半)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def dparse(s):
    return datetime.strptime(str(s), "%Y%m%d")


# ---------- 信号 ----------
def load_signal(key: str):
    d = json.load(open(ROOT / "data/processed/timing_viz.json"))[key]
    dates, price = d["dates"], d["price"]
    n = len(dates)
    buy, panic_arr = [None] * n, [None] * n
    for i in range(n):
        mp = d["mom_pct"][i]
        cs = []
        for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
            v = d[k][i]
            if v is None:
                continue
            cs.append(v if hi else 1 - v)
        if not cs or mp is None:
            continue
        panic = sum(cs) / len(cs)
        gate = clip((0.30 - mp) / 0.30, 0, 1)
        panic_arr[i] = panic
        buy[i] = panic * gate
    return dates, price, buy, panic_arr


# ---------- 期权链 ----------
def load_chain(prefix: str):
    basic = pd.read_parquet(ROOT / f"data/raw/ts_optbasic_{prefix}.parquet")
    basic = basic[basic.call_put == "C"][["ts_code", "exercise_price", "maturity_date"]].copy()
    ohlc = sorted((ROOT / "data/raw").glob(f"ts_optdaily_{prefix}_ohlc_*.parquet"))
    if ohlc:   # 有带 open 的权威文件就只用它 (覆盖全样本)
        daily = pd.concat([pd.read_parquet(f) for f in ohlc], ignore_index=True)
        has_open = True
    else:
        files = sorted((ROOT / "data/raw").glob(f"ts_optdaily_{prefix}_*.parquet"))
        daily = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        has_open = False
    daily = daily.drop_duplicates(subset=["ts_code", "trade_date"])
    df = daily.merge(basic, on="ts_code", how="inner")
    df["trade_date"] = df["trade_date"].astype(str)
    df["maturity_date"] = df["maturity_date"].astype(str)
    df["dte"] = [(dparse(m) - dparse(t)).days for m, t in zip(df.maturity_date, df.trade_date)]
    px = {(r.ts_code, r.trade_date): r.close for r in df.itertuples()}
    px_open = {}
    if has_open:
        px_open = {(r.ts_code, r.trade_date): (r.open if r.open and r.open > 0 else r.close)
                   for r in df.itertuples()}
    return df, px, px_open


def pick_expiry(day_df: pd.DataFrame, target_dte: int, min_dte: int):
    cand = day_df[day_df.dte >= min_dte]
    if cand.empty:
        return None
    mats = cand.groupby("maturity_date").dte.first()
    return (mats - target_dte).abs().idxmin()


def pick_strike(exp_df: pd.DataFrame, target_k: float):
    i = (exp_df.exercise_price - target_k).abs().idxmin()
    return exp_df.loc[i]


# ---------- 回测 ----------
def simulate(dates, price, buy, chain, px, px_open=None, *, theta, target_dte,
             m_long, width, hold=15, min_panic=0.0, panic_arr=None, entry_lag=0, entry_px="close",
             slip=SLIP, exit_rule="hold", exit_level=0.5, maxhold=25, regime_mask=None):
    """m_long: 长腿相对现价的moneyness (0=ATM, +0.02=虚2%); width: 价差宽度(占现价比, None=裸买call)
    entry_lag: 出信号后隔几天建仓 (0=同bar, 1=t+1, 去前视用)
    entry_px: 建仓价 'close'=收盘 / 'open'=开盘(需 px_open, 配 lag=1 即真实 t+1 开盘)
    slip: 单边滑点(占权利金比例)
    exit_rule: 'hold'=死扛hold日 / 'panic'=恐慌收敛(panic≤exit_level则次日出, maxhold兜底)"""
    by_day = {dt: g for dt, g in chain.groupby("trade_date")}
    epx = px_open if (entry_px == "open" and px_open) else px
    trades = []
    i, n = 0, len(dates)
    while i < n - 1:
        b = buy[i]
        ok = b is not None and b >= theta
        if ok and min_panic > 0 and panic_arr[i] is not None:
            ok = panic_arr[i] >= min_panic
        if ok and regime_mask is not None:      # regime开关: 只在允许的regime里进场
            ok = bool(regime_mask[i])
        if not ok:
            i += 1
            continue
        ei = i + entry_lag                 # 实际建仓日
        if ei > n - 2:
            break
        dt = dates[ei]
        spot = price[i]                    # 选strike用信号日现价(决策时已知)
        if dt not in by_day or spot is None:
            i += 1
            continue
        day_df = by_day[dt]
        if exit_rule in ("panic", "trail"):
            need = maxhold + 5
        elif exit_rule == "expiry":
            need = 12                       # 持有到期: 入场时至少留~12日到期
        elif exit_rule == "dte_left":
            need = exit_level + 15          # 入场DTE要高于出场剩余DTE, 留足持有窗口
        else:
            need = hold + 5
        exp = pick_expiry(day_df, target_dte, min_dte=need)
        if exp is None:
            i += 1
            continue
        exp_df = day_df[day_df.maturity_date == exp]
        long_c = pick_strike(exp_df, spot * (1 + m_long))
        entry_long = epx.get((long_c.ts_code, dt))
        if entry_long is None or entry_long <= 0:
            i += 1
            continue
        short_c = None
        if width is not None:                      # 价差空头腿先选好(移动止盈要盯每日市值)
            short_c = pick_strike(exp_df, spot * (1 + m_long + width))
            if short_c.ts_code == long_c.ts_code or not epx.get((short_c.ts_code, dt)):
                i += 1
                continue

        def val(k):                                # 第k日持仓市值(收盘, 缺价→内在)
            vl = px.get((long_c.ts_code, dates[k]), max(price[k] - long_c.exercise_price, 0.0))
            if short_c is None:
                return vl
            vs = px.get((short_c.ts_code, dates[k]), max(price[k] - short_c.exercise_price, 0.0))
            return vl - vs

        # 出场日 (平仓用收盘)
        if exit_rule == "panic":
            cap = min(ei + maxhold, n - 1)
            j = cap
            for k in range(ei + 1, cap):        # panic在第k日冷却→次日k+1出(无前视)
                if panic_arr[k] is not None and panic_arr[k] <= exit_level:
                    j = min(k + 1, n - 1)
                    break
        elif exit_rule == "expiry":             # 持有到期(OPEX): 最后一个≤到期日的交易日
            mat = long_c.maturity_date
            j = ei
            for k in range(ei + 1, n):
                if dates[k] <= mat:
                    j = k
                else:
                    break
        elif exit_rule == "dte_left":           # 到期前若干日出(躲theta悬崖, OPEX锚定)
            mat = long_c.maturity_date
            j = min(ei + maxhold, n - 1)
            for k in range(ei + 1, n):
                if (dparse(mat) - dparse(dates[k])).days <= exit_level:
                    j = k
                    break
        elif exit_rule == "trail":              # 移动止盈: 市值从峰值回撤exit_level则次日出
            mat = long_c.maturity_date          # 兜底: 剩~12DTE 或 maxhold
            cap = min(ei + maxhold, n - 1)
            peak = val(ei); j = cap
            for k in range(ei + 1, cap):
                vk = val(k)
                peak = max(peak, vk)
                hit_trail = peak > 0 and vk <= peak * (1 - exit_level)
                near_exp = (dparse(mat) - dparse(dates[k])).days <= 12
                if hit_trail or near_exp:
                    j = min(k + 1, n - 1) if hit_trail else k
                    break
        else:
            j = min(ei + hold, n - 1)
        ex_dt = dates[j]
        exit_long = px.get((long_c.ts_code, ex_dt))
        if exit_long is None:  # 到期或缺价 -> 用内在价值
            exit_long = max(price[j] - long_c.exercise_price, 0.0)

        if width is None:  # 裸买 call
            entry = entry_long * (1 + slip)
            exit_ = exit_long * (1 - slip)
        else:              # call spread: 空头腿已在上方选好
            entry_short = epx.get((short_c.ts_code, dt)) or 0.0
            exit_short = px.get((short_c.ts_code, ex_dt))
            if exit_short is None:
                exit_short = max(price[j] - short_c.exercise_price, 0.0)
            entry = entry_long * (1 + slip) - entry_short * (1 - slip)
            exit_ = exit_long * (1 - slip) - exit_short * (1 + slip)
            if entry <= 0:
                i += 1
                continue
        ret = exit_ / entry - 1.0
        trades.append(dict(entry_date=dt, exit_date=ex_dt, spot=spot,
                           k_long=long_c.exercise_price,
                           k_short=(None if width is None else short_c.exercise_price),
                           dte=long_c.dte, hold_d=j - ei, entry_px=round(entry, 4),
                           exit_px=round(exit_, 4), ret=ret))
        i = j + 1  # 非重叠: 平仓后才可再进
    return trades


def stats(trades, f=0.15):
    """f: 每笔占总资金比例 (固定分数下注); 买方单笔最多亏 f, 不会爆本。"""
    if not trades:
        return None
    r = np.array([t["ret"] for t in trades])
    win = r[r > 0]
    loss = r[r <= 0]
    payoff = (win.mean() / -loss.mean()) if len(win) and len(loss) else np.nan
    eq = np.cumprod(1 + f * r)                 # 固定 f 比例下注的资金曲线
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(n=len(r), win=(r > 0).mean(), avg=r.mean(), med=np.median(r),
                payoff=payoff, best=r.max(), worst=r.min(),
                tot=eq[-1] - 1, mdd=dd)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", default="zz1000")
    ap.add_argument("--theta", type=float, default=0.18)
    ap.add_argument("--min-panic", type=float, default=0.0)
    ap.add_argument("--lag", type=int, default=0, help="0=同bar建仓, 1=t+1(去前视)")
    ap.add_argument("--entry-px", default="close", choices=["close", "open"], help="建仓价, open需带open的ohlc文件")
    args = ap.parse_args()

    key, prefix = UNIV[args.u]
    dates, price, buy, panic = load_signal(key)
    chain, px, px_open = load_chain(prefix)
    if args.entry_px == "open" and not px_open:
        print("⚠ 无带 open 的 ohlc 文件, 回退到 close 建仓")
        args.entry_px = "close"
    pxtxt = {"close": "收盘", "open": "开盘"}[args.entry_px]
    lagtxt = "同bar" if args.lag == 0 else f"t+{args.lag}"
    print(f"标的={args.u}  样本 {dates[0]}~{dates[-1]}  θ={args.theta}  min_panic={args.min_panic}  建仓={lagtxt}{pxtxt}  滑点={SLIP:.1%}/边\n")

    grid = []
    for m_long in [-0.02, 0.0, 0.02, 0.04]:
        for width in [None, 0.04, 0.08]:
            for hold in [10, 15, 20]:
                for tdte in [30, 45]:
                    tr = simulate(dates, price, buy, chain, px, px_open, theta=args.theta,
                                  hold=hold, target_dte=tdte, m_long=m_long, width=width,
                                  min_panic=args.min_panic, panic_arr=panic,
                                  entry_lag=args.lag, entry_px=args.entry_px)
                    s = stats(tr)
                    if s and s["n"] >= 6:
                        struct = "裸call" if width is None else f"spread{int(width*100)}%"
                        s.update(struct=f"{'ITM' if m_long<0 else ('ATM' if m_long==0 else 'OTM')}{int(m_long*100):+d} {struct} H{hold} D{tdte}")
                        grid.append(s)

    grid.sort(key=lambda s: s["avg"], reverse=True)  # 按单笔期望排序
    print("(资金曲线=每笔押 15% 固定比例; 单笔最多亏该 15%)\n")
    hdr = (f"{'结构':<26}{'笔数':>5}{'胜率':>7}{'均值':>8}{'中位':>8}{'盈亏比':>7}"
           f"{'最好':>8}{'最差':>8}{'终值x':>8}{'回撤':>7}")
    print(hdr); print("-" * len(hdr))
    for s in grid[:22]:
        print(f"{s['struct']:<26}{s['n']:>5}{s['win']*100:>6.0f}%{s['avg']*100:>7.0f}%{s['med']*100:>7.0f}%"
              f"{s['payoff']:>7.2f}{s['best']*100:>7.0f}%{s['worst']*100:>7.0f}%"
              f"{1+s['tot']:>7.2f}{s['mdd']*100:>7.0f}%")


if __name__ == "__main__":
    main()
