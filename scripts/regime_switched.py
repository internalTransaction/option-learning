"""双腿 regime 切换策略(HVWMA + 200日线 结合), 算策略自己的净值。
架构(对话收敛结论):
  BULL       = 200线上行           → 多头凸性腿(panic抄底买call价差), 剩12DTE出
  BEAR       = 200线下行 & HVWMA红  → 空头腿(反弹卖bear call价差)
  拐点预警    = 200线下行 & HVWMA绿  → 空仓(HVWMA快撤空头躲melt-up)
两腿用统一风险口径合成: 每笔押 15% 的"最大亏损", 便于合成一条净值。
regime 用 2018+ 指数价(避开200线warmup)。滑点1%/边, t+1开盘。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("cs", ROOT / "scripts/call_spread_entry.py")
cs = u.module_from_spec(spec); spec.loader.exec_module(cs)
SLIP, FRISK = 0.01, 0.15
# 标的 -> 指数代码(算regime用)
IDXMAP = {"zz1000": "000852", "zz500": "000905", "hs300": "000300", "cyb": "399006",
          "sz50": "000016"}


def wma(s, n):
    n = max(1, int(n)); w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hvwma_green(p, H=16, SM=5):
    s = pd.Series(np.log(p)); half, sq = max(1, H // 2), max(1, round(math.sqrt(H)))
    hma = wma(2 * wma(s, half) - wma(s, H), sq)
    out = np.exp(hma.ewm(alpha=1 / SM, adjust=False).mean())
    return (np.sign(out.diff()).fillna(0).to_numpy() > 0)


def ma_slope_up(p, w=200, k=20):
    m = pd.Series(p).rolling(w, min_periods=w // 2).mean().to_numpy()
    sl = np.full(len(m), np.nan); sl[k:] = m[k:] - m[:-k]
    return sl > 0


def regime_by_date(idx_code):
    """返回 {date: (bull, bear_red, green)} , 用指数价算。"""
    df = pd.read_parquet(ROOT / f"data/raw/idx_{idx_code}.parquet").sort_values("trade_date")
    p = df.close.to_numpy(); up = ma_slope_up(p); grn = hvwma_green(p)
    out = {}
    for i, d in enumerate(df.trade_date.astype(str).tolist()):
        bull = bool(up[i]); green = bool(grn[i])
        out[d] = (bull, (not bull) and (not green), (not bull) and green)
    return out


# ---------- 空头腿(bear call价差, regime门控 + HVWMA绿快撤) ----------
def short_leg(dates, price, chain, px, po, reg, *, mom=5, mc=0.0, width=0.08, hold=15):
    by_day = {dt: g for dt, g in chain.groupby("trade_date")}
    trades = []; i, n = mom, len(dates)
    while i < n - 1:
        st = reg.get(dates[i])
        bear_red = st and st[1]
        bounce = price[i] is not None and price[i - mom] is not None and price[i] > price[i - mom]
        if not (bear_red and bounce):
            i += 1; continue
        ei = i + 1
        if ei > n - 2:
            break
        dt, spot = dates[ei], price[i]
        if dt not in by_day or spot is None:
            i += 1; continue
        day = by_day[dt]; exp = cs.pick_expiry(day, 30, min_dte=hold + 5)
        if exp is None:
            i += 1; continue
        ed = day[day.maturity_date == exp]
        scc = cs.pick_strike(ed, spot * (1 + mc)); lcc = cs.pick_strike(ed, spot * (1 + mc + width))
        if scc.ts_code == lcc.ts_code:
            i += 1; continue
        s_en, l_en = po.get((scc.ts_code, dt)), po.get((lcc.ts_code, dt))
        if not s_en or s_en <= 0 or l_en is None:
            i += 1; continue
        # 出场: 持hold日, 或中途HVWMA翻绿(拐点)则次日快撤
        j = min(ei + hold, n - 1)
        for k in range(ei + 1, j):
            stk = reg.get(dates[k])
            if stk and stk[2]:               # green=拐点预警 → 快撤
                j = min(k + 1, n - 1); break
        xd = dates[j]
        s_ex = px.get((scc.ts_code, xd), max(price[j] - scc.exercise_price, 0.0))
        l_ex = px.get((lcc.ts_code, xd), max(price[j] - lcc.exercise_price, 0.0))
        pnl = (s_en * (1 - SLIP) - s_ex * (1 + SLIP)) + (l_ex * (1 - SLIP) - l_en * (1 + SLIP))
        maxloss = (lcc.exercise_price - scc.exercise_price) - (s_en - l_en)   # 价差宽 - 净收权利金
        if maxloss <= 0:
            i += 1; continue
        trades.append(dict(exit=xd, r_risk=pnl / maxloss, leg="short"))
        i = j + 1
    return trades


def equity(trades):
    tr = sorted(trades, key=lambda t: t["exit"])
    cap = 1.0; curve = []
    for t in tr:
        cap *= (1 + FRISK * t["r_risk"]); curve.append(cap)
    if not curve:
        return None
    c = np.array(curve); dd = (c / np.maximum.accumulate(c) - 1).min()
    rr = np.array([t["r_risk"] for t in tr])
    nS = sum(t["leg"] == "short" for t in tr); nL = len(tr) - nS
    return dict(n=len(tr), nL=nL, nS=nS, term=c[-1], dd=dd, win=(rr > 0).mean())


def run(mode):
    """mode: 'naive'=多头无regime / 'bull_only'=多头仅BULL / 'combined'=双腿切换"""
    allt = []
    for uk, idxc in IDXMAP.items():
        key, prefix = cs.UNIV[uk]
        d, p, b, pa = cs.load_signal(key)
        chain, px, po = cs.load_chain(prefix)
        reg = regime_by_date(idxc)
        bull_mask = [(reg.get(dt, (True, 0, 0))[0]) for dt in d]   # 缺则默认可做
        # 多头腿
        rmask = None if mode == "naive" else bull_mask
        lt = cs.simulate(d, p, b, chain, px, po, theta=0.18, target_dte=30, m_long=0.0,
                         width=0.08, min_panic=0.70, panic_arr=pa, entry_lag=1, entry_px="open",
                         slip=SLIP, exit_rule="dte_left", exit_level=12, maxhold=30, regime_mask=rmask)
        for t in lt:
            allt.append(dict(exit=t["exit_date"], r_risk=t["ret"], leg="long"))
        # 空头腿(仅combined)
        if mode == "combined":
            allt += short_leg(d, p, chain, px, po, reg)
    return equity(allt)


def main():
    print("双腿regime切换(HVWMA+200线)  4标的(zz1000/zz500/hs300/cyb)  统一风险口径押15%\n")
    print(f"  {'方案':<22}{'总笔数':>6}{'多头':>5}{'空头':>5}{'胜率':>6}{'终值x':>7}{'回撤':>7}")
    print("  " + "-" * 58)
    for mode, lab in [("naive", "多头·无regime(基线)"), ("bull_only", "多头·仅BULL做"),
                      ("combined", "双腿·HVWMA+200线切换")]:
        s = run(mode)
        print(f"  {lab:<22}{s['n']:>6}{s['nL']:>5}{s['nS']:>5}{s['win']*100:>5.0f}%{s['term']:>6.2f}{s['dd']*100:>6.0f}%")


if __name__ == "__main__":
    main()
