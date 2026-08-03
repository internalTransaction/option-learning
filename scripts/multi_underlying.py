"""多标的验证: 定案配置在 zz1000/hs300/kc50/cyb 上各跑一遍, 并把四家交易合并成大样本。
定案: t+1开盘 / panic≥0.70 / 近月~30日 / ATM(裸call 或 +8%价差) / 剩~12DTE出 / 滑点1%/边 / 押15%。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("cs", ROOT / "scripts/call_spread_entry.py")
cs = u.module_from_spec(spec); spec.loader.exec_module(cs)

COMMON = dict(min_panic=0.70, panic_arr=None, entry_lag=1, entry_px="open", slip=0.01,
              theta=0.18, target_dte=30, exit_rule="dte_left", exit_level=12, maxhold=30)
STRUCTS = [("裸ATM call", dict(m_long=0.0, width=None)),
           ("ATM/+8% 价差", dict(m_long=0.0, width=0.08))]


def stat_line(rets):
    r = np.array(rets)
    if len(r) == 0:
        return None
    win, loss = r[r > 0], r[r <= 0]
    payoff = (win.mean() / -loss.mean()) if len(win) and len(loss) else float("nan")
    eq = np.cumprod(1 + 0.15 * r); dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(n=len(r), win=(r > 0).mean(), avg=r.mean(), med=np.median(r),
                payoff=payoff, best=r.max(), worst=r.min(), tot=eq[-1] - 1, mdd=dd)


def show(label, s):
    if s is None or s["n"] == 0:
        print(f"  {label:<20}  (无触发/无数据)"); return
    print(f"  {label:<20}{s['n']:>4}{s['win']*100:>5.0f}%{s['avg']*100:>6.0f}%{s['med']*100:>6.0f}%"
          f"{s['payoff']:>6.2f}{s['best']*100:>7.0f}%{s['worst']*100:>7.0f}%{1+s['tot']:>6.2f}{s['mdd']*100:>6.0f}%")


def main():
    unders = [("zz1000", "中证1000"), ("zz500", "中证500"), ("hs300", "沪深300ETF"),
              ("cyb", "创业板"), ("kc50", "科创50")]
    pooled = {name: [] for name, _ in STRUCTS}
    hdr = f"  {'':<20}{'笔数':>4}{'胜率':>5}{'均值':>6}{'中位':>6}{'盈亏':>6}{'最好':>7}{'最差':>7}{'终值x':>6}{'回撤':>6}"
    for ukey, uname in unders:
        prefix = cs.UNIV[ukey][1]
        if not list((ROOT / "data/raw").glob(f"ts_optdaily_{prefix}_ohlc_*.parquet")):
            print(f"\n【{uname} ({ukey})】 ⚠ 尚无OHLC文件, 跳过"); continue
        key = cs.UNIV[ukey][0]
        dates, price, buy, panic = cs.load_signal(key)
        chain, px, po = cs.load_chain(prefix)
        print(f"\n【{uname} ({ukey})  {dates[0]}~{dates[-1]}】"); print(hdr)
        for sname, skw in STRUCTS:
            kw = {**COMMON, "panic_arr": panic}
            tr = cs.simulate(dates, price, buy, chain, px, po, **kw, **skw)
            rets = [t["ret"] for t in tr]
            pooled[sname] += rets
            show(sname, stat_line(rets))

    print("\n" + "=" * 70)
    print("【四标的合并大样本】"); print(hdr)
    for sname in [n for n, _ in STRUCTS]:
        show(sname + "(合并)", stat_line(pooled[sname]))


if __name__ == "__main__":
    main()
