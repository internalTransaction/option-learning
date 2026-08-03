"""出场对比: 死扛hold日 vs 恐慌收敛(panic≤level次日出, maxhold兜底)。
主debit结构(裸call / 宽call价差), t+1开盘建仓, panic≥0.70, 对权利金收益(每笔押15%)。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("cs", ROOT / "scripts/call_spread_entry.py")
cs = u.module_from_spec(spec); spec.loader.exec_module(cs)


def holddays(dates, tr):
    idx = {d: i for i, d in enumerate(dates)}
    return np.mean([idx[t["exit_date"]] - idx[t["entry_date"]] for t in tr])


def main():
    dates, price, buy, panic = cs.load_signal("zz1000")
    chain, px, po = cs.load_chain("zz1000")
    common = dict(min_panic=0.70, panic_arr=panic, entry_lag=1, entry_px="open", slip=0.01)
    structs = [("裸ATM call", dict(m_long=0.0, width=None)),
               ("ATM/+8% 价差", dict(m_long=0.0, width=0.08))]
    exits = [("死扛15日", dict(theta=0.18, target_dte=30, exit_rule="hold", hold=15)),
             ("死扛20日", dict(theta=0.18, target_dte=30, exit_rule="hold", hold=20)),
             ("持到期(入~45日)", dict(theta=0.18, target_dte=45, exit_rule="expiry")),
             ("入45→剩21DTE出", dict(theta=0.18, target_dte=45, exit_rule="dte_left", exit_level=21, maxhold=30)),
             ("入45→剩15DTE出", dict(theta=0.18, target_dte=45, exit_rule="dte_left", exit_level=15, maxhold=30)),
             ("恐慌收敛≤0.5", dict(theta=0.18, target_dte=30, exit_rule="panic", exit_level=0.5, maxhold=25, hold=15)),
             ("恐慌收敛≤0.4", dict(theta=0.18, target_dte=30, exit_rule="panic", exit_level=0.4, maxhold=25, hold=15))]
    print("zz1000  t+1开盘 panic≥0.70 滑点1%/边  对权利金收益(每笔押15%)\n")
    for sname, skw in structs:
        print(f"【{sname}】")
        h = f"  {'出场规则':<14}{'笔数':>5}{'胜率':>6}{'均值':>7}{'中位':>7}{'最好':>7}{'最差':>8}{'终值x':>7}{'回撤':>7}{'均持有日':>8}"
        print(h); print("  " + "-" * (len(h) - 2))
        for ename, ekw in exits:
            tr = cs.simulate(dates, price, buy, chain, px, po, **common, **skw, **ekw)
            s = cs.stats(tr)
            hd = holddays(dates, tr)
            print(f"  {ename:<14}{s['n']:>5}{s['win']*100:>5.0f}%{s['avg']*100:>6.0f}%{s['med']*100:>6.0f}%"
                  f"{s['best']*100:>6.0f}%{s['worst']*100:>7.0f}%{1+s['tot']:>6.2f}{s['mdd']*100:>6.0f}%{hd:>7.1f}")
        print()


if __name__ == "__main__":
    main()
