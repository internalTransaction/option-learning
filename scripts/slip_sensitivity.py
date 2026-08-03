"""滑点敏感性: 固定 t+1 开盘建仓(零前视), 对代表配置扫描不同单边滑点。
看恐慌时点差变宽后策略还剩多少边际, 并估出"打平滑点"(终值回到1.0的滑点)。
价差腿滑点吃双份(两条腿各穿一次点差, 往返共4次), 会比裸call衰减更快。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("cs", ROOT / "scripts/call_spread_entry.py")
cs = u.module_from_spec(spec); spec.loader.exec_module(cs)

UND = "zz1000"
SLIPS = [0.0, 0.015, 0.03, 0.05, 0.07, 0.10]
# (标签, m_long, width, hold, dte)
CONFIGS = [
    ("均衡  ATM spread8% H20 D30", 0.0, 0.08, 20, 30),
    ("均衡短 ATM spread8% H15 D30", 0.0, 0.08, 15, 30),
    ("裸call ATM naked   H20 D30", 0.0, None, 20, 30),
    ("激进  OTM+4 spread4% H15 D30", 0.04, 0.04, 15, 30),
    ("激进裸 OTM+2 naked   H20 D30", 0.02, None, 20, 30),
]
THETA, MINPANIC = 0.18, 0.70


def breakeven(slips, tots):
    """线性插值出终值(=1+tot)穿过 1.0 的滑点。"""
    xs = np.array(slips); ys = np.array(tots)  # tot = 终值-1
    for k in range(len(xs) - 1):
        if ys[k] >= 0 >= ys[k + 1]:
            x0, x1, y0, y1 = xs[k], xs[k + 1], ys[k], ys[k + 1]
            return x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    return None if ys[-1] > 0 else 0.0


def main():
    key, prefix = cs.UNIV[UND]
    dates, price, buy, panic = cs.load_signal(key)
    chain, px, px_open = cs.load_chain(prefix)
    print(f"标的={UND}  建仓=t+1开盘(零前视)  θ={THETA} panic≥{MINPANIC}  资金曲线=每笔押15%\n")

    for label, m, w, hold, dte in CONFIGS:
        print(label)
        print(f"  {'单边滑点':>8}{'胜率':>7}{'均值':>8}{'中位':>8}{'终值x':>8}{'回撤':>7}")
        tots = []
        for s in SLIPS:
            tr = cs.simulate(dates, price, buy, chain, px, px_open, theta=THETA, hold=hold,
                             target_dte=dte, m_long=m, width=w, min_panic=MINPANIC,
                             panic_arr=panic, entry_lag=1, entry_px="open", slip=s)
            st = cs.stats(tr)
            tots.append(st["tot"])
            print(f"  {s*100:>7.1f}%{st['win']*100:>6.0f}%{st['avg']*100:>7.0f}%"
                  f"{st['med']*100:>7.0f}%{1+st['tot']:>7.2f}{st['mdd']*100:>7.0f}%")
        be = breakeven(SLIPS, tots)
        print(f"  → 打平滑点 ≈ {'>10%' if be is None else f'{be*100:.1f}%/边'}\n")


if __name__ == "__main__":
    main()
