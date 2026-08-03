"""跨周期验证: 50ETF(2015+, 唯一够老) 上跑多头凸性腿, 按三次regime周期拆开看。
只有50ETF期权覆盖 2019转牛 / 2022转熊 / 924 三个周期。
对比: 无regime vs 200线BULL门控。分周期报净值/胜率/回撤。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("rs", ROOT / "scripts/regime_switched.py")
rs = u.module_from_spec(spec); spec.loader.exec_module(rs); cs = rs.cs

# 周期切段 (按大盘节奏, 收盘日期界)
PERIODS = [
    ("2015-18 熊/震荡", "20150209", "20181231"),
    ("2019 转牛",       "20190101", "20211231"),
    ("2022 转熊",       "20220101", "20240923"),
    ("924后 转牛",      "20240924", "20260720"),
]


def substat(trades, w0, w1):
    r = np.array([t["ret"] for t in trades if w0 <= t["entry_date"] <= w1])
    if len(r) == 0:
        return None
    eq = np.cumprod(1 + 0.15 * r)
    return len(r), (r > 0).mean(), np.median(r), eq[-1], (eq / np.maximum.accumulate(eq) - 1).min()


def run(gated):
    key, prefix = cs.UNIV["sz50"]
    d, p, b, pa = cs.load_signal(key)
    chain, px, po = cs.load_chain(prefix)
    reg = rs.regime_by_date("000016")
    mask = [reg.get(dt, (True, 0, 0))[0] for dt in d] if gated else None
    tr = cs.simulate(d, p, b, chain, px, po, theta=0.18, target_dte=30, m_long=0.0, width=0.08,
                     min_panic=0.70, panic_arr=pa, entry_lag=1, entry_px="open", slip=0.01,
                     exit_rule="dte_left", exit_level=12, maxhold=30, regime_mask=mask)
    return tr


def main():
    print("50ETF 多头凸性腿 跨周期验证 (对权利金, 押15%)\n")
    for gated, lab in [(False, "无regime(每次panic都做)"), (True, "200线BULL门控(仅上行做)")]:
        tr = run(gated)
        print(f"【{lab}】 总{len(tr)}笔")
        print(f"  {'周期':<16}{'笔数':>5}{'胜率':>6}{'中位':>7}{'终值x':>7}{'回撤':>7}")
        for pn, w0, w1 in PERIODS:
            s = substat(tr, w0, w1)
            if s:
                print(f"  {pn:<16}{s[0]:>5}{s[1]*100:>5.0f}%{s[2]*100:>+6.0f}%{s[3]:>6.2f}{s[4]*100:>6.0f}%")
            else:
                print(f"  {pn:<16}    —  (无触发)")
        allr = np.array([t["ret"] for t in tr]); eq = np.cumprod(1 + 0.15 * allr)
        print(f"  {'全程':<16}{len(allr):>5}{(allr>0).mean()*100:>5.0f}%{np.median(allr)*100:>+6.0f}%"
              f"{eq[-1]:>6.2f}{(eq/np.maximum.accumulate(eq)-1).min()*100:>6.0f}%\n")


if __name__ == "__main__":
    main()
