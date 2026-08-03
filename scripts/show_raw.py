"""打印某标的的原始指标值(非百分位)时序 + 分位对照。
用法: python scripts/show_raw.py [key] [天数]     默认 zz1000 20
"""
from __future__ import annotations
import json, sys
from pathlib import Path

key = sys.argv[1] if len(sys.argv) > 1 else "zz1000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
T = json.load(open(Path(__file__).resolve().parents[1] / "data/processed/timing_viz.json"))[key]
n = len(T["dates"])
s = max(0, n - N)


def num(x, d=4, pct=False, sign=False):
    if x is None: return "—"
    v = x * 100 if pct else x
    f = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return f + "%" if pct else f


print(f"=== {T['name']} · 原始指标值(非百分位) · 最近 {N} 个交易日 ===")
hdr = (f"{'日期':>9}{'指数':>9}{'涨跌%':>7}{'ATM IV':>8}{'RV20':>7}{'VRP':>8}"
       f"{'情绪比':>8}{'RR':>9}{'斜率':>8}{'PCR':>7}{'20日动量':>9}{'20日回撤':>9}{'GEX_z':>7}")
print(hdr)
for i in range(s, n):
    chg = (T["price"][i]/T["price"][i-1]-1)*100 if i > 0 and T["price"][i-1] else 0.0
    print(f"{T['dates'][i]:>9}{T['price'][i]:>9.0f}{chg:>+7.2f}"
          f"{num(T['atm_iv'][i],1,pct=True):>8}{num(T['rv'][i],1,pct=True):>7}{num(T['vrp'][i],3,sign=True):>8}"
          f"{num(T['sent'][i],3):>8}{num(T['rr'][i],4,sign=True):>9}{num(T['slope'][i],3,sign=True):>8}"
          f"{num(T['pcr'][i],3):>7}{num(T['mom20'][i],1,pct=True,sign=True):>9}"
          f"{num(T['dd20'][i],1,pct=True,sign=True):>9}{num(T['gex_z'][i],2,sign=True):>7}")

print(f"\n=== 同期分位对照 (阈值: IV/情绪比 ≥0.90 亮 · RR/斜率/VRP ≤0.10 亮) ===")
print(f"{'日期':>9}{'IV':>7}{'情绪比':>8}{'RR':>7}{'斜率':>7}{'VRP':>7}{'动量':>7}{'灯':>4}")
for i in range(s, n):
    p = lambda k: "—" if T[k][i] is None else f"{T[k][i]:.2f}"
    k = sum(1 for key2, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]
            for v in [T[key2][i]] if v is not None and ((hi and v >= .9) or (not hi and v <= .1)))
    print(f"{T['dates'][i]:>9}{p('iv_pct'):>7}{p('sent_pct'):>8}{p('rr_pct'):>7}"
          f"{p('slope_pct'):>7}{p('vrp_pct'):>7}{p('mom_pct'):>7}{k:>4}")
