"""GEX 三问 · 4标的:
  A. 强化恐慌抄底复现 (低GEX抄底是否反弹更猛)
  B. GEX_z 与现有因子的相关性 (是正交新维度还是冗余 —— 回应 Lasso 共线性问题)
  C. 价格点位: 到期临近时 spot 是否向 max-OI 行权价收敛 (pin 效应)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PROC = Path(__file__).resolve().parents[1] / "data/processed"
TV = json.load(open(PROC/"timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
H = 20
FACTORS = [("iv_pct", "IV分位"), ("sent_pct", "情绪比"), ("rr_pct", "RR"),
           ("slope_pct", "斜率"), ("vrp_pct", "VRP"), ("pcr_pct", "PCR")]


def lights(T, i):
    c = 0
    for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = T[k][i]
        if v is None: continue
        if (hi and v >= .9) or (not hi and v <= .1): c += 1
    return c


def path(px, i):
    if i+H >= len(px) or px[i] is None: return None
    b = px[i]; fwd = [px[j]/b-1 for j in range(i+1, i+1+H) if px[j] is not None]
    if len(fwd) < H//2: return None
    return px[i+H]/b-1, max(fwd), min(fwd)


def agg(rows):
    rows = [r for r in rows if r]
    if not rows: return None
    a = np.array(rows); ret, mfe, mae = a[:, 0], a[:, 1], a[:, 2]
    amfe, amae = mfe.mean(), mae.mean()
    return dict(n=len(rows), hit=(ret > 0).mean(), ret=ret.mean(),
                payoff=(amfe/abs(amae) if amae else 0))


print("========== A. 强化恐慌抄底 · 低GEX vs 高GEX (未来20日) ==========")
for key in KEYS:
    T = TV[key]; G = json.load(open(PROC/f"gex_{key}.json"))
    gz = dict(zip(G["date"], G["gex_z"]))
    px = T["price"]; n = len(T["dates"]); mp = T["mom_pct"]
    ev = [i for i in range(n) if lights(T, i) >= 1 and mp[i] is not None and mp[i] <= .15
          and gz.get(T["dates"][i]) is not None]
    if len(ev) < 20:
        print(f" {T['name']}: 事件不足({len(ev)})"); continue
    z = [gz[T["dates"][i]] for i in ev]; med = np.median(z)
    lo = agg([path(px, i) for i in ev if gz[T["dates"][i]] <= med])
    hi = agg([path(px, i) for i in ev if gz[T["dates"][i]] > med])
    print(f" {T['name']:<8} 事件{len(ev):>3}")
    print(f"   低GEX  n={lo['n']:>3} 胜率{lo['hit']*100:>5.1f}% 均值{lo['ret']*100:>6.2f}% 盈亏比{lo['payoff']:>5.2f}")
    print(f"   高GEX  n={hi['n']:>3} 胜率{hi['hit']*100:>5.1f}% 均值{hi['ret']*100:>6.2f}% 盈亏比{hi['payoff']:>5.2f}")

print("\n========== B. GEX_z 与现有因子相关性 (|r|>0.5 才算冗余) ==========")
print(f"  {'标的':<8}" + "".join(f"{lab:>8}" for _, lab in FACTORS))
for key in KEYS:
    T = TV[key]; G = json.load(open(PROC/f"gex_{key}.json"))
    gz = dict(zip(G["date"], G["gex_z"]))
    di = {dt: i for i, dt in enumerate(T["dates"])}
    common = [dt for dt in G["date"] if dt in di and gz[dt] is not None]
    zvec = np.array([gz[dt] for dt in common])
    row = f"  {T['name']:<8}"
    for fk, _ in FACTORS:
        fv = np.array([T[fk][di[dt]] if T[fk][di[dt]] is not None else np.nan for dt in common])
        m = np.isfinite(zvec) & np.isfinite(fv)
        r = np.corrcoef(zvec[m], fv[m])[0, 1] if m.sum() > 30 else np.nan
        row += f"{r:>8.2f}"
    print(row)

print("\n========== C. Pin: 到期临近 spot 是否向 max-OI 收敛 ==========")
print("   |spot−maxOI|/spot 均值, 按到期剩余天数分组")
print(f"  {'标的':<8}{'DTE≤5':>9}{'6-12':>9}{'13-30':>9}{'>30':>9}")
for key in KEYS:
    G = json.load(open(PROC/f"gex_{key}.json"))
    sp = np.array(G["spot"]); mk = np.array([x if x is not None else np.nan for x in G["max_oi_k"]])
    dte = np.array(G["near_dte"]); dist = np.abs(sp-mk)/sp
    row = f"  {TV[key]['name']:<8}"
    for lo, hi in [(0, 5), (6, 12), (13, 30), (31, 999)]:
        m = (dte >= lo) & (dte <= hi) & np.isfinite(dist)
        row += f"{dist[m].mean()*100:>8.2f}%" if m.sum() else f"{'—':>9}"
    print(row)
