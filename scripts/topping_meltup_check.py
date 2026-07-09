"""验证观察: 高beta成长(尤其科创)的顶部是否为 melt-up 型——高位 + IV高 + VRP高。

对比之前逃顶实验的盲区: 我们只测过"低IV/贪婪镜像""IV从低位回升",从没测过
"高位 + IV分位高 + VRP分位高"这个组合(因为默认顶是低波)。

前提高位: 20日动量分位≥.75。做空视角未来20日(跌率=ret<0, 空盈亏比=|MAE|/MFE)。
另: 诊断各标的"最近一个顶部"(近120日最高点)当时的 IV/VRP/动量 分位。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
H, WIN = 20, 252


def mom_pct(px):
    n = len(px); mom = [None] * n
    for i in range(H, n):
        if px[i] is not None and px[i - H] is not None:
            mom[i] = px[i] / px[i - H] - 1
    pct = [None] * n
    for i in range(n):
        if mom[i] is None: continue
        w = [mom[j] for j in range(max(0, i - WIN + 1), i + 1) if mom[j] is not None]
        if len(w) < WIN // 2: continue
        pct[i] = sum(1 for x in w if x <= mom[i]) / len(w)
    return pct


def fmt(v):
    return "  na" if v is None else f"{v*100:4.0f}%"


def path(px, i):
    if i + H >= len(px) or px[i] is None: return None
    base = px[i]
    fwd = [px[j] / base - 1 for j in range(i + 1, i + 1 + H) if px[j] is not None]
    if len(fwd) < H // 2: return None
    return px[i + H] / base - 1, max(fwd), min(fwd)


def agg(rows):
    rows = [r for r in rows if r]
    if not rows: return None
    a = np.array(rows); ret, mfe, mae = a[:, 0], a[:, 1], a[:, 2]
    amfe, amae = mfe.mean(), mae.mean()
    return dict(n=len(rows), hit=round((ret < 0).mean(), 3), ret=round(ret.mean() * 100, 2),
                mfe=round(amfe * 100, 2), mae=round(amae * 100, 2),
                payoff=round(abs(amae) / amfe, 2) if amfe else np.nan)


def show(name, r):
    if not r: print(f"  {name:<22}(无样本)"); return
    print(f"  {name:<22}{r['n']:>4}{r['hit']*100:>8.1f}%{r['ret']:>8}{r['mfe']:>7}{r['mae']:>8}{r['payoff']:>8}")


print(f"===== 顶部 melt-up 检验 · 未来{H}日做空视角 (跌率 / 空盈亏比>1才划算) =====")
poolH, poolIVVRP = [], []
per = {}
for key, d in DATA.items():
    px = d["price"]; n = len(px)
    mp = mom_pct(px); ivp = d["iv_pct"]; vrp = d["vrp_pct"]
    gh, gv = [], []
    for i in range(n):
        p = path(px, i)
        if p is None or mp[i] is None: continue
        if mp[i] >= .75:
            gh.append(p); poolH.append(p)
            if ivp[i] is not None and vrp[i] is not None and ivp[i] >= .75 and vrp[i] >= .75:
                gv.append(p); poolIVVRP.append(p)
    per[key] = (d["name"], agg(gh), agg(gv))

print(f"  {'标的/组':<22}{'n':>4}{'跌率':>9}{'均值%':>8}{'MFE%':>7}{'MAE%':>8}{'空盈亏':>8}")
for key, (name, rh, rv) in per.items():
    show(f"{name} 高位", rh)
    show(f"{name} 高位+IV高+VRP高", rv)
print("  " + "-" * 66)
show("★合并 高位", agg(poolH))
show("★合并 高位+IV高VRP高", agg(poolIVVRP))

# 诊断各标的最近一个顶部(近120日最高点)当时分位
print(f"\n===== 最近顶部诊断 (近120日最高点当日分位) =====")
for key, d in DATA.items():
    px = d["price"]; n = len(px)
    mp = mom_pct(px)
    seg0 = max(0, n - 120)
    vals = [(i, px[i]) for i in range(seg0, n) if px[i] is not None]
    ti = max(vals, key=lambda t: t[1])[0]
    print(f"  {d['name']:<9} 顶 {d['dates'][ti]}  价{px[ti]:>9.2f}  "
          f"IV分位={fmt(d['iv_pct'][ti])} VRP分位={fmt(d['vrp_pct'][ti])} "
          f"情绪比分位={fmt(d['sent_pct'][ti])} 动量分位={fmt(mp[ti])}")
