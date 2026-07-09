"""动态阈值 + 逃顶消融: 把 hardcode 回撤阈值换成"20日动量的滚动分位",
并对称地做上涨(逃顶)方向; 检验镜像"贪婪灯"是否有增量。

动态阈值(路A·滚动分位): mom20 = price[i]/price[i-20]-1
  抄底候选 = mom20 在过去 WIN 日的分位 <= 0.10 (跌得比过去大多数时候狠)
  逃顶候选 = mom20 分位 >= 0.90                (涨得比过去大多数时候猛)
另打印路B(IV标准化 z = mom20 / (atm_iv*sqrt(20/252))) 的触发数对照。

恐慌灯(抄底质量): iv≥.9 sent≥.9 rr≤.1 slope≤.1 vrp≤.1
贪婪灯(逃顶质量): 镜像  iv≤.1 sent≤.1 rr≥.9 slope≥.9 vrp≥.9

未来20日: 抄底看 ret>0 胜率/做多盈亏比; 逃顶看 ret<0 胜率/做空盈亏比。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
H, WIN = 20, 252


def fear_lights(d, i):
    c = 0
    for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = d[k][i]
        if v is None: continue
        if (hi and v >= .9) or (not hi and v <= .1): c += 1
    return c


def greed_lights(d, i):
    c = 0
    for k, lo in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = d[k][i]
        if v is None: continue
        if (lo and v <= .1) or (not lo and v >= .9): c += 1
    return c


def mom_pct(px):
    n = len(px)
    mom = [None] * n
    for i in range(H, n):
        if px[i] is not None and px[i - H] is not None:
            mom[i] = px[i] / px[i - H] - 1
    pct = [None] * n
    for i in range(n):
        if mom[i] is None: continue
        w = [mom[j] for j in range(max(0, i - WIN + 1), i + 1) if mom[j] is not None]
        if len(w) < WIN // 2: continue
        pct[i] = sum(1 for x in w if x <= mom[i]) / len(w)
    return mom, pct


def path(px, i):
    if i + H >= len(px) or px[i] is None: return None
    base = px[i]
    fwd = [px[j] / base - 1 for j in range(i + 1, i + 1 + H) if px[j] is not None]
    if len(fwd) < H // 2: return None
    return px[i + H] / base - 1, max(fwd), min(fwd)


def agg(px, idxs, short=False):
    rows = [path(px, i) for i in idxs]
    rows = [r for r in rows if r]
    if not rows: return None
    a = np.array(rows); ret, mfe, mae = a[:, 0], a[:, 1], a[:, 2]
    amfe, amae = mfe.mean(), mae.mean()
    if short:  # 做空: 有利=下跌(mae), 不利=上涨(mfe)
        payoff = round(abs(amae) / amfe, 2) if amfe else np.nan
        hit = round((ret < 0).mean(), 3)
    else:
        payoff = round(amfe / abs(amae), 2) if amae else np.nan
        hit = round((ret > 0).mean(), 3)
    return dict(n=len(rows), hit=hit, ret=round(ret.mean() * 100, 2),
                mfe=round(amfe * 100, 2), mae=round(amae * 100, 2), payoff=payoff)


def line(name, r):
    if not r: print(f"  {name:<18} (无样本)"); return
    print(f"  {name:<18}{r['n']:>5}{r['hit']*100:>7.1f}%{r['ret']:>8}{r['mfe']:>7}{r['mae']:>8}{r['payoff']:>7}")


for key, d in DATA.items():
    px = d["price"]; n = len(px)
    mom, pct = mom_pct(px)
    fear = [fear_lights(d, i) for i in range(n)]
    greed = [greed_lights(d, i) for i in range(n)]
    idx = range(n)
    g = {
        "fall(分位≤.10)":    [i for i in idx if pct[i] is not None and pct[i] <= .10],
        "  +恐慌灯≥1":       [i for i in idx if pct[i] is not None and pct[i] <= .10 and fear[i] >= 1],
        "  +无恐慌灯":       [i for i in idx if pct[i] is not None and pct[i] <= .10 and fear[i] == 0],
        "rise(分位≥.90)":    [i for i in idx if pct[i] is not None and pct[i] >= .90],
        "  +贪婪灯≥1":       [i for i in idx if pct[i] is not None and pct[i] >= .90 and greed[i] >= 1],
        "  +无贪婪灯":       [i for i in idx if pct[i] is not None and pct[i] >= .90 and greed[i] == 0],
    }
    print(f"\n===== {d['name']}  未来{H}日  (抄底看ret>0 / 逃顶rise看ret<0·做空盈亏比) =====")
    print(f"  {'组':<18}{'n':>5}{'胜率':>8}{'均值%':>8}{'MFE%':>7}{'MAE%':>8}{'盈亏比':>7}")
    line("fall(分位≤.10)", agg(px, g["fall(分位≤.10)"]))
    line("  +无恐慌灯", agg(px, g["  +无恐慌灯"]))
    line("  +恐慌灯≥1", agg(px, g["  +恐慌灯≥1"]))
    line("rise(分位≥.90)", agg(px, g["rise(分位≥.90)"], short=True))
    line("  +无贪婪灯", agg(px, g["  +无贪婪灯"], short=True))
    line("  +贪婪灯≥1", agg(px, g["  +贪婪灯≥1"], short=True))
