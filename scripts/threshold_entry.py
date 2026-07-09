"""阈值建仓: 平时空仓/低仓, 抄底强度到阈值才进场, 越跌越恐慌加越多。

抄底强度 buy = panic × gate  (panic=五灯等权恐慌度, gate=跌幅门, 越跌越大)
仓位: w = floor + (1-floor) × clip((buy-θ)/(bfull-θ), 0, 1)
  floor  长期底仓(0=平时空仓)
  θ      进场阈值(buy<θ 不建仓)
  bfull  加到满仓所需强度
对比 满仓 / base0.6连续 / 阈值建仓各档。指标含"在场天数%"(踏空/纪律权衡)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))


def clip(x, lo, hi): return max(lo, min(hi, x))


def signals(d):
    n = len(d["dates"]); buy = [None]*n; wc06 = [None]*n
    for i in range(n):
        mp = d["mom_pct"][i]
        contribs = []
        for key, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
            v = d[key][i]
            if v is None: continue
            contribs.append(v if hi else 1 - v)
        if not contribs or mp is None: continue
        panic = sum(contribs)/len(contribs)
        gate = clip((0.30 - mp)/0.30, 0, 1)
        buy[i] = panic*gate
        wc06[i] = clip(0.6 + buy[i]*0.4, 0.3, 1.0)   # base0.6 连续参照
    return buy, wc06


def ret_of(d):
    px = d["price"]; n = len(px); r = [None]*n
    for i in range(1, n):
        if px[i] is not None and px[i-1] is not None: r[i] = px[i]/px[i-1]-1
    return r


def bt(d, wsig, since=None):
    px = d["price"]; n = len(px); r = ret_of(d)
    s = 0
    if since:
        while s < n and d["dates"][s] < since: s += 1
    rs, ws = [], []
    for i in range(s+1, n):
        if r[i] is None or wsig[i-1] is None: continue
        rs.append(wsig[i-1]*r[i]); ws.append(wsig[i-1])
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1
    shp = rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0
    eq = np.cumprod(1+rs); mdd = (eq/np.maximum.accumulate(eq)-1).min()
    inmkt = np.mean(np.array(ws) > 0.05)
    return ann, shp, mdd, np.mean(ws), inmkt


def wfun(buy, floor, th, bfull):
    return [None if b is None else clip(floor + (1-floor)*clip((b-th)/(bfull-th), 0, 1), 0, 1) for b in buy]


# buy 分布
allbuy = []
for k, d in DATA.items():
    b, _ = signals(d); allbuy += [x for x in b if x is not None and x > 0]
allbuy = np.array(allbuy)
print("[抄底强度 buy=panic×gate 分布 (仅 buy>0, 即已跌日)]")
print("  非零天占比 ≈ {:.0%}".format(len(allbuy)/sum(len(DATA[k]['dates']) for k in DATA)))
for q in [50, 70, 80, 90, 95, 99]:
    print(f"  P{q} = {np.percentile(allbuy, q):.3f}")

CONFIGS = [
    ("满仓", None),
    ("base0.6连续", "wc06"),
    ("阈值floor0 θ.12", (0.0, 0.12, 0.45)),
    ("阈值floor0 θ.18", (0.0, 0.18, 0.45)),
    ("底仓.25 θ.12", (0.25, 0.12, 0.45)),
    ("底仓.30 θ.15", (0.30, 0.15, 0.45)),
]

for label, since in [("全样本", None), ("924后", "20240924")]:
    print(f"\n===== {label}  年化 / 夏普 / MaxDD / 均仓 / 在场% =====")
    for k, d in DATA.items():
        buy, wc06 = signals(d)
        print(f" · {d['name']}")
        for name, cfg in CONFIGS:
            if cfg is None: w = [1.0]*len(d["dates"])
            elif cfg == "wc06": w = wc06
            else: w = wfun(buy, *cfg)
            a, s, m, mw, im = bt(d, w, since)
            print(f"    {name:<16}{a*100:>7.1f}{s:>7.2f}{m*100:>8.1f}{mw*100:>7.0f}%{im*100:>7.0f}%")
