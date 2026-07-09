"""结构化连续"恐慌指数"→ 仓位, 对比 满仓 / 离散三档 / 连续。

设计(全部可解释, 保留已建立的因果结构):
  panic = mean(iv_pct, sent_pct, 1-rr_pct, 1-slope_pct, 1-vrp_pct)  五因子等权连续合成
  gate  = clip((0.30 - mom_pct)/0.30, 0, 1)   方向门: 跌得越狠门越开(编码"跌幅是必要条件")
  buy   = panic * gate                         恐慌×已跌 才加仓
  hot   = 高位度 × IV高度 × VRP高度            melt-up 连续惩罚
  w     = clip(0.6 + buy*0.4 - hot*0.3, 0.3, 1.0)
无前视: t日信号 -> t+1日仓位。对比含单边5bps换手成本的版本。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
BASE, WMIN, WMAX = 0.6, 0.3, 1.0
COST = 0.0005  # 单边换手成本


def clip(x, lo, hi): return max(lo, min(hi, x))


def lights(d, i):
    k = 0
    for key, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = d[key][i]
        if v is None: continue
        if (hi and v >= .9) or (not hi and v <= .1): k += 1
    return k


def series(d):
    n = len(d["dates"])
    wc = [None]*n   # 连续
    wd = [None]*n   # 离散三档(旧)
    pan = [None]*n
    for i in range(n):
        mp = d["mom_pct"][i]; ivp = d["iv_pct"][i]; vp = d["vrp_pct"][i]
        contribs = []
        for key, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
            v = d[key][i]
            if v is None: continue
            contribs.append(v if hi else 1 - v)
        if not contribs or mp is None:
            continue
        panic = sum(contribs) / len(contribs); pan[i] = round(panic, 3)
        gate = clip((0.30 - mp) / 0.30, 0, 1)
        buy = panic * gate
        top = clip((mp - .70) / .30, 0, 1)
        ivh = clip(((ivp if ivp is not None else 0) - .60) / .40, 0, 1)
        vph = clip(((vp if vp is not None else 0) - .60) / .40, 0, 1)
        hot = top * ivh * vph
        wc[i] = clip(BASE + buy * (WMAX - BASE) - hot * (BASE - WMIN), WMIN, WMAX)
        # 离散三档(旧规则)
        k = lights(d, i); t = 0
        if k >= 3 and mp <= .04: t = 3
        elif k >= 2 and mp <= .08: t = 2
        elif k >= 1 and mp <= .15: t = 1
        hotb = 1 if (mp >= .85 and ivp is not None and vp is not None and ivp >= .75 and vp >= .75) else 0
        wd[i] = 0.35 if hotb else (1.0 if t == 3 else .9 if t == 2 else .8 if t == 1 else .6)
    return wc, wd, pan


def bt(d, wsig, since=None, cost=0.0):
    px = d["price"]; n = len(px)
    ret = [None]*n
    for i in range(1, n):
        if px[i] is not None and px[i-1] is not None: ret[i] = px[i]/px[i-1]-1
    s = 0
    if since:
        while s < n and d["dates"][s] < since: s += 1
    rs, ws = [], []; prev = None
    for i in range(s+1, n):
        if ret[i] is None or wsig[i-1] is None: continue
        w = wsig[i-1]; ws.append(w)
        r = w*ret[i]
        if cost and prev is not None: r -= cost*abs(w-prev)
        prev = w; rs.append(r)
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1
    shp = rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0
    eq = np.cumprod(1+rs); mdd = (eq/np.maximum.accumulate(eq)-1).min()
    turn = np.mean(np.abs(np.diff(ws)))*252 if len(ws) > 1 else 0
    return ann, shp, mdd, np.mean(ws), turn


def row(tag, r):
    ann, shp, mdd, mw, tn = r
    print(f"  {tag:<22}{ann*100:>7.1f}{shp:>7.2f}{mdd*100:>8.1f}{mw*100:>7.0f}%{tn:>7.1f}")


for label, since in [("全样本", None), ("924后", "20240924")]:
    print(f"\n===== {label}  年化 / 夏普 / MaxDD / 均仓 / 年换手 =====")
    for k, d in DATA.items():
        wc, wd, _ = series(d)
        ones = [1.0]*len(d["dates"])
        print(f" · {d['name']}")
        row("满仓", bt(d, ones, since))
        row("离散三档", bt(d, wd, since))
        row("连续指数", bt(d, wc, since))
        row("连续指数(含5bps成本)", bt(d, wc, since, COST))
