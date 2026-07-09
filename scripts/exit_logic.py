"""出场逻辑对比: 解决"一建仓就平仓"。入场快、出场慢(棘轮 + 明确离场信号)。

入场目标仓位 target = floor + (1-floor)×ramp(buy),  buy = panic × gate(动量分位跌幅门)
过热(hot)时 target 压到 floor*0.5。四种出场:

  V0 瞬时(现状)  w = target                      —— gate一关就掉回底仓
  V1 棘轮         加仓即时; 非过热则每天最多减 decay  —— 给反弹时间
  V2 棘轮+见顶    同V1, 但"动量涨到高位(≥.85)或过热"时快速减 —— 涨够/过热才兑现
  V3 恐慌保持     加仓即时; 恐慌未退(panic≥.40)则维持, 恐慌退才按目标减

对比 年化/夏普/MaxDD/均仓, 并打印 2025 春季那波抄底的逐日仓位路径。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DATA = json.load(open(Path(__file__).resolve().parents[1] / "data/processed/timing_viz.json"))
FLOOR, TH, FULL, DECAY = 0.30, 0.15, 0.45, 0.04


def clamp(x, a, b): return max(a, min(b, x))


def feats(d):
    n = len(d["dates"]); buy = [None]*n; panic = [None]*n
    for i in range(n):
        parts = []
        for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
            v = d[k][i]
            if v is None: continue
            parts.append(v if hi else 1-v)
        mp = d["mom_pct"][i]
        if not parts or mp is None: continue
        pa = sum(parts)/len(parts); panic[i] = pa
        buy[i] = pa*clamp((0.30-mp)/0.30, 0, 1)
    return buy, panic


def hot_of(d):
    n = len(d["dates"]); h = [0]*n
    for i in range(n):
        mp, iv, vp = d["mom_pct"][i], d["iv_pct"][i], d["vrp_pct"][i]
        if mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75:
            h[i] = 1
    return h


def weights(d, mode):
    n = len(d["dates"]); buy, panic = feats(d); hot = hot_of(d); mp = d["mom_pct"]
    w = [FLOOR]*n; prev = FLOOR
    for i in range(n):
        b = buy[i]
        tgt = FLOOR + (1-FLOOR)*clamp(((b if b else 0)-TH)/(FULL-TH), 0, 1)
        if hot[i]: tgt = min(tgt, FLOOR*0.5)
        if mode == "V0":
            wi = tgt
        elif mode == "V1":
            wi = tgt if tgt >= prev or hot[i] else max(tgt, prev-DECAY)
        elif mode == "V2":
            fast = hot[i] or (mp[i] is not None and mp[i] >= .85)
            wi = tgt if tgt >= prev or fast else max(tgt, prev-DECAY)
        elif mode == "V3":
            if tgt >= prev or hot[i]: wi = tgt
            elif panic[i] is not None and panic[i] < .40: wi = tgt
            else: wi = prev
        w[i] = clamp(wi, 0, 1); prev = w[i]
    return w


def bt(d, w, since):
    px = d["price"]; n = len(px); s = 0
    while s < n and d["dates"][s] < since: s += 1
    rs, ws = [], []
    for i in range(s+1, n):
        if px[i] is None or px[i-1] is None: continue
        rs.append(w[i-1]*(px[i]/px[i-1]-1)); ws.append(w[i-1])
    rs = np.array(rs); nn = len(rs)
    ann = (np.prod(1+rs))**(252/nn)-1
    shp = rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0
    eq = np.cumprod(1+rs); mdd = (eq/np.maximum.accumulate(eq)-1).min()
    return ann*100, shp, mdd*100, np.mean(ws)*100


for since, lab in [("20240924", "924后")]:
    print(f"===== 出场逻辑对比 · {lab} · 年化/夏普/MaxDD/均仓 =====")
    for k, d in DATA.items():
        print(f" · {d['name']}")
        for m in ["V0", "V1", "V2", "V3"]:
            a, s, mdd, mw = bt(d, weights(d, m), since)
            note = {"V0": "瞬时(现状)", "V1": "棘轮", "V2": "棘轮+见顶", "V3": "恐慌保持"}[m]
            print(f"    {m} {note:<12}{a:>7.1f}{s:>7.2f}{mdd:>8.1f}{mw:>7.0f}%")

# 逐日仓位路径: 2025 春季抄底
print("\n===== 中证1000 · 2025-04-07~05-09 逐日建议仓位(V0 现状 vs V2 棘轮+见顶) =====")
d = DATA["zz1000"]; w0 = weights(d, "V0"); w2 = weights(d, "V2")
for i, dt in enumerate(d["dates"]):
    if "20250407" <= dt <= "20250509":
        print(f"  {dt}  价 {d['price'][i]:.0f}  灯 {sum(1 for k,hi in [('iv_pct',1),('sent_pct',1),('rr_pct',0),('slope_pct',0),('vrp_pct',0)] if d[k][i] is not None and ((hi and d[k][i]>=.9) or (not hi and d[k][i]<=.1)))}  V0 {w0[i]*100:>3.0f}%   V2 {w2[i]*100:>3.0f}%")
