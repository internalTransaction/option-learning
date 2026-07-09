"""逃顶"二阶拐点"信号实验 + 百分位 vs z-score 对照。

一阶镜像(贪婪极值)已证伪。这里在"已涨到高位"(20日动量分位≥.75)前提下,
测试拐点/背离类信号能否捕捉顶部(做空/降仓视角, 未来H日看 ret<0):

  T0 高位             动量分位≥.75 (参照)
  T1 IV低位回升(pct)  高位 且 iv_pct 5日前≤.30 且 Δ5(iv_pct)≥.20   —— vol regime 醒
  T1z IV低位回升(z)   高位 且 iv_z 5日前≤-.3 且 Δ5(iv_z)≥.7          —— 同概念, robust-z 口径
  T2 动量拐头          高位 且 5日前动量分位≥.90 且 现分位<5日前        —— 冲高回落
  T3 背离              近60日新高 且 Δ5(情绪比分位)≥.20              —— 价新高但避险悄升

做空度量: hit_down=未来ret<0比例, 做空盈亏比=|均值MAE|/均值MFE (>1才划算)。
另单列 T1 vs T1z 的触发重叠, 直接回答"z-score有没有区别"。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
H, WIN = 20, 252


def mom_and_pct(px):
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
    return mom, pct


def robust_z(x):
    n = len(x); z = [None] * n
    for i in range(n):
        if x[i] is None: continue
        w = [v for v in x[max(0, i - WIN + 1):i + 1] if v is not None]
        if len(w) < WIN // 2: continue
        med = np.median(w); mad = np.median(np.abs(np.array(w) - med))
        s = 1.4826 * mad
        z[i] = (x[i] - med) / s if s > 1e-9 else 0.0
    return z


def newhigh(px, w=60):
    n = len(px); nh = [False] * n
    for i in range(n):
        seg = [v for v in px[max(0, i - w + 1):i + 1] if v is not None]
        if px[i] is not None and seg and px[i] >= max(seg): nh[i] = True
    return nh


def d5(a, i):
    if i - 5 < 0 or a[i] is None or a[i - 5] is None: return None
    return a[i] - a[i - 5]


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


# 收集各信号的 path(可跨标的 pool)
POOL = {k: [] for k in ["T0", "T1", "T1z", "T2", "T3"]}
TRIG = {"T1": {}, "T1z": {}}   # 记录触发index用于重叠

for key, d in DATA.items():
    px = d["price"]; n = len(px)
    _, mp = mom_and_pct(px)
    ivp = d["iv_pct"]; sp = d["sent_pct"]
    ivz = robust_z(d["atm_iv"])
    nh = newhigh(px)
    hi = lambda i: mp[i] is not None and mp[i] >= .75
    t1, t1z = set(), set()
    for i in range(n):
        p = path(px, i)
        if p is None: continue
        if hi(i): POOL["T0"].append(p)
        # T1 pct
        if hi(i) and i - 5 >= 0 and ivp[i - 5] is not None and ivp[i - 5] <= .30 and (d5(ivp, i) or -9) >= .20:
            POOL["T1"].append(p); t1.add(i)
        # T1z
        if hi(i) and i - 5 >= 0 and ivz[i - 5] is not None and ivz[i - 5] <= -.3 and (d5(ivz, i) or -9) >= .7:
            POOL["T1z"].append(p); t1z.add(i)
        # T2 动量拐头
        if hi(i) and i - 5 >= 0 and mp[i - 5] is not None and mp[i - 5] >= .90 and mp[i] < mp[i - 5]:
            POOL["T2"].append(p)
        # T3 背离
        if nh[i] and (d5(sp, i) or -9) >= .20:
            POOL["T3"].append(p)
    TRIG["T1"][key] = t1; TRIG["T1z"][key] = t1z


def show(name, r):
    if not r: print(f"  {name:<20} (无样本)"); return
    print(f"  {name:<20}{r['n']:>5}{r['hit']*100:>8.1f}%{r['ret']:>8}{r['mfe']:>7}{r['mae']:>8}{r['payoff']:>8}")


print(f"===== 逃顶二阶信号 · 四标的合并 · 未来{H}日(做空视角) =====")
print(f"  {'信号':<20}{'n':>5}{'跌率':>9}{'均值%':>8}{'MFE%':>7}{'MAE%':>8}{'空盈亏':>8}")
show("T0 高位(参照)", agg(POOL["T0"]))
show("T1 IV回升(百分位)", agg(POOL["T1"]))
show("T1z IV回升(z-score)", agg(POOL["T1z"]))
show("T2 动量拐头", agg(POOL["T2"]))
show("T3 价新高+避险背离", agg(POOL["T3"]))

# T1 vs T1z 重叠
a = sum(len(v) for v in TRIG["T1"].values())
b = sum(len(v) for v in TRIG["T1z"].values())
ov = sum(len(TRIG["T1"][k] & TRIG["T1z"][k]) for k in DATA)
uni = sum(len(TRIG["T1"][k] | TRIG["T1z"][k]) for k in DATA)
print(f"\n  T1(pct)触发 {a} 天, T1z(z)触发 {b} 天, 重叠 {ov} 天, Jaccard={ov/uni:.2f}" if uni else "")
