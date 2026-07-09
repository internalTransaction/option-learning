"""仓位管理视角(非做空): 用信号动态加减仓, 看风险调整收益 vs 满仓持有。

用户真实目标: 个股组合的择时加减仓(A股无做空)。评估标准从"做空盈亏比"
改为"风险调整收益/下行风险"。指数作为组合beta代理。

两部分:
  A. 减仓视角风险检验: melt-up过热点 vs 高位基准 vs 全样本, 未来H日的
     均值收益 / 年化波动 / 平均最大回撤 / P(未来H日内回撤≤-10%)。
  B. 动态仓位回测: t日信号决定 t+1日仓位(无前视), 对比恒定满仓。
     仓位规则(示意): 基础0.6; 抄底 tier1/2/3→0.8/0.9/1.0; melt-up过热→0.35。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
WIN = 252


def lights(d, i):
    k = 0
    for key, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0), ("slope_pct", 0), ("vrp_pct", 0)]:
        v = d[key][i]
        if v is None: continue
        if (hi and v >= .9) or (not hi and v <= .1): k += 1
    return k


def states(d):
    """每日 (tier抄底档0-3, hot过热0/1)。"""
    n = len(d["dates"]); tier = [0]*n; hot = [0]*n
    mp, ivp, vp = d["mom_pct"], d["iv_pct"], d["vrp_pct"]
    for i in range(n):
        k = lights(d, i); m = mp[i]
        if m is not None:
            if k >= 3 and m <= .04: tier[i] = 3
            elif k >= 2 and m <= .08: tier[i] = 2
            elif k >= 1 and m <= .15: tier[i] = 1
        if m is not None and ivp[i] is not None and vp[i] is not None and m >= .85 and ivp[i] >= .75 and vp[i] >= .75:
            hot[i] = 1
    return tier, hot


def fwd_path(px, i, H):
    if i + H >= len(px) or px[i] is None: return None
    base = px[i]
    seq = [px[j] / base - 1 for j in range(i + 1, i + 1 + H) if px[j] is not None]
    if len(seq) < H // 2: return None
    # 路径最大回撤(相对入场后的动态峰值)
    peak = 1.0; mdd = 0.0; cur = 1.0
    for r in seq:
        cur = 1 + r
        peak = max(peak, cur)
        mdd = min(mdd, cur / peak - 1)
    return seq[-1], min(seq), mdd


def risk_check(d, H=40):
    px = d["price"]; n = len(px); tier, hot = states(d)
    mp = d["mom_pct"]
    grp = {"全样本": range(n),
           "高位(动量≥.75)": [i for i in range(n) if mp[i] is not None and mp[i] >= .75 and not hot[i]],
           "melt-up过热": [i for i in range(n) if hot[i]]}
    print(f"\n=== {d['name']}  未来{H}日风险检验(减仓视角) ===")
    print(f"  {'组':<16}{'n':>4}{'均值%':>8}{'年化波动':>9}{'均值MDD%':>9}{'P(MDD≤-10%)':>12}")
    for name, idxs in grp.items():
        rows = [fwd_path(px, i, H) for i in idxs]
        rows = [r for r in rows if r]
        if not rows: continue
        a = np.array(rows); ret, mae, mdd = a[:,0], a[:,1], a[:,2]
        vol = np.std(ret) * np.sqrt(252 / H)
        p10 = (mdd <= -0.10).mean()
        print(f"  {name:<16}{len(rows):>4}{ret.mean()*100:>8.2f}{vol*100:>8.1f}%{mdd.mean()*100:>9.2f}{p10*100:>11.1f}%")


def backtest(d, since=None):
    px = d["price"]; n = len(px); tier, hot = states(d)
    ret = [None]*n
    for i in range(1, n):
        if px[i] is not None and px[i-1] is not None: ret[i] = px[i]/px[i-1]-1
    start = 0
    if since:
        while start < n and d["dates"][start] < since: start += 1
    # 仓位: t日信号 -> t+1日仓位
    def wpos(i):
        if hot[i]: return 0.35
        if tier[i] == 3: return 1.0
        if tier[i] == 2: return 0.9
        if tier[i] == 1: return 0.8
        return 0.6
    sr, br, ws = [], [], []
    for i in range(start+1, n):
        if ret[i] is None: continue
        w = wpos(i-1); ws.append(w)
        sr.append(w*ret[i]); br.append(ret[i])
    def stats(rs):
        rs = np.array(rs); nn = len(rs)
        ann = (np.prod(1+rs))**(252/nn) - 1
        vol = rs.std()*np.sqrt(252)
        shp = rs.mean()/rs.std()*np.sqrt(252) if rs.std() else 0
        eq = np.cumprod(1+rs); peak = np.maximum.accumulate(eq); mdd = (eq/peak-1).min()
        return ann, vol, shp, mdd
    sa, sv, ss, sm = stats(sr); ba, bv, bs, bm = stats(br)
    tag = f"{d['name']}" + (f" (924后)" if since else "")
    print(f"  {tag:<16}{sa*100:>7.1f}{ss:>7.2f}{sm*100:>8.1f}   | {ba*100:>7.1f}{bs:>7.2f}{bm*100:>8.1f}   | 均仓{np.mean(ws)*100:>4.0f}%")


print("========== A. 减仓视角·下行风险检验 ==========")
for k, d in DATA.items():
    risk_check(d, 40)

print("\n========== B. 动态仓位 vs 满仓持有 ==========")
print(f"  {'标的':<16}{'年化%':>7}{'夏普':>7}{'MaxDD%':>8}   | {'满仓年化':>7}{'夏普':>7}{'MaxDD%':>8}   | 平均仓位")
for k, d in DATA.items():
    backtest(d)
print("  --- 仅924后 ---")
for k in ["hs300", "zz1000", "kc50", "cyb"]:
    backtest(DATA[k], since="20240924")
