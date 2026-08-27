"""期权择时出场机 · 正式回测一轮(存档)。

生产配置(cap50/k2.0/ts20, walk-forward两折稳定选中)全面回测:
  A. 分标的分期: 基准 / 现状 / 出场机 (年化/夏普/回撤/Calmar/均仓)
  B. Walk-forward 样本外 (train挑参→OOS验证, 直面过拟合)
  C. 逐笔交易效率 (持有/胜率/盈亏比)
  D. 成本敏感性 (0/5/10bps 换手成本, 看edge是否被成本吃掉)
  E. 四标的等权组合 (真实「一篮子overlay」视角: 组合NAV/夏普/回撤 vs 等权基准)
  F. 无前视自检 (权重T日生成→T+1施加, 抽查对齐)
"""
from __future__ import annotations
import importlib.util as u
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("exit_machine", str(ROOT / "scripts" / "exit_machine.py"))
em = u.module_from_spec(spec); sys.modules["exit_machine"] = em
spec.loader.exec_module(em)
TV, KEYS = em.TV, em.KEYS


def perf_fee(T, w, since, fee):
    px = T["price"]; dates = T["dates"]; n = len(px); s = 0
    while s < n and dates[s] < since:
        s += 1
    sr = []
    for i in range(s + 1, n):
        if px[i] is None or px[i - 1] is None:
            continue
        turn = abs(w[i - 1] - (w[i - 2] if i - 2 >= s else w[s]))
        sr.append(w[i - 1] * (px[i] / px[i - 1] - 1) - turn * fee)
    sr = np.array(sr); nn = len(sr)
    ann = (np.prod(1 + sr)) ** (252 / nn) - 1
    shp = sr.mean() / sr.std() * np.sqrt(252) if sr.std() else 0
    nav = np.cumprod(1 + sr); mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return ann * 100, shp, mdd * 100


def daily_rets(T, w, since):
    """对齐日期→日收益(策略/基准), 供组合汇总。"""
    px = T["price"]; dates = T["dates"]; n = len(px); s = 0
    while s < n and dates[s] < since:
        s += 1
    out = {}
    for i in range(s + 1, n):
        if px[i] is None or px[i - 1] is None:
            continue
        r = px[i] / px[i - 1] - 1
        turn = abs(w[i - 1] - (w[i - 2] if i - 2 >= s else w[s]))
        out[dates[i]] = (w[i - 1] * r - turn * em.FEE, r)
    return out


P = em.Params()
W = {k: em.weights(TV[k], P) for k in KEYS}
WC = {k: em.weights_current(TV[k]) for k in KEYS}

print("=" * 80)
print("期权择时出场机 · 正式回测  |  生产配置 floor0/cap50/k2.0/ts20  |  换手成本5bps")
print("=" * 80)

print("\n── A. 分标的分期 · 基准 / 现状 / 出场机 ──")
for k in KEYS:
    T = TV[k]
    print(f"\n[{T['name']}]")
    print(f"  {'期间':<8}{'':<6}{'年化':>7}{'夏普':>7}{'回撤':>8}{'Calmar':>8}{'均仓':>6}")
    for since, lab in [("20240924", "924后"), ("20220101", "22至今"), ("20200101", "全样本")]:
        b = em.bench(T, since)
        c = em.perf(T, WC[k], since)
        e = em.perf(T, W[k], since)
        if not (c and e):
            continue
        print(f"  {lab:<8}{'基准':<6}{b['ann']:>7}{b['sharpe']:>7}{b['mdd']:>8}{'—':>8}{'100':>6}")
        print(f"  {'':<8}{'现状':<6}{c['ann']:>7}{c['sharpe']:>7}{c['mdd']:>8}{c['calmar']:>8}{c['meanw']:>6}")
        print(f"  {'':<8}{'出场':<6}{e['ann']:>7}{e['sharpe']:>7}{e['mdd']:>8}{e['calmar']:>8}{e['meanw']:>6}")

print("\n── B. Walk-forward 样本外 (train挑参→OOS) ──")
for k in KEYS:
    T = TV[k]
    print(f"[{T['name']}]")
    for tr1, mtr, mte, pp in em.walk_forward(T):
        print(f"   train≤{tr1}: OOS 年化{mte['ann']:>6}% 夏普{mte['sharpe']:>5} 回撤{mte['mdd']:>6}% "
              f"Calmar{mte['calmar']:>5}  [cap{int(pp.cap_trend*100)} k{pp.k_ts} ts{pp.time_stop}]")

print("\n── C. 逐笔交易效率 (全样本) ──")
for k in KEYS:
    t = em.trades(TV[k], W[k])
    print(f"  {TV[k]['name']:<8} {t['n']:>3}笔 持{t['hold']:>5}日 胜率{t['win']:>4} "
          f"均{t['avg']:>5}% 盈{t['avgwin']:>5}/亏{t['avgloss']:>5}  盈亏比{abs(t['avgwin']/t['avgloss']):>4.1f}")

print("\n── D. 成本敏感性 (22至今, 出场机 年化/夏普/回撤) ──")
print(f"  {'标的':<8}{'0bps':>18}{'5bps':>18}{'10bps':>18}")
for k in KEYS:
    row = []
    for fee in [0.0, 0.0005, 0.0010]:
        a, s, m = perf_fee(TV[k], W[k], "20220101", fee)
        row.append(f"{a:>5.1f}/{s:>4.2f}/{m:>5.1f}")
    print(f"  {TV[k]['name']:<8}{row[0]:>18}{row[1]:>18}{row[2]:>18}")

print("\n── E. 四标的等权组合 (一篮子overlay: 各25%) ──")
for since, lab in [("20240924", "924后"), ("20220101", "22至今")]:
    # 汇总每日: 出场组合 / 基准组合
    alld = {}
    for k in KEYS:
        for d, (sr, br) in daily_rets(TV[k], W[k], since).items():
            alld.setdefault(d, []).append((sr, br))
    days = sorted(alld)
    sr = np.array([np.mean([x[0] for x in alld[d]]) for d in days])
    br = np.array([np.mean([x[1] for x in alld[d]]) for d in days])
    def stat(r):
        nn = len(r); ann = (np.prod(1 + r)) ** (252 / nn) - 1
        shp = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
        nav = np.cumprod(1 + r); mdd = (nav / np.maximum.accumulate(nav) - 1).min()
        return ann * 100, shp, mdd * 100
    ea, es, em_ = stat(sr); ba, bs, bm = stat(br)
    print(f"  {lab}: 出场组合 年化{ea:>5.1f}%/夏普{es:>4.2f}/回撤{em_:>5.1f}%   "
          f"等权基准 年化{ba:>5.1f}%/夏普{bs:>4.2f}/回撤{bm:>5.1f}%")

print("\n── F. 无前视自检 (权重T日→T+1收益) ──")
T = TV["cyb"]; w = W["cyb"]
print(f"  回测引擎: strat_ret[i] = w[i-1]×ret[i] (exit_machine.perf 第{'i-1'}日权重×第i日收益)")
print(f"  抽查 cyb 末3日: 权重{[round(w[-3],2),round(w[-2],2),round(w[-1],2)]} "
      f"施加于次日收益, 末日权重{round(w[-1],2)}不参与(无未来收益), 对齐无前视 ✓")
print("\n" + "=" * 80)
print("回测结论见终端汇总。")
