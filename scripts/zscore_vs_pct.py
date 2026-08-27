"""口径实验室 · 滚动百分位 vs 滚动 z-score(正态CDF映射) × 窗口 63/126/252。

问题: 五灯极值现在用「63日滚动经验百分位」判定。换成 z-score 会更好吗? 窗口多长合适?

做法(单变量对照, 只换"极值强度"的算法, 其余全部钉死):
  强度 s ∈ [0,1] = 该因子当前值在滚动窗口内的相对高度。
    pct 口径: s = 经验百分位 (x <= x_now 的占比)
    z   口径: s = Φ((x - 滚动均值)/滚动标准差)   ← 正态假设下的"参数化分位"
  两者阈值同为 0.90/0.10, 故差异纯粹来自「经验分布 vs 正态假设」(厚尾/右偏的处理)。
  仓位模型、GEX 调节、棘轮出场、成本 全部复用 build_equity.py 的常量与逻辑。

输出: 4 标的 × {pct,z} × {63,126,252} 的 年化/夏普/回撤/均仓/超额/IR + 触发日 Jaccard 重叠,
      写 data/processed/calib_lab.json 供前端「口径实验室」面板消费。
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TV = json.load(open(PROC / "timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]

# ── 与 build_equity.py 完全一致的参数(不允许在本实验里漂移) ────────────────────
FLOOR, TH, FULL, DECAY, GK, GLO, GHI, FEE = 0.30, 0.15, 0.45, 0.04, 0.4, 0.4, 1.5, 0.0005
HI, LO = 0.90, 0.10          # 五灯极值阈值(强度口径, 两种算法共用)
MOM_WIN = 252                # 动量 gate 窗口固定(只扫因子窗口, 变量更干净)
WINS = [63, 126, 252]
MODES = ["pct", "z"]
SINCE = {"924": "20240924", "all": "20000101"}
# 五因子及其"恐慌方向": 1=数值高即恐慌, 0=数值低即恐慌
FACTORS = [("atm_iv", 1), ("sent", 1), ("rr", 0), ("slope", 0), ("vrp", 0)]


def clamp(x, a, b):
    return max(a, min(b, x))


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def strength(vals, win, mode):
    """原始序列 → 0~1 强度序列(高=数值相对高)。缺失=None。"""
    s = pd.Series([np.nan if v is None else float(v) for v in vals], dtype="float64")
    minp = max(10, win // 2)
    if mode == "pct":
        out = s.rolling(win, min_periods=minp).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    else:
        mu = s.rolling(win, min_periods=minp).mean()
        sd = s.rolling(win, min_periods=minp).std(ddof=0)
        z = (s - mu) / sd.replace(0.0, np.nan)
        out = z.apply(lambda v: np.nan if pd.isna(v) else _phi(float(v)))
    return [None if pd.isna(v) else float(v) for v in out]


def build_signals(T, win, mode):
    """返回 (panic 强度矩阵 dict, mom 强度, 灯数, tier)。"""
    st = {k: strength(T[k], win, mode) for k, _ in FACTORS}
    mom = strength(T["mom20"], MOM_WIN, mode)
    n = len(T["dates"])
    lights, tier = [0] * n, [0] * n
    for i in range(n):
        k = 0
        for name, hi in FACTORS:
            v = st[name][i]
            if v is None:
                continue
            if (hi and v >= HI) or (not hi and v <= LO):
                k += 1
        lights[i] = k
        m = mom[i]
        if m is not None:
            if k >= 3 and m <= 0.04:
                tier[i] = 3
            elif k >= 2 and m <= 0.08:
                tier[i] = 2
            elif k >= 1 and m <= 0.15:
                tier[i] = 1
    return st, mom, lights, tier


def weights_opt(T, st, mom):
    """新库纯期权模式: 恐慌度×跌幅门×GEX → floor30 + 棘轮出场。"""
    n = len(T["dates"])
    gz = T.get("gex_z")
    w, prev = [FLOOR] * n, FLOOR
    for i in range(n):
        parts = []
        for name, hi in FACTORS:
            v = st[name][i]
            if v is None:
                continue
            parts.append(v if hi else 1 - v)
        dip = 0.0
        m = mom[i]
        if parts and m is not None:
            b = (sum(parts) / len(parts)) * clamp((0.30 - m) / 0.30, 0, 1)
            if gz and gz[i] is not None:
                b *= clamp(1 - GK * gz[i], GLO, GHI)
            dip = clamp((b - TH) / (FULL - TH), 0, 1)
        tgt = FLOOR + (1 - FLOOR) * dip
        iv, vp = st["atm_iv"][i], st["vrp"][i]
        hot = m is not None and iv is not None and vp is not None and m >= .85 and iv >= .75 and vp >= .75
        if hot:
            tgt = min(tgt, FLOOR * 0.5)
        fast = hot or (m is not None and m >= .85)
        w[i] = clamp(tgt if (tgt >= prev or fast) else max(tgt, prev - DECAY), 0, 1)
        prev = w[i]
    return w


def stat(rets):
    r = np.array(rets)
    nav = np.cumprod(1 + r)
    ann = nav[-1] ** (252 / len(r)) - 1
    shp = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return dict(ann=round(ann * 100, 1), sharpe=round(shp, 2), mdd=round(mdd * 100, 1))


def evaluate(T, w, since):
    """策略 vs 等均仓恒定(剥离仓位水平) — 与 build_equity.strat_series 同口径。"""
    dates, pr, n = T["dates"], T["price"], len(T["price"])
    s = 0
    while s < n and dates[s] < since:
        s += 1
    mw = float(np.mean([w[i] for i in range(s, n)]))
    sr, cr = [], []
    for i in range(s + 1, n):
        if pr[i] is None or pr[i - 1] is None:
            continue
        r = pr[i] / pr[i - 1] - 1
        turn = abs(w[i - 1] - (w[i - 2] if i - 2 >= s else w[s]))
        sr.append(w[i - 1] * r - turn * FEE)
        cr.append(mw * r)
    st_, ct = stat(sr), stat(cr)
    d = np.array(sr) - np.array(cr)
    ir = d.mean() / d.std() * np.sqrt(252) if d.std() else 0
    st_.update(meanw=round(mw * 100), excess=round(st_["ann"] - ct["ann"], 1), ir=round(float(ir), 2))
    return st_, s


def jaccard(a, b):
    A, B = set(a), set(b)
    return round(len(A & B) / len(A | B), 3) if (A | B) else None


def main():
    out = {"params": {"floor": FLOOR, "th": TH, "full": FULL, "decay": DECAY, "gex_k": GK,
                      "hi": HI, "lo": LO, "mom_win": MOM_WIN, "fee": FEE},
           "wins": WINS, "modes": MODES, "underlyings": {}}
    for key in KEYS:
        T = TV[key]
        rec = {"name": T["name"], "cells": {}}
        trig = {}
        for mode in MODES:
            for win in WINS:
                st, mom, lights, tier = build_signals(T, win, mode)
                w = weights_opt(T, st, mom)
                cell = {}
                for rng, since in SINCE.items():
                    s_, s0 = evaluate(T, w, since)
                    if rng == "924":
                        trig[f"{mode}{win}"] = [i for i in range(s0, len(tier)) if tier[i] >= 1]
                        cell["trig_days"] = len(trig[f"{mode}{win}"])
                        cell["lit_days"] = sum(1 for i in range(s0, len(lights)) if lights[i] >= 1)
                    cell[rng] = s_
                rec["cells"][f"{mode}{win}"] = cell
        base = trig["pct63"]
        for k, v in trig.items():
            rec["cells"][k]["jaccard_vs_base"] = 1.0 if k == "pct63" else jaccard(base, v)
        out["underlyings"][key] = rec

        print(f"\n══ {T['name']} ══  (924后, vs 等均仓恒定)")
        print(f"{'口径':<10}{'年化%':>8}{'夏普':>7}{'回撤%':>8}{'均仓%':>7}{'超额%':>8}{'IR':>7}{'触发日':>7}{'Jac':>7}")
        for k, c in rec["cells"].items():
            a = c["924"]
            print(f"{k:<10}{a['ann']:>8}{a['sharpe']:>7}{a['mdd']:>8}{a['meanw']:>7}{a['excess']:>+8}"
                  f"{a['ir']:>7}{c['trig_days']:>7}{c['jaccard_vs_base']:>7}")

    (PROC / "calib_lab.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("\n写出 calib_lab.json")


if __name__ == "__main__":
    main()
