"""冗余处理验证: 偏斜类因子重复计 3 次(情绪比/RR/斜率 相关性0.91~0.97)是否有害?

在"跌到位"(mom_pct ≤ 0.15)的日子里, 比较几种恐慌度定义对未来20日收益的区分力:
  P5   现状: mean(iv, sent, 1-rr, 1-slope, 1-vrp)      → 偏斜权重 60%
  P3eq 去重: mean(iv, skew_dim, 1-vrp), skew_dim=偏斜三者均值 → 各维 33%
  P3one 去重: mean(iv, 1-rr, 1-vrp)                     → 偏斜只留一个代表
  P2   极简: mean(iv, skew_dim)                          → 去掉VRP看它是否必要
  单因子 iv / skew_dim / 1-vrp                            → 各维单独贡献

指标: Spearman IC(恐慌度 vs 未来20日收益)、高恐慌半区 vs 低恐慌半区 的胜率/均值/盈亏比。
灯数口径: L5(现状五灯) vs L3(三维度灯: 水平/偏斜/VRP)。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

TV = json.load(open(Path(__file__).resolve().parents[1] / "data/processed/timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
H, DIP = 20, 0.15


def fear(T, i):
    """返回方向对齐后的恐慌分位 dict(越大越恐慌), 缺失返回 None"""
    g = lambda k: T[k][i]
    iv, sent, rr, sl, vp = g("iv_pct"), g("sent_pct"), g("rr_pct"), g("slope_pct"), g("vrp_pct")
    if None in (iv, sent, rr, sl, vp): return None
    return {"iv": iv, "sent": sent, "rr": 1-rr, "slope": 1-sl, "vrp": 1-vp}


def scores(f):
    skew = (f["sent"] + f["rr"] + f["slope"]) / 3
    return {
        "P5(现状)":    (f["iv"]+f["sent"]+f["rr"]+f["slope"]+f["vrp"])/5,
        "P3eq(去重)":  (f["iv"]+skew+f["vrp"])/3,
        "P3one(去重)": (f["iv"]+f["rr"]+f["vrp"])/3,
        "P2(无VRP)":   (f["iv"]+skew)/2,
        "单·高IV":      f["iv"],
        "单·偏斜":      skew,
        "单·VRP":      f["vrp"],
    }


def lights(f):
    L5 = sum(1 for k in ("iv", "sent", "rr", "slope", "vrp") if f[k] >= 0.90)
    dim_iv = f["iv"] >= 0.90
    dim_skew = sum(f[k] >= 0.90 for k in ("sent", "rr", "slope")) >= 2   # 偏斜: 三者多数
    dim_vrp = f["vrp"] >= 0.90
    return L5, int(dim_iv)+int(dim_skew)+int(dim_vrp)


def fwd(T, i):
    px = T["price"]
    if i+H >= len(px) or px[i] is None or px[i+H] is None: return None
    return px[i+H]/px[i]-1


# 收集所有标的的"跌到位"事件
rows = []
for k in KEYS:
    T = TV[k]
    for i in range(len(T["dates"])):
        mp = T["mom_pct"][i]
        if mp is None or mp > DIP: continue
        f = fear(T, i); r = fwd(T, i)
        if f is None or r is None: continue
        s = scores(f); L5, L3 = lights(f)
        rows.append({**s, "ret": r, "L5": L5, "L3": L3, "key": k})

n = len(rows)
print(f"===== 跌到位事件(mom_pct≤{DIP}) 四标的合并 n={n} · 未来{H}日 =====\n")
print(f"  {'恐慌度定义':<14}{'IC':>7}{'高半区胜率':>10}{'低半区胜率':>10}{'胜率差':>8}{'高半区均值':>10}{'盈亏比':>7}")
for name in scores(fear(TV["hs300"], 300) or {"iv":0,"sent":0,"rr":0,"slope":0,"vrp":0}):
    v = np.array([r[name] for r in rows]); ret = np.array([r["ret"] for r in rows])
    ic = spearmanr(v, ret).correlation
    med = np.median(v)
    hi, lo = ret[v > med], ret[v <= med]
    win, loss = hi[hi > 0], hi[hi <= 0]
    payoff = win.mean()/abs(loss.mean()) if len(win) and len(loss) else np.nan
    print(f"  {name:<14}{ic:>+7.3f}{(hi>0).mean()*100:>9.1f}%{(lo>0).mean()*100:>9.1f}%"
          f"{((hi>0).mean()-(lo>0).mean())*100:>+7.1f}%{hi.mean()*100:>9.2f}%{payoff:>7.2f}")

print(f"\n===== 灯数口径对比 =====")
for tag, key, mx in [("L5 现状五灯", "L5", 5), ("L3 三维度灯", "L3", 3)]:
    print(f"\n  {tag}:")
    print(f"    {'灯数':<6}{'n':>5}{'胜率':>8}{'均值':>8}{'盈亏比':>8}")
    for lv in range(mx+1):
        sel = [r["ret"] for r in rows if r[key] == lv]
        if len(sel) < 5: print(f"    {lv:<6}{len(sel):>5}   (样本不足)"); continue
        a = np.array(sel); win, loss = a[a > 0], a[a <= 0]
        po = win.mean()/abs(loss.mean()) if len(win) and len(loss) else np.nan
        print(f"    {lv:<6}{len(a):>5}{(a>0).mean()*100:>7.1f}%{a.mean()*100:>7.2f}%{po:>8.2f}")
    # 按现行阈值分档
    print(f"    -- 分档(≥1 / ≥2 / ≥{mx if mx==3 else 3}) --")
    for th in [1, 2, 3 if mx == 5 else 3]:
        if th > mx: continue
        sel = np.array([r["ret"] for r in rows if r[key] >= th])
        if len(sel) < 5: continue
        win, loss = sel[sel > 0], sel[sel <= 0]
        po = win.mean()/abs(loss.mean()) if len(win) and len(loss) else np.nan
        print(f"    ≥{th}    {len(sel):>5}{(sel>0).mean()*100:>7.1f}%{sel.mean()*100:>7.2f}%{po:>8.2f}")

print(f"\n  (全样本基准: n={n} 胜率 {(np.array([r['ret'] for r in rows])>0).mean()*100:.1f}% "
      f"均值 {np.mean([r['ret'] for r in rows])*100:.2f}%)")
