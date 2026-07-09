"""消融实验: 期权"恐慌灯"相对纯"跌幅"是否有增量?

回答用户质疑: 先用20日回撤做闸门后, 期权数据(五灯)还有没有独立贡献?
方法: 用未来 h 日路径的 胜率/均值收益/盈亏比(MFE÷|MAE|) 对比几组触发:

  baseline       全样本(每天买)              —— 参照
  light_any      ≥1灯 (忽略跌幅)             —— 纯期权信号: 检验"领先性/假信号"
  light_nofall   ≥1灯 且 20日回撤 > -5%       —— "灯亮但没怎么跌"的日子, 若无用则印证同步性
  fall           20日回撤 <= -5% (忽略灯)      —— 纯价格信号(基准闸门)
  fall_light     回撤<=-5% 且 ≥1灯            —— 控制跌幅后"有灯"
  fall_nolight   回撤<=-5% 且  0灯            —— 控制跌幅后"无灯"

核心对比 = fall_light vs fall_nolight: 同样跌到位, 期权是否带来额外 edge。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
HI, LO, H = 0.90, 0.10, 20


def lights_of(d, i):
    k = 0
    for key, hi in [("iv_pct", True), ("sent_pct", True),
                    ("rr_pct", False), ("slope_pct", False), ("vrp_pct", False)]:
        v = d[key][i]
        if v is None:
            continue
        if (hi and v >= HI) or (not hi and v <= LO):
            k += 1
    return k


def path(px, i):
    if i + H >= len(px) or px[i] is None:
        return None
    base = px[i]
    fwd = [px[j] / base - 1 for j in range(i + 1, i + 1 + H) if px[j] is not None]
    if len(fwd) < H // 2:
        return None
    return (px[i + H] / base - 1, max(fwd), min(fwd))


def agg(px, idxs):
    rows = [path(px, i) for i in idxs]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    a = np.array(rows)
    ret, mfe, mae = a[:, 0], a[:, 1], a[:, 2]
    amfe, amae = mfe.mean(), mae.mean()
    return dict(n=len(rows), hit=round((ret > 0).mean(), 3),
                ret=round(ret.mean() * 100, 2), med=round(np.median(ret) * 100, 2),
                mfe=round(amfe * 100, 2), mae=round(amae * 100, 2),
                payoff=round(amfe / abs(amae), 2) if amae else np.nan)


def run(d, since=None):
    n = len(d["dates"])
    px = d["price"]
    dd = d["dd20"]
    start = 0
    if since:
        while start < n and d["dates"][start] < since:
            start += 1
    L = [lights_of(d, i) for i in range(n)]
    idx = range(start, n)
    groups = {
        "baseline":     [i for i in idx],
        "light_any":    [i for i in idx if L[i] >= 1],
        "light_nofall": [i for i in idx if L[i] >= 1 and dd[i] is not None and dd[i] > -0.05],
        "fall":         [i for i in idx if dd[i] is not None and dd[i] <= -0.05],
        "fall_light":   [i for i in idx if dd[i] is not None and dd[i] <= -0.05 and L[i] >= 1],
        "fall_nolight": [i for i in idx if dd[i] is not None and dd[i] <= -0.05 and L[i] == 0],
        "fall_2l":      [i for i in idx if dd[i] is not None and dd[i] <= -0.08 and L[i] >= 2],
    }
    return {g: agg(px, ii) for g, ii in groups.items()}


def show(title, res):
    print(f"\n===== {title} =====")
    print(f"{'组':<14}{'n':>5}{'胜率':>7}{'均值%':>8}{'中位%':>8}{'MFE%':>7}{'MAE%':>8}{'盈亏比':>7}")
    order = ["baseline", "light_any", "light_nofall", "fall",
             "fall_nolight", "fall_light", "fall_2l"]
    for g in order:
        r = res.get(g)
        if not r:
            print(f"{g:<14}  (无样本)"); continue
        print(f"{g:<14}{r['n']:>5}{r['hit']*100:>6.1f}%{r['ret']:>8}{r['med']:>8}"
              f"{r['mfe']:>7}{r['mae']:>8}{r['payoff']:>7}")


for key, d in DATA.items():
    show(f"{d['name']} · 全样本({d['dates'][0]}~)  未来{H}日", run(d))
show(f"沪深300 · 仅924后", run(DATA["hs300"], since="20240924"))
show(f"中证1000 · 仅924后", run(DATA["zz1000"], since="20240924"))
