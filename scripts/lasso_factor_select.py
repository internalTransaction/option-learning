"""用 Lasso(L1) 筛选五盏灯 —— 目的是降维/剔因子, 不是提性能。

因子统一成"恐慌度"(正=越恐慌, 预期正贡献):
  f_iv=iv_pct  f_sent=sent_pct  f_rr=1-rr_pct  f_slope=1-slope_pct  f_vrp=1-vrp_pct
标签: 未来20日收益。主分析在"已跌"子样本(mom_pct≤.30, 灯的用武之地), 附全样本对照。
每标的内部标准化 X,y 后 pool(去标的间尺度差异)。LassoCV 用 TimeSeriesSplit(禁随机CV)。
同时报单因子 Spearman IC, 以区分"本身无用" vs "共线被代表"。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(open(ROOT / "data/processed/timing_viz.json"))
NAMES = ["高IV", "情绪比", "RR", "斜率", "VRP"]
H = 20


def rows_of(d, sub=True):
    n = len(d["dates"]); px = d["price"]
    X, y = [], []
    for i in range(n):
        if i + H >= n or px[i] is None or px[i + H] is None: continue
        mp = d["mom_pct"][i]
        if mp is None: continue
        if sub and mp > 0.30: continue
        vals = [d["iv_pct"][i], d["sent_pct"][i], d["rr_pct"][i], d["slope_pct"][i], d["vrp_pct"][i]]
        if any(v is None for v in vals): continue
        iv, se, rr, sl, vp = vals
        X.append([iv, se, 1 - rr, 1 - sl, 1 - vp])
        y.append(px[i + H] / px[i] - 1)
    return np.array(X), np.array(y)


def zpool(sub=True):
    """每标的内部标准化后 pool。"""
    XS, YS, per = [], [], {}
    for k, d in DATA.items():
        X, y = rows_of(d, sub)
        if len(y) < 30: continue
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
        yz = (y - y.mean()) / (y.std() + 1e-9)
        XS.append(Xz); YS.append(yz); per[k] = (d["name"], X, y)
    return np.vstack(XS), np.concatenate(YS), per


def lasso_fit(X, y):
    m = LassoCV(cv=TimeSeriesSplit(5), n_alphas=100, max_iter=50000, random_state=0).fit(X, y)
    return m


for sub in [True, False]:
    tag = "已跌子样本(mom_pct≤.30)" if sub else "全样本"
    X, y, per = zpool(sub)
    print(f"\n########## {tag}  · pooled n={len(y)} ##########")

    # 1. 因子间相关(证共线)
    print("\n[因子相关矩阵]  " + " ".join(f"{n:>6}" for n in NAMES))
    C = np.corrcoef(X.T)
    for i, n in enumerate(NAMES):
        print(f"  {n:<5} " + " ".join(f"{C[i, j]:>6.2f}" for j in range(5)))

    # 2. 单因子 Spearman IC (与未来收益)
    print("\n[单因子 IC (Spearman, 越正=越恐慌越涨)]")
    for i, n in enumerate(NAMES):
        ic, _ = spearmanr(X[:, i], y)
        print(f"  {n:<6} IC={ic:+.3f}")

    # 3. Lasso 联合筛选
    m = lasso_fit(X, y)
    print(f"\n[LassoCV 系数]  alpha={m.alpha_:.4f}")
    for i, n in enumerate(NAMES):
        mark = "  ← 剔除" if abs(m.coef_[i]) < 1e-4 else ("  ★保留" if abs(m.coef_[i]) >= 0.02 else "")
        print(f"  {n:<6} coef={m.coef_[i]:+.4f}{mark}")

    if sub:
        # 4. 分标的一致性
        print("\n[分标的 Lasso 系数 (一致性检查)]")
        print("  标的        " + " ".join(f"{n:>7}" for n in NAMES))
        for k, (name, X0, y0) in per.items():
            Xz = (X0 - X0.mean(0)) / (X0.std(0) + 1e-9)
            yz = (y0 - y0.mean()) / (y0.std() + 1e-9)
            mm = lasso_fit(Xz, yz)
            print(f"  {name:<10} " + " ".join(f"{c:>7.3f}" for c in mm.coef_))
