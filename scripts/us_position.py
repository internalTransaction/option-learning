"""美股期权择时 · 波动率制度仓位管理(逃顶=按波动降仓, 抄底=capitulation加仓)。

不再靠价格动量逃顶(已证伪), 而是**按波动率制度调仓**:
  - 降仓侧(逃顶替代): ATM IV(前瞻)越高→仓位越低。vol-target: w = TARGET/IV。
    高波动制度自动降敞口, 不需要预测顶。
  - 加仓侧(抄底): panic×跌幅门 capitulation 时叠加 ADD, 持有 H 日(骑反弹)。
    这一步刚好在 IV 见顶(vol-target 压最低)时把敞口补回来, 修正"卖在地板"。

无前视: w_t 由 t 日及之前信息决定, 吃 t+1 日收益 (strat = w.shift(1)*ret)。
公平比较: 夏普与标的无关(scale invariant); 另给"波动对齐"(把策略缩放到与买入持有
同年化波动)后的总收益/最大回撤, 回答"同样的风险, 回撤更小、收益更高吗?"。

用法: python -m scripts.us_position SPY 2024-08-01 2026-07-23
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.us_backtest import prep
from src.data import cache

ANN = 252
WLO, WHI = 0.30, 1.30      # vol-target 基仓上下限
ADD, H = 0.40, 20          # capitulation 叠加仓位 / 持有天数
PANIC, GATE = 0.60, 0.15
WMAX = 1.50                # 总仓上限


def metrics(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(ANN)
    return {"ret": eq.iloc[-1] - 1, "cagr": eq.iloc[-1] ** (ANN / len(r)) - 1,
            "vol": vol, "sharpe": (r.mean() * ANN) / vol if vol > 0 else np.nan,
            "dd": (eq / eq.cummax() - 1).min()}


def vol_matched(ret: pd.Series, target_vol: float) -> dict:
    """把策略日收益整体缩放到 target_vol 年化波动后再算指标(同风险对比)。"""
    v = ret.std() * np.sqrt(ANN)
    k = target_vol / v if v > 0 else 1.0
    return metrics(ret * k)


def capit_hold(d: pd.DataFrame) -> pd.Series:
    fired = (d["panic"] >= PANIC) & (d["mom_pct"] <= GATE)
    return fired.astype(float).rolling(H, min_periods=1).max().fillna(0.0)


def positions(d: pd.DataFrame, kind: str) -> pd.Series:
    iv = d["atm_iv"]
    target = iv.median()
    w_vt = (target / iv).clip(WLO, WHI)          # 波动率目标基仓(高IV→低仓)
    add = ADD * capit_hold(d)
    if kind == "voltarget":
        w = w_vt
    elif kind == "voltarget_add":
        w = w_vt + add
    elif kind == "add_only":                      # 上一版: 满仓 + 抄底叠加
        w = 1.0 + add
    else:
        w = pd.Series(1.0, index=d.index)
    return w.clip(0, WMAX)


def line(name: str, m: dict, vm: dict, expo: float) -> str:
    if not m:
        return f"  {name:26s} (样本不足)"
    return (f"  {name:26s} 夏普 {m['sharpe']:+.2f}  年化 {m['cagr']*100:+6.1f}%  "
            f"回撤 {m['dd']*100:6.1f}%  仓位 {expo:.2f} │ 同波动: 收益 {vm['ret']*100:+6.1f}% "
            f"回撤 {vm['dd']*100:6.1f}%")


def run(key: str, start: str, end: str) -> None:
    name = f"us_surface_{key}_day_{start}_{end}"
    if not cache.exists(name, "processed"):
        sys.exit(f"缺少 {name}")
    d = prep(cache.load(name, "processed"))
    ret = d["price"].pct_change()
    bh = metrics(ret)
    tv = bh["vol"]                                # 以买入持有波动为对齐目标
    print(f"\n{'='*104}\n{key.upper()}  {d['date'].iloc[0]}~{d['date'].iloc[-1]}  {len(d)}日"
          f"  (同波动=缩放到买入持有年化波动 {tv*100:.0f}% 后比较)\n{'='*104}")
    print(line("买入持有(基准)", bh, vol_matched(ret, tv), 1.00))
    for kind, tag in [("add_only", "满仓 + 抄底叠加(上版)"),
                      ("voltarget", "波动率降仓(仅逃顶侧)"),
                      ("voltarget_add", "波动率降仓 + 抄底加仓(完整仓管)")]:
        w = positions(d, kind)
        sr = (w.shift(1) * ret).rename("s")
        print(line(tag, metrics(sr), vol_matched(sr, tv), w.mean()))


if __name__ == "__main__":
    key = (sys.argv[1] if len(sys.argv) > 1 else "SPY").lower()
    start = sys.argv[2] if len(sys.argv) > 2 else "2024-08-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-07-23"
    run(key, start, end)
