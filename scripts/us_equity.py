"""美股期权择时 · 仓位/净值回测(把事件研究的边际转成可交易的权益曲线)。

策略动机(来自事件研究): 边际是**条件性抄底加仓**——恐慌×跌幅门(或 VVIX×跌幅门)
后约 20 日有超额; 逃顶无效, 故**只在恐慌-回撤时加仓, 不做空/不主动减到基仓以下**。

规则(无前视):
  fired_t  = panic_t≥PANIC 且 跌幅分位_t≤GATE       (可选并入 VVIX 分位≥.90)
  hold_t   = 过去 H 日内出现过 fired                  (加仓持有 H 日, 骑反弹)
  仓位 w_t = BASE + ADD·hold_t                        (常态 BASE, 触发后升到 BASE+ADD)
  次日收益 = w_{t-1} · ret_t                          (t-1 收盘定仓, 吃 t 日收益)

对比基准: 买入持有(w=1)。报告 年化/波动/夏普/最大回撤/在场时间。
用法: python -m scripts.us_equity SPY 2024-08-01 2026-07-23
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

H = 20            # 加仓持有天数(边际集中在 5-20 日)
PANIC = 0.60
GATE = 0.15
ANN = 252


def metrics(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ANN)
    sharpe = (r.mean() * ANN) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"总收益": eq.iloc[-1] - 1, "年化": cagr, "波动": vol,
            "夏普": sharpe, "最大回撤": dd}


def simulate(d: pd.DataFrame, base: float, add: float, use_vvix: bool) -> pd.Series:
    panic, momp = d["panic"], d["mom_pct"]
    fired = (panic >= PANIC) & (momp <= GATE)
    if use_vvix and "vvix_pct" in d:
        fired = fired | ((d["vvix_pct"] >= 0.90) & (momp <= GATE))
    hold = fired.astype(float).rolling(H, min_periods=1).max().fillna(0.0)
    w = base + add * hold
    ret = d["price"].pct_change()
    return (w.shift(1) * ret).rename("strat")


def line(name: str, m: dict) -> str:
    if not m:
        return f"  {name:22s}  (样本不足)"
    return (f"  {name:22s}  总收益 {m['总收益']*100:+7.1f}%  年化 {m['年化']*100:+6.1f}%  "
            f"波动 {m['波动']*100:5.1f}%  夏普 {m['夏普']:+.2f}  最大回撤 {m['最大回撤']*100:6.1f}%")


def run(key: str, start: str, end: str) -> None:
    name = f"us_surface_{key}_day_{start}_{end}"
    if not cache.exists(name, "processed"):
        sys.exit(f"缺少 {name}")
    d = prep(cache.load(name, "processed"))
    bh = d["price"].pct_change()
    fired = ((d["panic"] >= PANIC) & (d["mom_pct"] <= GATE))
    print(f"\n{'='*92}\n{key.upper()}  {d['date'].iloc[0]}~{d['date'].iloc[-1]}  {len(d)}日  "
          f"触发日 {int(fired.sum())}  (加仓持有{H}日)\n{'='*92}")
    print(line("买入持有(基准)", metrics(bh)))
    # 同均仓变体(base<1, 触发升到~1.x): 择时再配置, 平均仓位≈1
    for base, add, vv, tag in [
        (1.00, 0.50, False, "杠杆叠加 1.0→1.5 · 灯×跌幅"),
        (1.00, 0.50, True,  "杠杆叠加 1.0→1.5 · +VVIX"),
        (0.85, 0.50, False, "同均仓 0.85→1.35 · 灯×跌幅"),
        (0.85, 0.50, True,  "同均仓 0.85→1.35 · +VVIX"),
    ]:
        r = simulate(d, base, add, vv)
        expo = (base + add * ((d["panic"] >= PANIC) & (d["mom_pct"] <= GATE)
                              ).astype(float).rolling(H, min_periods=1).max().fillna(0)).mean()
        print(line(tag, metrics(r)) + f"  平均仓位 {expo:.2f}")


if __name__ == "__main__":
    key = (sys.argv[1] if len(sys.argv) > 1 else "SPY").lower()
    start = sys.argv[2] if len(sys.argv) > 2 else "2024-08-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-07-23"
    run(key, start, end)
