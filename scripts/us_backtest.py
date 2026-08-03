"""美股期权择时信号 · 极值反转事件研究(回测)。

方法(对齐用户口径, 见 memory: discretionary-extreme-lens / options-signal-is-conditional-alpha):
  不看全样本 IC, 而看**极值处**的前瞻收益: 触发次数、胜率、盈亏比、相对无条件基线的超额。
  分别测: 单因子极值 / 恐慌灯计数 / 恐慌×跌幅门(条件alpha) / melt-up 逃顶。

因子(63日滚动分位):
  atm_iv↑  sent(iv_ratio_25d)↑  rr↓  slope↓  vrp↓  pcr↑  ts_ratio(期限结构)↑
  mom20 及其 252日分位(跌幅门)。恐慌五灯 = A股同口径(iv/sent高 + rr/slope/vrp低)。

用法:
    python -m scripts.us_backtest SPY 2022-01-03 2026-07-23
    python -m scripts.us_backtest SOXX 2022-01-03 2026-07-23
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import cache
from src.utils.logger import get_logger

log = get_logger("us_backtest")

WIN, MINP = 63, 31          # 分位窗口(与信号台一致)
RV_WIN = 20
HORIZONS = [5, 10, 20, 60]
HI, LO = 0.90, 0.10


def roll_pct(s: pd.Series, win=WIN, minp=MINP) -> pd.Series:
    """滚动分位, NaN 稳健(窗口内剔除 NaN 后计算; 当日为 NaN 则返回 NaN)。"""
    def f(x):
        last = x[-1]
        if np.isnan(last):
            return np.nan
        v = x[~np.isnan(x)]
        return (v <= last).mean() if len(v) else np.nan
    return s.rolling(win, min_periods=minp).apply(f, raw=True)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    d["price"] = d["spot"]
    logret = np.log(d["price"] / d["price"].shift(1))
    d["rv"] = logret.rolling(RV_WIN, min_periods=10).std() * np.sqrt(252)
    d["vrp"] = d["atm_iv"] - d["rv"]
    d["sent"] = d["iv_ratio_25d"]
    d["rr"] = d["rr_25d"]
    d["slope"] = d["smile_slope"]
    d["pcr"] = d["pcr_vol"]
    d["ts"] = d.get("ts_ratio")
    d["mom20"] = d["price"] / d["price"].shift(20) - 1.0

    # --- 补充因子(美股常用, A股较少见) ---
    ivchg = np.log(d["atm_iv"] / d["atm_iv"].shift(1))
    d["vvix"] = ivchg.rolling(10, min_periods=5).std() * np.sqrt(252)     # 波动的波动(VVIX 代理)
    d["bf"] = d.get("bf_25d")                                             # 25Δ蝶式 = 微笑凸度(尾部定价)
    d["ivmom"] = d["atm_iv"] / d["atm_iv"].shift(5) - 1.0                 # IV 5日变化率(vol spike)
    neg = np.minimum(logret, 0.0)
    d["drv"] = np.sqrt((neg ** 2).rolling(RV_WIN, min_periods=10).mean() * 252)  # 下行已实现半波动
    d["dvrp"] = d["atm_iv"] - d["drv"]                                    # 下行(bad)VRP
    d["skew"] = d["skew_cboe"] if "skew_cboe" in d.columns else np.nan    # model-free SKEW(CBOE口径)

    d["iv_pct"] = roll_pct(d["atm_iv"])
    d["sent_pct"] = roll_pct(d["sent"])
    d["rr_pct"] = roll_pct(d["rr"])
    d["slope_pct"] = roll_pct(d["slope"])
    d["vrp_pct"] = roll_pct(d["vrp"])
    d["pcr_pct"] = roll_pct(d["pcr"])
    d["ts_pct"] = roll_pct(d["ts"]) if d["ts"] is not None else np.nan
    d["vvix_pct"] = roll_pct(d["vvix"])
    d["bf_pct"] = roll_pct(d["bf"]) if d["bf"] is not None else np.nan
    d["ivmom_pct"] = roll_pct(d["ivmom"])
    d["dvrp_pct"] = roll_pct(d["dvrp"])
    d["skew_pct"] = roll_pct(d["skew"]) if "skew_cboe" in d.columns else pd.Series(np.nan, index=d.index)
    d["mom_pct"] = roll_pct(d["mom20"], 252, 126)

    # 恐慌五灯(A股同口径) + 连续恐慌度
    d["lights"] = (
        (d["iv_pct"] >= HI).astype(int) + (d["sent_pct"] >= HI).astype(int)
        + (d["rr_pct"] <= LO).astype(int) + (d["slope_pct"] <= LO).astype(int)
        + (d["vrp_pct"] <= LO).astype(int))
    d["panic"] = np.nanmean(np.vstack([
        d["iv_pct"], d["sent_pct"], 1 - d["rr_pct"], 1 - d["slope_pct"], 1 - d["vrp_pct"]]), axis=0)

    for h in HORIZONS:
        d[f"fwd{h}"] = d["price"].shift(-h) / d["price"] - 1.0
    return d


def stats(sub: pd.Series) -> dict:
    s = sub.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "win": np.nan, "payoff": np.nan}
    win = (s > 0).mean()
    up = s[s > 0].mean() if (s > 0).any() else 0.0
    dn = -s[s < 0].mean() if (s < 0).any() else np.nan
    payoff = up / dn if dn and dn > 0 else np.nan
    return {"n": len(s), "mean": s.mean(), "win": win, "payoff": payoff}


def event(d: pd.DataFrame, mask: pd.Series, label: str) -> list[dict]:
    rows = []
    for h in HORIZONS:
        st = stats(d.loc[mask, f"fwd{h}"])
        base = stats(d[f"fwd{h}"])
        rows.append({"signal": label, "h": h, **st,
                     "base_mean": base["mean"], "base_win": base["win"],
                     "excess": (st["mean"] - base["mean"]) if np.isfinite(st["mean"]) else np.nan})
    return rows


def fmt_block(rows: list[dict], title: str) -> str:
    out = [f"\n【{title}】"]
    out.append(f"  {'因子/条件':28s} {'H':>3s} {'次数':>5s} {'均值':>8s} {'胜率':>6s} "
               f"{'盈亏比':>6s} {'基线均值':>8s} {'超额':>8s}")
    last = None
    for r in rows:
        sig = r["signal"] if r["signal"] != last else ""
        last = r["signal"]
        mean = f"{r['mean']*100:+.2f}%" if np.isfinite(r["mean"]) else "  -  "
        win = f"{r['win']*100:.0f}%" if np.isfinite(r["win"]) else " - "
        pay = f"{r['payoff']:.2f}" if np.isfinite(r["payoff"]) else " - "
        bm = f"{r['base_mean']*100:+.2f}%" if np.isfinite(r["base_mean"]) else "  -  "
        ex = f"{r['excess']*100:+.2f}%" if np.isfinite(r["excess"]) else "  -  "
        out.append(f"  {sig:28s} {r['h']:>3d} {r['n']:>5d} {mean:>8s} {win:>6s} "
                   f"{pay:>6s} {bm:>8s} {ex:>8s}")
    return "\n".join(out)


def run(key: str, start: str, end: str) -> None:
    name = f"us_surface_{key}_day_{start}_{end}"
    if not cache.exists(name, "processed"):
        sys.exit(f"缺少曲面缓存 {name}, 先跑 build_us_surface")
    d = prep(cache.load(name, "processed"))
    print(f"\n{'='*88}\n{key.upper()}  {d['date'].iloc[0]}~{d['date'].iloc[-1]}  "
          f"{len(d)} 交易日  (基线: 无条件前瞻收益)\n{'='*88}")

    # A. 单因子极值(恐慌方向)
    single = []
    single += event(d, d["iv_pct"] >= HI, "高IV 分位≥.90")
    single += event(d, d["sent_pct"] >= HI, "情绪比 分位≥.90")
    single += event(d, d["rr_pct"] <= LO, "RR 分位≤.10")
    single += event(d, d["slope_pct"] <= LO, "微笑斜率 分位≤.10")
    single += event(d, d["vrp_pct"] <= LO, "VRP 分位≤.10")
    single += event(d, d["pcr_pct"] >= HI, "PCR 分位≥.90")
    if d["ts_pct"].notna().any():
        single += event(d, d["ts_pct"] >= HI, "期限结构 分位≥.90(backwardation)")
    print(fmt_block(single, "A. 单因子极值 → 前瞻收益(测有无领先性)"))

    # B. 恐慌灯计数 & 连续恐慌度
    lights_rows = []
    for k in (1, 2, 3):
        lights_rows += event(d, d["lights"] >= k, f"恐慌灯 ≥{k}")
    lights_rows += event(d, d["panic"] >= 0.70, "连续恐慌度 ≥.70")
    print(fmt_block(lights_rows, "B. 恐慌灯计数 / 连续恐慌度"))

    # C. 恐慌 × 跌幅门(条件 alpha: 跌幅=触发器, 期权=质量过滤)
    cond = []
    cond += event(d, d["mom_pct"] <= 0.15, "仅跌幅门 分位≤.15")
    cond += event(d, d["mom_pct"] <= 0.08, "仅跌幅门 分位≤.08")
    cond += event(d, (d["lights"] >= 2) & (d["mom_pct"] <= 0.15), "灯≥2 且 跌幅≤.15")
    cond += event(d, (d["lights"] >= 2) & (d["mom_pct"] <= 0.08), "灯≥2 且 跌幅≤.08")
    cond += event(d, (d["panic"] >= 0.70) & (d["mom_pct"] <= 0.15), "恐慌≥.70 且 跌幅≤.15")
    print(fmt_block(cond, "C. 抄底: 恐慌×跌幅门(对比仅跌幅门, 看期权是否加分)"))

    # D. melt-up 逃顶(测对称镜像是否失效)
    top = []
    top += event(d, d["mom_pct"] >= 0.85, "仅高动量 分位≥.85")
    top += event(d, (d["mom_pct"] >= 0.85) & (d["iv_pct"] >= 0.75) & (d["vrp_pct"] >= 0.75),
                 "melt-up: 动量≥.85 & IV≥.75 & VRP≥.75")
    top += event(d, (d["iv_pct"] <= LO) & (d["mom_pct"] >= 0.85), "低IV高位(自满): IV≤.10 & 动量≥.85")
    print(fmt_block(top, "D. 逃顶/melt-up → 前瞻收益(负=下跌; 测逃顶信号是否有效)"))

    # E. 补充因子(美股常用): VVIX代理 / 凸度 / IV动量 / 下行VRP
    ext = []
    ext += event(d, d["vvix_pct"] >= HI, "波动的波动(VVIX代理) 分位≥.90")
    if d["bf_pct"].notna().any():
        ext += event(d, d["bf_pct"] >= HI, "蝶式/凸度(尾部) 分位≥.90")
    ext += event(d, d["ivmom_pct"] >= HI, "IV 5日急升 分位≥.90")
    ext += event(d, d["dvrp_pct"] <= LO, "下行VRP(bad) 分位≤.10")
    ext += event(d, (d["dvrp_pct"] <= LO) & (d["mom_pct"] <= 0.15), "下行VRP≤.10 且 跌幅≤.15")
    ext += event(d, (d["vvix_pct"] >= HI) & (d["mom_pct"] <= 0.15), "VVIX≥.90 且 跌幅≤.15")
    if d["skew_pct"].notna().any():
        ext += event(d, d["skew_pct"] >= HI, "model-free SKEW 分位≥.90(尾部定价高)")
        ext += event(d, d["skew_pct"] <= LO, "model-free SKEW 分位≤.10(尾部无恐惧)")
        ext += event(d, (d["skew_pct"] >= HI) & (d["mom_pct"] <= 0.15), "SKEW≥.90 且 跌幅≤.15")
    print(fmt_block(ext, "E. 补充因子(VVIX / 凸度 / IV动量 / 下行VRP / model-free SKEW)"))


if __name__ == "__main__":
    key = (sys.argv[1] if len(sys.argv) > 1 else "SPY").lower()
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-03"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-07-23"
    run(key, start, end)
