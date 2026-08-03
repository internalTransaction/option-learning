"""VVIX 思路在 A 股的代理检验(A股无VIX期权, 故用两个 vol-of-vol 代理)。

  bf_25d   蝶式 = (IV_c25+IV_p25)/2 − ATM_IV   微笑凸度(横截面 vol-of-vol), 曲面里一直算但从未进五灯
  ivvol    = std(Δln ATM_IV, 20日) × √252      隐含 vol-of-vol(IV 自身的波动)
  rvvol    = std(Δln RV20,   20日) × √252      **已实现 vol-of-vol**(只用价格, 不依赖IV)
  vvrp     = ivvol − rvvol                      vol-of-vol 风险溢价(VRP 的二阶版本)

检验:
  A. 与现有五因子的相关性 —— 是否正交的新维度(像 GEX 那样)
  B. 跌到位事件(mom_pct≤.15)上的 IC / 胜率 / 盈亏比 —— 与偏斜、高IV 对比
  C. 背离(对标美股 VVIX 用法): IV 高但 vol-of-vol 低 = 假恐慌? vs 两者齐飙 = 真恐慌?
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TV = json.load(open(ROOT/"data/processed/timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
SURF = {"hs300": "surface_300etf", "zz1000": "surface_zz1000",
        "kc50": "surface_kc50", "cyb": "surface_cyb"}
H, DIP, WIN, MINP = 20, 0.15, 63, 31
FACT = [("iv_pct", "高IV"), ("sent_pct", "情绪比"), ("rr_pct", "RR"),
        ("slope_pct", "斜率"), ("vrp_pct", "VRP")]


def roll_pct(s):
    return s.rolling(WIN, min_periods=MINP).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def latest(prefix):
    """用项目自带的 catalog(优先最长历史), 避免误选只有几行的增量文件。"""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.catalog import latest_ranged_file
    f = latest_ranged_file(ROOT/"data/processed", prefix)
    if f is None: raise FileNotFoundError(prefix)
    return f.path


def build(key):
    """返回 DataFrame: date, price, 五因子分位, bf/bf_pct, ivvol/ivvol_pct, mom_pct, fwd"""
    T = TV[key]
    d = pd.DataFrame({"date": T["dates"], "price": T["price"], "mom_pct": T["mom_pct"],
                      "atm_iv": T["atm_iv"], "rv": T["rv"], **{k: T[k] for k, _ in FACT}})
    sf = pd.read_parquet(latest(SURF[key]))[["date", "bf_25d"]]
    sf["date"] = sf["date"].astype(str)
    d = d.merge(sf, on="date", how="left")
    # 隐含 vol-of-vol: ATM IV 的对数变化波动
    d["ivvol"] = np.log(d["atm_iv"].astype(float)).diff().rolling(20, min_periods=10).std()*np.sqrt(252)
    # 已实现 vol-of-vol: RV20 的对数变化波动(只用价格)
    d["rvvol"] = np.log(d["rv"].astype(float)).diff().rolling(20, min_periods=10).std()*np.sqrt(252)
    d["vvrp"] = d["ivvol"] - d["rvvol"]            # vol-of-vol 风险溢价
    d["bf_pct"] = roll_pct(d["bf_25d"].astype(float))
    d["ivvol_pct"] = roll_pct(d["ivvol"])
    d["rvvol_pct"] = roll_pct(d["rvvol"])
    d["vvrp_pct"] = roll_pct(d["vvrp"])
    px = d["price"].astype(float)
    d["fwd"] = px.shift(-H)/px - 1
    return d


ALL = {k: build(k) for k in KEYS}

print("========== A. 相关性: vol-of-vol 代理 vs 现有因子(分位序列) ==========")
print(f"  {'标的':<9}{'':<4}" + "".join(f"{lab:>8}" for _, lab in FACT) + f"{'蝶式':>8}")
for k in KEYS:
    d = ALL[k].dropna(subset=["bf_pct", "ivvol_pct", "rvvol_pct", "vvrp_pct"] + [c for c, _ in FACT])
    for nm, col in [("蝶式", "bf_pct"), ("隐含vv", "ivvol_pct"), ("已实现vv", "rvvol_pct"), ("vvRP", "vvrp_pct")]:
        row = f"  {TV[k]['name']:<9}{nm:<4}"
        for c, _ in FACT:
            row += f"{d[col].corr(d[c]):>+8.2f}"
        row += f"{d[col].corr(d['bf_pct']):>+8.2f}"
        print(row)

print("\n========== B. 跌到位事件(mom_pct≤.15) 的预测力 · 未来20日 ==========")
pool = []
for k in KEYS:
    d = ALL[k]
    m = d[(d.mom_pct <= DIP) & d.fwd.notna() & d.bf_pct.notna() & d.ivvol_pct.notna() & d.rvvol_pct.notna()].copy()
    m["key"] = k
    pool.append(m)
P = pd.concat(pool, ignore_index=True)
skew = (P.sent_pct + (1-P.rr_pct) + (1-P.slope_pct))/3
cands = {"偏斜(基准)": skew, "高IV": P.iv_pct, "VRP恐慌": 1-P.vrp_pct,
         "蝶式凸度": P.bf_pct, "蝶式(反向)": 1-P.bf_pct,
         "隐含vol-of-vol": P.ivvol_pct, "隐含vv(反向)": 1-P.ivvol_pct,
         "已实现vol-of-vol": P.rvvol_pct, "已实现vv(反向)": 1-P.rvvol_pct,
         "vol-of-vol RP": P.vvrp_pct, "vvRP(反向)": 1-P.vvrp_pct}
print(f"  合并 n={len(P)}   全样本胜率 {(P.fwd>0).mean()*100:.1f}%  均值 {P.fwd.mean()*100:+.2f}%")
print(f"  {'因子':<20}{'IC':>8}{'高半区胜率':>10}{'低半区':>8}{'胜率差':>8}{'高半区均值':>10}{'盈亏比':>7}")
for nm, v in cands.items():
    v = v.to_numpy(); r = P.fwd.to_numpy()
    ok = np.isfinite(v) & np.isfinite(r)
    v, r = v[ok], r[ok]
    med = np.median(v); hi, lo = r[v > med], r[v <= med]
    w, l = hi[hi > 0], hi[hi <= 0]
    po = w.mean()/abs(l.mean()) if len(w) and len(l) else np.nan
    print(f"  {nm:<20}{spearmanr(v, r).correlation:>+8.3f}{(hi>0).mean()*100:>9.1f}%"
          f"{(lo>0).mean()*100:>7.1f}%{((hi>0).mean()-(lo>0).mean())*100:>+7.1f}%{hi.mean()*100:>9.2f}%{po:>7.2f}")

print("\n========== C. 背离检验(对标美股 VVIX 用法) ==========")
print("  在 IV 已高(iv_pct≥0.75)的跌到位日, 按 vol-of-vol 分组:")
for nm, col in [("蝶式凸度", "bf_pct"), ("隐含vol-of-vol", "ivvol_pct"), ("已实现vol-of-vol", "rvvol_pct")]:
    sub = P[P.iv_pct >= 0.75]
    if len(sub) < 30: print(f"   {nm}: 样本不足"); continue
    med = sub[col].median()
    print(f"\n   —— 按 {nm} 中位数({med:.2f})分组, n={len(sub)} ——")
    for tag, sel in [(f"{nm} 高(两者齐飙=真恐慌?)", sub[sub[col] > med]),
                     (f"{nm} 低(IV高但vov低=假恐慌?)", sub[sub[col] <= med])]:
        r = sel.fwd.to_numpy()
        w, l = r[r > 0], r[r <= 0]
        po = w.mean()/abs(l.mean()) if len(w) and len(l) else np.nan
        print(f"     {tag:<28} n={len(r):>3}  胜率{(r>0).mean()*100:>5.1f}%  均值{r.mean()*100:>+6.2f}%  盈亏比{po:>5.2f}")
