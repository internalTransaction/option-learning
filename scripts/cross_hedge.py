"""用其他品种代理对冲科创50(科创无股指期货)。
先看相关性/beta, 再模拟"科创多头 + beta中性做空代理品种"在科创过热后的对冲效果。
代理候选: 沪深300(IF期货/300ETF期权)、中证1000(IM期货)、创业板(159915期权)。
"""
from __future__ import annotations
import json
import numpy as np

TV = json.load(open(__file__.rsplit("/", 2)[0]+"/data/processed/timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
NAME = {k: TV[k]["name"] for k in KEYS}

# 对齐共同交易日
common = set(TV["kc50"]["dates"])
for k in KEYS: common &= set(TV[k]["dates"])
common = sorted(common)
idx = {k: {d: i for i, d in enumerate(TV[k]["dates"])} for k in KEYS}
ret = {k: np.array([TV[k]["price"][idx[k][d]] for d in common]) for k in KEYS}
ret = {k: np.diff(np.log(v)) for k, v in ret.items()}          # 对数日收益(对齐)
cdates = common[1:]

print("== 科创50 与各品种 日收益相关性 / beta(科创对该品种) ==")
for k in ["hs300", "zz1000", "cyb"]:
    r = np.corrcoef(ret["kc50"], ret[k])[0, 1]
    beta = np.cov(ret["kc50"], ret[k])[0, 1]/np.var(ret[k])
    print(f"  科创50 ~ {NAME[k]:<8} 相关 {r:.2f}   beta {beta:.2f}")

# 过热(科创)日 → 未来20日: 裸科创 vs beta中性对冲组合
def is_hot(i):
    kc = TV["kc50"]; d = cdates[i]; j = idx["kc50"][d]
    mp, iv, vp = kc["mom_pct"][j], kc["iv_pct"][j], kc["vrp_pct"][j]
    return mp is not None and iv is not None and vp is not None and mp >= .85 and iv >= .75 and vp >= .75

H = 20
betas = {k: np.cov(ret["kc50"], ret[k])[0, 1]/np.var(ret[k]) for k in ["hs300", "zz1000", "cyb"]}
hot = [i for i in range(len(cdates)-H) if is_hot(i)]
print(f"\n== 科创过热日({len(hot)}个) 未来{H}日: 裸多头 vs 代理对冲(beta中性做空) ==")
def summ(paths):
    a = np.array(paths)
    cum = a.sum(axis=1)                      # 20日累计收益
    mdd = np.array([(np.minimum.accumulate(np.cumsum(p)*0+np.maximum.accumulate(np.cumsum(p)))-np.cumsum(p)).max() for p in a])
    dn = np.array([np.cumsum(p).min() for p in a])   # 路径内最大回撤(相对入场)
    return cum.mean()*100, (cum > 0).mean()*100, dn.mean()*100, cum.std()*100

kc_paths = [ret["kc50"][i+1:i+1+H] for i in hot]
m, w, dn, sd = summ(kc_paths)
print(f"  裸科创多头       20日均值{m:>6.1f}%  胜率{w:>5.1f}%  路径最深{dn:>6.1f}%  波动{sd:>5.1f}%")
for k in ["hs300", "zz1000", "cyb"]:
    b = betas[k]
    hp = [ret["kc50"][i+1:i+1+H] - b*ret[k][i+1:i+1+H] for i in hot]
    m, w, dn, sd = summ(hp)
    print(f"  +做空{NAME[k]:<6}(β={b:.2f}) 20日均值{m:>6.1f}%  胜率{w:>5.1f}%  路径最深{dn:>6.1f}%  波动{sd:>5.1f}%")
