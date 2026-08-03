"""regime开关测试: 用理论先行的简单趋势规则(非拟合分类器)门控抄底进场。
核心诚实性检验: 单一regime转折(924)无法统计验证分类器, 故只测"事先就该成立"的趋势规则,
并看它是否(a)自然剔除pre-924失血、(b)仍保留部分pre-924赢家(=真趋势, 非924代理)。
"""
from __future__ import annotations
import importlib.util as u
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = u.spec_from_file_location("cs", ROOT / "scripts/call_spread_entry.py")
cs = u.module_from_spec(spec); spec.loader.exec_module(cs)

COMMON = dict(min_panic=0.70, entry_lag=1, entry_px="open", slip=0.01, theta=0.18,
              target_dte=30, exit_rule="dte_left", exit_level=12, maxhold=30, m_long=0.0, width=0.08)
UNDERS = [("zz1000", "中证1000"), ("zz500", "中证500"), ("cyb", "创业板"),
          ("kc50", "科创50"), ("hs300", "沪深300")]


def ma(price, w):
    p = np.array([np.nan if x is None else x for x in price], float)
    out = np.full(len(p), np.nan)
    for i in range(len(p)):
        seg = p[max(0, i - w + 1):i + 1]
        if np.isfinite(seg).sum() >= max(5, w // 2):
            out[i] = np.nanmean(seg)
    return out


def regime(name, price):
    p = np.array([np.nan if x is None else x for x in price], float)
    if name == "none":
        return [True] * len(p)
    if name == "px>MA200":
        m = ma(price, 200); return list(p > m)
    if name == "MA200升":
        m = ma(price, 200)
        sl = np.full(len(p), np.nan); sl[20:] = m[20:] - m[:-20]
        return list(sl > 0)
    if name == "MA200降":                       # 严格下行(排除warmup的nan)
        m = ma(price, 200)
        sl = np.full(len(p), np.nan); sl[20:] = m[20:] - m[:-20]
        return list(sl < 0)
    if name == "MA50>MA200":
        return list(ma(price, 50) > ma(price, 200))
    if name == "px>MA120":
        return list(p > ma(price, 120))
    return [True] * len(p)


def stat(rets):
    r = np.array(rets)
    if len(r) == 0:
        return None
    eq = np.cumprod(1 + 0.15 * r)
    return len(r), (r > 0).mean(), np.median(r), r.mean(), eq[-1]


def main():
    data = {}
    for ukey, _ in UNDERS:
        key, prefix = cs.UNIV[ukey]
        d, p, b, pa = cs.load_signal(key)
        chain, px, po = cs.load_chain(prefix)
        data[ukey] = (d, p, b, pa, chain, px, po)

    print("价差 剩12DTE  各regime门控下的【五标的合并】(全样本)")
    print(f"  {'regime规则':<14}{'笔数':>5}{'胜率':>6}{'中位':>7}{'均值':>7}{'终值x':>7}{'│pre924':>9}{'post924':>8}")
    print("  " + "-" * 62)
    for rname in ["none", "px>MA200", "MA200升", "MA50>MA200", "px>MA120"]:
        pool, pre, post = [], 0, 0
        for ukey, _ in UNDERS:
            d, p, b, pa, chain, px, po = data[ukey]
            mask = regime(rname, p)
            tr = cs.simulate(d, p, b, chain, px, po, **COMMON, panic_arr=pa, regime_mask=mask)
            for t in tr:
                pool.append(t["ret"])
                if t["entry_date"] >= "20240924":
                    post += 1
                else:
                    pre += 1
        s = stat(pool)
        print(f"  {rname:<14}{s[0]:>5}{s[1]*100:>5.0f}%{s[2]*100:>+6.0f}%{s[3]*100:>+6.0f}%{s[4]:>6.2f}"
              f"{pre:>7}笔{post:>6}笔")

    print("\n按标的看最稳的一条(MA200升) vs 无门控:")
    print(f"  {'标的':<9}{'无门控(笔/中位/终值)':<24}{'MA200升(笔/中位/终值)':<24}")
    for ukey, un in UNDERS:
        d, p, b, pa, chain, px, po = data[ukey]
        line = f"  {un:<9}"
        for rname in ["none", "MA200升"]:
            mask = regime(rname, p)
            tr = cs.simulate(d, p, b, chain, px, po, **COMMON, panic_arr=pa, regime_mask=mask)
            s = stat([t["ret"] for t in tr])
            line += (f"{s[0]:>2}笔 中{s[2]*100:>+4.0f}% {s[4]:.2f}x      " if s else "  无            ")
        print(line)


if __name__ == "__main__":
    main()
