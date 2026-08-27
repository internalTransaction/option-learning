"""独立出场机 — 修「一建仓就平仓」的结构病。

诊断: 现状(build_equity.weights_opt)把出场寄生在入场信号(dip)的衰减上,
而 dip = panic × clamp((0.30-mom_pct)/0.30) 在抄底奏效(动量回升)时必然归零,
于是刚赚钱就减仓。补丁(线性棘轮 DECAY=4%/天)是路径盲的时间衰减,不合理。

本模块把入场与出场彻底解耦成两套独立机制:
  入场(沿用期权因子擅长的): dip 抄底分档 + OVERLAY 趋势打底(留参与)。
  出场(全新, 与入场信号无关, 纯价格+波动率驱动):
    1. σ 缩放跟踪止损 —— 距离 = k_ts × 日波动(rv/√252) × √H_ts,
       自动对科创放宽、对沪深300收紧, 解决固定 5% 对高波动标的是噪声的问题。
    2. HVWMA 趋势闸 —— 绿则持有骑趋势, 红转破位才释放(dirmap 可插分钟级)。
    3. time-stop —— 进场后 N 日无进展的死钱仓位强制释放, 省机会成本。
  风控总闸: 指数 MA200 regime, off 时封顶降仓(vault 已验证 edge 多在此)。

核心特性: 持有期权重只**棘轮上行**(max(prev, want)), 绝不线性下滑;
仓位下台阶只由**显式出场触发**驱动。这是与现状最本质的区别。

回测纪律: 权重由 ≤ i 日信息生成, 施加于 i→i+1 收益, 无前视。
评估用 expanding-window walk-forward, 同时报 IS/OOS, 并与现状 weights_opt 对比,
直面「参数只对 924 后过拟合」的批评。
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TV = json.load(open(PROC / "timing_viz.json"))
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
FEE = 0.0005          # 单边换手成本 5bps


def clamp(x, a, b):
    return max(a, min(b, x))


# ── 入场信号(沿用现状 dip): panic × 动量跌幅门 × GEX ─────────────────────
TH, FULL, GK, GLO, GHI = 0.15, 0.45, 0.4, 0.4, 1.5


def dip_arr(T):
    """每日抄底信念 0..1。等价 build_equity.dip_arr, 保留期权因子的入场强项。"""
    n = len(T["dates"]); mp = T["mom_pct"]; gz = T.get("gex_z"); dip = [0.0] * n
    for i in range(n):
        parts = [(v if hi else 1 - v)
                 for k, hi in [("iv_pct", 1), ("sent_pct", 1), ("rr_pct", 0),
                               ("slope_pct", 0), ("vrp_pct", 0)]
                 for v in [T[k][i]] if v is not None]
        if parts and mp[i] is not None:
            b = (sum(parts) / len(parts)) * clamp((0.30 - mp[i]) / 0.30, 0, 1)
            if gz and gz[i] is not None:
                b *= clamp(1 - GK * gz[i], GLO, GHI)
            dip[i] = clamp((b - TH) / (FULL - TH), 0, 1)
    return dip


def hot_arr(T):
    """melt-up 过热(见顶兑现用)。"""
    n = len(T["dates"]); mp, iv, vp = T["mom_pct"], T["iv_pct"], T["vrp_pct"]
    return [bool(mp[i] is not None and iv[i] is not None and vp[i] is not None
                 and mp[i] >= .85 and iv[i] >= .75 and vp[i] >= .75) for i in range(n)]


# ── 价格派生: 日波动 σ、MA200 regime、日线 HVWMA 趋势(分钟级不可用时的回退)──
def daily_sigma(T):
    """日收益波动 = rv(20日已实现,年化) / √252。缺失回退用价格滚动 std。"""
    rv = T["rv"]; px = T["price"]; n = len(px)
    sig = [None] * n
    for i in range(n):
        if rv[i] is not None and rv[i] > 0:
            sig[i] = rv[i] / math.sqrt(252)
    # 回退填充
    fallback = 0.015
    last = None
    for i in range(n):
        if sig[i] is None:
            sig[i] = last if last is not None else fallback
        else:
            last = sig[i]
    return sig


def ma_regime(T, win=200):
    """price >= MA(win) 视为 risk-on。ETF 收盘 ≈ 指数, 作 regime 代理。"""
    px = np.array([p if p is not None else np.nan for p in T["price"]], float)
    n = len(px); reg = [True] * n
    s = np.copy(px)
    for i in range(n):
        lo = max(0, i - win + 1)
        w = px[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= win // 2 and not np.isnan(px[i]):
            reg[i] = px[i] >= w.mean()
    return reg


def _wma(x, n):
    n = max(1, int(n)); w = np.arange(1, n + 1)
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        seg = x[i - n + 1:i + 1]
        out[i] = np.dot(seg, w) / w.sum()
    return out


def hvwma_daily_dir(T, hlen=16, smooth=5):
    """日线近似 HVWMA 方向(+1/-1)。dirmap 缺失时的回退趋势闸。"""
    px = np.array([p if p is not None else np.nan for p in T["price"]], float)
    px = np.where(np.isnan(px), np.nanmedian(px), px)
    s = np.log(px)
    half, sq = max(1, hlen // 2), max(1, round(math.sqrt(hlen)))
    hma = _wma(2 * _wma(s, half) - _wma(s, hlen), sq)
    # ewm 平滑
    out = np.copy(hma); a = 1 / smooth
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
        elif not np.isnan(out[i - 1]):
            out[i] = a * out[i] + (1 - a) * out[i - 1]
    d = np.sign(np.diff(out, prepend=out[0]))
    return [int(x) if not np.isnan(x) else 0 for x in d]


# ── 参数 ──────────────────────────────────────────────────────────────
@dataclass
class Params:
    # 默认 = walk-forward 两折稳定选中的稳健参数(cap50/k2.0/ts20), 非手调 924 后。
    floor: float = 0.0        # 组合最低仓(0=纯信号驱动, 允许完全避险; 30=打底)
    cap_trend: float = 0.50   # 趋势基准上限(留 (1-cap) 给择时上探)
    trlo: float = 0.55        # 趋势参与动量分位下沿
    trhi: float = 0.92        # 上沿
    k_ts: float = 2.0         # σ 缩放跟踪止损系数
    h_ts: int = 10            # 止损时间尺度(√天)
    time_stop: int = 20       # 死钱强平天数
    dead_eps: float = 0.01    # 死钱判定: 相对进场 ≤ +1%
    regime_cap: float = 0.30  # risk-off 封顶
    hot_cut: float = 0.5      # 过热时对基准的压制系数
    use_regime: bool = True
    use_trend_gate: bool = True


# ── 按标的分参: 默认紧; kc50 处罕见产业趋势, 偏松放大风险偏好(骑趋势/关MA200闸)──
PARAMS_BY_KEY = {
    "kc50": Params(cap_trend=0.85, k_ts=4.0, time_stop=50, use_regime=False, trlo=0.40),
}


def params_for(key) -> Params:
    return PARAMS_BY_KEY.get(key, Params())


# ── 出场机主体: 逐日状态机, 返回每日目标仓位 ─────────────────────────────
def weights(T, P: Params, dirmap: dict | None = None, trace: bool = False):
    n = len(T["dates"]); dates = T["dates"]; px = T["price"]
    dip = dip_arr(T); hot = hot_arr(T); sig = daily_sigma(T)
    reg = ma_regime(T) if P.use_regime else [True] * n
    mp = T["mom_pct"]
    dfall = hvwma_daily_dir(T)   # 回退趋势
    w = [P.floor] * n
    trrec = [None] * n if trace else None
    prev = P.floor
    peak = None; days = 0; entry_px = None

    for i in range(n):
        # 趋势方向: 优先分钟级 dirmap, 回退日线
        if dirmap is not None and dates[i] in dirmap:
            tdir = dirmap[dates[i]]
        else:
            tdir = dfall[i]

        # 入场目标 = max(趋势打底, 抄底档)
        tr = clamp((mp[i] - P.trlo) / (P.trhi - P.trlo), 0, 1) if mp[i] is not None else 0
        base = P.cap_trend * tr if reg[i] else 0.0
        entry = P.floor + (1 - P.floor) * dip[i]
        want = max(base, entry)

        p = px[i]
        ts_dist = P.k_ts * sig[i] * math.sqrt(P.h_ts)
        trigger = ""
        if want > prev + 1e-6:
            # 新建/加仓: 即时到位, 重置出场状态
            wi = want
            peak = p; days = 0; entry_px = p
            trigger = "entry"
        else:
            # 持有: 出场机独占决策
            if p is not None:
                peak = p if peak is None else max(peak, p)
            days += 1
            broke = (peak is not None and p is not None and p < peak * (1 - ts_dist))
            trend_red = (P.use_trend_gate and tdir < 0 and dip[i] < 0.05)
            dead = (entry_px is not None and p is not None and days > P.time_stop
                    and p <= entry_px * (1 + P.dead_eps))
            if broke or trend_red or dead:
                wi = max(base, P.floor)     # 释放超额仓 → 趋势基准(或底仓)
                trigger = "trail" if broke else ("trend" if trend_red else "timestop")
                peak = None; days = 0; entry_px = None
            else:
                wi = max(prev, want)        # 骑趋势: 只棘轮上行, 绝不线性衰减
                trigger = "hold"

        # 过热兑现: 压制到基准
        if hot[i]:
            wi = min(wi, base if base > 0 else P.floor + (1 - P.floor) * 0.15)
            if trigger == "hold":
                trigger = "hot"
        # regime 总闸
        if not reg[i]:
            wi = min(wi, P.regime_cap)
        wi = clamp(wi, 0, 1)
        w[i] = wi; prev = wi
        if trace:
            trrec[i] = {"dip": round(dip[i], 3), "base": round(base, 3),
                     "target": round(entry, 3), "want": round(want, 3),
                     "trend": int(tdir), "regime": bool(reg[i]), "hot": bool(hot[i]),
                     "sigma": round(sig[i], 4), "ts_dist": round(ts_dist, 4),
                     "peak": round(peak, 4) if peak else None,
                     "stop_px": round(peak * (1 - ts_dist), 4) if peak else None,
                     "in_pos": wi > (base + 1e-6) or (P.floor > 0 and wi > P.floor + 1e-6),
                     "days_held": days, "trigger": trigger, "weight": round(wi, 3)}
    return (w, trrec) if trace else w


# ── 现状基线(build_equity.weights_opt): 对照组 ──────────────────────────
def weights_current(T):
    FLOOR, DECAY = 0.30, 0.04
    n = len(T["dates"]); mp = T["mom_pct"]; dip = dip_arr(T)
    w = [FLOOR] * n; prev = FLOOR
    for i in range(n):
        tgt = FLOOR + (1 - FLOOR) * dip[i]
        iv, vp = T["iv_pct"][i], T["vrp_pct"][i]
        hot = mp[i] is not None and iv is not None and vp is not None and mp[i] >= .85 and iv >= .75 and vp >= .75
        if hot:
            tgt = min(tgt, FLOOR * 0.5)
        fast = hot or (mp[i] is not None and mp[i] >= .85)
        w[i] = clamp(tgt if (tgt >= prev or fast) else max(tgt, prev - DECAY), 0, 1)
        prev = w[i]
    return w


# ── 回测 / 绩效 ─────────────────────────────────────────────────────────
def _slice(T, since, until):
    dates = T["dates"]; n = len(dates)
    s = 0
    while s < n and dates[s] < since:
        s += 1
    e = n
    if until:
        e = 0
        while e < n and dates[e] <= until:
            e += 1
    return s, e


def perf(T, w, since="20000101", until=None):
    px = T["price"]; s, e = _slice(T, since, until)
    sr, ws = [], []
    for i in range(s + 1, e):
        if px[i] is None or px[i - 1] is None:
            continue
        turn = abs(w[i - 1] - (w[i - 2] if i - 2 >= s else w[s]))
        sr.append(w[i - 1] * (px[i] / px[i - 1] - 1) - turn * FEE)
        ws.append(w[i - 1])
    if len(sr) < 20:
        return None
    sr = np.array(sr); nn = len(sr)
    ann = (np.prod(1 + sr)) ** (252 / nn) - 1
    shp = sr.mean() / sr.std() * np.sqrt(252) if sr.std() else 0
    nav = np.cumprod(1 + sr); mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    calmar = ann / abs(mdd) if mdd else 0
    return dict(ann=round(ann * 100, 1), sharpe=round(shp, 2), mdd=round(mdd * 100, 1),
                calmar=round(calmar, 2), meanw=round(np.mean(ws) * 100), n=nn)


def bench(T, since="20000101", until=None):
    px = T["price"]; s, e = _slice(T, since, until)
    br = [px[i] / px[i - 1] - 1 for i in range(s + 1, e)
          if px[i] is not None and px[i - 1] is not None]
    br = np.array(br); nn = len(br)
    ann = (np.prod(1 + br)) ** (252 / nn) - 1
    shp = br.mean() / br.std() * np.sqrt(252) if br.std() else 0
    nav = np.cumprod(1 + br); mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return dict(ann=round(ann * 100, 1), sharpe=round(shp, 2), mdd=round(mdd * 100, 1), n=nn)


# ── 逐笔交易统计(把连续权重切成「加仓段」看持有效率)───────────────────
def trades(T, w, since="20000101"):
    px = T["price"]; dates = T["dates"]; s, e = _slice(T, since, None)
    segs = []; in_pos = False; ent_i = None
    thresh = 0.05
    for i in range(s, e):
        if not in_pos and w[i] > thresh:
            in_pos = True; ent_i = i
        elif in_pos and w[i] <= thresh:
            in_pos = False
            if px[ent_i] and px[i]:
                segs.append((dates[ent_i], dates[i], i - ent_i, px[i] / px[ent_i] - 1))
    if in_pos and ent_i is not None and px[ent_i] and px[e - 1]:
        segs.append((dates[ent_i], dates[e - 1], (e - 1) - ent_i, px[e - 1] / px[ent_i] - 1))
    if not segs:
        return dict(n=0)
    rets = np.array([x[3] for x in segs]); hold = np.array([x[2] for x in segs])
    return dict(n=len(segs), hold=round(hold.mean(), 1), win=round((rets > 0).mean(), 2),
                avg=round(rets.mean() * 100, 1),
                avgwin=round(rets[rets > 0].mean() * 100, 1) if (rets > 0).any() else 0,
                avgloss=round(rets[rets <= 0].mean() * 100, 1) if (rets <= 0).any() else 0)


# ── walk-forward: expanding window, 训练挑参 → 样本外验证 ────────────────
GRID = [
    Params(floor=f, cap_trend=c, k_ts=k, time_stop=ts)
    for f in (0.0, 0.30)
    for c in (0.50, 0.60)
    for k in (2.0, 2.5, 3.0)
    for ts in (20, 30)
]


def champion(T, since, until, metric="calmar"):
    best = None; bestp = None
    for P in GRID:
        m = perf(T, weights(T, P), since, until)
        if m is None:
            continue
        score = m[metric]
        if best is None or score > best:
            best = score; bestp = P; bestm = m
    return bestp, bestm


def walk_forward(T):
    """两折: train=2020~2023末挑参, test=2024~今; 再 train=至2024中, test=2024中~今。"""
    folds = [("20200101", "20231231", "20240101", None),
             ("20200101", "20240630", "20240701", None)]
    rows = []
    for tr0, tr1, te0, te1 in folds:
        P, mtr = champion(T, tr0, tr1)
        if P is None:
            continue
        mte = perf(T, weights(T, P), te0, te1)
        rows.append((tr1, mtr, mte, P))
    return rows


def dirmaps():
    """若已 build 分钟级 HVWMA, 读入; 否则 None(回退日线)。"""
    dm = {}
    for key in KEYS:
        f = PROC / f"hvwma_dir_{key}.json"
        dm[key] = json.load(open(f)) if f.exists() else None
    return dm


def main():
    DM = dirmaps()
    fixed = Params()   # 一套固定「合理」参数, 全程不调, 看跨期稳定性

    print("═" * 78)
    print("A. 固定参数(不调) · 分标的分期 · 现状 vs 出场机 · 年化/夏普/回撤/Calmar/均仓")
    print("═" * 78)
    for key in KEYS:
        T = TV[key]; dm = DM.get(key)
        print(f"\n【{T['name']}】")
        for since, lab in [("20240924", "924后"), ("20220101", "22年至今"), ("20200101", "全样本")]:
            b = bench(T, since)
            cur = perf(T, weights_current(T), since)
            new = perf(T, weights(T, fixed, dm), since)
            if not (cur and new):
                continue
            print(f"  {lab:<8} 基准 年化{b['ann']:>6}%/夏普{b['sharpe']:>5}/回撤{b['mdd']:>6}%")
            print(f"           现状 年化{cur['ann']:>6}%/夏普{cur['sharpe']:>5}/回撤{cur['mdd']:>6}%/Calmar{cur['calmar']:>5}/均仓{cur['meanw']}%")
            print(f"           出场 年化{new['ann']:>6}%/夏普{new['sharpe']:>5}/回撤{new['mdd']:>6}%/Calmar{new['calmar']:>5}/均仓{new['meanw']}%")

    print("\n" + "═" * 78)
    print("B. Walk-forward 样本外(train挑参→test验证, 直面过拟合质疑)")
    print("═" * 78)
    for key in KEYS:
        T = TV[key]
        print(f"\n【{T['name']}】")
        for tr1, mtr, mte, P in walk_forward(T):
            print(f"  train≤{tr1}: IS Calmar{mtr['calmar']:>5}/夏普{mtr['sharpe']:>5}  →  "
                  f"OOS 年化{mte['ann']:>6}%/夏普{mte['sharpe']:>5}/回撤{mte['mdd']:>6}%/Calmar{mte['calmar']:>5}  "
                  f"[floor{int(P.floor*100)} cap{int(P.cap_trend*100)} k{P.k_ts} ts{P.time_stop}]")

    print("\n" + "═" * 78)
    print("C. 逐笔交易效率(全样本): 现状 vs 出场机")
    print("═" * 78)
    for key in KEYS:
        T = TV[key]; dm = DM.get(key)
        tc = trades(T, weights_current(T)); tn = trades(T, weights(T, fixed, dm))
        print(f"  {T['name']:<8} 现状: {tc.get('n')}笔 持有{tc.get('hold')}日 胜率{tc.get('win')} 均{tc.get('avg')}% (盈{tc.get('avgwin')}/亏{tc.get('avgloss')})")
        print(f"  {'':<8} 出场: {tn.get('n')}笔 持有{tn.get('hold')}日 胜率{tn.get('win')} 均{tn.get('avg')}% (盈{tn.get('avgwin')}/亏{tn.get('avgloss')})")


if __name__ == "__main__":
    main()
