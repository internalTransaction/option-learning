"""研究向期权择时信号台 · 汇总后端 → option_signal_desk.json (供 cnstock 前端消费)。

一份 JSON 打包信号台需要的全部研究视图:
  1. 入场因子板     price + 6 个期权因子滚动分位(极值带), 来自 timing_viz。
  2. 出场机仓位      新出场机 vs 现状基线的每日目标仓位 + 当前出场状态(dip/目标/趋势/regime/跟踪止损位)。
  3. 净值与超额      基准 / 出场机 / 现状 的 NAV、回撤、相对基准超额曲线。
  4. 分期绩效        924后 / 22年至今 / 全样本 的 年化/夏普/回撤/Calmar/均仓。
  5. 样本外验证      walk-forward IS→OOS, 直面过拟合质疑。
  6. 逐笔效率        持有天数/胜率/盈亏比, 证明「让盈利跑、砍亏损」。
  7. HVWMA 趋势态    对齐实盘 3H HVWMA(hvwma_live.json), 红绿。

只读上游产物, 不联网; 由 daily 编排在 build_timing_viz + build_hvwma_live 之后调用。
"""
from __future__ import annotations
import importlib.util as u
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# 载入 exit_machine 作为库
spec = u.spec_from_file_location("exit_machine", str(ROOT / "scripts" / "exit_machine.py"))
em = u.module_from_spec(spec); sys.modules["exit_machine"] = em
spec.loader.exec_module(em)

TV = em.TV
KEYS = em.KEYS
FEE = em.FEE
DISPLAY_SINCE = "20220101"    # 净值/仓位展示起点(样本外为主, 避开旧系统过拟合窗口)


def nav_series(T, w, since):
    """给定权重序列 → NAV / 回撤 / 均仓, 从 since 起。含换手成本。"""
    dates = T["dates"]; px = T["price"]; n = len(px)
    s = 0
    while s < n and dates[s] < since:
        s += 1
    ds = [dates[s]]; sr = []
    for i in range(s + 1, n):
        if px[i] is None or px[i - 1] is None:
            continue
        turn = abs(w[i - 1] - (w[i - 2] if i - 2 >= s else w[s]))
        sr.append(w[i - 1] * (px[i] / px[i - 1] - 1) - turn * FEE)
        ds.append(dates[i])
    nav = np.cumprod([1.0] + [1 + x for x in sr])
    dd = nav / np.maximum.accumulate(nav) - 1
    return ds, [round(float(x), 4) for x in nav], [round(float(x) * 100, 2) for x in dd]


def bench_series(T, since):
    dates = T["dates"]; px = T["price"]; n = len(px)
    s = 0
    while s < n and dates[s] < since:
        s += 1
    ds = [dates[s]]; br = []
    for i in range(s + 1, n):
        if px[i] is None or px[i - 1] is None:
            continue
        br.append(px[i] / px[i - 1] - 1); ds.append(dates[i])
    nav = np.cumprod([1.0] + [1 + x for x in br])
    dd = nav / np.maximum.accumulate(nav) - 1
    return ds, [round(float(x), 4) for x in nav], [round(float(x) * 100, 2) for x in dd]


def excess(strat_nav, bench_nav):
    m = min(len(strat_nav), len(bench_nav))
    return [round(strat_nav[i] / bench_nav[i], 4) if bench_nav[i] else 1.0 for i in range(m)]


def build_one(key, hv):
    T = TV[key]
    P = em.params_for(key)         # 按标的分参(kc50 偏松)
    dm = em.dirmaps().get(key)     # 分钟级 dirmap(有则用, 无回退日线)
    w_exit, trace = em.weights(T, P, dm, trace=True)
    w_cur = em.weights_current(T)

    # 因子板(全序列, 前端自行裁窗)
    factors = {k: T[k] for k in ["iv_pct", "sent_pct", "rr_pct", "slope_pct",
                                 "vrp_pct", "pcr_pct", "mom_pct"]}
    raw = {k: T[k] for k in ["atm_iv", "sent", "rr", "vrp", "rv", "pcr"]}

    # 净值(展示窗)
    bds, bnav, bdd = bench_series(T, DISPLAY_SINCE)
    eds, enav, edd = nav_series(T, w_exit, DISPLAY_SINCE)
    cds, cnav, cdd = nav_series(T, w_cur, DISPLAY_SINCE)

    # 仓位序列(展示窗, 与净值同日期轴对齐用 eds)
    def wslice(w):
        s = 0
        while s < len(T["dates"]) and T["dates"][s] < DISPLAY_SINCE:
            s += 1
        return [round(w[i] * 100, 1) for i in range(s, len(w))]

    # 当前出场状态
    stnow = trace[-1] if trace and trace[-1] else {}
    stnow = dict(stnow); stnow["date"] = T["dates"][-1]

    # 分期绩效
    stats = {}
    for tag, since in [("924", "20240924"), ("22", "20220101"), ("all", "20000101")]:
        stats[tag] = {"bench": em.bench(T, since),
                      "exit": em.perf(T, w_exit, since),
                      "current": em.perf(T, w_cur, since)}

    # 样本外验证
    wf = []
    for tr1, mtr, mte, pp in em.walk_forward(T):
        wf.append({"train_end": tr1, "is_calmar": mtr["calmar"], "is_sharpe": mtr["sharpe"],
                   "oos": mte, "params": {"floor": pp.floor, "cap": pp.cap_trend,
                                          "k_ts": pp.k_ts, "time_stop": pp.time_stop}})

    # 逐笔
    tr_exit = em.trades(T, w_exit); tr_cur = em.trades(T, w_cur)

    hvk = hv.get(key, {})
    return {
        "name": T["name"],
        "dates": T["dates"], "price": T["price"],
        "factors": factors, "raw": raw,
        "hvwma": {"dir_now": hvk.get("dir", 0), "source": hvk.get("source"),
                  "asof": hvk.get("asof"), "recent": hvk.get("recent", [])},
        "position": {
            "date_axis": eds,
            "weight_exit": wslice(w_exit),
            "weight_current": wslice(w_cur),
            "stop_px": [t["stop_px"] if t else None for t in trace][-len(eds):],
            "state_now": stnow,
        },
        "equity": {
            "dates": bds,
            "bench": bnav, "bench_dd": bdd,
            "exit": enav, "exit_dd": edd, "exit_excess": excess(enav, bnav),
            "current": cnav, "current_dd": cdd, "current_excess": excess(cnav, bnav),
        },
        "stats": stats,
        "walkforward": wf,
        "trades": {"exit": tr_exit, "current": tr_cur},
    }


def main():
    hv = {}
    hvf = PROC / "hvwma_live.json"
    if hvf.exists():
        hv = json.load(open(hvf))

    from datetime import datetime
    from zoneinfo import ZoneInfo
    out = {"meta": {"asof": TV[KEYS[0]]["dates"][-1],
                    "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M"),
                    "underlyings": KEYS,
                    "display_since": DISPLAY_SINCE,
                    "params_default": em.Params().__dict__,
                    "params_by_key": {k: em.params_for(k).__dict__ for k in KEYS},
                    "note": "入场=期权恐慌dip(擅长); 出场=σ缩放跟踪止损+HVWMA趋势闸+MA200 regime+time-stop(与入场解耦); kc50处罕见产业趋势偏松放大风险偏好"},
        "underlyings": {}}
    for key in KEYS:
        out["underlyings"][key] = build_one(key, hv)
        s = out["underlyings"][key]["stats"]["22"]
        print(f"{key:7s} {TV[key]['name']:8s} 22年至今: "
              f"出场 年化{s['exit']['ann']}%/夏普{s['exit']['sharpe']}/回撤{s['exit']['mdd']}%  "
              f"vs 现状 {s['current']['ann']}%/{s['current']['sharpe']}/{s['current']['mdd']}%  "
              f"HVWMA {'🟢' if out['underlyings'][key]['hvwma']['dir_now']>0 else '🔴'}", flush=True)

    dest = PROC / "option_signal_desk.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"\n写出 {dest} ({dest.stat().st_size/1024:.0f}KB)")


if __name__ == "__main__":
    main()
