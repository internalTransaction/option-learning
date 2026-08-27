"""期权择时 v2 数据层 → option_desk_v2.json (供 cnstock 前端 /option-timing-v2 与 /option-raw-v2 消费)。

对齐上游新库 842adda(对冲研究 + 今日动作提示 + 双策略净值), 相对 v1 的三处变化:
  1. 口径统一: 25Δ 偏斜改报 认沽IV − 认购IV (= −RR), 与情绪比(认沽/认购)同为"认沽相对认购的贵贱",
     两者同向: 高 = 恐慌。RR 原值(call−put)仍保留在 raw.rr 里备查。
  2. 原始值随分位一同下发, 前端可切「分位 / 原始值 / z-score(窗口可选)」三视图。
  3. 不再下发"建议仓位"作为主展示 —— 前端改为随十字光标给「当日期权动作」(抄底/过热/低波保险/常态),
     仓位只作为双策略净值的副图(纯期权 floor30+棘轮 / HVWMA结合)。

只读上游产物(timing_viz.json / equity.json / calib_lab.json), 不联网;
由 daily_update.sh 在 build_timing_viz + build_equity 之后调用。
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
KEYS = ["hs300", "zz1000", "kc50", "cyb"]
WIN, MINP = 63, 31          # 与 build_timing_viz 一致的滚动分位口径
HI, LO = 0.90, 0.10
UPSTREAM = {"commit": "842adda", "date": "2026-07-20",
            "title": "信号台移除建议仓位, 聚焦看指标+今日动作"}

# 因子展示规格: key, 标签, 原始值键, 恐慌方向(1=高即恐慌), 说明
FACTOR_SPEC = [
    ["iv",    "ATM IV",      "atm_iv",  1, "平值隐含波动率 · 高=恐慌买保护"],
    ["sent",  "情绪比",       "sent",    1, "认沽IV ÷ 认购IV · 高=抢认沽"],
    ["skew",  "25Δ 偏斜",     "skew_pd", 1, "认沽IV − 认购IV · 高=下跌偏斜极端(= −RR, 与情绪比同向)"],
    ["slope", "微笑斜率",     "slope",   0, "smile 斜率 · 低=尾部恐惧"],
    ["vrp",   "方差溢价 VRP", "vrp",     0, "ATM IV − RV20 · 低=capitulation"],
    ["pcr",   "PCR",         "pcr",     None, "认沽/认购成交比 · 辅助, 不计灯"],
]
LIGHT_KEYS = [("iv", 1), ("sent", 1), ("skew", 1), ("slope", 0), ("vrp", 0)]

# 「今日动作」四态 —— 落地新库对冲研究结论(依据见 meta.hedge)
ACTIONS = {
    "buy":    {"ico": "🟢", "tag": "抄底 / 买 call 加仓",
               "odds": "恐慌抄底历史胜率 ~63–77% · 低 GEX 时反弹更猛"},
    "trim":   {"ico": "🟣", "tag": "过热 · 减仓 / 收紧止盈",
               "odds": "对冲科创下行: 做空中证1000(IM) 或直接减仓, 别买 put"},
    "insure": {"ico": "🔵", "tag": "低波 · 可布局 2 周 put 保险",
               "odds": "低IV买put(2周): 创业板 +15.6% / 沪深300 打平 / 赔率≈盈亏平衡"},
    "hold":   {"ico": "⚪", "tag": "常态 · 持有观望", "odds": ""},
}
HEDGE = [
    {"code": "buy", "ico": "🟢", "title": "恐慌 → 抄底 / 买 call",
     "body": "恐慌灯 + 跌到位。期权买方的主场：IV 即将见顶回落、方向与 vega 都顺。",
     "evidence": "恐慌抄底历史胜率 ~63–77%，低 GEX(空 gamma·波动放大)时反弹更猛。"},
    {"code": "trim", "ico": "🟣", "title": "过热 → 减仓，别买 put",
     "body": "melt-up 过热时买 put 是灾难——方向错(还在涨) + 高 IV 买贵 + IV 回落/theta，三杀。",
     "evidence": "科创过热买 ATM put 持 20 日均值 −55%、赔率 0.62；对冲科创下行改用做空中证1000(IM)"
                 "或直接减仓(beta 中性、不赌方向，回撤 −5.1%→−1.2%)。"},
    {"code": "insure", "ico": "🔵", "title": "低波 → 布局 2 周 put 保险",
     "body": "保险要在天晴时买：低 IV 时 put 便宜、且 IV 有上升空间。优先创业板 / 沪深300，持约 2 周。",
     "evidence": "低 IV 买 put(2周)：创业板 +15.6% / 沪深300 打平 / 科创 +1.3%，赔率≈盈亏平衡；"
                 "高 IV 买则 −10~−38%。换的是时机+持有期，不是行权价。"},
]
CHAIN = [
    {"n": "1", "h": "信号采集", "tag": "曲面因子",
     "b": "Tushare 期权链构建波动率曲面：ATM IV、情绪比(认沽/认购 25Δ IV)、25Δ 偏斜(认沽−认购)、微笑斜率、VRP、PCR，全部取 63 日滚动百分位。",
     "f": "为何用百分位不用 z-score：IV/VRP 右偏厚尾，z 会被自身尖峰污染 std。本地实测(见口径实验室)：z 只是更钝，不是更准。"},
    {"n": "2", "h": "恐慌五灯", "tag": "极值判定",
     "b": "ATM IV≥.90 · 情绪比≥.90 · 25Δ偏斜≥.90 · 微笑斜率≤.10 · VRP≤.10，五信号合成亮灯数，全部指向「恐慌→反转看多」。",
     "f": "五灯单独没有领先性(没跌之前假信号多)——它是条件性 alpha，价值只在「已经跌了」时释放。"},
    {"n": "3", "h": "触发闸门", "tag": "灯 × 动态跌幅",
     "b": "抄底 = 恐慌灯 + 动态跌幅(20日动量的 1 年滚动分位，替代 hardcode −5/−8/−10%)，gate 随跌幅增大。",
     "f": "消融证明：控制跌幅后「有灯 vs 无灯」胜率 40%→77%(中证1000)。跌幅=触发器、期权=质量过滤器。"},
    {"n": "4", "h": "建仓", "tag": "底仓30%+阈值",
     "b": "buy = 恐慌度 × 跌幅门 × GEX 调节；仓位 = 30% + 70%×ramp(buy)，越跌越恐慌越接近满仓。",
     "f": "离散 3 档 ≈ 5 档 ≈ 连续爬坡(精细化零增量)；但纯跌幅分批(丢掉恐慌质量)明显更差。"},
    {"n": "5", "h": "出场", "tag": "棘轮+见顶",
     "b": "入场快、出场慢：减仓每天最多降 4%(棘轮，给反弹时间)，仅动量涨到高位或 melt-up 过热才快速兑现(降到 15%)。",
     "f": "逃顶做空被证伪(波动率不对称：底有尖峰、顶无)。另一变体是 HVWMA 结合(趋势打底+期权抄底)，见双策略对比。"},
    {"n": "6", "h": "GEX 增强", "tag": "dealer gamma",
     "b": "GEX 作 buy 的连续调节：低 GEX(空 gamma·波动放大·剧烈 capitulation)抄底加成、高 GEX 压制，K=0.4。",
     "f": "与现有因子几乎正交(|r|<0.3)；同样恐慌抄底，低 GEX vs 高 GEX 胜率大幅分离(中证1000 95% vs 47%)。A股无 pin。"},
]
PARAMS = [["滚动分位窗口", "63 日"], ["极值阈值(五灯)", "0.90 / 0.10"],
          ["跌幅门 gate", "(0.30−动量分位)/0.30"], ["建仓 buy 阈值 / 满仓", "0.15 / 0.45"],
          ["GEX 调节 K / clamp", "0.4 / [0.4, 1.5]"], ["纯期权 底仓 / 棘轮", "30% / 每日 ≤4%"],
          ["HVWMA 趋势打底", "0.65 (日线近似 len16)"], ["换手成本", "5 bps"]]


def flip_pct(vals):
    """RR 分位 → 偏斜分位。偏斜 = −RR, 故 P(偏斜低) = P(RR高), 直接取补。

    刻意不对 −RR 重算滚动分位: 经验分位对并列值不对称(P(X≤x) ≠ 1−P(X<x)),
    重算会让「偏斜≥.90」与上游「RR≤.10」错开 1–3% 的交易日, 灯与净值就不同源了。
    取补则严格等价 —— 本次统一是展示层变换, 不动信号。
    """
    return [None if v is None else round(1 - float(v), 3) for v in vals]


def neg(vals):
    return [None if v is None else round(-float(v), 4) for v in vals]


def action_of(tier, hot, iv_pct, mom_pct, lights):
    """与上游 timing_template.html actionAt() 同逻辑。"""
    pp = lambda v: "—" if v is None else f"{v * 100:.0f}"
    if tier and tier >= 1:
        return dict(code="buy", why=f"恐慌灯 {lights} 盏 + 已跌到位(动量分位 {pp(mom_pct)}) → 极值反转窗口。"
                                    f"这是期权买方的主场：IV 即将见顶回落、方向与 vega 都顺。", **ACTIONS["buy"])
    if hot:
        return dict(code="trim", why="melt-up 过热(动量≥85 · IV≥75 · VRP≥75)。别买 put 对冲——高 IV 三杀"
                                     "(方向错+买贵+IV回落)，过热买 put 历史均值 −36~−55%。降敞口最实在。", **ACTIONS["trim"])
    if iv_pct is not None and iv_pct <= 0.25:
        return dict(code="insure", why=f"IV 分位 {pp(iv_pct)}(平静、保险便宜)。天晴时买伞：优先创业板 / 沪深300 "
                                       f"的近月 ATM put，持约 2 周。", **ACTIONS["insure"])
    return dict(code="hold", why=f"无极值信号(动量分位 {pp(mom_pct)} · IV 分位 {pp(iv_pct)})。"
                                 f"无恐慌抄底 / 过热 / 低波保险信号，不额外动作。", **ACTIONS["hold"])


def main():
    TV = json.load(open(PROC / "timing_viz.json"))
    EQ = json.load(open(PROC / "equity.json"))
    calib = json.loads((PROC / "calib_lab.json").read_text()) if (PROC / "calib_lab.json").exists() else None

    out = {"meta": {}, "underlyings": {}}
    asof = ""
    for key in KEYS:
        T = TV[key]
        n = len(T["dates"])
        skew_pd = neg(T["rr"])                       # 认沽IV − 认购IV, 高=恐慌(与情绪比同向)
        pct = {
            "iv": T["iv_pct"], "sent": T["sent_pct"], "skew": flip_pct(T["rr_pct"]),
            "slope": T["slope_pct"], "vrp": T["vrp_pct"], "pcr": T["pcr_pct"], "mom": T["mom_pct"],
        }
        lights, tier, hot = [0] * n, [0] * n, [0] * n
        for i in range(n):
            k = 0
            for fk, hi in LIGHT_KEYS:
                v = pct[fk][i]
                if v is None:
                    continue
                if (hi and v >= HI) or (not hi and v <= LO):
                    k += 1
            lights[i] = k
            m, iv, vp = pct["mom"][i], pct["iv"][i], pct["vrp"][i]
            if m is not None:
                if k >= 3 and m <= 0.04:
                    tier[i] = 3
                elif k >= 2 and m <= 0.08:
                    tier[i] = 2
                elif k >= 1 and m <= 0.15:
                    tier[i] = 1
            if m is not None and iv is not None and vp is not None and m >= .85 and iv >= .75 and vp >= .75:
                hot[i] = 1
        # 一致性硬校验: 新口径(偏斜≥.90) 必须与上游(RR≤.10) 完全同集合, 否则灯与净值脱钩
        old = {i for i in range(n) if (T["rr_pct"][i] is not None and T["rr_pct"][i] <= LO)}
        new = {i for i in range(n) if (pct["skew"][i] is not None and pct["skew"][i] >= HI)}
        assert old == new, f"{key} 偏斜灯与上游 RR 灯不等价: 差 {len(old ^ new)} 日"
        print(f"{key:7s} 偏斜灯 {len(new)} 日 ≡ 上游 RR 灯 ✓")

        i = n - 1
        rec = {
            "name": T["name"], "dates": T["dates"], "price": T["price"],
            "raw": {"atm_iv": T["atm_iv"], "sent": T["sent"], "skew_pd": skew_pd, "rr": T["rr"],
                    "slope": T["slope"], "vrp": T["vrp"], "pcr": T["pcr"], "rv": T["rv"],
                    "gex_z": T.get("gex_z"), "mom20": T["mom20"], "dd20": T["dd20"]},
            "pct": pct, "lights": lights, "tier": tier, "hot": hot,
            "equity": EQ[key]["ranges"],
            "now": {"date": T["dates"][i], "price": T["price"][i], "lights": lights[i],
                    "tier": tier[i], "hot": hot[i],
                    "action": action_of(tier[i], hot[i], pct["iv"][i], pct["mom"][i], lights[i]),
                    "w_opt": EQ[key]["ranges"]["924"]["opt"]["weight"][-1],
                    "w_hv": EQ[key]["ranges"]["924"]["hv"]["weight"][-1]},
        }
        out["underlyings"][key] = rec
        asof = max(asof, T["dates"][-1])

    now = datetime.now(timezone(timedelta(hours=8)))
    out["meta"] = {
        "asof": asof, "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "underlyings": KEYS, "names": {k: out["underlyings"][k]["name"] for k in KEYS},
        "upstream": UPSTREAM, "factors": FACTOR_SPEC, "hedge": HEDGE, "chain": CHAIN,
        "params": PARAMS, "hi": HI, "lo": LO, "win": WIN,
        "strategies": [["opt", "纯期权择时", "恐慌灯×动态跌幅 + GEX + 底仓30% + 棘轮出场"],
                       ["hv", "HVWMA 结合", "HVWMA 趋势绿→65% 打底, 红→期权抄底 overlay(日线近似)"]],
    }
    if calib:
        out["calib"] = calib

    dest = PROC / "option_desk_v2.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print(f"写出 {dest.name}  ({dest.stat().st_size / 1024:.0f} KB)  asof={asof}")
    for k in KEYS:
        a = out["underlyings"][k]["now"]
        print(f"  {out['underlyings'][k]['name']:<8} {a['action']['ico']} {a['action']['tag']:<18}"
              f" 灯{a['lights']} tier{a['tier']} 纯期权{a['w_opt']}% HVWMA{a['w_hv']}%")


if __name__ == "__main__":
    main()
