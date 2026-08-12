"""构建波动概率锥页面的数据集(前瞻锥 + 挂单回测)。

页面主体是 SOXX expected-move 那种锥: 以今天收盘为顶点, 用今天的 IV,
向未来张开 P*(1 ± m*IV*sqrt(d/252))。回测则回答"在下沿挂买单管不管用":
每天挂一单、H 日内有效、用日内最低价判成交, 统计成交率/成交后收益/还要再跌多少。

输出 data/processed/cone_viz.json:
  <key>.dates/open/high/low/price     历史 OHLC(price=收盘, 画蜡烛与判成交)
  <key>.iv/iv_put/panic               IV 与恐慌度(前端现算锥)
  <key>.first_W / first_M             周/月首日标记(历史分期带模式用, 其余前端算)
  <key>.anchor / z                    高点锚定模式(保留)
  <key>.bt / bt_panic                 挂单回测(本标的), _pooledBt / _pooledBtPanic 为池化
用法: python -m scripts.build_cone_viz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.catalog import latest_ranged_file
from src.utils.config import abspath

PROC = abspath("data/processed")
LOOKBACK = 60          # 高点锚定模式的回看窗口
MIN_DAYS = 3
H = 20                 # 挂单有效期(交易日)
K = 20                 # 成交后持有天数
LEVELS = [0.5, 1.0, 1.5, 2.0]
PANIC_GATE = 0.70

SOURCES = {
    "zz1000": ("中证1000", "surface_zz1000", "zz1000"),
    "kc50": ("科创50", "surface_kc50", "kc50"),
    "cyb": ("创业板", "surface_cyb", "cyb"),
    "hs300": ("沪深300ETF", "surface_300etf", "hs300"),
}


def clean(s, nd=4):
    out = []
    for v in s:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            out.append(None)
        else:
            out.append(round(float(v), nd))
    return out


def build_one(key: str) -> pd.DataFrame:
    name, prefix, tv_key = SOURCES[key]
    tv = json.load(open(PROC / "timing_viz.json"))[tv_key]
    df = pd.DataFrame({
        "date": tv["dates"], "iv": tv["atm_iv"], "iv_pct": tv["iv_pct"],
        "sent_pct": tv["sent_pct"], "rr_pct": tv["rr_pct"],
        "slope_pct": tv["slope_pct"], "vrp_pct": tv["vrp_pct"],
    })
    sf = latest_ranged_file(PROC, prefix)
    sv = pd.read_parquet(sf.path, columns=["date", "iv_put_25d"])
    sv["date"] = sv["date"].astype(str)
    df = df.merge(sv.rename(columns={"iv_put_25d": "iv_put"}), on="date", how="left")

    # OHLC: 挂单成交要用日内最低价, timing_viz 只有收盘
    ohlc = pd.read_parquet(abspath("data/raw") / f"ohlc_{key}.parquet")
    ohlc = ohlc.rename(columns={"trade_date": "date"})
    ohlc["date"] = ohlc["date"].astype(str)
    df = df.merge(ohlc[["date", "open", "high", "low", "close"]], on="date", how="inner")

    df = df.dropna(subset=["close", "iv"]).sort_values("date").reset_index(drop=True)
    px = df["close"].astype(float)
    iv = df["iv"].astype(float)

    anchor = []
    for i in range(len(df)):
        j = int(px.iloc[max(0, i - LOOKBACK + 1):i + 1].idxmax())
        anchor.append(j if i - j >= MIN_DAYS else None)
    df["anchor"] = anchor
    days = pd.Series([i - a if a is not None else np.nan for i, a in enumerate(anchor)])
    apx = pd.Series([px.iloc[a] if a is not None else np.nan for a in anchor])
    aiv = pd.Series([iv.iloc[a] if a is not None else np.nan for a in anchor])
    df["z"] = (px / apx - 1) / (aiv * np.sqrt(days / 252))

    df["panic"] = pd.concat(
        [df["iv_pct"], df["sent_pct"], 1 - df["rr_pct"],
         1 - df["slope_pct"], 1 - df["vrp_pct"]], axis=1).mean(axis=1)
    dt = pd.to_datetime(df["date"], format="%Y%m%d")
    for f in ("W", "M"):
        per = dt.dt.to_period(f)
        df[f"first_{f}"] = (per != per.shift(1)).tolist()
    df["name"] = name
    df["key"] = key
    return df


def simulate(df: pd.DataFrame, m: float) -> pd.DataFrame:
    """每天在 -m σ 挂一单, H 日内有效, 日内最低价触及即成交。"""
    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    iv = df["iv"].to_numpy(float)
    panic = df["panic"].to_numpy(float)
    n, rows = len(df), []
    for t in range(n - H - K):
        lvl = close[t] * (1 - m * iv[t] * np.sqrt(H / 252))
        win = low[t + 1:t + 1 + H]
        hit = np.nonzero(win <= lvl)[0]
        rec = {"panic": panic[t], "filled": bool(len(hit)),
               "base_ret": close[t + K] / close[t] - 1}
        if len(hit):
            f = t + 1 + int(hit[0])
            rec["wait"] = f - t
            rec["ret"] = close[min(f + K, n - 1)] / lvl - 1
            rec["mae"] = low[f:min(f + K, n)].min() / lvl - 1
            rec["vs_low"] = lvl / win.min() - 1
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> dict:
    f = res[res["filled"]]
    out = {"n": int(len(res)), "fill": round(float(res["filled"].mean()), 4),
           "n_fill": int(len(f)),
           "base": round(float(res["base_ret"].mean()), 4),
           "base_win": round(float((res["base_ret"] > 0).mean()), 3)}
    if len(f) >= 20:
        out.update({
            "ret": round(float(f["ret"].mean()), 4),
            "win": round(float((f["ret"] > 0).mean()), 3),
            "mae": round(float(f["mae"].mean()), 4),
            "mae_worst": round(float(f["mae"].min()), 4),
            "vs_low": round(float(f["vs_low"].mean()), 4),
            "vs_low_med": round(float(f["vs_low"].median()), 4),
            "near_bottom": round(float((f["mae"] >= -0.01).mean()), 3),
            "wait": round(float(f["wait"].mean()), 1),
        })
    return out


def main() -> None:
    frames, out = {}, {}
    for key in SOURCES:
        df = build_one(key)
        frames[key] = df
        bt, btp = {}, {}
        for m in LEVELS:
            res = simulate(df, m)
            bt[str(m)] = summarize(res)
            btp[str(m)] = summarize(res[res["panic"] >= PANIC_GATE])
        out[key] = {
            "name": df["name"].iloc[0],
            "dates": [str(x) for x in df["date"]],
            "price": clean(df["close"], 4),
            "open": clean(df["open"], 4),
            "high": clean(df["high"], 4),
            "low": clean(df["low"], 4),
            "iv": clean(df["iv"], 4),
            "iv_put": clean(df["iv_put"], 4),
            "panic": clean(df["panic"], 3),
            "z": clean(df["z"], 3),
            "anchor": [None if pd.isna(a) else int(a) for a in df["anchor"]],
            "first_W": [bool(x) for x in df["first_W"]],
            "first_M": [bool(x) for x in df["first_M"]],
            "bt": bt, "bt_panic": btp,
        }
        print(f"{key:7s} {df['name'].iloc[0]:10s} {len(df):5d} 天  "
              f"{df['date'].iloc[0]}->{df['date'].iloc[-1]}  "
              f"成交率 -1σ {bt['1.0']['fill']:.0%} / -2σ {bt['2.0']['fill']:.0%}")

    pbt, pbtp = {}, {}
    for m in LEVELS:
        allres = pd.concat([simulate(df, m) for df in frames.values()], ignore_index=True)
        pbt[str(m)] = summarize(allres)
        pbtp[str(m)] = summarize(allres[allres["panic"] >= PANIC_GATE])
    out["_pooledBt"] = pbt
    out["_pooledBtPanic"] = pbtp
    out["_meta"] = {"lookback": LOOKBACK, "min_days": MIN_DAYS, "H": H, "K": K,
                    "levels": LEVELS, "panic_gate": PANIC_GATE,
                    "period_n": {"W": 5, "M": 21}}

    dest = PROC / "cone_viz.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"\n池化(H={H},K={K}): " + "  ".join(
        f"-{m}σ 成交{pbt[str(m)]['fill']:.0%}/收益{pbt[str(m)].get('ret', 0):+.1%}" for m in LEVELS))
    print(f"写出 {dest} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
