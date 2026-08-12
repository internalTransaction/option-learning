"""构建波动概率锥页面的数据集。

每个标的输出:
  dates/price/iv/iv_put/panic   基础序列(iv_put = 25Δ put IV, 供下沿 skew 修正)
  anchor                        每日对应的锚点下标(过去 LOOKBACK 日最高收盘, 且距今≥MIN_DAYS)
  z                             锚定锥的 z 值 = (P/P_anchor-1) / (IV_anchor*sqrt(days/252))
  stats                         按 z 分层的前瞻统计(该标的自身 + 池化), 供页面显示决策依据

锥位由前端按 anchor 现算, 便于用户改锚点/切 ATM↔put IV, 无需重跑。

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
LOOKBACK = 60
MIN_DAYS = 3
HORIZONS = (5, 10, 20)

# 页面 key -> (显示名, surface 文件前缀, timing_viz key)
SOURCES = {
    "zz1000": ("中证1000", "surface_zz1000", "zz1000"),
    "kc50": ("科创50", "surface_kc50", "kc50"),
    "cyb": ("创业板", "surface_cyb", "cyb"),
    "hs300": ("沪深300ETF", "surface_300etf", "hs300"),
}

BANDS = [(-99, -2.5, "≤-2.5σ"), (-2.5, -2, "-2.5~-2σ"), (-2, -1.5, "-2~-1.5σ"),
         (-1.5, -1, "-1.5~-1σ"), (-1, -0.5, "-1~-0.5σ"), (-0.5, 99, ">-0.5σ")]


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
        "date": tv["dates"], "price": tv["price"], "iv": tv["atm_iv"],
        "iv_pct": tv["iv_pct"], "sent_pct": tv["sent_pct"], "rr_pct": tv["rr_pct"],
        "slope_pct": tv["slope_pct"], "vrp_pct": tv["vrp_pct"],
    })

    # 25Δ put IV 从 surface 并入(下沿 skew 修正用)
    sf = latest_ranged_file(PROC, prefix)
    sv = pd.read_parquet(sf.path, columns=["date", "iv_put_25d"])
    sv["date"] = sv["date"].astype(str)
    df = df.merge(sv.rename(columns={"iv_put_25d": "iv_put"}), on="date", how="left")

    df = df.dropna(subset=["price", "iv"]).reset_index(drop=True)
    px = df["price"].astype(float)
    iv = df["iv"].astype(float)

    # 锚点: 过去 LOOKBACK 日最高收盘, 距今至少 MIN_DAYS(锥要张得开)
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
    for h in HORIZONS:
        df[f"fwd{h}"] = px.shift(-h) / px - 1
    df["name"] = name
    return df


def band_stats(d: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi, lab in BANDS:
        g = d[(d["z"] > lo) & (d["z"] <= hi)]
        rec = {"band": lab}
        for h in HORIZONS:
            r = g[f"fwd{h}"].dropna()
            rec[f"n{h}"] = len(r)
            rec[f"m{h}"] = round(float(r.mean()), 4) if len(r) >= 15 else None
            rec[f"w{h}"] = round(float((r > 0).mean()), 3) if len(r) >= 15 else None
        rows.append(rec)
    return rows


def main() -> None:
    frames, out = {}, {}
    for key in SOURCES:
        df = build_one(key)
        frames[key] = df
        out[key] = {
            "name": df["name"].iloc[0],
            "dates": [str(x) for x in df["date"]],
            "price": clean(df["price"], 4),
            "iv": clean(df["iv"], 4),
            "iv_put": clean(df["iv_put"], 4),
            "panic": clean(df["panic"], 3),
            "z": clean(df["z"], 3),
            "anchor": [None if pd.isna(a) else int(a) for a in df["anchor"]],
            "stats": band_stats(df.dropna(subset=["z"])),
        }
        n_valid = df["z"].notna().sum()
        print(f"{key:7s} {df['name'].iloc[0]:10s} {len(df)} 天 (z 有效 {n_valid})  "
              f"{df['date'].iloc[0]}->{df['date'].iloc[-1]}")

    pooled = pd.concat(frames.values(), ignore_index=True).dropna(subset=["z"])
    out["_pooled"] = band_stats(pooled)
    out["_meta"] = {"lookback": LOOKBACK, "min_days": MIN_DAYS,
                    "bands": [b[2] for b in BANDS], "horizons": list(HORIZONS)}

    dest = PROC / "cone_viz.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"\n池化样本 {len(pooled)} 天")
    print(f"写出 {dest} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
