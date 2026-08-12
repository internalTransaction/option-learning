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

# 周期重置带的分档(不用 -2.5σ, 固定期限下极少触及, 样本太薄)
PBANDS = [(-99, -2, "≤-2σ"), (-2, -1.5, "-2~-1.5σ"), (-1.5, -1, "-1.5~-1σ"),
          (-1, -0.5, "-1~-0.5σ"), (-0.5, 0, "-0.5~0σ"), (0, 99, ">0σ")]
PERIOD_N = {"W": 5, "M": 21}


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


def period_cols(df: pd.DataFrame, freq: str) -> dict:
    """周期重置带: 期初(上期最后一个交易日)定死 base 与 sigma, 期内不再变。

    -1σ 因此在期初就是一个确定价位, 可以直接挂单 —— 这是它相对锚定锥的关键差别:
    锚定锥的 sigma 随"已跌了多少天"张开, 而开跌当天并不知道会跌多久。
    """
    n = PERIOD_N[freq]
    dt = pd.to_datetime(df["date"], format="%Y%m%d")
    per = dt.dt.to_period(freq)
    periods = list(dict.fromkeys(per))
    base = pd.Series(np.nan, index=df.index)
    sig = pd.Series(np.nan, index=df.index)
    px, iv = df["price"].astype(float), df["iv"].astype(float)
    for pi, p in enumerate(periods):
        idx = df.index[per == p]
        if pi == 0:
            continue
        prev = df.index[per == periods[pi - 1]][-1]
        base.loc[idx] = px.loc[prev]
        sig.loc[idx] = iv.loc[prev] * np.sqrt(n / 252)
    return {"base": base, "sigma": sig, "z": (px / base - 1) / sig,
            "first": [bool(x) for x in (per != per.shift(1)).tolist()]}


def band_stats(d: pd.DataFrame, bands=BANDS, zcol: str = "z") -> list[dict]:
    rows = []
    for lo, hi, lab in bands:
        g = d[(d[zcol] > lo) & (d[zcol] <= hi)]
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
        rec = {
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
        for freq in PERIOD_N:
            col = period_cols(df, freq)
            df[f"z{freq}"] = col["z"]
            rec[freq] = {
                "base": clean(col["base"], 4), "sigma": clean(col["sigma"], 4),
                "z": clean(col["z"], 3), "first": col["first"],
                "stats": band_stats(df.dropna(subset=[f"z{freq}"]), PBANDS, f"z{freq}"),
            }
        frames[key] = df
        out[key] = rec
        print(f"{key:7s} {df['name'].iloc[0]:10s} {len(df)} 天  "
              f"(锚定 {df['z'].notna().sum()} / 周 {df['zW'].notna().sum()} "
              f"/ 月 {df['zM'].notna().sum()})  {df['date'].iloc[0]}->{df['date'].iloc[-1]}")

    pool = pd.concat(frames.values(), ignore_index=True)
    out["_pooled"] = band_stats(pool.dropna(subset=["z"]))
    out["_pooledW"] = band_stats(pool.dropna(subset=["zW"]), PBANDS, "zW")
    out["_pooledM"] = band_stats(pool.dropna(subset=["zM"]), PBANDS, "zM")
    out["_meta"] = {"lookback": LOOKBACK, "min_days": MIN_DAYS,
                    "bands": [b[2] for b in BANDS], "pbands": [b[2] for b in PBANDS],
                    "period_n": PERIOD_N, "horizons": list(HORIZONS)}

    dest = PROC / "cone_viz.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"\n池化样本 锚定{pool['z'].notna().sum()} / 周{pool['zW'].notna().sum()} / 月{pool['zM'].notna().sum()}")
    print(f"写出 {dest} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
