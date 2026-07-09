"""信号可信度检验：把"看着有效"收敛到"经得起检验"。

针对前瞻收益重叠夸大显著性、单一 regime 主导等问题, 提供:
  ic_overlap        全样本重叠 IC(会高估显著性, 仅作参照)
  ic_nonoverlap     每 h 日取一个不重叠样本的 IC + t 值 + p 值(诚实的显著性)
  ic_by_year        分年度 IC(看是否稳定, 还是被某年主导)
  ic_winsorized     对信号与收益各截尾 1%/99% 后的 IC(看是否被极端崩盘日撑起)
  ic_drop_crisis    剔除已实现波动最高的 5% 交易日后的 IC(去危机日)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 10:
        return float("nan"), len(d)
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())), len(d)


def _t_p(ic: float, n: int) -> tuple[float, float]:
    if not np.isfinite(ic) or n < 5 or abs(ic) >= 1:
        return float("nan"), float("nan")
    t = ic * np.sqrt((n - 2) / (1 - ic ** 2))
    p = 2 * (1 - stats.t.cdf(abs(t), n - 2))
    return float(t), float(p)


def reliability(df: pd.DataFrame, signal: str, h: int = 10) -> dict:
    """对单个信号做全套可信度检验。df 需含 signal 列、fwd_ret_h 列、date、rv(可选)。"""
    ret = f"fwd_ret_{h}"
    d = df.dropna(subset=[signal, ret]).copy()
    out = {"signal": signal, "h": h}

    ic_o, n_o = _ic(d[signal], d[ret])
    out["ic_overlap"] = round(ic_o, 4)
    out["n_overlap"] = n_o

    # 不重叠: 每 h 行取一个, 使前瞻窗口互不交叠
    nz = d.iloc[::h]
    ic_n, n_n = _ic(nz[signal], nz[ret])
    t, p = _t_p(ic_n, n_n)
    out["ic_nonoverlap"] = round(ic_n, 4)
    out["n_nonoverlap"] = n_n
    out["t_stat"] = round(t, 2)
    out["p_value"] = round(p, 4)

    # 分年度(不重叠不做, 直接重叠但看方向一致性)
    d["year"] = d["date"].astype(str).str[:4]
    by = {}
    for yr, g in d.groupby("year"):
        ic_y, n_y = _ic(g[signal], g[ret])
        by[yr] = round(ic_y, 3)
    out["ic_by_year"] = by
    signs = [v for v in by.values() if np.isfinite(v)]
    out["year_sign_consistency"] = round(np.mean([np.sign(s) == np.sign(ic_o) for s in signs]), 2) if signs else None

    # 截尾: 信号与收益各 winsorize 到 1/99 分位
    dw = d.copy()
    for c in (signal, ret):
        lo, hi = dw[c].quantile([0.01, 0.99])
        dw[c] = dw[c].clip(lo, hi)
    ic_w, _ = _ic(dw[signal], dw[ret])
    out["ic_winsorized"] = round(ic_w, 4)

    # 去危机日: 剔除 rv 最高 5%
    if "rv" in d.columns and d["rv"].notna().any():
        thr = d["rv"].quantile(0.95)
        dc = d[d["rv"] < thr]
        ic_c, _ = _ic(dc[signal], dc[ret])
        out["ic_drop_crisis"] = round(ic_c, 4)

    return out
