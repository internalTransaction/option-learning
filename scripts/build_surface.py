"""重建某标的的历史波动率曲面时序并落盘。

用法:
    python -m scripts.build_surface 300etf 20200101 20260707
"""
from __future__ import annotations

import sys

from src.research.surface_history import reconstruct
from src.utils.logger import get_logger

log = get_logger("build_surface")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "300etf"
    start = sys.argv[2] if len(sys.argv) > 2 else "20200101"
    end = sys.argv[3] if len(sys.argv) > 3 else "20260707"
    log.info("重建曲面时序: %s %s~%s", key, start, end)
    df = reconstruct(key, start, end, use_cache=True)
    log.info("完成: %d 日", len(df))
    cols = ["date", "spot", "atm_iv", "iv_ratio_25d", "iv_ratio_15m", "skew_25d", "pcr_oi", "pcr_amount"]
    print(df[cols].tail(10).to_string())


if __name__ == "__main__":
    main()
