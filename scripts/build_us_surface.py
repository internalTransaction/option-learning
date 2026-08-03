"""重建并缓存指定美股标的的历史曲面(供回测用, 不出可视化)。

用法:
    python -m scripts.build_us_surface SPY,SOXX 2022-01-03 2026-07-23 day
缓存名: us_surface_{key}_{gran}_{start}_{end}  (与 build_us_timing_viz 一致)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import cache
from src.research import us_surface_history as ush
from src.utils.logger import get_logger

log = get_logger("build_us_surface")


def main() -> None:
    tickers = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SPY", "SOXX"]
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-03"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-07-23"
    gran = sys.argv[4] if len(sys.argv) > 4 else "day"
    for t in tickers:
        key = t.lower()
        name = f"us_surface_{key}_{gran}_{start}_{end}"
        if cache.exists(name, "processed"):
            log.info("已缓存, 跳过 %s", name)
            continue
        log.info("重建 %s %s~%s (%s)", t, start, end, gran)
        df = ush.reconstruct(t, start, end, gran)
        cache.save(df, name, "processed")
        log.info("完成 %s: %d 时点 -> %s", t, len(df), name)


if __name__ == "__main__":
    main()
