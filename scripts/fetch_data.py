"""拉取并缓存所有 enabled 标的的历史数据 (QVIX + ETF 日线)。

用法:
    python -m scripts.fetch_data              # 用缓存(若有)
    python -m scripts.fetch_data --refresh    # 强制重新拉取
    python -m scripts.fetch_data --snapshot   # 额外拉一份当日期权链快照
"""
from __future__ import annotations

import argparse

from src.data import akshare_loader as loader
from src.utils.config import enabled_underlyings
from src.utils.logger import get_logger

log = get_logger("fetch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取")
    ap.add_argument("--snapshot", action="store_true", help="额外拉取当日期权链快照")
    args = ap.parse_args()
    use_cache = not args.refresh

    for key, u in enabled_underlyings().items():
        log.info("=== %s (%s) ===", key, u["name"])
        try:
            q = loader.fetch_qvix(u["qvix"], use_cache=use_cache)
            log.info("  QVIX: %d 行 (%s -> %s)", len(q), q["date"].min().date(), q["date"].max().date())
        except Exception as e:  # noqa: BLE001
            log.error("  QVIX 失败: %s", e)
        try:
            e_df = loader.fetch_underlying(u["etf_code"], exchange=u["exchange"], use_cache=use_cache)
            log.info("  ETF %s: %d 行", u["etf_code"], len(e_df))
        except Exception as e:  # noqa: BLE001
            log.error("  ETF 失败: %s", e)

    if args.snapshot:
        try:
            chain = loader.fetch_option_chain(use_cache=False)
            log.info("期权链快照: %d 行, 列: %s", len(chain), list(chain.columns))
        except Exception as e:  # noqa: BLE001
            log.error("期权链快照失败: %s", e)


if __name__ == "__main__":
    main()
