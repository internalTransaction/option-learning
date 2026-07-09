"""本地数据缓存：把拉取到的行情落盘，避免重复请求。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import abspath, load_config
from src.utils.logger import get_logger

log = get_logger("data.cache")


def _cache_path(name: str, subdir: str = "raw") -> Path:
    cfg = load_config()
    fmt = cfg["cache"]["format"]
    base = abspath(cfg["cache"][f"{subdir}_dir"])
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.{fmt}"


def save(df: pd.DataFrame, name: str, subdir: str = "raw") -> Path:
    path = _cache_path(name, subdir)
    fmt = load_config()["cache"]["format"]
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("cached %s rows -> %s", len(df), path.name)
    return path


def load(name: str, subdir: str = "raw") -> pd.DataFrame | None:
    path = _cache_path(name, subdir)
    if not path.exists():
        return None
    fmt = load_config()["cache"]["format"]
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)


def exists(name: str, subdir: str = "raw") -> bool:
    return _cache_path(name, subdir).exists()
