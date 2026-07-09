"""配置加载工具。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# 项目根目录 = 本文件上溯三级 (src/utils/config.py -> 根)
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """读取并缓存全局配置。"""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def enabled_underlyings(cfg: dict[str, Any] | None = None) -> dict[str, dict]:
    """返回 enabled=true 的标的字典。"""
    cfg = cfg or load_config()
    return {k: v for k, v in cfg["underlyings"].items() if v.get("enabled")}


def abspath(rel: str) -> Path:
    """相对项目根目录的路径 -> 绝对路径。"""
    return ROOT / rel
