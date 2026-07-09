"""Helpers for finding ranged market-data files.

Many generated files are named like ``surface_300etf_20200101_20260707.parquet``.
These helpers keep scripts from hard-coding the end date.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RangedFile:
    path: Path
    start: str
    end: str


def _parse_ranged_name(path: Path, prefix: str) -> RangedFile | None:
    stem = path.stem
    marker = f"{prefix}_"
    if not stem.startswith(marker):
        return None
    tail = stem[len(marker):]
    parts = tail.rsplit("_", 1)
    if len(parts) != 2:
        return None
    start, end = parts
    if len(start) != 8 or len(end) != 8 or not start.isdigit() or not end.isdigit():
        return None
    return RangedFile(path=path, start=start, end=end)


def latest_ranged_file(directory: Path, prefix: str) -> RangedFile | None:
    """Return the file with the latest end date, preferring the longest history."""
    matches = [
        parsed
        for path in directory.glob(f"{prefix}_*.parquet")
        if (parsed := _parse_ranged_name(path, prefix)) is not None
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: (item.end, -int(item.start)))


def read_ranged_parquets(directory: Path, prefix: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read and de-duplicate all parquet chunks for a ranged-file prefix."""
    frames = []
    for path in sorted(directory.glob(f"{prefix}_*.parquet")):
        if _parse_ranged_name(path, prefix) is None:
            continue
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError(f"No parquet files found for {prefix} in {directory}")
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
