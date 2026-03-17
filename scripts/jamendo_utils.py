"""
jamendo_utils.py - Helpers for MTG-Jamendo metadata and tar archives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_metadata(path: Path) -> pd.DataFrame:
    """Load Jamendo metadata (TSV/CSV) into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    if path.suffix.lower() in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the columns found: {list(candidates)}")


def normalize_ids(df: pd.DataFrame, id_col: str | None = None) -> pd.Series:
    if id_col is None:
        id_col = find_column(df, ["track_id", "id", "song_id", "file_id"])
    return df[id_col].astype(str)


def normalize_paths(df: pd.DataFrame, path_col: str | None = None) -> pd.Series:
    if path_col is None:
        path_col = find_column(df, ["path", "filepath", "audio_path", "file_path"])
    return df[path_col].astype(str)
