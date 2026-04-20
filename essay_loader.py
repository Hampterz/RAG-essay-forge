"""
Shared helpers to load and normalize the three essay CSV sources into one schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CSV_FILES = [
    "collegebase_essays.csv",
    "essaysthatworked_essays.csv",
    "openessays_essays.csv",
]


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def _enrich_collegebase_metadata(row: pd.Series) -> str:
    base = row.get("metadata")
    if base is None or (isinstance(base, float) and pd.isna(base)):
        base = ""
    else:
        base = str(base).strip()
    extra: dict[str, object] = {}
    for key in ("category", "prompt", "word_count"):
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            s = str(v).strip()
            if s:
                extra[key] = v
    if not extra:
        return base
    try:
        merged = json.loads(base) if base else {}
        if not isinstance(merged, dict):
            merged = {"original_metadata": base}
    except json.JSONDecodeError:
        merged = {"original_metadata": base}
    merged.update({k: str(v) for k, v in extra.items()})
    return json.dumps(merged, ensure_ascii=False)


def load_merged_essays(project_dir: Path | None = None) -> pd.DataFrame:
    """Load all CSVs and return a unified DataFrame with standard columns."""
    root = project_dir or _project_dir()
    frames: list[pd.DataFrame] = []
    for name in CSV_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing CSV: {path}. Place {name} in the project directory."
            )
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        df["_source_file"] = name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    out = pd.DataFrame()
    out["school"] = merged.get("school", "").fillna("").astype(str).str.strip()
    out["essay_type"] = merged.get("essay_type", "").fillna("").astype(str).str.strip()
    out["essay_text"] = merged.get("essay_text", "").fillna("").astype(str)

    def _meta_cell(r: pd.Series) -> str:
        if str(r.get("_source_file", "")) == "collegebase_essays.csv":
            return _enrich_collegebase_metadata(r)
        m = r.get("metadata")
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return ""
        return str(m).strip()

    meta_series = merged.apply(_meta_cell, axis=1)
    out["metadata"] = meta_series

    if "grade" in merged.columns:
        g = merged["grade"].replace("", pd.NA)
        out["grade"] = g
    else:
        out["grade"] = pd.NA

    if "grade_reason" in merged.columns:
        out["grade_reason"] = merged["grade_reason"].replace("", pd.NA)
    else:
        out["grade_reason"] = pd.NA

    out["_source_file"] = merged["_source_file"]
    return out


def filter_valid_essays(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with empty or very short essay_text."""
    text = df["essay_text"].fillna("").astype(str)
    mask = text.str.len() >= 100
    return df.loc[mask].copy()


def needs_grade(value: object) -> bool:
    """True if grade is missing, blank, or unknown (case-insensitive)."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    s = str(value).strip()
    if not s:
        return True
    if s.lower() == "unknown":
        return True
    return False
