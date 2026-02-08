from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileConfig:
    """
    Controls profiling output size / safety.

    max_unique_for_categorical:
      If a column has <= this number of unique values, we treat it as categorical-like
      and can compute value counts.
    max_value_counts_rows:
      Cap the number of rows written per value-counts file.
    correlation_method:
      "pearson" (default) is standard for numeric correlations.
    """
    max_unique_for_categorical: int = 50
    max_value_counts_rows: int = 30
    correlation_method: str = "pearson"


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float)):
            if np.isnan(x):
                return None
            return float(x)
        if isinstance(x, (np.integer, int)):
            return float(x)
        return float(x)
    except Exception:
        return None


def _infer_kind(series: pd.Series, unique_count: int) -> str:
    # "kind" is stable and easy to understand in artifacts
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # strings / objects / categories
    if unique_count <= 50:
        return "categorical"
    return "text"


def profile_table(df: pd.DataFrame, cfg: ProfileConfig) -> dict[str, Any]:
    """
    Returns a dict with:
      - dataset_summary: dict
      - column_profile: pd.DataFrame
      - corr_numeric: pd.DataFrame (may be empty)
      - value_counts: dict[str, pd.DataFrame]
    """
    n_rows, n_cols = df.shape

    # Column profile rows
    rows: list[dict[str, Any]] = []
    value_counts_tables: dict[str, pd.DataFrame] = {}

    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        missing_rate = float(n_missing / n_rows) if n_rows else 0.0
        unique_count = int(s.nunique(dropna=True))

        kind = _infer_kind(s, unique_count)

        rec: dict[str, Any] = {
            "column": str(col),
            "dtype": str(s.dtype),
            "kind": kind,
            "rows": int(n_rows),
            "missing": n_missing,
            "missing_rate": missing_rate,
            "unique": unique_count,
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            rec.update(
                {
                    "mean": _safe_float(s_num.mean()),
                    "std": _safe_float(s_num.std(ddof=1)),
                    "min": _safe_float(s_num.min()),
                    "p25": _safe_float(s_num.quantile(0.25)),
                    "median": _safe_float(s_num.median()),
                    "p75": _safe_float(s_num.quantile(0.75)),
                    "max": _safe_float(s_num.max()),
                }
            )
        else:
            # Non-numeric: keep numeric fields empty for clean CSV
            rec.update(
                {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "max": None,
                }
            )

        # Value counts (bounded & safe)
        if kind in {"categorical", "boolean"} and unique_count <= cfg.max_unique_for_categorical:
            vc = (
                s.astype("string")
                .fillna("<NA>")
                .value_counts(dropna=False)
                .head(cfg.max_value_counts_rows)
                .rename_axis("value")
                .reset_index(name="count")
            )
            vc["share"] = vc["count"] / max(n_rows, 1)
            value_counts_tables[str(col)] = vc

        rows.append(rec)

    column_profile = pd.DataFrame(rows)

    # Dataset summary
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dataset_summary: dict[str, Any] = {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "numeric_cols": int(len(numeric_cols)),
        "non_numeric_cols": int(n_cols - len(numeric_cols)),
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Correlation matrix (numeric only)
    if len(numeric_cols) >= 2:
        corr_numeric = df[numeric_cols].corr(method=cfg.correlation_method)
    else:
        corr_numeric = pd.DataFrame()

    return {
        "dataset_summary": dataset_summary,
        "column_profile": column_profile,
        "corr_numeric": corr_numeric,
        "value_counts": value_counts_tables,
    }

