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
        # numpy scalar handling
        if isinstance(x, (np.floating, float)):
            if np.isnan(x):
                return None
            return float(x)
        if isinstance(x, (np.integer, int)):
            return float(x)
        return float(x)
    except Exception:
        return None


def _infer_kind(series: pd.Series, unique_count: int, cfg: ProfileConfig) -> str:
    """
    Stable “kind” labels used in artifacts/reports.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # strings / objects / categories
    if unique_count <= cfg.max_unique_for_categorical:
        return "categorical"
    return "text"


def _iqr_outlier_frac(s_num: pd.Series) -> float | None:
    """
    Outlier fraction based on IQR rule. Returns fraction of non-null rows flagged.
    """
    try:
        x = pd.to_numeric(s_num, errors="coerce").dropna()
        n = len(x)
        if n < 8:
            return None
        q1 = float(x.quantile(0.25))
        q3 = float(x.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            return 0.0
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        frac = float(((x < lo) | (x > hi)).mean())
        return frac
    except Exception:
        return None


def profile_table(df: pd.DataFrame, cfg: ProfileConfig) -> dict[str, Any]:
    """
    Returns a dict with:
      - dataset_summary: dict
      - column_profile: pd.DataFrame
      - corr_numeric: pd.DataFrame (may be empty)
      - value_counts: dict[str, pd.DataFrame]
    """
    n_rows, n_cols = df.shape

    rows: list[dict[str, Any]] = []
    value_counts_tables: dict[str, pd.DataFrame] = {}

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_frac = float(missing_count / n_rows) if n_rows else 0.0
        unique_count = int(s.nunique(dropna=True))
        kind = _infer_kind(s, unique_count, cfg)

        rec: dict[str, Any] = {
            "column": str(col),
            "dtype": str(s.dtype),
            "kind": kind,

            # keep BOTH names for compatibility
            "rows": int(n_rows),
            "n_rows": int(n_rows),

            "missing": missing_count,
            "missing_count": missing_count,

            "missing_rate": missing_frac,   # old name
            "missing_frac": missing_frac,   # new name expected by report

            "unique": unique_count,
        }

        # -------------------------
        # Numeric stats
        # -------------------------
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")

            rec.update(
                {
                    "mean": _safe_float(s_num.mean()),
                    "std": _safe_float(s_num.std(ddof=1)),
                    "min": _safe_float(s_num.min()),
                    "p01": _safe_float(s_num.quantile(0.01)),
                    "p05": _safe_float(s_num.quantile(0.05)),
                    "p25": _safe_float(s_num.quantile(0.25)),
                    "median": _safe_float(s_num.median()),
                    "p75": _safe_float(s_num.quantile(0.75)),
                    "p95": _safe_float(s_num.quantile(0.95)),
                    "p99": _safe_float(s_num.quantile(0.99)),
                    "max": _safe_float(s_num.max()),
                    "skew": _safe_float(s_num.skew()),
                    "kurtosis": _safe_float(s_num.kurtosis()),
                    "outlier_frac_iqr": _iqr_outlier_frac(s_num),
                }
            )
        else:
            # Keep numeric fields present (clean CSV schema)
            rec.update(
                {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p01": None,
                    "p05": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "p95": None,
                    "p99": None,
                    "max": None,
                    "skew": None,
                    "kurtosis": None,
                    "outlier_frac_iqr": None,
                }
            )

            # Extra “text/categorical” hints (small but useful)
            try:
                s_str = s.astype("string")
                non_null = s_str.dropna()
                if len(non_null) > 0:
                    rec["avg_len"] = _safe_float(non_null.str.len().mean())
                else:
                    rec["avg_len"] = None
            except Exception:
                rec["avg_len"] = None

        # -------------------------
        # Value counts (bounded & safe)
        # -------------------------
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

            # Also attach top-category summary to column_profile row
            top_val = str(vc.loc[0, "value"]) if len(vc) else None
            top_cnt = int(vc.loc[0, "count"]) if len(vc) else None
            top_share = float(vc.loc[0, "share"]) if len(vc) else None
            rec["top_value"] = top_val
            rec["top_value_count"] = top_cnt
            rec["top_value_share"] = top_share
        else:
            rec["top_value"] = None
            rec["top_value_count"] = None
            rec["top_value_share"] = None

        rows.append(rec)

    column_profile = pd.DataFrame(rows)

    # -------------------------
    # Dataset summary (IMPORTANT: match report expectations)
    # -------------------------
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_like_cols = [
        c for c in df.columns
        if _infer_kind(df[c], int(df[c].nunique(dropna=True)), cfg) == "categorical"
    ]

    dataset_summary: dict[str, Any] = {
        # new names expected by report/template
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_numeric": int(len(numeric_cols)),
        "n_categorical": int(len(cat_like_cols)),

        # keep older names too (backward compatible)
        "rows": int(n_rows),
        "cols": int(n_cols),
        "numeric_cols": int(len(numeric_cols)),
        "non_numeric_cols": int(n_cols - len(numeric_cols)),

        "total_missing": int(df.isna().sum().sum()),
        "missing_frac_total": float(df.isna().sum().sum() / max(n_rows * max(n_cols, 1), 1)),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # -------------------------
    # Correlation matrix (numeric only)
    # -------------------------
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
