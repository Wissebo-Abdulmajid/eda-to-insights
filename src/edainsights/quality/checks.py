from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
import pandas as pd
import numpy as np


@dataclass
class Issue:
    column: str
    issue_type: str
    severity: str
    message: str
    metric: Any


def check_missingness(df: pd.DataFrame, threshold: float) -> List[Issue]:
    issues: List[Issue] = []

    missing_pct = df.isna().mean()

    for col, pct in missing_pct.items():
        if pct > threshold:
            issues.append(
                Issue(
                    column=col,
                    issue_type="missingness",
                    severity="warning" if pct < 0.5 else "error",
                    message=f"Missing rate {pct:.2%} exceeds threshold {threshold:.2%}",
                    metric=float(pct),
                )
            )
    return issues


def check_duplicates(df: pd.DataFrame) -> List[Issue]:
    issues: List[Issue] = []

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(
            Issue(
                column="__row_level__",
                issue_type="duplicates",
                severity="warning",
                message=f"{dup_count} duplicate rows detected",
                metric=int(dup_count),
            )
        )
    return issues


def check_constant_columns(df: pd.DataFrame) -> List[Issue]:
    issues: List[Issue] = []

    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 1:
            issues.append(
                Issue(
                    column=col,
                    issue_type="constant_column",
                    severity="warning",
                    message="Column has no variance",
                    metric=int(nunique),
                )
            )
    return issues


def check_numeric_outliers(df: pd.DataFrame) -> List[Issue]:
    issues: List[Issue] = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_rate = ((series < lower) | (series > upper)).mean()

        if outlier_rate > 0:
            issues.append(
                Issue(
                    column=col,
                    issue_type="outliers",
                    severity="info",
                    message=f"Outliers detected using IQR method",
                    metric=float(outlier_rate),
                )
            )

    return issues

