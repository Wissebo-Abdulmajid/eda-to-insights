from __future__ import annotations

from typing import List
import pandas as pd

from .checks import (
    Issue,
    check_missingness,
    check_duplicates,
    check_constant_columns,
    check_numeric_outliers,
)


def run_quality_checks(
    df: pd.DataFrame,
    missing_threshold: float,
) -> List[Issue]:
    issues: List[Issue] = []

    issues.extend(check_missingness(df, missing_threshold))
    issues.extend(check_duplicates(df))
    issues.extend(check_constant_columns(df))
    issues.extend(check_numeric_outliers(df))

    return issues

