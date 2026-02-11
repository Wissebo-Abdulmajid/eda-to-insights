from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Insight:
    title: str
    severity: str  # "info" | "warning" | "action"
    message: str


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _has_col(df: pd.DataFrame | None, col: str) -> bool:
    return df is not None and not df.empty and col in df.columns


def _top_n_str(items: list[str], n: int = 3) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return ", ".join(items[:n])


def generate_insights(
    df: pd.DataFrame,
    issues_df: pd.DataFrame | None,
    profile_summary: dict[str, Any] | None,
    col_profile: pd.DataFrame | None,
    corr_df: pd.DataFrame | None,
    top_k: int = 12,
) -> list[Insight]:
    insights: list[Insight] = []

    # -------------------------
    # Dataset headline (stakeholder-friendly)
    # -------------------------
    if profile_summary:
        n_rows = int(profile_summary.get("n_rows", df.shape[0]))
        n_cols = int(profile_summary.get("n_cols", df.shape[1]))
        n_num = int(profile_summary.get("n_numeric", 0))
        n_cat = int(profile_summary.get("n_categorical", 0))
        dup = int(profile_summary.get("duplicate_rows", df.duplicated().sum()))
        miss_total = int(profile_summary.get("total_missing", int(df.isna().sum().sum())))

        msg = (
            f"Dataset contains {n_rows:,} rows × {n_cols} columns "
            f"({n_num} numeric, {n_cat} categorical-like). "
            f"Total missing cells: {miss_total:,}. Duplicate rows: {dup:,}."
        )

        sev = "info"
        if dup > 0 or miss_total > 0:
            sev = "warning"

        insights.append(
            Insight(
                title="Executive snapshot",
                severity=sev,
                message=msg,
            )
        )

    # -------------------------
    # Quality insights from issues.csv
    # -------------------------
    if issues_df is not None and not issues_df.empty and "issue_type" in issues_df.columns:
        # duplicates
        dup_row = issues_df[issues_df["issue_type"] == "duplicates"]
        if not dup_row.empty:
            ndup = int(float(dup_row.iloc[0].get("metric", 0)))
            severity = "action" if ndup > 0 else "info"
            insights.append(
                Insight(
                    title="Duplicate rows detected",
                    severity=severity,
                    message=(
                        f"{ndup} duplicate rows were detected. "
                        f"If duplicates are unintended, drop them or define a unique key to prevent re-ingestion."
                    ),
                )
            )

        # outliers summary (your quality module emits these as 'info')
        outlier_rows = issues_df[issues_df["issue_type"] == "outliers"]
        if not outlier_rows.empty:
            # show top 3 by metric (fraction)
            top = outlier_rows.sort_values("metric", ascending=False).head(3)
            parts = []
            for _, r in top.iterrows():
                col = str(r.get("column", ""))
                frac = _safe_float(r.get("metric"))
                if frac is not None:
                    parts.append(f"{col} ({_fmt_pct(frac)})")
            if parts:
                insights.append(
                    Insight(
                        title="Outlier hotspots (IQR)",
                        severity="warning",
                        message=(
                            f"Columns with the most outliers: {', '.join(parts)}. "
                            f"Outliers can be legitimate signals or data errors—verify against domain ranges before removing."
                        ),
                    )
                )

    # -------------------------
    # Column-level intelligence (uses new profiling fields)
    # -------------------------
    if col_profile is not None and not col_profile.empty:
        # 1) Missingness clustering
        if _has_col(col_profile, "missing_frac"):
            worst_missing = col_profile.sort_values("missing_frac", ascending=False).head(3)
            top_frac = _safe_float(worst_missing.iloc[0]["missing_frac"]) or 0.0
            if top_frac > 0:
                cols = ", ".join(
                    [f"{r['column']} ({_fmt_pct(float(r['missing_frac']))})" for _, r in worst_missing.iterrows()]
                )
                sev = "action" if top_frac >= 0.10 else "warning"
                insights.append(
                    Insight(
                        title="Missingness concentration",
                        severity=sev,
                        message=(
                            f"Highest missingness is concentrated in: {cols}. "
                            f"Best practice: impute (if missing-at-random), drop (if sparse), or fix upstream ingestion/schema."
                        ),
                    )
                )

        # 2) Possible identifier columns (near unique)
        if _has_col(col_profile, "unique") and (_has_col(col_profile, "n_rows") or _has_col(col_profile, "rows")):
            n_rows_col = "n_rows" if _has_col(col_profile, "n_rows") else "rows"
            ratios = (col_profile["unique"] / col_profile[n_rows_col]).fillna(0)
            maybe_id = col_profile[ratios > 0.95].head(3)
            if not maybe_id.empty:
                cols = ", ".join(maybe_id["column"].astype(str).tolist())
                insights.append(
                    Insight(
                        title="Potential ID/proxy columns",
                        severity="warning",
                        message=(
                            f"These columns look near-unique and may behave like IDs: {cols}. "
                            f"IDs can cause leakage or overfitting—exclude unless they have real predictive meaning."
                        ),
                    )
                )

        # 3) Dominant category detection (low entropy)
        if _has_col(col_profile, "top_value_share") and _has_col(col_profile, "kind"):
            cat = col_profile[col_profile["kind"].isin(["categorical", "boolean"])].copy()
            if not cat.empty:
                cat = cat.sort_values("top_value_share", ascending=False).head(3)
                top_share = _safe_float(cat.iloc[0]["top_value_share"])
                if top_share is not None and top_share >= 0.85:
                    parts = []
                    for _, r in cat.iterrows():
                        share = _safe_float(r.get("top_value_share"))
                        if share is None:
                            continue
                        parts.append(f"{r['column']} (top value = {r.get('top_value')}, share = {_fmt_pct(share)})")
                    insights.append(
                        Insight(
                            title="Highly imbalanced categorical columns",
                            severity="warning",
                            message=(
                                f"Some categorical columns are dominated by one value: {'; '.join(parts)}. "
                                f"This can reduce model usefulness and may indicate default values or data collection bias."
                            ),
                        )
                    )

        # 4) Numeric “range sanity” using p01/p99 (helps stakeholders)
        numeric = col_profile[col_profile.get("kind", "").eq("numeric")] if "kind" in col_profile.columns else col_profile
        if _has_col(numeric, "p01") and _has_col(numeric, "p99"):
            # pick 2-3 columns with widest relative spread (p99 - p01)
            tmp = numeric.copy()
            tmp["p01_f"] = pd.to_numeric(tmp["p01"], errors="coerce")
            tmp["p99_f"] = pd.to_numeric(tmp["p99"], errors="coerce")
            tmp["spread"] = (tmp["p99_f"] - tmp["p01_f"]).abs()
            tmp = tmp.sort_values("spread", ascending=False).head(3)

            parts = []
            for _, r in tmp.iterrows():
                c = str(r.get("column"))
                p01 = _safe_float(r.get("p01_f"))
                p99 = _safe_float(r.get("p99_f"))
                if p01 is None or p99 is None:
                    continue
                parts.append(f"{c}: p01={p01:.3g}, p99={p99:.3g}")

            if parts:
                insights.append(
                    Insight(
                        title="Typical numeric ranges (p01 → p99)",
                        severity="info",
                        message=(
                            "These ranges approximate where ~98% of values live and help spot unrealistic values: "
                            + "; ".join(parts)
                            + "."
                        ),
                    )
                )

        # 5) Skew/outlier warning using new fields
        if _has_col(col_profile, "outlier_frac_iqr") or _has_col(col_profile, "skew"):
            tmp = col_profile.copy()
            if "outlier_frac_iqr" in tmp.columns:
                tmp["out_f"] = pd.to_numeric(tmp["outlier_frac_iqr"], errors="coerce")
            else:
                tmp["out_f"] = None
            if "skew" in tmp.columns:
                tmp["skew_f"] = pd.to_numeric(tmp["skew"], errors="coerce")
            else:
                tmp["skew_f"] = None

            # choose top signals
            out_top = tmp.dropna(subset=["out_f"]).sort_values("out_f", ascending=False).head(2)
            skew_top = tmp.dropna(subset=["skew_f"]).assign(abs_skew=lambda d: d["skew_f"].abs()).sort_values("abs_skew", ascending=False).head(2)

            parts = []
            for _, r in out_top.iterrows():
                parts.append(f"{r['column']} outliers≈{_fmt_pct(float(r['out_f']))}")
            for _, r in skew_top.iterrows():
                parts.append(f"{r['column']} skew≈{float(r['skew_f']):+.2f}")

            if parts:
                insights.append(
                    Insight(
                        title="Distribution risks (skew/outliers)",
                        severity="warning",
                        message=(
                            "Some numeric columns are heavy-tailed or skewed: "
                            + "; ".join(parts)
                            + ". Consider robust scaling, winsorization, or transformations for modeling."
                        ),
                    )
                )

    # -------------------------
    # Correlation insights (with “business interpretation” tone)
    # -------------------------
    if corr_df is not None and not corr_df.empty:
        corr = corr_df.copy()
        cols = list(corr.columns)

        pairs: list[tuple[str, str, float]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    pairs.append((cols[i], cols[j], float(v)))

        if pairs:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top = pairs[:5]
            msg = "; ".join([f"{a} ↔ {b}: {v:+.2f}" for a, b, v in top])

            insights.append(
                Insight(
                    title="Strongest numeric relationships",
                    severity="info",
                    message=(
                        f"Top correlations (absolute): {msg}. "
                        f"Useful for feature engineering—but correlation is not causation."
                    ),
                )
            )

            # action flag if extreme correlation
            a, b, v = top[0]
            if abs(v) >= 0.85:
                insights.append(
                    Insight(
                        title="Multicollinearity / redundancy alert",
                        severity="action",
                        message=(
                            f"{a} and {b} are highly correlated ({v:+.2f}). "
                            f"For modeling: try dropping one, applying regularization, or compressing features (PCA)."
                        ),
                    )
                )

    # -------------------------
    # Clean “next actions” (not boring)
    # -------------------------
    # Make next steps conditional instead of generic
    action_steps: list[str] = []
    if issues_df is not None and not issues_df.empty:
        if (issues_df["issue_type"] == "duplicates").any():
            action_steps.append("Remove duplicates (or define a unique key) and re-run the pipeline.")
        if (issues_df["issue_type"] == "outliers").any():
            action_steps.append("Validate outlier ranges against domain constraints before trimming.")

    # If missingness exists
    if col_profile is not None and _has_col(col_profile, "missing_frac"):
        max_miss = pd.to_numeric(col_profile["missing_frac"], errors="coerce").max()
        if pd.notna(max_miss) and float(max_miss) > 0:
            action_steps.append("Decide: impute vs drop vs upstream fix for missing fields (document the choice).")

    # If correlation exists, mention leakage/proxy checks
    if corr_df is not None and not corr_df.empty and "quality" in df.columns:
        action_steps.append("If 'quality' is a target: run target-correlation checks to detect leakage/proxies.")

    if not action_steps:
        action_steps.append("Promote this run to a baseline: freeze config + dataset hash in run_metadata.json.")

    insights.append(
        Insight(
            title="Recommended next moves",
            severity="action",
            message=" ".join([f"({i+1}) {s}" for i, s in enumerate(action_steps[:4])]),
        )
    )

    return insights[:top_k]
