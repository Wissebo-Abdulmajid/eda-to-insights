from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .insights_engine import generate_insights


# -------------------------
# Helpers: reading artifacts
# -------------------------
def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


def _read_corr_csv(path: Path) -> pd.DataFrame | None:
    """
    correlation_numeric.csv is written with index.
    Read using index_col=0 to restore matrix.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------------------------
# Helpers: Plotly JS (offline embed)
# -------------------------
def _get_plotlyjs_inline() -> str:
    """
    Return Plotly.js bundle as a string for fully-offline HTML.
    Compatible across Plotly versions.
    """
    try:
        from plotly.offline import get_plotlyjs  # type: ignore

        return get_plotlyjs()
    except Exception:
        pass

    try:
        html = pio.to_html(go.Figure(), include_plotlyjs=True, full_html=False)
        marker = '<script type="text/javascript">'
        if marker in html:
            return html.split(marker, 1)[1].split("</script>", 1)[0]
        if "<script" in html:
            return html.split("<script", 1)[1].split("</script>", 1)[0]
    except Exception:
        pass

    raise RuntimeError(
        "Unable to embed Plotly.js offline. "
        "Upgrade plotly (pip install -U plotly) or switch template to CDN mode."
    )


# -------------------------
# Plot theme: ALWAYS readable in dark UI
# -------------------------
def _apply_readable_theme(fig: go.Figure) -> go.Figure:
    """
    Force a stakeholder-friendly LIGHT plot theme (white plot area),
    even if the overall report page is dark.
    Compatible with newer Plotly versions (no xaxis.titlefont).
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111", size=12),
        title=dict(font=dict(color="#111")),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.10)",
            borderwidth=1,
        ),
    )

    # 2D axes styling (NO titlefont here)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.12)",
        linecolor="rgba(0,0,0,0.18)",
        tickfont=dict(color="#111"),
        title=dict(font=dict(color="#111")),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.12)",
        linecolor="rgba(0,0,0,0.18)",
        tickfont=dict(color="#111"),
        title=dict(font=dict(color="#111")),
    )

    # 3D scene styling
    if hasattr(fig.layout, "scene") and fig.layout.scene is not None:
        fig.update_layout(
            scene=dict(
                bgcolor="white",
                xaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(0,0,0,0.10)",
                    zerolinecolor="rgba(0,0,0,0.14)",
                    color="#111",
                    title=dict(font=dict(color="#111")),
                ),
                yaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(0,0,0,0.10)",
                    zerolinecolor="rgba(0,0,0,0.14)",
                    color="#111",
                    title=dict(font=dict(color="#111")),
                ),
                zaxis=dict(
                    backgroundcolor="white",
                    gridcolor="rgba(0,0,0,0.10)",
                    zerolinecolor="rgba(0,0,0,0.14)",
                    color="#111",
                    title=dict(font=dict(color="#111")),
                ),
            )
        )

    return fig

# -------------------------
# Helpers: plotting
# -------------------------
def _plot_to_div(fig: go.Figure, *, include_plotlyjs: str | bool = False) -> str:
    """
    Convert Plotly figure to embeddable div.
    IMPORTANT: Apply readable theme here so EVERY chart is fixed.
    """
    fig = _apply_readable_theme(fig)
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False)


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").dropna(axis=1, how="all")


def _safe_sample(df: pd.DataFrame, max_rows: int = 7000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _pick_top_numeric_columns(num: pd.DataFrame, k: int = 8) -> list[str]:
    if num.empty:
        return []
    variances = num.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
    variances = variances[variances > 0]
    return variances.sort_values(ascending=False).head(k).index.tolist()


def _best_corr_pair(corr_df: pd.DataFrame) -> tuple[str, str, float] | None:
    if corr_df is None or corr_df.empty:
        return None
    cols = list(corr_df.columns)
    best: tuple[str, str, float] | None = None
    best_abs = -1.0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr_df.iloc[i, j]
            if pd.notna(v):
                av = abs(float(v))
                if av > best_abs:
                    best_abs = av
                    best = (cols[i], cols[j], float(v))
    return best


def _missingness_heatmap(df: pd.DataFrame, cols: list[str], max_rows: int = 240) -> go.Figure:
    subset = df[cols].head(max_rows)
    mat = subset.isna().astype(int)
    fig = px.imshow(
        mat.T,
        aspect="auto",
        title=f"Missingness heatmap (first {min(max_rows, len(df))} rows)",
        labels=dict(x="Row index", y="Column", color="Missing"),
        color_continuous_scale="Blues",
    )
    return fig


def _kpi_cards_figure(
    profile_summary: dict[str, Any] | None,
    quality_summary: dict[str, Any] | None,
) -> go.Figure:
    ps = profile_summary or {}
    qs = quality_summary or {}

    rows = int(ps.get("n_rows", ps.get("rows", 0)))
    cols = int(ps.get("n_cols", ps.get("cols", 0)))
    n_num = int(ps.get("n_numeric", ps.get("numeric_cols", 0)))
    n_cat = int(ps.get("n_categorical", 0))
    dup = int(ps.get("duplicate_rows", 0))
    miss = int(ps.get("total_missing", 0))

    issues = int(qs.get("total_issues", 0))
    sev = (qs.get("by_severity", {}) or {})
    warn = int(sev.get("warning", 0))
    err = int(sev.get("error", 0))

    metrics = [
        ("Rows", f"{rows:,}"),
        ("Columns", f"{cols:,}"),
        ("Numeric", f"{n_num:,}"),
        ("Categorical", f"{n_cat:,}"),
        ("Missing cells", f"{miss:,}"),
        ("Duplicate rows", f"{dup:,}"),
        ("Total issues", f"{issues:,}"),
        ("Warnings / Errors", f"{warn:,} / {err:,}"),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="rgba(0,0,0,0.05)",
                font=dict(size=12, color="#111"),
                align=["left", "right"],
                height=28,
            ),
            cells=dict(
                values=[[m[0] for m in metrics], [m[1] for m in metrics]],
                fill_color="white",
                font=dict(size=14, color="#111"),
                align=["left", "right"],
                height=28,
            ),
            columnwidth=[0.65, 0.35],
        )
    )
    fig.update_layout(title="Run KPIs", height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def _numeric_range_radar(col_profile: pd.DataFrame | None, top_n: int = 6) -> go.Figure | None:
    if col_profile is None or col_profile.empty:
        return None
    if not {"p01", "p99", "kind", "column"}.issubset(set(col_profile.columns)):
        return None

    num = col_profile[col_profile["kind"] == "numeric"].copy()
    if num.empty:
        return None

    num["p01_f"] = pd.to_numeric(num["p01"], errors="coerce")
    num["p99_f"] = pd.to_numeric(num["p99"], errors="coerce")
    num["spread"] = (num["p99_f"] - num["p01_f"]).abs()
    num = num.dropna(subset=["spread"]).sort_values("spread", ascending=False).head(top_n)
    if num.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=num["spread"].tolist(),
            theta=num["column"].astype(str).tolist(),
            fill="toself",
            name="p99 - p01 spread",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Typical spread (p01 → p99) • Radar view",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def _outlier_bubbles(col_profile: pd.DataFrame | None, top_n: int = 18) -> go.Figure | None:
    if col_profile is None or col_profile.empty:
        return None
    needed = {"outlier_frac_iqr", "skew", "missing_frac", "column", "kind"}
    if not needed.issubset(set(col_profile.columns)):
        return None

    tmp = col_profile[col_profile["kind"] == "numeric"].copy()
    if tmp.empty:
        return None

    tmp["out_f"] = pd.to_numeric(tmp["outlier_frac_iqr"], errors="coerce")
    tmp["skew_f"] = pd.to_numeric(tmp["skew"], errors="coerce")
    tmp["miss_f"] = pd.to_numeric(tmp["missing_frac"], errors="coerce").fillna(0.0)
    tmp = tmp.dropna(subset=["out_f", "skew_f"])
    if tmp.empty:
        return None

    tmp["score"] = tmp["out_f"].abs() + tmp["skew_f"].abs() / 6.0 + tmp["miss_f"]
    tmp = tmp.sort_values("score", ascending=False).head(top_n)

    fig = px.scatter(
        tmp,
        x="skew_f",
        y="out_f",
        size="miss_f",
        hover_name="column",
        title="Distribution risk map • skew vs outliers (bubble size = missingness)",
        labels={"skew_f": "Skew", "out_f": "Outlier fraction (IQR)", "miss_f": "Missingness"},
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# -------------------------
# Main report builder
# -------------------------
def build_html_report(
    df: pd.DataFrame,
    artifacts_dir: Path,
    out_path: Path,
    project_name: str,
    run_id: str,
    config_path: str,
    include_corr: bool = True,
    top_k_categories: int = 10,
    corr_method: str = "pearson",
    plotly_mode: str = "offline",  # "offline" (embed) or "cdn" (smaller html)
) -> Path:
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Load artifacts
    profile_summary = _read_json(artifacts_dir / "profile_summary.json")
    quality_summary = _read_json(artifacts_dir / "quality_summary.json")
    issues_df = _read_csv(artifacts_dir / "issues.csv")
    col_profile = _read_csv(artifacts_dir / "column_profile.csv")
    corr_df = _read_corr_csv(artifacts_dir / "correlation_numeric.csv") if include_corr else None

    # Plotly JS strategy
    mode = (plotly_mode or "offline").strip().lower()
    use_offline = mode in ("offline", "embed", "inline")
    plotly_js_inline = _get_plotlyjs_inline() if use_offline else None
    include_js = False if use_offline else "cdn"

    charts: dict[str, str] = {}

    # KPI strip
    fig_kpi = _kpi_cards_figure(profile_summary, quality_summary)
    charts["kpi_cards"] = _plot_to_div(fig_kpi, include_plotlyjs=include_js)

    # Prepare data for plotting
    df_plot = _safe_sample(df, max_rows=7000, seed=42)
    num = _safe_numeric(df_plot)
    top_num_cols = _pick_top_numeric_columns(num, k=8)

    # Chart A: Missingness bar + heatmap
    if col_profile is not None and {"missing_frac", "column"}.issubset(set(col_profile.columns)):
        miss = (
            col_profile[["column", "missing_frac"]]
            .copy()
            .sort_values("missing_frac", ascending=False)
            .head(30)
        )
        fig = px.bar(
            miss,
            x="column",
            y="missing_frac",
            title="Missingness by column (top 30)",
            hover_data={"missing_frac": ":.3f"},
        )
        fig.update_layout(xaxis_title="Column", yaxis_title="Missing fraction", height=420)
        charts["missingness_bar"] = _plot_to_div(fig, include_plotlyjs=include_js)

        top_missing_cols = miss["column"].head(min(12, len(miss))).tolist()
        if len(top_missing_cols) >= 2:
            fig_hm = _missingness_heatmap(df_plot, top_missing_cols, max_rows=240)
            fig_hm.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            charts["missingness_heatmap"] = _plot_to_div(fig_hm, include_plotlyjs=include_js)

    # Chart B: premium “risk” visuals
    fig_radar = _numeric_range_radar(col_profile, top_n=6)
    if fig_radar is not None:
        charts["range_radar"] = _plot_to_div(fig_radar, include_plotlyjs=include_js)

    fig_bubbles = _outlier_bubbles(col_profile, top_n=18)
    if fig_bubbles is not None:
        charts["risk_bubbles"] = _plot_to_div(fig_bubbles, include_plotlyjs=include_js)

    # Chart C: distributions
    for c in top_num_cols[:4]:
        fig = px.histogram(df_plot, x=c, nbins=50, marginal="box", title=f"Distribution (with box): {c}")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        charts[f"hist_{c}"] = _plot_to_div(fig, include_plotlyjs=include_js)

    if top_num_cols:
        melt_cols = top_num_cols[:6]
        mdf = df_plot[melt_cols].melt(var_name="feature", value_name="value")
        fig_box = px.box(
            mdf,
            x="feature",
            y="value",
            title="Outlier overview (boxplots of top numeric features)",
            points="outliers",
        )
        fig_box.update_layout(xaxis_title="Feature", yaxis_title="Value", height=460)
        charts["box_outliers_overview"] = _plot_to_div(fig_box, include_plotlyjs=include_js)

    # Chart D: scatter matrix
    if len(top_num_cols) >= 3:
        sm_cols = top_num_cols[:4]
        color_col = "quality" if "quality" in df_plot.columns else None
        fig_sm = px.scatter_matrix(
            df_plot,
            dimensions=sm_cols,
            color=color_col,
            title="Scatter Matrix (top numeric features)",
        )
        fig_sm.update_traces(diagonal_visible=False)
        fig_sm.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        charts["scatter_matrix"] = _plot_to_div(fig_sm, include_plotlyjs=include_js)

    # Chart E: correlation + top pairs + 3D plot
    if include_corr and corr_df is not None and not corr_df.empty:
        fig_corr = px.imshow(
            corr_df,
            title=f"Correlation Heatmap ({corr_method})",
            aspect="auto",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
        )
        fig_corr.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        charts["corr_heatmap"] = _plot_to_div(fig_corr, include_plotlyjs=include_js)

        pairs = []
        cols_ = list(corr_df.columns)
        for i in range(len(cols_)):
            for j in range(i + 1, len(cols_)):
                v = corr_df.iloc[i, j]
                if pd.notna(v):
                    pairs.append((cols_[i], cols_[j], float(v), abs(float(v))))
        if pairs:
            top_pairs = (
                pd.DataFrame(pairs, columns=["col_a", "col_b", "corr", "abs_corr"])
                .sort_values("abs_corr", ascending=False)
                .head(12)
            )
            fig_pairs = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=["Column A", "Column B", "Correlation", "|corr|"]),
                        cells=dict(
                            values=[
                                top_pairs["col_a"].tolist(),
                                top_pairs["col_b"].tolist(),
                                [f"{v:.3f}" for v in top_pairs["corr"].tolist()],
                                [f"{v:.3f}" for v in top_pairs["abs_corr"].tolist()],
                            ]
                        ),
                    )
                ]
            )
            fig_pairs.update_layout(title="Top correlation pairs (numeric)", height=420)
            charts["corr_top_pairs_table"] = _plot_to_div(fig_pairs, include_plotlyjs=include_js)

        best = _best_corr_pair(corr_df)
        if best and len(top_num_cols) >= 3:
            x, y, corr_val = best
            third = [c for c in top_num_cols if c not in (x, y)][0]
            color_col = "quality" if "quality" in df_plot.columns else None

            fig3d = px.scatter_3d(
                df_plot,
                x=x,
                y=y,
                z=third,
                color=color_col,
                opacity=0.8,
                title=f"3D Relationship: {x} vs {y} vs {third} (corr={corr_val:.3f})"
                + ("" if not color_col else " • colored by quality"),
            )
            fig3d.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10))
            charts["scatter_3d"] = _plot_to_div(fig3d, include_plotlyjs=include_js)

    # Chart F: categorical snapshot
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    chosen_cat = None
    for c in cat_cols:
        nunq = df[c].nunique(dropna=False)
        if 2 <= nunq <= 50:
            chosen_cat = c
            break

    if chosen_cat:
        vc = df[chosen_cat].value_counts(dropna=False).head(top_k_categories).reset_index()
        vc.columns = [chosen_cat, "count"]
        fig_cat = px.bar(vc, x=chosen_cat, y="count", title=f"Top categories: {chosen_cat}")
        fig_cat.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        charts["top_categories"] = _plot_to_div(fig_cat, include_plotlyjs=include_js)

    # Narrative insights
    insights = generate_insights(
        df=df,
        issues_df=issues_df,
        profile_summary=profile_summary,
        col_profile=col_profile,
        corr_df=corr_df,
        top_k=12,
    )

    # Render HTML
    template = env.get_template("report.html.j2")
    html = template.render(
        project_name=project_name,
        run_id=run_id,
        config_path=config_path,
        plotly_mode=plotly_mode,
        profile_summary=profile_summary,
        quality_summary=quality_summary,
        insights=insights,
        charts=charts,
        plotly_js_inline=plotly_js_inline,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
