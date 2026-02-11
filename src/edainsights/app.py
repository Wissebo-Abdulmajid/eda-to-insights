from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from edainsights.config import load_config
from edainsights.io import load_table, make_run_context
from edainsights.profiling import ProfileConfig, profile_table
from edainsights.quality.runner import run_quality_checks
from edainsights.reporting import build_html_report


def sniff_csv_delimiter(path: Path, default: str = ",") -> str:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:5000]
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return default


def write_issues_csv(path: Path, issues) -> None:
    """
    Robust CSV writer: safely handles commas/newlines/quotes in message fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["column", "issue_type", "severity", "message", "metric"])
        for i in issues:
            w.writerow([i.column, i.issue_type, i.severity, i.message, i.metric])


st.set_page_config(page_title="EDA-to-Insights", layout="wide")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("EDA-to-Insights")
st.caption("Upload a dataset → Get a decision-ready data report")

st.markdown(
    """
This tool automatically checks:
- dataset structure & size
- data quality issues (missingness, duplicates, outliers)
- distributions & anomalies
- relationships & correlations

**You upload one file. You download the report.**
"""
)

# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload dataset",
    type=["csv", "xlsx", "xls", "parquet"],
    help="Supported formats: CSV, Excel, Parquet",
)

# Preview (safe + consistent)
if uploaded_file is not None:
    st.success("Dataset uploaded")

    try:
        with tempfile.TemporaryDirectory() as prev_tmp:
            prev_tmp = Path(prev_tmp)
            prev_path = prev_tmp / uploaded_file.name
            prev_path.write_bytes(uploaded_file.getvalue())

            if prev_path.suffix.lower() == ".csv":
                sep = sniff_csv_delimiter(prev_path, default=",")
                preview_df = pd.read_csv(prev_path, sep=sep, nrows=10)
            elif prev_path.suffix.lower() in (".xlsx", ".xls"):
                # NOTE: requires openpyxl for .xlsx on most setups
                preview_df = pd.read_excel(prev_path, nrows=10)
            else:
                preview_df = pd.read_parquet(prev_path).head(10)

        st.caption("Preview (first 10 rows)")
        st.dataframe(preview_df, use_container_width=True)

    except Exception as e:
        st.info("Preview unavailable for this file type/encoding on this machine.")
        st.caption(f"Preview error: {e}")

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
run_clicked = st.button("Generate Report", type="primary", disabled=(uploaded_file is None))

if run_clicked and uploaded_file is not None:
    with st.spinner("Analyzing data and generating report..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save dataset
            data_path = tmpdir / uploaded_file.name
            data_path.write_bytes(uploaded_file.getvalue())

            # Load default config (hidden from users)
            cfg_path = Path("configs/default.yml")
            cfg = load_config(cfg_path)

            # CSV delimiter auto-detect (critical)
            if data_path.suffix.lower() == ".csv":
                detected = sniff_csv_delimiter(data_path, default=cfg.io.csv.delimiter)
                cfg.io.csv.delimiter = detected

            # Run context
            ctx = make_run_context(tmpdir / "outputs")
            ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
            ctx.reports_dir.mkdir(parents=True, exist_ok=True)

            # Load data (real pipeline load)
            df = load_table(data_path, cfg)

            # Quality checks
            issues = run_quality_checks(df=df, missing_threshold=cfg.quality.missing_threshold)

            # Profiling
            prof_cfg = ProfileConfig(
                max_unique_for_categorical=cfg.profiling.max_unique_for_categorical,
                max_value_counts_rows=cfg.profiling.top_k_categories,
                correlation_method=cfg.profiling.corr_method,
            )
            prof = profile_table(df, prof_cfg)

            # Write report-required artifacts (robust)
            write_issues_csv(ctx.artifacts_dir / "issues.csv", issues)

            (ctx.artifacts_dir / "quality_summary.json").write_text(
                pd.Series(
                    {
                        "total_issues": len(issues),
                        "by_severity": {
                            "error": sum(i.severity == "error" for i in issues),
                            "warning": sum(i.severity == "warning" for i in issues),
                            "info": sum(i.severity == "info" for i in issues),
                        },
                        "missing_threshold": float(cfg.quality.missing_threshold),
                    }
                ).to_json(indent=2),
                encoding="utf-8",
            )

            (ctx.artifacts_dir / "profile_summary.json").write_text(
                pd.Series(prof["dataset_summary"]).to_json(indent=2),
                encoding="utf-8",
            )

            prof["column_profile"].to_csv(ctx.artifacts_dir / "column_profile.csv", index=False)

            # Keep index for correlation matrix (your reader expects index_col=0)
            if "corr_numeric" in prof and prof["corr_numeric"] is not None:
                prof["corr_numeric"].to_csv(ctx.artifacts_dir / "correlation_numeric.csv")

            # Build report
            report_path = ctx.reports_dir / "report.html"
            build_html_report(
                df=df,
                artifacts_dir=ctx.artifacts_dir,
                out_path=report_path,
                project_name=cfg.project.name,
                run_id=ctx.run_id,
                config_path=str(cfg_path),
                include_corr=cfg.reporting.include_corr,
                top_k_categories=cfg.profiling.top_k_categories,
                corr_method=cfg.profiling.corr_method,
                plotly_mode=cfg.reporting.plotly_mode,
            )

            st.success("Report generated")

            # Downloads
            report_bytes = report_path.read_bytes()
            st.download_button(
                "Download HTML Report",
                data=report_bytes,
                file_name="eda_report.html",
                mime="text/html",
            )

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in ctx.artifacts_dir.rglob("*"):
                    if p.is_file():
                        z.write(p, arcname=f"artifacts/{p.name}")
                z.writestr("reports/report.html", report_bytes)

            st.download_button(
                "Download Full Package (ZIP)",
                data=zip_buf.getvalue(),
                file_name="eda_package.zip",
                mime="application/zip",
            )
