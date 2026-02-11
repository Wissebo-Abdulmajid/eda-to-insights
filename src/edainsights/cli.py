from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from edainsights.config import load_config
from edainsights.io import load_table, make_run_context
from edainsights.profiling import ProfileConfig, profile_table
from edainsights.quality.runner import run_quality_checks
from edainsights.utils.hashing import file_sha256
from edainsights.utils.logging import setup_logger

# Milestone E
from edainsights.reporting import build_html_report

console = Console()
app = typer.Typer(add_completion=False, help="EDA-to-Insights Framework CLI")


@app.command("run")
def run(
    data: Path = typer.Option(..., "--data", exists=True, readable=True, help="Path to dataset file"),
    config: Path = typer.Option(..., "--config", exists=True, readable=True, help="Path to YAML config"),
    out: Path = typer.Option(Path("outputs/run"), "--out", help="Output directory"),
) -> None:
    """
    Run the EDA-to-Insights pipeline on a single dataset.
    Produces:
      - artifacts (csv/json)
      - logs
      - reports (interactive HTML)
    """
    cfg = load_config(config)
    ctx = make_run_context(out)

    logger = setup_logger(ctx.logs_dir, level=cfg.logging.level)
    logger.info("Starting run_id=%s", ctx.run_id)
    logger.info("Data path: %s", data)
    logger.info("Config path: %s", config)
    logger.info("Output dir: %s", out)

    # -------------------------
    # Load
    # -------------------------
    df = load_table(data, cfg)
    logger.info("Loaded dataset: rows=%d cols=%d", df.shape[0], df.shape[1])

    # -------------------------
    # Run metadata (audit trail)
    # -------------------------
    meta = {
        "project": {"name": cfg.project.name, "version": cfg.project.version},
        "run": {"run_id": ctx.run_id, "started_at_utc": ctx.started_at_utc},
        "input": {
            "data_path": str(data),
            "data_sha256": file_sha256(data),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
        },
        "config_path": str(config),
        "notes": [
            "This run loads data, runs data-quality checks, generates profiling artifacts, and writes an interactive HTML report.",
        ],
    }

    meta_path = ctx.artifacts_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote: %s", meta_path)

    # -------------------------
    # Preview export (small sample, safe)
    # -------------------------
    preview_rows = min(cfg.reporting.sample_rows, len(df))
    preview_path = ctx.artifacts_dir / "preview.csv"
    df.head(preview_rows).to_csv(preview_path, index=False)
    logger.info("Wrote: %s (rows=%d)", preview_path, preview_rows)

    # -------------------------
    # Milestone B: Quality checks
    # -------------------------
    issues = run_quality_checks(
        df=df,
        missing_threshold=cfg.quality.missing_threshold,
    )
    logger.info("Quality checks complete: issues=%d", len(issues))

    # Write issues.csv
    issues_path = ctx.artifacts_dir / "issues.csv"
    with open(issues_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["column", "issue_type", "severity", "message", "metric"],
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.__dict__)
    logger.info("Wrote: %s", issues_path)

    # Write quality_summary.json
    summary = {
        "total_issues": len(issues),
        "by_severity": {
            "error": sum(i.severity == "error" for i in issues),
            "warning": sum(i.severity == "warning" for i in issues),
            "info": sum(i.severity == "info" for i in issues),
        },
        "missing_threshold": float(cfg.quality.missing_threshold),
    }

    summary_path = ctx.artifacts_dir / "quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote: %s", summary_path)

    # -------------------------
    # Milestone C: Profiling (config-driven)
    # -------------------------
    # NOTE:
    # cfg.profiling.top_k_categories is about "top categories to display",
    # not "max unique for categorical inference". We'll set reasonable defaults:
    #
    # - max_unique_for_categorical: keep at 50 (heuristic)
    # - max_value_counts_rows: controlled by top_k_categories
    #
    prof_cfg = ProfileConfig(
        max_unique_for_categorical=50,
        max_value_counts_rows=cfg.profiling.top_k_categories,
        correlation_method=cfg.profiling.corr_method,
    )
    prof = profile_table(df, prof_cfg)

    # 1) dataset-level summary JSON
    profile_summary_path = ctx.artifacts_dir / "profile_summary.json"
    profile_summary_path.write_text(
        json.dumps(prof["dataset_summary"], indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote: %s", profile_summary_path)

    # 2) per-column profile CSV
    col_profile_path = ctx.artifacts_dir / "column_profile.csv"
    prof["column_profile"].to_csv(col_profile_path, index=False)
    logger.info("Wrote: %s", col_profile_path)

    # 3) numeric correlation CSV
    corr_path = ctx.artifacts_dir / "correlation_numeric.csv"
    if not prof["corr_numeric"].empty:
        prof["corr_numeric"].to_csv(corr_path)
        logger.info("Wrote: %s", corr_path)
    else:
        pd.DataFrame().to_csv(corr_path, index=False)
        logger.info("Wrote: %s (empty)", corr_path)

    # 4) bounded value-counts files (categorical-ish)
    for col, vc_df in prof["value_counts"].items():
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in col)[:80]
        vc_path = ctx.artifacts_dir / f"value_counts__{safe}.csv"
        vc_df.to_csv(vc_path, index=False)
        logger.info("Wrote: %s", vc_path)

    # -------------------------
    # Milestone E: HTML report (interactive)
    # -------------------------
    # IMPORTANT: report belongs in reports/ (not artifacts/)
    report_path = ctx.reports_dir / "report.html"
    try:
        build_html_report(
            df=df,
            artifacts_dir=ctx.artifacts_dir,
            out_path=report_path,
            project_name=cfg.project.name,
            run_id=ctx.run_id,
            config_path=str(config),
            plotly_mode=cfg.reporting.plotly_mode,
            include_corr=cfg.reporting.include_corr,
            top_k_categories=cfg.profiling.top_k_categories,
            corr_method=cfg.profiling.corr_method,
        )
        logger.info("Wrote: %s", report_path)
    except Exception as e:
        # We do not want the entire run to fail just because report generation broke.
        logger.exception("HTML report generation failed: %s", e)
    
    # -------------------------
    # Run Summary (single file for automation + portfolio)
    # -------------------------
    run_summary = {
        "run_id": ctx.run_id,
        "project": {"name": cfg.project.name, "version": cfg.project.version},
        "paths": {
            "artifacts_dir": str(ctx.artifacts_dir),
            "reports_dir": str(ctx.reports_dir),
            "logs_dir": str(ctx.logs_dir),
            "report_html": str(report_path),
        },
        "dataset": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "quality": summary,  # your quality_summary.json content
    }

    run_summary_path = ctx.artifacts_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    logger.info("Wrote: %s", run_summary_path)

    # -------------------------
    # Finish
    # -------------------------
    console.print("[green]Run complete.[/green]")
    console.print(f"Artifacts: {ctx.artifacts_dir}")
    console.print(f"Reports: {ctx.reports_dir}")
    console.print(f"Logs: {ctx.logs_dir}")


@app.command("version")
def version() -> None:
    console.print("edainsights 0.1.0")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
