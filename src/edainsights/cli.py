from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from edainsights.config import load_config
from edainsights.io import load_table, make_run_context
from edainsights.utils.hashing import file_sha256
from edainsights.utils.logging import setup_logger

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
    """
    cfg = load_config(config)
    ctx = make_run_context(out)

    logger = setup_logger(ctx.logs_dir, level=cfg.logging.level)
    logger.info("Starting run_id=%s", ctx.run_id)
    logger.info("Data path: %s", data)
    logger.info("Config path: %s", config)
    logger.info("Output dir: %s", out)

    # Load
    df = load_table(data, cfg)
    logger.info("Loaded dataset: rows=%d cols=%d", df.shape[0], df.shape[1])

    # Run metadata (audit trail)
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
            "This run only validates config + loads data + writes run metadata.",
            "Quality checks and profiling are added in later milestones.",
        ],
    }

    meta_path = ctx.artifacts_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote: %s", meta_path)

    # Preview export (small sample, safe)
    preview_rows = min(cfg.reporting.sample_rows, len(df))
    preview_path = ctx.artifacts_dir / "preview.csv"
    df.head(preview_rows).to_csv(preview_path, index=False)
    logger.info("Wrote: %s (rows=%d)", preview_path, preview_rows)

    console.print("[green]Run complete.[/green]")
    console.print(f"Artifacts: {ctx.artifacts_dir}")
    console.print(f"Logs: {ctx.logs_dir}")


@app.command("version")
def version() -> None:
    console.print("edainsights 0.1.0")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
