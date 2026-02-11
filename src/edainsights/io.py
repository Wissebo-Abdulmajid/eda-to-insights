from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edainsights.config import Config


@dataclass(frozen=True)
class RunContext:
    run_id: str
    started_at_utc: str
    out_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    reports_dir: Path  # NEW: where HTML reports (and future exports) will go


def _utc_run_id() -> str:
    """
    Generate a readable UTC run_id suitable for folder naming.
    Example: 2026-02-08T13-11-53Z
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def make_run_context(out_dir: Path) -> RunContext:
    """
    Create a run context with deterministic output folders.
    We always create:
      - artifacts/  (csv/json outputs)
      - logs/       (run.log)
      - reports/    (HTML report + future exports)
    """
    out_dir = Path(out_dir)

    started = _utc_run_id()
    run_id = started  # readable; later we can switch to UUID safely without breaking structure

    artifacts_dir = out_dir / "artifacts"
    logs_dir = out_dir / "logs"
    reports_dir = out_dir / "reports"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        started_at_utc=started,
        out_dir=out_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
    )


def load_table(data_path: Path, cfg: Config) -> pd.DataFrame:
    """
    Load a tabular dataset based on config.
    Supported: CSV / Parquet / Excel.

    Notes:
    - CSV uses delimiter/encoding/na_values from cfg.io.csv
    - keep_default_na=True ensures pandas also recognizes its default NA tokens.
    """
    data_path = Path(data_path)
    fmt = cfg.io.format

    if fmt == "csv":
        c = cfg.io.csv
        df = pd.read_csv(
            data_path,
            sep=c.delimiter,
            encoding=c.encoding,
            na_values=c.na_values,
            keep_default_na=True,
        )
        return df

    if fmt == "parquet":
        return pd.read_parquet(data_path)

    if fmt == "excel":
        return pd.read_excel(data_path)

    raise ValueError(f"Unsupported format: {fmt}")
