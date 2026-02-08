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


def make_run_context(out_dir: Path) -> RunContext:
    """
    Create a run context with deterministic output folders.
    """
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_id = started  # simple, readable; we can change later to UUID if needed

    artifacts_dir = out_dir / "artifacts"
    logs_dir = out_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        started_at_utc=started,
        out_dir=out_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
    )


def load_table(data_path: Path, cfg: Config) -> pd.DataFrame:
    """
    Load a tabular dataset based on config.
    Current scope: CSV/Parquet/Excel.
    """
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

