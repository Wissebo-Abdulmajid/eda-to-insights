from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


# ----------------------------
# Sub-configs
# ----------------------------
class QualityConfig(BaseModel):
    # Fraction of missing values allowed per column before flagging (used by your CLI today)
    missing_threshold: float = Field(0.10, ge=0.0, le=1.0)

    # Extra policy knobs (optional, future-proof)
    missingness_warn: float = Field(0.05, ge=0.0, le=1.0)
    missingness_fail: float = Field(0.40, ge=0.0, le=1.0)
    duplicate_warn: float = Field(0.01, ge=0.0, le=1.0)


class ProfilingConfig(BaseModel):
    top_k_categories: int = Field(10, ge=1, le=200)
    corr_method: Literal["pearson", "spearman", "kendall"] = "pearson"


class CsvConfig(BaseModel):
    delimiter: str = Field(default=",", min_length=1)
    encoding: str = Field(default="utf-8")
    na_values: list[str] = Field(default_factory=lambda: ["", "NA", "N/A", "null", "NULL", "None"])


class IoConfig(BaseModel):
    format: Literal["csv", "parquet", "excel"] = "csv"
    csv: CsvConfig = Field(default_factory=CsvConfig)


class ProjectConfig(BaseModel):
    name: str = "EDA-to-Insights Framework"
    version: str = "0.1.0"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ReportingConfig(BaseModel):
    sample_rows: int = Field(default=20, ge=0, le=500)
    include_corr: bool = True


# ----------------------------
# Root config
# ----------------------------
class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    io: IoConfig = Field(default_factory=IoConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        msg = f"Invalid config at {path}:\n{e}"
        raise SystemExit(msg) from e
