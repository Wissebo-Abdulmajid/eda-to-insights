from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


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


class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    io: IoConfig = Field(default_factory=IoConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def load_config(path: Path) -> Config:
    """
    Load YAML config and validate it with Pydantic.
    Hard-fail on invalid config (production-adjacent behavior).
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        msg = f"Invalid config at {path}:\n{e}"
        raise SystemExit(msg) from e

