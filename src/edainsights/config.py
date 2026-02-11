from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict


# ----------------------------
# Sub-configs
# ----------------------------
class QualityConfig(BaseModel):
    # Fraction of missing values allowed per column before flagging (used by CLI today)
    missing_threshold: float = Field(0.10, ge=0.0, le=1.0)

    # Extra policy knobs (optional, future-proof)
    missingness_warn: float = Field(0.05, ge=0.0, le=1.0)
    missingness_fail: float = Field(0.40, ge=0.0, le=1.0)
    duplicate_warn: float = Field(0.01, ge=0.0, le=1.0)


class ProfilingConfig(BaseModel):
    top_k_categories: int = Field(10, ge=1, le=200)
    corr_method: Literal["pearson", "spearman", "kendall"] = "pearson"

    # Optional: cap heavy outputs
    max_unique_for_categorical: int = Field(50, ge=1, le=5000)
    max_value_counts_rows: int = Field(30, ge=1, le=500)


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


class VisualsConfig(BaseModel):
    """
    Controls for interactive/3D visuals (Option B).
    """
    enabled: bool = True

    # Plot backend (Option B wants interactive)
    backend: Literal["plotly", "matplotlib"] = "plotly"

    # Allow 3D visuals when meaningful (scatter_3d, surface correlations, etc.)
    enable_3d: bool = True

    # Control how many plots we generate per run
    max_plots: int = Field(20, ge=0, le=200)

    # Output format for static exports (if we add them later)
    static_format: Literal["png", "svg"] = "png"


class AIInsightsConfig(BaseModel):
    """
    Controls for auto-generated narrative insights (no external API required).
    We can produce rule-based "AI-like" insights now; later you can plug an LLM.
    """
    enabled: bool = True

    # How many key insights to include in the report
    max_insights: int = Field(12, ge=0, le=100)

    # Keep language appropriate for business + non-technical readers
    tone: Literal["business", "technical", "mixed"] = "mixed"


class ReportingConfig(BaseModel):
    """
    Report output controls.
    """
    sample_rows: int = Field(default=20, ge=0, le=500)
    include_corr: bool = True
    plotly_mode: str = "offline"

    # Option B: interactive HTML report
    html_report: bool = True

    # File name inside ctx.reports_dir/
    html_filename: str = "report.html"

    # Include issues + profiling tables in report
    include_quality_section: bool = True
    include_profiling_section: bool = True

    # Include generated plots
    include_visuals: bool = True

    # Optional: title override for the HTML
    title: Optional[str] = None


# ----------------------------
# Root config
# ----------------------------
class Config(BaseModel):
    """
    Root config object.
    - extra="ignore" makes config forward/backward compatible:
      if YAML has extra keys, we don't crash.
    """
    model_config = ConfigDict(extra="ignore")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    io: IoConfig = Field(default_factory=IoConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)
    ai_insights: AIInsightsConfig = Field(default_factory=AIInsightsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


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
