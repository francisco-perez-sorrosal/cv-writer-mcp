"""Shared models for CV Writer MCP Server.

This module contains models shared across all packages.
Package-specific models are in their respective packages:
- Compilation models: cv_writer_mcp.compilation.models
- Conversion models: cv_writer_mcp.conversion.models
- Style models: cv_writer_mcp.style.models
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from cv_writer_mcp.logger import LogLevel


class CompletionStatus(str, Enum):
    """Generic completion status enumeration."""

    SUCCESS = "success"
    FAILED = "failed"


class HealthStatusResponse(BaseModel):
    """Response model for health check."""

    status: str = "healthy"
    service: str = "cv-writer-mcp"
    timestamp: str | None = None
    version: str | None = None


class ServerConfig(BaseModel):
    """Server configuration model."""

    host: str = "localhost"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    latex_timeout: int = 30
    openai_api_key: str | None = None
    templates_dir: Path = Path("./context")

    @field_validator("output_dir", "temp_dir", "templates_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to ensure base_url consistency."""
        # If base_url is still the default, update it based on host and port
        if self.base_url == "http://localhost:8000" and (
            self.host != "localhost" or self.port != 8000
        ):
            self.base_url = f"http://{self.host}:{self.port}"

    def get_base_url(self) -> str:
        """Get the base URL, ensuring it reflects current host and port."""
        return f"http://{self.host}:{self.port}"


def get_output_type_class(output_type_name: str):
    """Get the actual output type class from the string name.

    Args:
        output_type_name: String name of the output type from config

    Returns:
        The actual class object for the output type

    Raises:
        ValueError: If the output type name is not recognized
    """
    # Import package-specific models
    from .compilation.models import CompilationErrorOutput, CompilerAgentOutput
    from .conversion.models import LaTeXOutput
    from .style.models import (
        FormattingOutput,
        PageCaptureOutput,
        PDFAnalysisOutput,
        SingleVariantEvaluationOutput,
        VariantEvaluationOutput,
    )

    # Centralized mapping of output type names to actual classes
    output_type_mapping = {
        "LaTeXOutput": LaTeXOutput,
        "CompilerAgentOutput": CompilerAgentOutput,
        "CompilationErrorOutput": CompilationErrorOutput,
        "PageCaptureOutput": PageCaptureOutput,
        "FormattingOutput": FormattingOutput,
        "PDFAnalysisOutput": PDFAnalysisOutput,
        # Multi-variant style improvement outputs
        "SingleVariantEvaluationOutput": SingleVariantEvaluationOutput,
        "VariantEvaluationOutput": VariantEvaluationOutput,
    }

    if output_type_name not in output_type_mapping:
        raise ValueError(
            f"Unknown output type: {output_type_name}. Available types: {list(output_type_mapping.keys())}"
        )

    return output_type_mapping[output_type_name]
