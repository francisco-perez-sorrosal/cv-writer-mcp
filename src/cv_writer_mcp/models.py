"""Shared models for CV Writer MCP Server.

This module contains models shared across all packages.
Package-specific models are in their respective packages:
- Compilation models: cv_writer_mcp.compilation.models
- Conversion models: cv_writer_mcp.conversion.models
- Style models: cv_writer_mcp.style.models
"""

from enum import Enum
import os
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
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "./output"))
    temp_dir: Path = Path("./temp")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    latex_timeout_seconds: int = 900  # 15 minutes for complex compilations
    openai_api_key: str | None = None
    templates_dir: Path = Path("./context")

    @field_validator("output_dir", "temp_dir", "templates_dir")
    @classmethod
    def validate_directories(cls, v: Path) -> Path:
        """Validate and convert directory paths to absolute paths.

        Note: Directories are created lazily when first accessed, not during initialization.
        This prevents failures when running from read-only locations (e.g., MCP bundles).
        """
        # Convert to absolute path if relative
        if not v.is_absolute():
            v = v.resolve()
        return v

    def ensure_dir(self, dir_path: Path) -> Path:
        """Ensure a directory exists, creating it if possible.

        Args:
            dir_path: Path to the directory

        Returns:
            The directory path

        Raises:
            OSError: If directory cannot be created
        """
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # If we can't create in the intended location, try a temp directory
            import tempfile
            import os

            # Create in system temp dir as fallback
            temp_base = Path(tempfile.gettempdir()) / "cv-writer-mcp"
            fallback_dir = temp_base / dir_path.name
            fallback_dir.mkdir(parents=True, exist_ok=True)

            # Log the fallback (using stderr to avoid stdout pollution)
            import sys
            print(f"Warning: Could not create {dir_path}: {e}. Using fallback: {fallback_dir}", file=sys.stderr)

            return fallback_dir
        return dir_path

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to ensure base_url consistency and create directories."""
        # If base_url is still the default, update it based on host and port
        if self.base_url == "http://localhost:8000" and (
            self.host != "localhost" or self.port != 8000
        ):
            self.base_url = f"http://{self.host}:{self.port}"

        # Ensure directories exist (with fallback to temp dir if needed)
        self.output_dir = self.ensure_dir(self.output_dir)
        self.temp_dir = self.ensure_dir(self.temp_dir)
        # Note: templates_dir is read-only, so we don't create it

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
        PDFAnalysisOutput,
        SingleVariantEvaluationOutput,
        VariantEvaluationOutput,
        VisualCriticOutput,
    )

    # Centralized mapping of output type names to actual classes
    output_type_mapping = {
        "LaTeXOutput": LaTeXOutput,
        "CompilerAgentOutput": CompilerAgentOutput,
        "CompilationErrorOutput": CompilationErrorOutput,
        "VisualCriticOutput": VisualCriticOutput,
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


def create_agent_from_config(
    agent_config: dict,
    instructions: str,
    model: str | None = None,
    tools: list | None = None,
    name_suffix: str = "",
    strict_json_schema: bool = False,
):
    """Create an OpenAI Agent with standardized configuration and safe defaults.

    This helper function centralizes agent creation logic, automatically wrapping
    output types with AgentOutputSchema to handle complex Pydantic models
    (dict[int, ...], Union types, list[ComplexModel], etc.) gracefully.

    Args:
        agent_config: Agent configuration dict from YAML (loaded via load_agent_config)
        instructions: Formatted instructions string for the agent
        model: Model override (uses agent_config model if None)
        tools: List of tools for the agent (default: empty list)
        name_suffix: Optional suffix for agent name (e.g., "_v2" for variants)
        strict_json_schema: Whether to use strict JSON schema mode (default: False for safety)

    Returns:
        Configured Agent instance with properly wrapped output type

    Example:
        >>> agent_config = load_agent_config("error_agent.yaml")
        >>> instructions = agent_config["instructions"].format(...)
        >>> agent = create_agent_from_config(agent_config, instructions)
    """
    from agents import Agent, AgentOutputSchema

    # Get output type class and wrap with AgentOutputSchema
    output_type_class = get_output_type_class(
        agent_config["agent_metadata"]["output_type"]
    )
    wrapped_output_type = AgentOutputSchema(
        output_type_class, strict_json_schema=strict_json_schema
    )

    # Build agent name with optional suffix (for variants)
    agent_name = agent_config["agent_metadata"]["name"]
    if name_suffix:
        agent_name = f"{agent_name}{name_suffix}"

    return Agent(
        name=agent_name,
        instructions=instructions,
        tools=tools or [],
        model=model or agent_config["agent_metadata"]["model"],
        output_type=wrapped_output_type,
    )
