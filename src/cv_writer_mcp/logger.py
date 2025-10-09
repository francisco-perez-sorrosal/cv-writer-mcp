"""Logger configuration for CV Writer MCP Server."""

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as loguru_logger

if TYPE_CHECKING:
    from loguru import Logger
from pydantic import BaseModel


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogConfig(BaseModel):
    """Logger configuration model."""

    level: LogLevel = LogLevel.INFO
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    rotation: str = "10 MB"
    retention: str = "7 days"
    compression: str = "zip"
    log_file: Path | None = None
    console_output: bool = True


def configure_logger(config: LogConfig) -> "Logger":
    """Configure and return a loguru logger instance.

    Args:
        config: Logger configuration

    Returns:
        Configured logger instance
    """
    import sys

    # Remove default handler
    loguru_logger.remove()

    # Add console handler with custom colors
    # IMPORTANT: Use stderr to avoid polluting stdout (MCP uses stdout for JSON-RPC)
    if config.console_output:
        loguru_logger.add(
            sink=sys.stderr,
            format=config.format,
            level=config.level.value,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # Add file handler if specified
    if config.log_file:
        loguru_logger.add(
            sink=str(config.log_file),
            format=config.format,
            level=config.level.value,
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            backtrace=True,
            diagnose=True,
        )

    return loguru_logger


def get_logger(name: str | None = None) -> "Logger":
    """Get a logger instance with simplified module path.

    Args:
        name: Optional logger name (if None, extracts from calling module)

    Returns:
        Logger instance with simplified module path
    """
    import inspect

    if name:
        # Extract just the last part of the module path
        module_name = name.split(".")[-1]
        # Create a custom logger that overrides the name field
        return loguru_logger.patch(lambda record: record.update(name=module_name))

    # Auto-detect calling module
    frame = inspect.currentframe()
    if frame and frame.f_back:
        module_name = frame.f_back.f_globals.get("__name__", "unknown")
        # Extract just the last part of the module path
        simple_module = module_name.split(".")[-1]
        return loguru_logger.patch(lambda record: record.update(name=simple_module))

    # Fallback if frame detection fails
    return loguru_logger.patch(lambda record: record.update(name="unknown"))
