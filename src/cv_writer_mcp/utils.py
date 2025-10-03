"""Utility functions for the CV Writer MCP Server."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def create_timestamped_version(tex_file_path: Path) -> Path:
    """Create a timestamped backup version of the .tex file for tracking changes.

    Args:
        tex_file_path: Path to the original .tex file

    Returns:
        Path to the timestamped backup file, or original path if backup fails
    """
    # Generate formatted timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract the original filename stem (remove any existing timestamps)
    original_stem = tex_file_path.stem
    # Remove any existing timestamp pattern (_YYYYMMDD_HHMMSS)
    original_stem = re.sub(r"_\d{8}_\d{6}$", "", original_stem)

    suffix = tex_file_path.suffix
    new_filename = f"{original_stem}_{timestamp}{suffix}"

    # Create new path in the same directory
    timestamped_path = tex_file_path.parent / new_filename

    try:
        # Copy the current content to the new timestamped file
        with open(tex_file_path, encoding="utf-8") as source:
            content = source.read()

        with open(timestamped_path, "w", encoding="utf-8") as target:
            target.write(content)

        logger.info(f"Created timestamped version: {timestamped_path.name}")
        return timestamped_path

    except Exception as e:
        logger.error(f"Failed to create timestamped version: {str(e)}")
        return tex_file_path  # Return original path if backup fails


def read_text_file(
    file_path: Path,
    description: str = "file",
    expected_extension: str | None = None
) -> str:
    """Read text content from a file with proper error handling and logging.

    Args:
        file_path: Path to the file to read
        description: Human-readable description of the file for logging
        expected_extension: Optional expected file extension (e.g., '.tex', '.yaml', '.txt')
                           If provided, validates that the file has the expected extension

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't have the expected extension
        UnicodeDecodeError: If the file can't be decoded as UTF-8
        Exception: For other file reading errors
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{description.capitalize()} not found: {file_path}")

    # Validate file extension if specified
    if expected_extension:
        if not expected_extension.startswith('.'):
            expected_extension = f".{expected_extension}"

        if file_path.suffix.lower() != expected_extension.lower():
            raise ValueError(
                f"{description.capitalize()} must have {expected_extension} extension, "
                f"but found {file_path.suffix}: {file_path}"
            )

    try:
        content = file_path.read_text(encoding="utf-8")
        logger.info(f"Successfully read {description} ({len(content)} characters): {file_path}")
        return content
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode {description} as UTF-8: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read {description}: {file_path} - {e}")
        raise


def load_agent_config(config_file: str) -> dict[str, Any]:
    """Load agent configuration from a YAML file with multi-path search.

    Searches for configuration files in the following order:
    1. src/cv_writer_mcp/{config_file}  (root level - for non-refactored agents)
    2. src/cv_writer_mcp/compilation/configs/{config_file}  (compilation package)
    3. Future: other package configs/ directories

    This pragmatic approach ensures:
    - Backward compatibility with non-refactored agents
    - Support for new package-based structure
    - Easy extension for future packages

    Args:
        config_file: Name of the YAML configuration file (e.g., "compiler_agent.yaml")

    Returns:
        Dictionary containing the agent configuration

    Raises:
        FileNotFoundError: If the configuration file doesn't exist in any location
        Exception: If the YAML file is malformed
    """
    base_dir = Path(__file__).parent

    # Define search paths in priority order
    search_paths = [
        base_dir / config_file,  # Root level (current location for non-refactored agents)
        base_dir / "compilation" / "configs" / config_file,  # Compilation package
        base_dir / "conversion" / "configs" / config_file,  # Conversion package
        base_dir / "style" / "configs" / config_file,  # Style package
    ]

    # Find the first existing config file
    config_file_path = None
    for path in search_paths:
        if path.exists():
            config_file_path = path
            break

    if not config_file_path:
        searched_locations = "\n  - ".join(str(p) for p in search_paths)
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found in any of these locations:\n  - {searched_locations}"
        )

    try:
        with open(config_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded agent configuration from: {config_file_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading YAML configuration file {config_file_path}: {e}")
        raise
