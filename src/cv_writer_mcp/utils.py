"""Utility functions for the CV Writer MCP Server."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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


def load_agent_config(config_file: str) -> Dict[str, Any]:
    """Load agent configuration from a YAML file.

    Args:
        config_file_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the agent configuration

    Raises:
        Exception: If the configuration file doesn't exist or if the
        if the YAML file is malformed
    """
    
    config_file_path = Path(__file__).parent / Path(config_file)

    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    try:
        with open(config_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded agent configuration from: {config_file_path}")
        return config
    except Exception as e:
        logger.error(f"Error loadding YAML configuration file {config_file_path}: {e}")
        raise
