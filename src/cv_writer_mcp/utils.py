"""Utility functions for the CV Writer MCP Server."""

import re
from datetime import datetime
from pathlib import Path

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
