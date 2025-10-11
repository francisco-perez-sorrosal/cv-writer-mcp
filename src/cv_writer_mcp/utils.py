"""Utility functions for the CV Writer MCP Server."""

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

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


def create_versioned_file(base_path: Path, version: int) -> Path:
    """Create a versioned file with the pattern: base_name_ver<version>.ext

    Args:
        base_path: Base file path (e.g., iter1_var1.tex)
        version: Version number (e.g., 1, 2, 3)

    Returns:
        Path to the versioned file (e.g., iter1_var1_ver1.tex)
    """
    # Extract base name without any existing version patterns
    base_stem = base_path.stem

    # Remove any existing version patterns (_ver<number>)
    base_stem = re.sub(r"_ver\d+$", "", base_stem)

    # Remove any legacy patterns (_refined, _timestamp, etc.)
    base_stem = re.sub(r"_(refined|refined_\d{8}_\d{6}|\d{8}_\d{6})$", "", base_stem)

    # Remove any remaining version patterns after cleaning legacy patterns
    base_stem = re.sub(r"_ver\d+$", "", base_stem)

    suffix = base_path.suffix
    new_filename = f"{base_stem}_ver{version}{suffix}"

    return base_path.parent / new_filename


def get_next_version_number(base_path: Path) -> int:
    """Get the next version number for a base file path.

    Args:
        base_path: Base file path to check for existing versions

    Returns:
        Next version number (1 if no versions exist)
    """
    base_stem = base_path.stem

    # Remove any existing version patterns to get the clean base
    clean_base = re.sub(r"_ver\d+$", "", base_stem)
    clean_base = re.sub(r"_(refined|refined_\d{8}_\d{6}|\d{8}_\d{6})$", "", clean_base)

    # Remove any remaining version patterns after cleaning legacy patterns
    clean_base = re.sub(r"_ver\d+$", "", clean_base)

    # Find all existing version files
    parent_dir = base_path.parent
    pattern = f"{clean_base}_ver*.{base_path.suffix[1:]}"  # Remove the dot from suffix

    existing_versions = []
    for file_path in parent_dir.glob(pattern):
        match = re.search(r"_ver(\d+)$", file_path.stem)
        if match:
            existing_versions.append(int(match.group(1)))

    # Return next version number
    return max(existing_versions, default=0) + 1


def create_error_version(base_path: Path) -> Path:
    """Create an error version of a file with the pattern: base_name_error.ext

    Args:
        base_path: Base file path (e.g., iter1_var1_ver1.tex)

    Returns:
        Path to the error version (e.g., iter1_var1_ver1_error.tex)
    """
    # Extract base name without any existing error patterns
    base_stem = base_path.stem

    # Remove any existing error patterns (_error)
    base_stem = re.sub(r"_error$", "", base_stem)

    suffix = base_path.suffix
    new_filename = f"{base_stem}_error{suffix}"

    return base_path.parent / new_filename


def is_error_version(file_path: Path) -> bool:
    """Check if a file path represents an error version.

    Args:
        file_path: File path to check

    Returns:
        True if the file is an error version, False otherwise
    """
    return file_path.stem.endswith("_error")


def get_non_error_versions(directory: Path, base_pattern: str) -> list[Path]:
    """Get all non-error versions of files matching a base pattern.

    Args:
        directory: Directory to search in
        base_pattern: Base pattern to match (e.g., "iter1_var1")

    Returns:
        List of non-error file paths
    """
    pattern = f"{base_pattern}*"
    all_files = list(directory.glob(pattern))

    # Filter out error versions
    non_error_files = [f for f in all_files if not is_error_version(f)]

    return non_error_files


def create_organized_backup(tex_file_path: Path, backup_type: str = "backup") -> Path:
    """Create an organized backup version with clear naming.

    Args:
        tex_file_path: Path to the original .tex file
        backup_type: Type of backup (e.g., "backup", "before_fix", "after_fix")

    Returns:
        Path to the backup file, or original path if backup fails
    """
    # Generate formatted timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract the original filename stem (remove any existing timestamps or backup patterns)
    original_stem = tex_file_path.stem
    # Remove any existing timestamp pattern (_YYYYMMDD_HHMMSS)
    original_stem = re.sub(r"_\d{8}_\d{6}$", "", original_stem)
    # Remove any existing backup patterns (_backup, _before_fix, etc.)
    original_stem = re.sub(
        r"_(backup|before_fix|after_fix|refined)$", "", original_stem
    )

    suffix = tex_file_path.suffix
    new_filename = f"{original_stem}_{backup_type}_{timestamp}{suffix}"

    # Create new path in the same directory
    backup_path = tex_file_path.parent / new_filename

    try:
        # Copy the current content to the new backup file
        with open(tex_file_path, encoding="utf-8") as source:
            content = source.read()

        with open(backup_path, "w", encoding="utf-8") as target:
            target.write(content)

        logger.info(f"Created organized backup: {backup_path.name}")
        return backup_path

    except Exception as e:
        logger.error(f"Failed to create organized backup: {str(e)}")
        return tex_file_path  # Return original path if backup fails


def read_text_file(
    file_path: Path, description: str = "file", expected_extension: str | None = None
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
        if not expected_extension.startswith("."):
            expected_extension = f".{expected_extension}"

        if file_path.suffix.lower() != expected_extension.lower():
            raise ValueError(
                f"{description.capitalize()} must have {expected_extension} extension, "
                f"but found {file_path.suffix}: {file_path}"
            )

    try:
        content = file_path.read_text(encoding="utf-8")
        logger.info(
            f"Successfully read {description} ({len(content)} characters): {file_path}"
        )
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
        base_dir
        / config_file,  # Root level (current location for non-refactored agents)
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


# ============================================================================
# Progress Reporting Utilities
# ============================================================================

# Type alias for progress callbacks
ProgressCallback: TypeAlias = Callable[[int], Awaitable[None]] | None


class ProgressMapper:
    """Maps sub-operation progress (0-100) to parent operation's allocated range.

    This enables hierarchical progress reporting where nested operations can report
    their own 0-100% progress, which gets automatically mapped to their allocated
    portion of the parent's progress range.

    Example:
        # Parent allocates 30-60% for compilation phase
        mapper = ProgressMapper(parent_callback, start_percent=30, end_percent=60)
        await mapper.report(0)    # Reports 30% to parent
        await mapper.report(50)   # Reports 45% to parent (midpoint)
        await mapper.report(100)  # Reports 60% to parent
    """

    def __init__(
        self,
        parent_callback: ProgressCallback,
        start_percent: int,
        end_percent: int,
    ):
        """Initialize progress mapper.

        Args:
            parent_callback: Parent's progress callback (or None)
            start_percent: Start of this operation's range (0-100)
            end_percent: End of this operation's range (0-100)
        """
        if not 0 <= start_percent <= 100:
            raise ValueError(f"start_percent must be 0-100, got {start_percent}")
        if not 0 <= end_percent <= 100:
            raise ValueError(f"end_percent must be 0-100, got {end_percent}")
        if start_percent >= end_percent:
            raise ValueError(
                f"start_percent ({start_percent}) must be < end_percent ({end_percent})"
            )

        self.parent_callback = parent_callback
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.range_size = end_percent - start_percent

    async def report(self, sub_progress: int) -> None:
        """Report sub-operation progress mapped to parent range.

        Args:
            sub_progress: Progress within this operation (0-100)
        """
        if self.parent_callback is None:
            return

        # Clamp sub_progress to 0-100
        sub_progress = max(0, min(100, sub_progress))

        # Map 0-100 to start_percent-end_percent
        mapped_progress = self.start_percent + (sub_progress * self.range_size // 100)

        # Ensure we don't exceed end_percent
        mapped_progress = min(mapped_progress, self.end_percent)

        await self.parent_callback(mapped_progress)

    def create_sub_mapper(self, sub_start: int, sub_end: int) -> "ProgressMapper":
        """Create a nested progress mapper for sub-sub-operations.

        This allows for deeper nesting of progress reporting. The sub-mapper's
        range (sub_start to sub_end) is mapped to this mapper's range.

        Args:
            sub_start: Start percent within this operation (0-100)
            sub_end: End percent within this operation (0-100)

        Returns:
            New ProgressMapper that maps to this mapper's range

        Example:
            # Parent has 30-60% range
            parent_mapper = ProgressMapper(callback, 30, 60)
            # Child gets 0-50% of parent's range (30-45%)
            child_mapper = parent_mapper.create_sub_mapper(0, 50)
            await child_mapper.report(100)  # Reports 45% to root callback
        """
        if not 0 <= sub_start <= 100:
            raise ValueError(f"sub_start must be 0-100, got {sub_start}")
        if not 0 <= sub_end <= 100:
            raise ValueError(f"sub_end must be 0-100, got {sub_end}")
        if sub_start >= sub_end:
            raise ValueError(
                f"sub_start ({sub_start}) must be < sub_end ({sub_end})"
            )

        # Map sub-range to this operation's range
        actual_start = self.start_percent + (sub_start * self.range_size // 100)
        actual_end = self.start_percent + (sub_end * self.range_size // 100)

        return ProgressMapper(self.parent_callback, actual_start, actual_end)


class PeriodicProgressTicker:
    """Background task that reports progress periodically during long operations.

    This is crucial for preventing MCP client timeouts when operations take
    longer than ~10 seconds without any progress updates. The ticker increments
    progress gradually over the allocated range.

    Use as async context manager:
        async with PeriodicProgressTicker(callback, 10, 85, interval_seconds=10.0):
            result = await long_running_operation()  # 30 second operation
        # Ticker automatically stops when context exits

    During the long operation, progress will be reported every 10 seconds,
    gradually moving from 10% to 85%.
    """

    def __init__(
        self,
        progress_callback: ProgressCallback,
        start_percent: int,
        end_percent: int,
        interval_seconds: float = 10.0,
        step_size: int = 5,
    ):
        """Initialize periodic progress ticker.

        Args:
            progress_callback: Callback to report progress to
            start_percent: Starting progress value (0-100)
            end_percent: Target progress value (0-100)
            interval_seconds: How often to report progress (default: 10.0)
            step_size: How much to increment each tick (default: 5)
        """
        if not 0 <= start_percent <= 100:
            raise ValueError(f"start_percent must be 0-100, got {start_percent}")
        if not 0 <= end_percent <= 100:
            raise ValueError(f"end_percent must be 0-100, got {end_percent}")
        if start_percent >= end_percent:
            raise ValueError(
                f"start_percent ({start_percent}) must be < end_percent ({end_percent})"
            )

        self.progress_callback = progress_callback
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.interval_seconds = interval_seconds
        self.step_size = step_size
        self._task: asyncio.Task | None = None
        self._stop_flag = asyncio.Event()
        self._current_progress = start_percent

    async def _tick_loop(self):
        """Increment progress periodically until stopped."""
        # Reserve some headroom (don't reach 100% of allocated range)
        max_progress = self.end_percent - 2
        tick_count = 0

        logger.debug(f"Ticker loop starting (target: {self.start_percent}% -> {max_progress}%)")

        # Report initial progress immediately
        if self.progress_callback:
            try:
                await self.progress_callback(self._current_progress)
                logger.debug(f"Ticker: reported initial progress {self._current_progress}%")
            except Exception as e:
                logger.warning(f"Initial progress callback failed: {e}")

        while not self._stop_flag.is_set():
            # Use asyncio.sleep instead of Event.wait for better event loop yielding
            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                logger.debug("Ticker loop cancelled")
                break

            # Check if we should stop
            if self._stop_flag.is_set():
                logger.debug(f"Ticker loop: stop flag set, exiting after {tick_count} ticks")
                break

            # Increment progress
            tick_count += 1
            if self._current_progress < max_progress:
                self._current_progress = min(
                    self._current_progress + self.step_size, max_progress
                )
                if self.progress_callback:
                    try:
                        logger.info(
                            f"📊 Progress: {self._current_progress}% (ticker update #{tick_count})"
                        )
                        await self.progress_callback(self._current_progress)
                        logger.debug(f"Ticker tick #{tick_count}: callback completed successfully")
                    except Exception as e:
                        logger.warning(
                            f"Progress callback failed at tick #{tick_count}: {e}, continuing ticker"
                        )
            else:
                logger.debug(
                    f"Ticker at max progress ({self._current_progress}%), waiting for completion..."
                )

        logger.debug(f"Ticker loop exiting after {tick_count} ticks")

    async def __aenter__(self):
        """Start the ticker when entering context."""
        if self.progress_callback:
            self._stop_flag.clear()
            self._current_progress = self.start_percent
            self._task = asyncio.create_task(self._tick_loop())
            logger.debug(
                f"Started progress ticker: {self.start_percent}% -> {self.end_percent}% "
                f"(interval: {self.interval_seconds}s, step: {self.step_size}%)"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the ticker when exiting context."""
        if self._task:
            # Signal the ticker to stop
            self._stop_flag.set()

            try:
                # Wait for ticker to finish gracefully
                await asyncio.wait_for(self._task, timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Progress ticker didn't stop gracefully, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

            logger.debug("Stopped progress ticker")

        return False  # Don't suppress exceptions


async def report_progress(
    progress: int,
    progress_callback: ProgressCallback,
    stage: str = "",
) -> None:
    """Report progress with optional stage logging.

    This is a convenience function that combines progress reporting with
    logging, similar to the orchestrator's _report_progress method.

    Args:
        progress: Progress value (0-100)
        progress_callback: Optional callback to report progress
        stage: Optional stage description for logging
    """
    if progress_callback:
        await progress_callback(progress)
        if stage:
            logger.info(f"📊 Progress: {progress}% - {stage}")
        else:
            logger.info(f"📊 Progress: {progress}%")
