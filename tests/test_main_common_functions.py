"""Tests for common configuration and logging functions in main.py."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from cv_writer_mcp.main import create_config, setup_logging
from cv_writer_mcp.models import ServerConfig


class TestMainCommonFunctions:
    """Test cases for common configuration and logging functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_setup_config_with_debug_false(self):
        """Test configuration setup without debug mode."""
        with patch.dict(
            "os.environ",
            {
                "HOST": "testhost",
                "PORT": "9000",
                "LOG_LEVEL": "WARNING",
                "OUTPUT_DIR": str(Path(self.temp_dir) / "output"),
                "TEMP_DIR": str(Path(self.temp_dir) / "temp"),
            },
        ):
            config = create_config(debug=False)

            assert isinstance(config, ServerConfig)
            assert config.host == "testhost"
            assert config.port == 9000
            assert config.debug is False
            assert config.log_level.value == "WARNING"

    def test_setup_config_with_debug_true(self):
        """Test configuration setup with debug mode."""
        with patch.dict(
            "os.environ",
            {
                "HOST": "testhost",
                "PORT": "9000",
                "LOG_LEVEL": "INFO",
                "OUTPUT_DIR": str(Path(self.temp_dir) / "output"),
                "TEMP_DIR": str(Path(self.temp_dir) / "temp"),
            },
        ):
            config = create_config(debug=True)

            assert isinstance(config, ServerConfig)
            assert config.host == "testhost"
            assert config.port == 9000
            assert config.debug is True
            assert config.log_level.value == "DEBUG"

    def test_setup_config_defaults(self):
        """Test configuration setup with default values."""
        with patch.dict("os.environ", {}, clear=True):
            config = create_config()

            assert isinstance(config, ServerConfig)
            assert config.host == "localhost"
            assert config.port == 8000
            assert config.debug is False
            assert config.log_level.value == "INFO"

    def test_setup_logging(self):
        """Test logging setup."""
        config = ServerConfig(
            host="localhost",
            port=8000,
            base_url="http://localhost:8000",
            debug=False,
            log_level="INFO",
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
        )

        # This should not raise an exception
        setup_logging(config.log_level)

        # Test with debug mode
        setup_logging(config.log_level)

    def test_setup_logging_with_debug_config(self):
        """Test logging setup with debug configuration."""
        config = ServerConfig(
            host="localhost",
            port=8000,
            base_url="http://localhost:8000",
            debug=True,
            log_level="DEBUG",
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
        )

        # This should not raise an exception
        setup_logging(config.log_level)
