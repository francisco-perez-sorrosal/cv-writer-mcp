"""Tests for Simplified CV converter."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cv_writer_mcp.cv_converter import CVConverter
from cv_writer_mcp.models import (
    ConversionStatus,
    MarkdownToLaTeXRequest,
    MarkdownToLaTeXResponse,
    ServerConfig,
)


class TestCVConverter:
    """Test cases for SimplifiedCVConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ServerConfig(
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
            base_url="http://localhost:8000",
            openai_api_key="test-api-key",
        )

    def test_initialization(self):
        """Test converter initialization."""
        with patch("cv_writer_mcp.cv_converter.MD2LaTeXAgent") as mock_openai:
            mock_openai.return_value = MagicMock()
            converter = CVConverter(self.config)

            assert converter.config == self.config
            assert converter.md2latex_agent is not None

    def test_initialization_no_openai_key(self):
        """Test converter initialization without OpenAI API key."""
        config_no_key = ServerConfig(
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
            openai_api_key=None,
        )

        with patch("cv_writer_mcp.cv_converter.MD2LaTeXAgent") as mock_openai:
            mock_openai.side_effect = ValueError("API key required")
            converter = CVConverter(config_no_key)

            assert converter.config == config_no_key
            assert converter.md2latex_agent is None

    @pytest.mark.asyncio
    async def test_convert_markdown_to_latex_success(self):
        """Test successful markdown to LaTeX conversion."""
        with patch("cv_writer_mcp.cv_converter.MD2LaTeXAgent") as mock_openai:
            mock_agent = MagicMock()
            mock_agent.convert = AsyncMock(
                return_value=MarkdownToLaTeXResponse(
                    status=ConversionStatus.SUCCESS,
                    tex_url="cv-writer://tex/test.tex",
                    metadata={"output_filename": "test.tex"},
                )
            )
            mock_openai.return_value = mock_agent

            converter = CVConverter(self.config)
            request = MarkdownToLaTeXRequest(
                markdown_content="# Test CV\nThis is a test.",
                output_filename="test.tex",
            )

            response = await converter.convert_markdown_to_latex(request)

            assert response.status == ConversionStatus.SUCCESS
            assert response.tex_url == "cv-writer://tex/test.tex"
            mock_agent.convert.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_convert_markdown_to_latex_no_agent(self):
        """Test markdown to LaTeX conversion without OpenAI agent."""
        config_no_key = ServerConfig(
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
            openai_api_key=None,
        )

        with patch("cv_writer_mcp.cv_converter.MD2LaTeXAgent") as mock_openai:
            mock_openai.side_effect = ValueError("API key required")
            converter = CVConverter(config_no_key)

            request = MarkdownToLaTeXRequest(
                markdown_content="# Test CV\nThis is a test."
            )

            response = await converter.convert_markdown_to_latex(request)

            assert response.status == ConversionStatus.FAILED
            assert "Markdown to LaTeX agent not initialized" in response.message
