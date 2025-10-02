"""Tests for MD2LaTeXAgent."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cv_writer_mcp.conversion import MD2LaTeXAgent
from cv_writer_mcp.conversion.models import MarkdownToLaTeXRequest, MarkdownToLaTeXResponse
from cv_writer_mcp.models import CompletionStatus


class TestMD2LaTeXAgent:
    """Test cases for MD2LaTeXAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_initialization(self):
        """Test MD2LaTeXAgent initialization."""
        with patch("cv_writer_mcp.conversion.md2latex_agent.load_agent_config") as mock_load_config, \
             patch("cv_writer_mcp.conversion.md2latex_agent.read_text_file") as mock_read_file, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Agent") as mock_agent_class:
            
            # Mock config
            mock_config = {
                "agent_metadata": {"model": "gpt-4", "name": "test", "output_type": "LaTeXOutput"},
                "instructions": "Test instructions",
                "prompt_template": "Test prompt: {markdown_content}"
            }
            mock_load_config.return_value = mock_config
            
            # Mock file reading
            mock_read_file.return_value = "test content"
            
            # Mock agent creation
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            agent = MD2LaTeXAgent(api_key="test-key")
            
            assert agent.api_key == "test-key"
            assert agent.model == "gpt-4"
            assert agent.agent is not None
            mock_load_config.assert_called_once_with("md2latex_agent.yaml")
            mock_agent_class.assert_called_once()

    def test_initialization_no_openai_key(self):
        """Test MD2LaTeXAgent initialization without OpenAI API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                MD2LaTeXAgent()

    @pytest.mark.asyncio
    async def test_convert_success(self):
        """Test successful markdown to LaTeX conversion."""
        with patch("cv_writer_mcp.conversion.md2latex_agent.load_agent_config") as mock_load_config, \
             patch("cv_writer_mcp.conversion.md2latex_agent.read_text_file") as mock_read_file, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Agent") as mock_agent_class, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Runner") as mock_runner, \
             patch("pathlib.Path.write_text") as mock_write:
            
            # Mock config
            mock_config = {
                "agent_metadata": {"model": "gpt-4", "name": "test", "output_type": "LaTeXOutput"},
                "instructions": "Test instructions",
                "prompt_template": "Test prompt: {markdown_content}"
            }
            mock_load_config.return_value = mock_config
            
            # Mock file reading
            mock_read_file.return_value = "test content"
            
            # Mock agent creation
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Mock runner result
            mock_result = MagicMock()
            mock_result.final_output = MagicMock()
            mock_result.final_output.latex_content = "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            mock_result.final_output.conversion_notes = "Successfully converted"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            agent = MD2LaTeXAgent(api_key="test-key")
            
            request = MarkdownToLaTeXRequest(
                markdown_content="# Test CV\n\nSome content",
                output_filename="test.tex",
            )

            response = await agent.convert(request)

            assert response.status == CompletionStatus.SUCCESS
            assert response.tex_url == "cv-writer://tex/test.tex"
            assert "Successfully converted to test.tex" in response.message
            mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_empty_latex(self):
        """Test conversion with empty LaTeX content."""
        with patch("cv_writer_mcp.conversion.md2latex_agent.load_agent_config") as mock_load_config, \
             patch("cv_writer_mcp.conversion.md2latex_agent.read_text_file") as mock_read_file, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Agent") as mock_agent_class, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Runner") as mock_runner:
            
            # Mock config
            mock_config = {
                "agent_metadata": {"model": "gpt-4", "name": "test", "output_type": "LaTeXOutput"},
                "instructions": "Test instructions",
                "prompt_template": "Test prompt: {markdown_content}"
            }
            mock_load_config.return_value = mock_config
            
            # Mock file reading
            mock_read_file.return_value = "test content"
            
            # Mock agent creation
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Mock runner result with empty content
            mock_result = MagicMock()
            mock_result.final_output = MagicMock()
            mock_result.final_output.latex_content = ""
            mock_result.final_output.conversion_notes = "No content generated"
            mock_runner.run = AsyncMock(return_value=mock_result)
            
            agent = MD2LaTeXAgent(api_key="test-key")
            
            request = MarkdownToLaTeXRequest(
                markdown_content="# Test CV\n\nSome content",
                output_filename="test.tex",
            )

            response = await agent.convert(request)

            assert response.status == CompletionStatus.FAILED
            assert "Agent response does not contain valid LaTeX content" in response.message
            assert "No content generated" in response.message

    @pytest.mark.asyncio
    async def test_convert_exception(self):
        """Test conversion with exception."""
        with patch("cv_writer_mcp.conversion.md2latex_agent.load_agent_config") as mock_load_config, \
             patch("cv_writer_mcp.conversion.md2latex_agent.read_text_file") as mock_read_file, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Agent") as mock_agent_class, \
             patch("cv_writer_mcp.conversion.md2latex_agent.Runner") as mock_runner:
            
            # Mock config
            mock_config = {
                "agent_metadata": {"model": "gpt-4", "name": "test", "output_type": "LaTeXOutput"},
                "instructions": "Test instructions",
                "prompt_template": "Test prompt: {markdown_content}"
            }
            mock_load_config.return_value = mock_config
            
            # Mock file reading
            mock_read_file.return_value = "test content"
            
            # Mock agent creation
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Mock runner to raise exception
            mock_runner.run = AsyncMock(side_effect=Exception("Test error"))
            
            agent = MD2LaTeXAgent(api_key="test-key")
            
            request = MarkdownToLaTeXRequest(
                markdown_content="# Test CV\n\nSome content",
                output_filename="test.tex",
            )

            response = await agent.convert(request)

            assert response.status == CompletionStatus.FAILED
            assert "Conversion failed: Test error" in response.message
