"""Tests for LaTeX compiler."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cv_writer_mcp.latex_compiler import LaTeXCompiler
from cv_writer_mcp.models import (
    CompileLaTeXRequest,
    CompileLaTeXResponse,
    ConversionStatus,
    LaTeXEngine,
    ServerConfig,
)


class TestLaTeXCompiler:
    """Test cases for LaTeXCompiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ServerConfig(
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
            base_url="http://localhost:8000",
        )

    def test_initialization(self):
        """Test compiler initialization."""
        compiler = LaTeXCompiler(timeout=30, config=self.config)
        assert compiler.timeout == 30
        assert compiler.config == self.config

    def test_initialization_no_config(self):
        """Test compiler initialization without config."""
        compiler = LaTeXCompiler(timeout=30)
        assert compiler.timeout == 30
        assert compiler.config is None

    @pytest.mark.asyncio
    async def test_compile_from_request_success(self):
        """Test successful LaTeX to PDF compilation from request."""
        # Create a test LaTeX file
        tex_file = self.config.output_dir / "test.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        with patch.object(LaTeXCompiler, "compile_with_agent") as mock_agent:
            from cv_writer_mcp.models import LaTeXCompilationResult

            mock_agent.return_value = LaTeXCompilationResult(
                success=True,
                compilation_time=1.5,
                error_message=None,
                output_path=self.config.output_dir / "test.pdf",
            )

            compiler = LaTeXCompiler(timeout=30, config=self.config)
            request = CompileLaTeXRequest(
                tex_filename="test.tex",
                output_filename="test.pdf",
                latex_engine=LaTeXEngine.PDFLATEX,
            )

            response = await compiler.compile_from_request(request)

            assert response.status == ConversionStatus.SUCCESS
            assert response.pdf_url == "cv-writer://pdf/test.pdf"
            assert response.metadata["tex_filename"] == "test.tex"
            assert response.metadata["output_filename"] == "test.pdf"
            assert response.metadata["latex_engine"] == "pdflatex"
            assert response.metadata["compilation_time"] == 1.5

    @pytest.mark.asyncio
    async def test_compile_from_request_no_config(self):
        """Test compilation from request without config."""
        compiler = LaTeXCompiler(timeout=30)  # No config
        request = CompileLaTeXRequest(
            tex_filename="test.tex",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        response = await compiler.compile_from_request(request)

        assert response.status == ConversionStatus.FAILED
        assert "Server configuration not provided" in response.error_message

    @pytest.mark.asyncio
    async def test_compile_from_request_file_not_found(self):
        """Test compilation from request with missing file."""
        compiler = LaTeXCompiler(timeout=30, config=self.config)
        request = CompileLaTeXRequest(
            tex_filename="nonexistent.tex",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        response = await compiler.compile_from_request(request)

        assert response.status == ConversionStatus.FAILED
        assert "LaTeX file not found: nonexistent.tex" in response.error_message

    @pytest.mark.asyncio
    async def test_compile_from_request_compilation_failure(self):
        """Test compilation from request with compilation failure."""
        # Create a test LaTeX file
        tex_file = self.config.output_dir / "test.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        with patch.object(LaTeXCompiler, "compile_from_request") as mock_compile:
            mock_compile.return_value = CompileLaTeXResponse(
                status=ConversionStatus.FAILED,
                pdf_url=None,
                error_message="LaTeX compilation failed",
            )

            compiler = LaTeXCompiler(timeout=30, config=self.config)
            request = CompileLaTeXRequest(
                tex_filename="test.tex",
                latex_engine=LaTeXEngine.PDFLATEX,
            )

            response = await compiler.compile_from_request(request)

            assert response.status == ConversionStatus.FAILED
            assert "LaTeX compilation failed" in response.error_message

    @pytest.mark.asyncio
    async def test_compile_from_request_exception(self):
        """Test compilation from request with exception."""
        # Create a test LaTeX file
        tex_file = self.config.output_dir / "test.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        with patch.object(LaTeXCompiler, "compile_with_agent") as mock_agent:
            mock_agent.side_effect = Exception("Compilation error")

            compiler = LaTeXCompiler(timeout=30, config=self.config)
            request = CompileLaTeXRequest(
                tex_filename="test.tex",
                latex_engine=LaTeXEngine.PDFLATEX,
            )

            response = await compiler.compile_from_request(request)

            assert response.status == ConversionStatus.FAILED
            assert "Unexpected error: Compilation error" in response.error_message

    def test_check_latex_installation_with_engine(self):
        """Test LaTeX installation check with specific engine."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            compiler = LaTeXCompiler(timeout=30)
            result = compiler.check_latex_installation(LaTeXEngine.PDFLATEX)

            assert result is True
            mock_run.assert_called_once_with(
                ["pdflatex", "--version"], capture_output=True, timeout=10
            )

    def test_check_latex_installation_not_found(self):
        """Test LaTeX installation check when not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            compiler = LaTeXCompiler(timeout=30)
            result = compiler.check_latex_installation(LaTeXEngine.PDFLATEX)

            assert result is False

    def test_check_latex_installation_timeout(self):
        """Test LaTeX installation check with timeout."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("pdflatex", 10)

            compiler = LaTeXCompiler(timeout=30)
            result = compiler.check_latex_installation(LaTeXEngine.PDFLATEX)

            assert result is False

    def test_check_latex_installation_default_engine(self):
        """Test LaTeX installation check with default engine."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            compiler = LaTeXCompiler(timeout=30)
            result = compiler.check_latex_installation()

            assert result is True
            mock_run.assert_called_once_with(
                ["pdflatex", "--version"], capture_output=True, timeout=10
            )

    @pytest.mark.asyncio
    async def test_compile_from_request_with_agent_default(self):
        """Test compile_from_request uses agent by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test LaTeX file
            tex_file = temp_path / "test.tex"
            tex_file.write_text(
                "\\documentclass{article}\\begin{document}Test\\end{document}"
            )

            # Create config
            config = ServerConfig(output_dir=temp_path)
            compiler = LaTeXCompiler(config=config)

            # Create request (should use agent by default)
            request = CompileLaTeXRequest(tex_filename="test.tex")

            # Mock the compile_with_agent method to avoid actual agent execution
            with patch.object(compiler, "compile_with_agent") as mock_agent:
                from cv_writer_mcp.models import LaTeXCompilationResult

                mock_agent.return_value = LaTeXCompilationResult(
                    success=True,
                    compilation_time=1.0,
                    error_message=None,
                    output_path=temp_path / "test.pdf",
                )

                response = await compiler.compile_from_request(request)

                # Verify agent was called
                mock_agent.assert_called_once()

                # Verify response
                assert response.status == ConversionStatus.SUCCESS
                assert response.pdf_url == "cv-writer://pdf/test.pdf"
                assert response.metadata["compilation_method"] == "agent"

    @pytest.mark.asyncio
    async def test_compile_from_request_with_direct_compilation(self):
        """Test compile_from_request with direct compilation when use_agent=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test LaTeX file
            tex_file = temp_path / "test.tex"
            tex_file.write_text(
                "\\documentclass{article}\\begin{document}Test\\end{document}"
            )

            # Create config
            config = ServerConfig(output_dir=temp_path)
            compiler = LaTeXCompiler(config=config)

            # Create request with use_agent=False
            request = CompileLaTeXRequest(tex_filename="test.tex", use_agent=False)

            # Mock the compile_with_agent method to avoid actual LaTeX compilation
            with patch.object(compiler, "compile_with_agent") as mock_compile:
                from cv_writer_mcp.models import LaTeXCompilationResult

                mock_compile.return_value = LaTeXCompilationResult(
                    success=True,
                    compilation_time=1.0,
                    error_message=None,
                    output_path=temp_path / "test.pdf",
                )

                response = await compiler.compile_from_request(request)

                # Verify direct compile was called
                mock_compile.assert_called_once()

                # Verify response
                assert response.status == ConversionStatus.SUCCESS
                assert response.pdf_url == "cv-writer://pdf/test.pdf"
                assert response.metadata["compilation_method"] == "agent"
                assert "agent_response" not in response.metadata
