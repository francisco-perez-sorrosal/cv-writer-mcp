"""Tests for LaTeX compiler."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cv_writer_mcp.compilation import LaTeXExpert
from cv_writer_mcp.models import (
    CompileLaTeXRequest,
    CompileLaTeXResponse,
    CompletionStatus,
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
        compiler = LaTeXExpert(config=self.config)
        assert compiler.timeout == 30
        assert compiler.config == self.config

    def test_initialization_no_config(self):
        """Test compiler initialization without config."""
        compiler = LaTeXExpert(config=ServerConfig())
        assert compiler.timeout == 30
        assert compiler.config is not None

    @pytest.mark.asyncio
    async def test_compile_latex_file_success(self):
        """Test successful LaTeX to PDF compilation."""
        # Create a test LaTeX file
        tex_file = self.config.output_dir / "test.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        with patch.object(LaTeXExpert, "orchestrate_compilation") as mock_agent:
            from cv_writer_mcp.models import OrchestrationResult

            mock_agent.return_value = OrchestrationResult(
                success=True,
                compilation_time=1.5,
                error_message=None,
                output_path=self.config.output_dir / "test.pdf",
            )

            compiler = LaTeXExpert(config=self.config)
            request = CompileLaTeXRequest(
                tex_filename="test.tex",
                output_filename="test.pdf",
                latex_engine=LaTeXEngine.PDFLATEX,
            )

            response = await compiler.compile_latex_file(request)

            assert response.status == CompletionStatus.SUCCESS
            assert response.pdf_url == "cv-writer://pdf/test.pdf"
            assert response.metadata["tex_filename"] == "test.tex"
            assert response.metadata["output_filename"] == "test.pdf"
            assert response.metadata["latex_engine"] == "pdflatex"
            assert response.metadata["compilation_time"] == 1.5

    @pytest.mark.asyncio
    async def test_compile_latex_file_no_config(self):
        """Test compilation without config."""
        compiler = LaTeXExpert(config=ServerConfig())  # No config
        request = CompileLaTeXRequest(
            tex_filename="test.tex",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        response = await compiler.compile_latex_file(request)

        assert response.status == CompletionStatus.FAILED
        assert "LaTeX file not found" in response.message

    @pytest.mark.asyncio
    async def test_compile_latex_file_file_not_found(self):
        """Test compilation with missing file."""
        compiler = LaTeXExpert(config=self.config)
        request = CompileLaTeXRequest(
            tex_filename="nonexistent.tex",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        response = await compiler.compile_latex_file(request)

        assert response.status == CompletionStatus.FAILED
        assert "LaTeX file not found: nonexistent.tex" in response.message

    @pytest.mark.asyncio
    async def test_compile_latex_file_compilation_failure(self):
        """Test compilation with compilation failure."""
        # Create a test LaTeX file
        tex_file = self.config.output_dir / "test.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        with patch.object(LaTeXExpert, "compile_latex_file") as mock_compile:
            mock_compile.return_value = CompileLaTeXResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                error_message="LaTeX compilation failed",
            )

            compiler = LaTeXExpert(config=self.config)
            request = CompileLaTeXRequest(
                tex_filename="test.tex",
                latex_engine=LaTeXEngine.PDFLATEX,
            )

            response = await compiler.compile_latex_file(request)

            assert response.status == CompletionStatus.FAILED
            assert "LaTeX compilation failed" in response.message

    @pytest.mark.asyncio
    async def test_compile_latex_file_exception(self):
        """Test compilation with exception."""
        # Create a test LaTeX file
        tex_file = self.config.output_dir / "test.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        with patch.object(LaTeXExpert, "orchestrate_compilation") as mock_agent:
            mock_agent.side_effect = Exception("Compilation error")

            compiler = LaTeXExpert(config=self.config)
            request = CompileLaTeXRequest(
                tex_filename="test.tex",
                latex_engine=LaTeXEngine.PDFLATEX,
            )

            response = await compiler.compile_latex_file(request)

            assert response.status == CompletionStatus.FAILED
            assert "Unexpected error: Compilation error" in response.message

    def test_check_latex_installation_with_engine(self):
        """Test LaTeX installation check with specific engine."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            compiler = LaTeXExpert(config=ServerConfig())
            result = compiler.check_latex_installation(LaTeXEngine.PDFLATEX)

            assert result is True
            mock_run.assert_called_once_with(
                ["pdflatex", "--version"], capture_output=True, timeout=10
            )

    def test_check_latex_installation_not_found(self):
        """Test LaTeX installation check when not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            compiler = LaTeXExpert(config=ServerConfig())
            result = compiler.check_latex_installation(LaTeXEngine.PDFLATEX)

            assert result is False

    def test_check_latex_installation_timeout(self):
        """Test LaTeX installation check with timeout."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("pdflatex", 10)

            compiler = LaTeXExpert(config=ServerConfig())
            result = compiler.check_latex_installation(LaTeXEngine.PDFLATEX)

            assert result is False

    def test_check_latex_installation_default_engine(self):
        """Test LaTeX installation check with default engine."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            compiler = LaTeXExpert(config=ServerConfig())
            result = compiler.check_latex_installation()

            assert result is True
            mock_run.assert_called_once_with(
                ["pdflatex", "--version"], capture_output=True, timeout=10
            )

    @pytest.mark.asyncio
    async def test_compile_latex_file_with_agent_default(self):
        """Test compile_latex_file uses agent by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test LaTeX file
            tex_file = temp_path / "test.tex"
            tex_file.write_text(
                "\\documentclass{article}\\begin{document}Test\\end{document}"
            )

            # Create config
            config = ServerConfig(output_dir=temp_path)
            compiler = LaTeXExpert(config=config)

            # Create request (should use agent by default)
            request = CompileLaTeXRequest(tex_filename="test.tex")

            # Mock the orchestrate_compilation method to avoid actual agent execution
            with patch.object(compiler, "orchestrate_compilation") as mock_agent:
                from cv_writer_mcp.models import OrchestrationResult

                mock_agent.return_value = OrchestrationResult(
                    success=True,
                    compilation_time=1.0,
                    error_message=None,
                    output_path=temp_path / "test.pdf",
                )

                response = await compiler.compile_latex_file(request)

                # Verify agent was called
                mock_agent.assert_called_once()

                # Verify response
                assert response.status == CompletionStatus.SUCCESS
                assert response.pdf_url == "cv-writer://pdf/test.pdf"

    @pytest.mark.asyncio
    async def test_compile_latex_file_with_compilation(self):
        """Test compile_latex_file compilation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test LaTeX file
            tex_file = temp_path / "test.tex"
            tex_file.write_text(
                "\\documentclass{article}\\begin{document}Test\\end{document}"
            )

            # Create config
            config = ServerConfig(output_dir=temp_path)
            compiler = LaTeXExpert(config=config)

            # Create request
            request = CompileLaTeXRequest(tex_filename="test.tex")

            # Mock the orchestrate_compilation method to avoid actual LaTeX compilation
            with patch.object(compiler, "orchestrate_compilation") as mock_compile:
                from cv_writer_mcp.models import OrchestrationResult

                mock_compile.return_value = OrchestrationResult(
                    success=True,
                    compilation_time=1.0,
                    error_message=None,
                    output_path=temp_path / "test.pdf",
                )

                response = await compiler.compile_latex_file(request)

                # Verify direct compile was called
                mock_compile.assert_called_once()

                # Verify response
                assert response.status == CompletionStatus.SUCCESS
                assert response.pdf_url == "cv-writer://pdf/test.pdf"
                assert "agent_response" not in response.metadata
