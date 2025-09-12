"""Tests for CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from cv_writer_mcp.main import app
from cv_writer_mcp.models import (
    CompileLaTeXResponse,
    ConversionStatus,
    ServerConfig,
)


class TestCLICommands:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ServerConfig(
            output_dir=Path(self.temp_dir) / "output",
            temp_dir=Path(self.temp_dir) / "temp",
            base_url="http://localhost:8000",
        )
        self.runner = CliRunner()

    def test_compile_latex_success(self):
        """Test successful LaTeX compilation via CLI."""
        # Create a test LaTeX file
        tex_file = Path(self.temp_dir) / "test.tex"
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        # Mock the LaTeX compiler to return success
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = True

            # Mock the compilation response
            mock_response = CompileLaTeXResponse(
                status=ConversionStatus.SUCCESS,
                pdf_url="cv-writer://pdf/test.pdf",
                metadata={"output_filename": "test.pdf", "engine": "pdflatex"},
            )

            mock_compiler.compile_latex_file = AsyncMock(return_value=mock_response)

            result = self.runner.invoke(app, ["compile-latex", str(tex_file)])

            assert result.exit_code == 0
            assert "Successfully compiled LaTeX to PDF" in result.stdout
            assert "cv-writer://pdf/test.pdf" in result.stdout

    def test_compile_latex_file_not_found(self):
        """Test CLI compilation with non-existent file."""
        result = self.runner.invoke(app, ["compile-latex", "nonexistent.tex"])

        assert result.exit_code == 1
        assert "No .tex file found" in result.stdout

    def test_compile_latex_invalid_extension(self):
        """Test CLI compilation with invalid file extension."""
        # Create a non-LaTeX file
        txt_file = Path(self.temp_dir) / "test.txt"
        txt_file.write_text("This is not a LaTeX file")

        result = self.runner.invoke(app, ["compile-latex", str(txt_file)])

        assert result.exit_code == 1
        assert "No .tex file found" in result.stdout

    def test_compile_latex_latex_not_installed(self):
        """Test CLI compilation when LaTeX is not installed."""
        # Create a test LaTeX file
        tex_file = Path(self.temp_dir) / "test.tex"
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        # Mock the LaTeX compiler to return False for installation check
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = False

            result = self.runner.invoke(app, ["compile-latex", str(tex_file)])

            assert result.exit_code == 1
            assert "LaTeX is not installed" in result.stdout

    def test_compile_latex_compilation_failure(self):
        """Test CLI compilation when compilation fails."""
        # Create a test LaTeX file
        tex_file = Path(self.temp_dir) / "test.tex"
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        # Mock the LaTeX compiler to return success for installation check
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = True

            # Mock the compilation response to return failure
            mock_response = CompileLaTeXResponse(
                status=ConversionStatus.FAILED,
                pdf_url=None,
                error_message="Compilation failed: Missing package",
            )

            mock_compiler.compile_latex_file = AsyncMock(return_value=mock_response)

            result = self.runner.invoke(app, ["compile-latex", str(tex_file)])

            assert result.exit_code == 1
            assert "Compilation failed" in result.stdout
            assert "Missing package" in result.stdout

    def test_compile_latex_with_custom_output(self):
        """Test CLI compilation with custom output filename."""
        # Create a test LaTeX file
        tex_file = Path(self.temp_dir) / "test.tex"
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        # Mock the LaTeX compiler
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = True

            # Mock the compilation response
            mock_response = CompileLaTeXResponse(
                status=ConversionStatus.SUCCESS,
                pdf_url="cv-writer://pdf/custom.pdf",
                metadata={"output_filename": "custom.pdf", "engine": "pdflatex"},
            )

            mock_compiler.compile_latex_file = AsyncMock(return_value=mock_response)

            result = self.runner.invoke(
                app, ["compile-latex", str(tex_file), "--output", "custom.pdf"]
            )

            assert result.exit_code == 0
            assert "Successfully compiled LaTeX to PDF" in result.stdout
            assert "cv-writer://pdf/custom.pdf" in result.stdout

    def test_compile_latex_with_debug_mode(self):
        """Test CLI compilation with debug mode enabled."""
        # Create a test LaTeX file
        tex_file = Path(self.temp_dir) / "test.tex"
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        # Mock the LaTeX compiler
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = True

            # Mock the compilation response
            mock_response = CompileLaTeXResponse(
                status=ConversionStatus.SUCCESS,
                pdf_url="cv-writer://pdf/test.pdf",
                metadata={"output_filename": "test.pdf", "engine": "pdflatex"},
            )

            mock_compiler.compile_latex_file = AsyncMock(return_value=mock_response)

            result = self.runner.invoke(
                app, ["compile-latex", str(tex_file), "--debug"]
            )

            assert result.exit_code == 0
            assert "Successfully compiled LaTeX to PDF" in result.stdout

    def test_compile_latex_with_custom_engine(self):
        """Test CLI compilation with custom LaTeX engine."""
        # Create a test LaTeX file
        tex_file = Path(self.temp_dir) / "test.tex"
        tex_file.write_text(
            "\\documentclass{article}\\begin{document}Test\\end{document}"
        )

        # Mock the LaTeX compiler
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = True

            # Mock the compilation response
            mock_response = CompileLaTeXResponse(
                status=ConversionStatus.SUCCESS,
                pdf_url="cv-writer://pdf/test.pdf",
                metadata={"output_filename": "test.pdf", "engine": "pdflatex"},
            )

            mock_compiler.compile_latex_file = AsyncMock(return_value=mock_response)

            result = self.runner.invoke(
                app, ["compile-latex", str(tex_file), "--engine", "pdflatex"]
            )

            assert result.exit_code == 0
            assert "Successfully compiled LaTeX to PDF" in result.stdout

    def test_compile_latex_help(self):
        """Test CLI compile-latex help."""
        result = self.runner.invoke(app, ["compile-latex", "--help"])

        assert result.exit_code == 0
        assert "Compile a LaTeX file to PDF from the command line" in result.stdout
        assert "--output" in result.stdout
        assert "--engine" in result.stdout
        assert "--debug" in result.stdout

    def test_check_latex_command(self):
        """Test the check-latex CLI command."""
        with patch("cv_writer_mcp.main.LaTeXExpert") as mock_compiler_class:
            mock_compiler = MagicMock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.check_latex_installation.return_value = True

            result = self.runner.invoke(app, ["check-latex"])

            assert result.exit_code == 0
            assert "✅" in result.stdout
            assert "PDFLATEX" in result.stdout
