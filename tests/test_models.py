"""Tests for simplified Pydantic models."""

import pytest
from pydantic import ValidationError

from cv_writer_mcp.compilation.models import CompileLaTeXRequest, CompileLaTeXResponse, LaTeXEngine
from cv_writer_mcp.conversion.models import MarkdownToLaTeXRequest, MarkdownToLaTeXResponse
from cv_writer_mcp.models import CompletionStatus, HealthStatusResponse, ServerConfig


class TestMarkdownToLaTeXRequest:
    """Test cases for MarkdownToLaTeXRequest model."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = MarkdownToLaTeXRequest(
            markdown_content="# My CV\nThis is my CV content.",
            output_filename="my_cv.tex",
        )

        assert request.markdown_content == "# My CV\nThis is my CV content."
        assert request.output_filename == "my_cv.tex"

    def test_valid_request_no_filename(self):
        """Test valid request creation without filename."""
        request = MarkdownToLaTeXRequest(
            markdown_content="# My CV\nThis is my CV content."
        )

        assert request.markdown_content == "# My CV\nThis is my CV content."
        assert request.output_filename is None

    def test_empty_markdown_content(self):
        """Test validation of empty markdown content."""
        with pytest.raises(ValidationError) as exc_info:
            MarkdownToLaTeXRequest(markdown_content="")

        assert "cannot be empty" in str(exc_info.value)

    def test_whitespace_only_content(self):
        """Test validation of whitespace-only content."""
        with pytest.raises(ValidationError) as exc_info:
            MarkdownToLaTeXRequest(markdown_content="   \n\n   ")

        assert "cannot be empty" in str(exc_info.value)


class TestMarkdownToLaTeXResponse:
    """Test cases for MarkdownToLaTeXResponse model."""

    def test_success_response(self):
        """Test successful response creation."""
        response = MarkdownToLaTeXResponse(
            status=CompletionStatus.SUCCESS,
            tex_url="cv-writer://tex/test.tex",
        )

        assert response.status == CompletionStatus.SUCCESS
        assert response.tex_url == "cv-writer://tex/test.tex"
        assert response.message is None

    def test_failed_response(self):
        """Test failed response creation."""
        response = MarkdownToLaTeXResponse(
            status=CompletionStatus.FAILED,
            message="Conversion failed",
        )

        assert response.status == CompletionStatus.FAILED
        assert response.tex_url is None
        assert response.message == "Conversion failed"


class TestCompileLaTeXRequest:
    """Test cases for CompileLaTeXRequest model."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = CompileLaTeXRequest(
            tex_filename="test.tex",
            output_filename="test.pdf",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        assert request.tex_filename == "test.tex"
        assert request.output_filename == "test.pdf"
        assert request.latex_engine == LaTeXEngine.PDFLATEX

    def test_valid_request_no_output_filename(self):
        """Test valid request creation without output filename."""
        request = CompileLaTeXRequest(
            tex_filename="test.tex",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        assert request.tex_filename == "test.tex"
        assert request.output_filename == "test.pdf"  # Auto-generated from tex_filename
        assert request.latex_engine == LaTeXEngine.PDFLATEX

    def test_tex_filename_auto_extension(self):
        """Test that .tex extension is added automatically."""
        request = CompileLaTeXRequest(
            tex_filename="test",
            latex_engine=LaTeXEngine.PDFLATEX,
        )

        assert request.tex_filename == "test.tex"

    def test_empty_tex_filename(self):
        """Test validation of empty tex filename."""
        with pytest.raises(ValidationError) as exc_info:
            CompileLaTeXRequest(tex_filename="")

        assert "cannot be empty" in str(exc_info.value)


class TestCompileLaTeXResponse:
    """Test cases for CompileLaTeXResponse model."""

    def test_success_response(self):
        """Test successful response creation."""
        response = CompileLaTeXResponse(
            status=CompletionStatus.SUCCESS,
            pdf_url="cv-writer://pdf/test.pdf",
            message="Successfully compiled test.tex to test.pdf",
        )

        assert response.status == CompletionStatus.SUCCESS
        assert response.pdf_url == "cv-writer://pdf/test.pdf"
        assert response.message == "Successfully compiled test.tex to test.pdf"

    def test_failed_response(self):
        """Test failed response creation."""
        response = CompileLaTeXResponse(
            status=CompletionStatus.FAILED,
            message="Compilation failed",
        )

        assert response.status == CompletionStatus.FAILED
        assert response.pdf_url is None
        assert response.message == "Compilation failed"


class TestCompletionStatus:
    """Test cases for CompletionStatus enum."""

    def test_status_values(self):
        """Test that status values are correct."""
        assert CompletionStatus.SUCCESS == "success"
        assert CompletionStatus.FAILED == "failed"


class TestLaTeXEngine:
    """Test cases for LaTeXEngine enum."""

    def test_engine_values(self):
        """Test that engine values are correct."""
        # Primary engine (currently supported)
        assert LaTeXEngine.PDFLATEX == "pdflatex"

    def test_default_engine(self):
        """Test that PDFLATEX is the default engine."""
        # Test that CompileLaTeXRequest defaults to PDFLATEX
        request = CompileLaTeXRequest(tex_filename="test.tex")
        assert request.latex_engine == LaTeXEngine.PDFLATEX


class TestServerConfig:
    """Test cases for ServerConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ServerConfig()

        assert config.host == "localhost"
        assert config.port == 8000
        assert config.base_url == "http://localhost:8000"
        assert config.debug is False
        assert config.log_level.value == "INFO"
        assert config.max_file_size == 10 * 1024 * 1024
        assert config.latex_timeout == 30
        assert config.openai_api_key is None
        assert config.templates_dir.name == "context"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ServerConfig(
            host="0.0.0.0",
            port=9000,
            base_url="https://example.com",
            debug=True,
            log_level="DEBUG",
            max_file_size=20 * 1024 * 1024,
            latex_timeout=60,
            openai_api_key="sk-test-key",
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.base_url == "https://example.com"
        assert config.debug is True
        assert config.log_level.value == "DEBUG"
        assert config.max_file_size == 20 * 1024 * 1024
        assert config.latex_timeout == 60
        assert config.openai_api_key == "sk-test-key"

    def test_directory_creation(self, tmp_path):
        """Test that directories are created automatically."""
        output_dir = tmp_path / "output"
        temp_dir = tmp_path / "temp"
        templates_dir = tmp_path / "context"

        config = ServerConfig(
            output_dir=output_dir,
            temp_dir=temp_dir,
            templates_dir=templates_dir,
        )

        assert output_dir.exists()
        assert temp_dir.exists()
        assert templates_dir.exists()
        assert config.output_dir == output_dir
        assert config.temp_dir == temp_dir
        assert config.templates_dir == templates_dir

    def test_base_url_consistency(self):
        """Test that base_url is consistent with host and port."""
        # Test default case
        config = ServerConfig()
        assert config.get_base_url() == "http://localhost:8000"

        # Test custom host and port
        config = ServerConfig(host="0.0.0.0", port=9000)
        assert config.get_base_url() == "http://0.0.0.0:9000"

        # Test explicit base_url override
        config = ServerConfig(
            host="0.0.0.0", port=9000, base_url="https://custom.example.com"
        )
        assert (
            config.get_base_url() == "http://0.0.0.0:9000"
        )  # Should still reflect host/port
        assert (
            config.base_url == "https://custom.example.com"
        )  # But base_url field is preserved

    def test_model_post_init_base_url_update(self):
        """Test that model_post_init updates base_url when host/port change."""
        # Test that base_url gets updated when host/port are different from default
        config = ServerConfig(host="0.0.0.0", port=9000)
        # The model_post_init should have updated base_url to match host/port
        assert config.base_url == "http://0.0.0.0:9000"


class TestHealthStatusResponse:
    """Test cases for HealthStatusResponse."""

    def test_default_health_response(self):
        """Test default health response creation."""
        response = HealthStatusResponse()
        assert response.status == "healthy"
        assert response.service == "cv-writer-mcp"
        assert response.timestamp is None
        assert response.version is None

    def test_custom_health_response(self):
        """Test custom health response creation."""
        response = HealthStatusResponse(
            status="healthy",
            service="test-service",
            timestamp="2024-01-01T00:00:00",
            version="1.0.0",
        )
        assert response.status == "healthy"
        assert response.service == "test-service"
        assert response.timestamp == "2024-01-01T00:00:00"
        assert response.version == "1.0.0"

    def test_health_response_json_serialization(self):
        """Test health response JSON serialization."""
        response = HealthStatusResponse(
            status="healthy",
            service="test-service",
            timestamp="2024-01-01T00:00:00",
            version="1.0.0",
        )
        json_data = response.model_dump()
        assert json_data["status"] == "healthy"
        assert json_data["service"] == "test-service"
        assert json_data["timestamp"] == "2024-01-01T00:00:00"
        assert json_data["version"] == "1.0.0"
