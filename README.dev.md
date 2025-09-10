# CV Writer MCP Server - Development Guide

This document provides detailed information for developers working on the CV Writer MCP Server project, built using the FastMCP framework.

## Development Setup

### Prerequisites

- Python 3.11+
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/francisco-perez-sorrosal/cv-writer-mcp.git
cd cv-writer-mcp

# Install with pixi (recommended)
pixi install

# Install development dependencies
pixi install --environment dev

# Copy environment file
cp env.example .env
```

### Verify Installation

```bash
# Check LaTeX installation
pixi run start check-latex

# Run tests
pixi run test

# Run linting
pixi run lint
```

## Development Workflow

### Running the Server

```bash
# Development mode (with hot reload)
pixi run start --dev

# MCP server only
pixi run start --mcp-only

# Web server only
pixi run start --web-only
```

### Testing

```bash
# Run all tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Run specific test file
pixi run test tests/test_markdown_converter.py

# Run tests with verbose output
pixi run test -v
```

### Code Quality

```bash
# Format code
pixi run format

# Lint code
pixi run lint

# Type checking
pixi run type-check

# Run all quality checks
pixi run ci
```

## Project Architecture

### Core Components

1. **FastMCP Server** (`fastmcp_server.py`): Implements both MCP protocol and HTTP API using FastMCP
2. **CV Converter** (`cv_converter.py`): Main conversion orchestration
3. **Markdown Converter** (`markdown_converter.py`): Converts markdown to LaTeX
4. **LaTeX Compiler** (`latex_compiler.py`): Compiles LaTeX to PDF
5. **Models** (`models.py`): Pydantic data models
6. **Logger** (`logger.py`): Logging configuration

### Data Flow

```
Markdown Content → MarkdownToLaTeXConverter → LaTeX Content
                                                      ↓
LaTeX Template → Template Combiner → Full LaTeX Document
                                                      ↓
LaTeX Document → LaTeXCompiler → PDF File
                                                      ↓
PDF File → Web Server → HTTP URL
```

### MCP Integration

The server implements the MCP protocol with three main tools:

1. **convert_cv**: Main conversion tool
2. **get_job_status**: Job status tracking
3. **check_latex_installation**: System verification

## Testing Strategy

### Unit Tests

- **test_markdown_converter.py**: Tests markdown to LaTeX conversion
- **test_models.py**: Tests Pydantic model validation
- **test_cv_converter.py**: Tests main conversion logic

### Test Structure

```python
class TestComponentName:
    """Test cases for ComponentName."""
    
    def setup_method(self):
        """Set up test fixtures."""
        pass
    
    def test_specific_functionality(self):
        """Test specific functionality."""
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_markdown_converter.py::TestMarkdownToLaTeXConverter::test_convert_headers

# Run with verbose output
pytest -v
```

## Code Style and Standards

### Python Style

- Follow PEP 8
- Use type hints throughout
- Use Pydantic for data validation
- Use loguru for logging
- Use rich for console output

### Code Formatting

```bash
# Format with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

### Import Organization

```python
# Standard library imports
import asyncio
from pathlib import Path

# Third-party imports
from loguru import logger
from pydantic import BaseModel

# Local imports
from .models import ServerConfig
```

## Adding New Features

### 1. Define Models

Add new Pydantic models in `models.py`:

```python
class NewFeatureRequest(BaseModel):
    """Request model for new feature."""
    field1: str = Field(..., description="Description")
    field2: Optional[int] = Field(None, description="Optional field")
```

### 2. Implement Logic

Create the core logic in appropriate module:

```python
class NewFeatureService:
    """Service for new feature."""
    
    async def process(self, request: NewFeatureRequest) -> NewFeatureResponse:
        """Process the request."""
        pass
```

### 3. Add MCP Tool

Add tool to `mcp_server.py`:

```python
Tool(
    name="new_feature",
    description="Description of new feature",
    inputSchema={
        "type": "object",
        "properties": {
            "field1": {"type": "string", "description": "Description"}
        },
        "required": ["field1"]
    }
)
```

### 4. Add HTTP Endpoint

Add endpoint to `web_server.py`:

```python
@self.app.post("/api/new-feature")
async def new_feature(request: NewFeatureRequest) -> NewFeatureResponse:
    """New feature endpoint."""
    pass
```

### 5. Add Tests

Create comprehensive tests:

```python
class TestNewFeature:
    """Test cases for new feature."""
    
    def test_valid_request(self):
        """Test valid request."""
        pass
    
    def test_invalid_request(self):
        """Test invalid request."""
        pass
```

## Debugging

### Logging

The server uses loguru for logging. Configure log levels in `.env`:

```bash
LOG_LEVEL=DEBUG
```

### Debug Mode

Run in debug mode for detailed output:

```bash
pixi run start --dev
```

### LaTeX Debugging

For LaTeX compilation issues:

1. Check LaTeX installation: `pixi run start check-latex`
2. Review LaTeX logs in job metadata
3. Test with simple context files first
4. Check for missing LaTeX packages

### MCP Debugging

For MCP issues:

1. Check MCP server logs
2. Verify tool schemas
3. Test with MCP client
4. Check JSON serialization

## Performance Considerations

### LaTeX Compilation

- LaTeX compilation is CPU-intensive
- Consider timeout settings for large documents
- Use appropriate LaTeX engines for different content types

### File Management

- Clean up temporary files regularly
- Implement job cleanup for old completed jobs
- Monitor disk usage in output directories

### Memory Usage

- Large markdown files can consume significant memory
- Consider streaming for very large files
- Monitor memory usage during compilation

## Security Considerations

### File Access

- Validate file paths to prevent directory traversal
- Limit file sizes to prevent DoS attacks
- Sanitize user input in context files

### API Security

- Implement rate limiting for API endpoints
- Validate all input data
- Use HTTPS in production

### LaTeX Security

- LaTeX compilation can execute arbitrary code
- Validate LaTeX context files
- Run compilation in sandboxed environment if possible

## Deployment

### Production Configuration

```bash
# Production environment variables
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
BASE_URL=https://your-domain.com
SECRET_KEY=your-secure-secret-key
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install LaTeX
RUN apt-get update && apt-get install -y texlive-full

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -e .

# Run application
CMD ["python", "-m", "cv_writer_mcp"]
```

### Health Checks

The server provides health check endpoints:

- `GET /health` - Basic health check
- `GET /api/status/{job_id}` - Job status check

## Release Process

### Version Management

The project uses semantic versioning and python-semantic-release:

```bash
# Check current version
python -c "from cv_writer_mcp import __version__; print(__version__)"

# Release new version
pixi run semantic-release
```

### Release Checklist

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Run full test suite: `pixi run ci`
4. Create release PR
5. Tag release
6. Update documentation

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run quality checks: `pixi run ci`
5. Commit changes: `git commit -m "Add amazing feature"`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open pull request

### Code Review

- All code must pass CI checks
- Tests must have good coverage
- Documentation must be updated
- Follow established patterns

### Issue Reporting

When reporting issues:

1. Use the issue template
2. Include reproduction steps
3. Provide system information
4. Include relevant logs

## Resources

### Documentation

- [MCP Protocol Documentation](https://modelcontextprotocol.io/docs/getting-started/intro)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LaTeX Documentation](https://www.latex-project.org/help/documentation/)

### Tools

- [Pixi Documentation](https://pixi.sh/)
- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Rich Documentation](https://rich.readthedocs.io/)

### Community

- [MCP Discord](https://discord.gg/modelcontextprotocol)
- [Python Discord](https://discord.gg/python)
- [LaTeX Stack Exchange](https://tex.stackexchange.com/)
