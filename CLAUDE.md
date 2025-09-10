# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Environment
```bash
# Install dependencies and setup
pixi install

# Set environment variables (required for OpenAI integration)
export OPENAI_API_KEY="your-api-key-here"
export TRANSPORT="stdio"  # or "streamable-http" for HTTP transport
```

### Running the Server
```bash
# Start MCP server (default: stdio transport)
pixi run start

# Start in development mode
pixi run dev

# Start with HTTP transport
TRANSPORT="streamable-http" pixi run start
```

### Development Tasks
```bash
# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Code formatting and linting
pixi run format    # Format with black
pixi run lint      # Lint with ruff
pixi run type-check # Type check with mypy

# Run all CI checks
pixi run ci
```

### LaTeX Commands
```bash
# Check LaTeX installation
pixi run check-latex

# Compile LaTeX file (CLI)
pixi run python -m cv_writer_mcp compile-latex output/file.tex --output result.pdf
```

## Architecture Overview

This is a **Model Context Protocol (MCP) server** built with **FastMCP framework** that provides AI-powered CV generation from markdown to LaTeX/PDF.

### Core Components

- **`main.py`**: FastMCP server setup, MCP tools definitions, and CLI entry point
- **`md2latex_agent.py`**: OpenAI Agents SDK integration for intelligent markdown→LaTeX conversion
- **`latex_compiler.py`**: LaTeX compilation engine with support for multiple engines (pdflatex, xelatex, lualatex)  
- **`cv_converter.py`**: Main conversion orchestration logic
- **`models.py`**: Pydantic models for type-safe data handling
- **`logger.py`**: Centralized logging configuration

### MCP Tools Provided

1. **`markdown_to_latex`**: AI-powered markdown to LaTeX conversion using OpenAI agents
2. **`compile_latex_to_pdf`**: LaTeX compilation with multiple engine support
3. **`check_latex_installation`**: LaTeX environment verification
4. **`health_check`**: Server health monitoring

### Key Architecture Patterns

- **FastMCP Framework**: Unified MCP server and HTTP API in single codebase
- **OpenAI Agents SDK**: Uses agents for intelligent markdown interpretation and LaTeX generation
- **Transport Flexibility**: Supports both stdio (Claude Desktop) and HTTP transports
- **Pydantic Models**: Type-safe data validation throughout the pipeline
- **Async/Await**: Asynchronous processing for better performance

### Directory Structure
- `src/cv_writer_mcp/`: Main source code
- `tests/`: Test suite with pytest
- `templates/`: LaTeX templates
- `output/`: Generated PDF/LaTeX files
- `input/`: Sample input files and documentation
- `temp/`: Temporary compilation files

### Transport Configuration
The server supports two transport modes via `TRANSPORT` environment variable:
- `stdio`: For Claude Desktop integration (default)
- `streamable-http`: For HTTP-based MCP clients

### Dependencies
- **MCP**: Model Context Protocol framework
- **OpenAI Agents SDK**: AI-powered conversion logic
- **Pydantic**: Data validation and type safety
- **FastMCP**: Unified MCP/HTTP server framework
- **Typer**: CLI interface
- **LaTeX**: External dependency for PDF compilation