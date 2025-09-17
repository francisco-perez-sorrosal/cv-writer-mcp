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
pixi run start_mcps

# Start in development mode
pixi run dev

# Start with HTTP transport
TRANSPORT="streamable-http" pixi run start_mcps
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
pixi run lint-fix  # Lint with automatic fixes
pixi run type-check # Type check with mypy

# Run all CI checks
pixi run ci
```

### LaTeX Commands
```bash
# Check LaTeX installation
pixi run check-latex

# Compile LaTeX file (CLI)
pixi run compile-latex

# Fix PDF style issues
pixi run fix-style
```

## Architecture Overview

This is a **Model Context Protocol (MCP) server** built with **FastMCP framework** that provides AI-powered CV generation from markdown to LaTeX/PDF.

### Core Components

- **`main.py`**: FastMCP server setup, MCP tools definitions, and CLI entry point
- **`md2latex_agent.py`**: OpenAI Agents SDK integration for intelligent markdown→LaTeX conversion
- **`compiler_agent.py`**: LaTeX compilation engine with AI-powered error fixing
- **`latex_expert.py`**: LaTeX expertise and template management
- **`pdf_style_coordinator.py`**: PDF style analysis and coordination
- **`fixing_agent.py`**: Intelligent error fixing for LaTeX compilation issues
- **`page_capture_agent.py`**: Web page capture and analysis functionality
- **`latex_fix_agent.py`**: Specialized LaTeX error fixing agent
- **`models.py`**: Pydantic models for type-safe data handling
- **`tools.py`**: MCP tools implementation
- **`utils.py`**: Utility functions
- **`logger.py`**: Centralized logging configuration

### MCP Tools Provided

1. **`markdown_to_latex`**: AI-powered markdown to LaTeX conversion using OpenAI agents
2. **`compile_latex_to_pdf`**: LaTeX compilation with AI-powered error fixing
3. **`analyze_pdf_style`**: PDF style analysis and improvement suggestions
4. **`check_latex_installation`**: LaTeX environment verification
5. **`health_check`**: Server health monitoring

### Key Architecture Patterns

- **FastMCP Framework**: Unified MCP server and HTTP API in single codebase
- **OpenAI Agents SDK**: Uses multiple specialized agents for different tasks (conversion, compilation, fixing, analysis)
- **Agent-Based Architecture**: Specialized agents handle specific tasks (MD2LaTeX, LaTeX Expert, PDF Style Coordinator, Fixing Agent)
- **Transport Flexibility**: Supports both stdio (Claude Desktop) and HTTP transports
- **Pydantic Models**: Type-safe data validation throughout the pipeline
- **Async/Await**: Asynchronous processing for better performance
- **Multi-Stage Processing**: Markdown → LaTeX → PDF with intelligent error fixing at each stage

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