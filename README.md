# CV Writer MCP Server

A Model Context Protocol (MCP) server that converts markdown CV content to LaTeX and compiles it to PDF. This server provides both MCP tools and CLI commands for CV conversion functionality, built using the FastMCP framework for seamless integration.

## Features

- **AI-Powered Markdown to LaTeX Conversion**: Uses OpenAI Agents for intelligent conversion from markdown to LaTeX format
- **Intelligent LaTeX Compilation**: Compile LaTeX documents to PDF using AI agents with automatic error fixing
- **MCP Integration**: Provides MCP tools for seamless integration with AI assistants
- **FastMCP Framework**: Built using FastMCP for unified MCP and CLI support
- **CLI Interface**: Command-line tools for intelligent LaTeX compilation and server management
- **PDF Serving**: Serve generated PDFs and LaTeX files via MCP resources
- **Structured Output**: Pydantic models for type-safe data handling
- **LaTeX Engine Support**: Support for pdflatex with intelligent error fixing
- **OpenAI Agents SDK**: Leverages the power of OpenAI's agent framework for intelligent conversion
- **Transport Flexibility**: Support for both stdio and HTTP transport protocols
- **Health Monitoring**: Built-in health check and LaTeX installation verification

## Installation

### Prerequisites

- Python 3.11+
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- OpenAI API key (for AI-powered markdown to LaTeX conversion)

### Install LaTeX

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

### Install the Package

```bash
# Clone the repository
git clone https://github.com/francisco-perez-sorrosal/cv-writer-mcp.git
cd cv-writer-mcp

# Install with pixi (recommended)
pixi install

# Or install with pip
pip install -e .
```

### Development Setup

```bash
# Install development dependencies
pixi install --environment dev

# Run tests
pixi run test

# Run linting
pixi run lint

# Format code
pixi run format

# Type checking
pixi run type-check

# Run all checks
pixi run ci
```

### Environment Setup

Set up your OpenAI API key and transport configuration:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Configure transport protocol (optional)
export TRANSPORT="stdio"  # Default: stdio for Claude Desktop
# or
export TRANSPORT="streamable-http"  # For HTTP-based MCP clients

# Configure server settings (optional)
export HOST="localhost"  # Default: localhost
export PORT="8000"       # Default: 8000

# Or create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "TRANSPORT=stdio" >> .env
echo "HOST=localhost" >> .env
echo "PORT=8000" >> .env
```

## Usage

### Transport Configuration

The server supports two transport protocols:

- **`stdio`** (default): Standard input/output transport, suitable for local Claude Desktop integration
- **`streamable-http`**: HTTP-based transport for web-based MCP clients and remote access

Configure the transport using the `TRANSPORT` environment variable:

```bash
# For Claude Desktop integration (default)
export TRANSPORT="stdio"
pixi run start_mcps

# For HTTP-based clients
export TRANSPORT="streamable-http"
pixi run start_mcps
```

### FastMCP Server

Start the FastMCP server (provides both MCP tools and HTTP API):

```bash
# Run the server (default: localhost:8000)
pixi run start_mcps

# Run with custom host and port
pixi run start_mcps --host 0.0.0.0 --port 9000

# Run in development mode
pixi run start_mcps --dev
```

The FastMCP server provides MCP tools and resources:

- **MCP Tools**: `markdown_to_latex`, `compile_latex_to_pdf`, `check_latex_installation`, `health_check`
- **MCP Resources**: `cv-writer://pdf/{filename}`, `cv-writer://tex/{filename}`

### MCP Tools

The server provides the following MCP tools:

#### `markdown_to_latex`

Convert markdown CV content to LaTeX using AI-powered conversion.

**Parameters:**
- `markdown_content` (string, required): Markdown content of the CV
- `output_filename` (string, optional): Custom output filename for .tex file

**Example:**
```json
{
  "markdown_content": "# John Doe\n\n## Experience\n- Software Engineer at Company X",
  "output_filename": "john_doe_cv.tex"
}
```

#### `compile_latex_to_pdf`

Compile a LaTeX file to PDF using intelligent agents with automatic error fixing.

**Parameters:**
- `tex_filename` (string, required): Name of the .tex file to compile
- `output_filename` (string, optional): Custom output filename for PDF
- `latex_engine` (string, optional): LaTeX engine (currently only "pdflatex" supported)

#### `check_latex_installation`

Check if LaTeX is installed and accessible.

**Parameters:**
- `engine` (string, optional): LaTeX engine to check (currently only "pdflatex" supported)

#### `health_check`

Check the health status of the CV Writer MCP server.

**Returns:**
- JSON response with server status, service name, timestamp, and version

### Command Line Interface

The server provides CLI commands for intelligent LaTeX compilation:

```bash
# Start the MCP server
pixi run start_mcps

# Start server with custom host and port
pixi run start_mcps --host 0.0.0.0 --port 9000

# Start server in development mode
pixi run start_mcps --debug

# Check LaTeX installation
pixi run check-latex

# Convert markdown CV to LaTeX
pixi run convert-markdown input/cv.md

# Convert with custom output filename
pixi run convert-markdown input/cv.md --output my_cv.tex

# Compile a LaTeX file to PDF using intelligent agents
pixi run compile-latex output/test.tex

# Compile with custom output filename
pixi run compile-latex output/test.tex --output my_cv.pdf

# Compile with debug mode
pixi run compile-latex output/test.tex --debug

# Compile with custom LaTeX engine
pixi run compile-latex output/test.tex --engine pdflatex
```

## Configuration

Create a `.env` file based on `env.example`:

```bash
cp env.example .env
```

Key configuration options:

- `HOST`: Server host (default: localhost)
- `PORT`: Server port (default: 8000)
- `BASE_URL`: Base URL for PDF serving (default: http://localhost:8000)
- `OUTPUT_DIR`: Directory for generated PDFs (default: ./output)
- `TEMP_DIR`: Directory for temporary files (default: ./temp)
- `TEMPLATES_DIR`: Directory for LaTeX templates (default: ./context)
- `LATEX_TIMEOUT`: LaTeX compilation timeout in seconds (default: 180)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DEBUG`: Enable debug mode (default: false)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 10485760)
- `TRANSPORT`: Transport protocol ("stdio" or "streamable-http", default: stdio)

## LaTeX Template Format

The server supports several template formats:

### Content Placeholder
```latex
\documentclass{article}
\begin{document}
{{CONTENT}}
\end{document}
```

### Input Command
```latex
\documentclass{article}
\begin{document}
\input{content}
\end{document}
```

### Document Environment
```latex
\documentclass{article}
\begin{document}
\title{Curriculum Vitae}
\maketitle
% Content will be inserted here
\end{document}
```

## Development

The development setup is already covered in the Installation section above. Here are additional development commands:

### Project Structure

```
cv-writer-mcp/
├── src/cv_writer_mcp/
│   ├── __init__.py
│   ├── __main__.py          # Module entry point
│   ├── main.py              # CLI entry point and MCP server
│   ├── cv_converter.py      # Main conversion logic
│   ├── md2latex_agent.py    # Markdown to LaTeX conversion agent
│   ├── latex_compiler.py    # LaTeX compilation with AI agents
│   ├── models.py            # Pydantic models and data structures
│   └── logger.py            # Logging configuration
├── tests/                   # Test suite
│   ├── test_cli_commands.py
│   ├── test_cv_converter.py
│   ├── test_latex_compiler.py
│   ├── test_main_common_functions.py
│   └── test_models.py
├── context/                 # LaTeX templates and context files
│   └── latex/              # LaTeX templates and user guide
│       ├── moderncv_template.tex
│       └── moderncv_userguide.txt
├── input/                   # Input files and documentation
│   └── input_cv_test.md
├── output/                  # Generated files
├── temp/                    # Temporary files
├── pyproject.toml          # Project configuration
├── pixi.lock              # Pixi lock file
├── env.example            # Environment variables template
└── README.md              # This file
```

## Usage Examples

### Convert Markdown to LaTeX via MCP

The MCP tools can be used by AI assistants like Claude Desktop:

1. **Convert markdown to LaTeX**: Use the `markdown_to_latex` tool
2. **Compile LaTeX to PDF**: Use the `compile_latex_to_pdf` tool
3. **Check LaTeX installation**: Use the `check_latex_installation` tool

### CLI Usage Examples

```bash
# Start the MCP server
pixi run start_mcps

# Start server in development mode
pixi run start_mcps --debug

# Convert markdown to LaTeX
pixi run convert-markdown input/cv.md --output my_cv.tex

# Basic LaTeX compilation
pixi run compile-latex output/simple_test.tex

# Compile with custom output
pixi run compile-latex output/simple_test.tex --output my_cv.pdf

# Check LaTeX installation
pixi run check-latex

# Get help for any command
pixi run compile-latex --help
```

## Troubleshooting

### LaTeX Not Found

If you get "LaTeX engine not found" errors:

1. Install LaTeX distribution (see Installation section)
2. Ensure LaTeX binaries are in your PATH
3. Check installation with: `pixi run check-latex`

### Compilation Errors

Common LaTeX compilation issues:

1. **Missing packages**: Add required packages to your template
2. **Invalid characters**: Ensure proper escaping of special characters
3. **Template syntax**: Check template format and placeholders

### Permission Errors

If you get permission errors:

1. Ensure output and temp directories are writable
2. Check file permissions: `chmod 755 output temp`
3. Run with appropriate user permissions

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pixi run ci`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Francisco Perez-Sorrosal**
- Email: fperezsorrosal@gmail.com
- GitHub: [francisco-perez-sorrosal](https://github.com/francisco-perez-sorrosal)
