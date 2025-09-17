# CV Writer MCP Examples

This directory contains examples for using the CV Writer MCP Server tools.

## Available Tools

### 1. PDF Style Analysis (`fix-style`)

The `fix-style` command analyzes PDF layouts visually and improves LaTeX formatting using AI agents with browser automation.

#### Usage:

```bash
# Basic usage with default files
pixi run fix-style

# Custom input files
pixi run fix-style --pdf ./path/to/your.pdf --tex ./path/to/your.tex

# Custom output file
pixi run fix-style --pdf input.pdf --tex input.tex --output improved.tex

# With debug mode
pixi run fix-style --debug
```

#### Requirements:

1. **Input Files**: You need both a PDF file and its corresponding LaTeX source file
2. **OpenAI API Key**: Set the `OPENAI_API_KEY` environment variable
3. **Playwright**: The tool uses browser automation to analyze PDFs

#### Features:

- **Visual Analysis**: Opens PDF in browser and analyzes layout issues
- **AI-Powered Improvements**: Uses GPT-4o-mini to identify and fix formatting problems
- **LaTeX Expertise**: Applies moderncv best practices and proper LaTeX commands
- **Content Preservation**: Never changes text content, only improves formatting

#### Example Workflow:

1. Create a LaTeX CV and compile it to PDF
2. Use `fix-style` to analyze the PDF and improve the LaTeX
3. Compile the improved LaTeX to see the enhancements

```bash
# Step 1: Compile your LaTeX to PDF
pixi run compile-latex your_cv.tex --output your_cv.pdf

# Step 2: Analyze and improve the style
pixi run fix-style --pdf your_cv.pdf --tex your_cv.tex --output improved_cv.tex

# Step 3: Compile the improved version
pixi run compile-latex improved_cv.tex --output improved_cv.pdf
```

### 2. LaTeX Compilation (`compile-latex`)

Compile LaTeX files to PDF using intelligent agents.

```bash
# Basic compilation
pixi run compile-latex your_file.tex

# Custom output filename
pixi run compile-latex your_file.tex --output custom_name.pdf

# With specific engine
pixi run compile-latex your_file.tex --engine pdflatex
```

### 3. MCP Server (`start-mcps`)

Start the MCP server for integration with other tools.

```bash
# Start the server
pixi run start_mcps

# With debug mode
pixi run start_mcps --debug
```

## Prerequisites

- Python 3.11+
- LaTeX installation (for compilation)
- OpenAI API key (for AI features)
- Playwright (for PDF analysis)

## Installation

```bash
# Install dependencies
pixi install

# Install Playwright browsers
pixi run playwright install
```

## Environment Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Optional: Set custom directories
export OUTPUT_DIR='./output'
export TEMPLATES_DIR='./context'
```