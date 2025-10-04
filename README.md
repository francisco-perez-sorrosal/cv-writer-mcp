# CV Writer MCP Server

An AI-powered Model Context Protocol (MCP) server that generates professional CVs from markdown with intelligent style optimization. Built with OpenAI Agents SDK and FastMCP framework.

## 🚀 Features

### Complete End-to-End Pipeline
- **Markdown → LaTeX → PDF → Style Optimization → Final PDF**
- Multi-variant style generation with AI quality judge
- Automatic LaTeX error fixing with retry logic
- Iterative quality improvement with feedback loops

### AI-Powered Intelligence
- **OpenAI Agents SDK**: Multiple specialized agents for conversion, compilation, and styling
- **Quality Judge**: LLM-as-a-judge pattern for variant evaluation
- **Smart Defaults**: Cost-aware configuration (fast by default, quality on demand)
- **Parallel Processing**: Generate N style variants simultaneously

### MCP Integration
- **Primary Tools**: `generate_cv_from_markdown`, `compile_and_improve_style`
- **Debug Tools**: Individual phase tools for testing
- **Transport Flexibility**: stdio (Claude Desktop) and HTTP support
- **Resource Serving**: PDFs and LaTeX files via MCP resources

## 📦 Installation

### Prerequisites

- **Python 3.11+**
- **LaTeX distribution** (TeX Live, MiKTeX, or MacTeX)
- **OpenAI API key** (for AI agents)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/francisco-perez-sorrosal/cv-writer-mcp.git
cd cv-writer-mcp

# Install with pixi (recommended)
pixi install

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Verify installation
pixi run check-latex
```

### Install LaTeX

**macOS:**
```bash
brew install --cask mactex
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**Windows:**
Download [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)

## 🎯 Quick Start

### End-to-End CV Generation

```bash
# FAST MODE (Default) - Quick & Cheap (~3-4 LLM calls)
pixi run generate-cv-fast

# QUALITY MODE - 2 variants, judge picks best (~6-8 LLM calls)
pixi run generate-cv-quality

# ITERATIVE MODE - Best quality with feedback loops (~15-25 LLM calls)
pixi run generate-cv-iterative
```

### What Happens in Each Mode?

| Mode | Variants | Iterations | Judge | Cost | Use Case |
|------|----------|------------|-------|------|----------|
| **Fast** | 1 | 1 | No | ~3-4 calls | Testing, quick iterations |
| **Quality** | 2 | 1 | Yes | ~6-8 calls | Production CVs |
| **Iterative** | 2 | 3 | Yes | ~15-25 calls | Highest quality needed |

## 🏗️ Architecture

### Complete Pipeline

```
Phase 1: Markdown → LaTeX
  └─ MD2LaTeXAgent (OpenAI agent)

Phase 2: Initial Compilation (LOOP #1: compile-fix-compile)
  └─ LaTeXExpert
      FOR attempt in 1..max_attempts:
        ├─ CompilationAgent: compile LaTeX
        ├─ If errors: CompilationErrorAgent fixes them
        └─ Retry until success or max attempts

Phase 3: Style Improvement (LOOP #2: multi-variant with judge)
  └─ PDFStyleCoordinator
      FOR iteration in 1..max_iterations:
        ├─ PageCaptureAgent: analyze PDF visually
        ├─ Generate N variants IN PARALLEL
        │   FOR each variant:
        │     ├─ FormattingAgent: generate variant LaTeX
        │     └─ LOOP #1: compile-fix-compile (nested!)
        ├─ StyleQualityAgent: evaluate variants and pick best
        └─ Decision: pass/needs_improvement → stop or continue
```

### Smart Defaults

**Auto-Enables Judge**: When `num_variants >= 2` (needed to pick best)
**Auto-Disables Judge**: When `num_variants = 1` (nothing to compare)
**Cost-Aware**: Default is 1 variant, 1 iteration (fast & cheap)
**Quality-Aware**: Easy to enable quality mode with `--variants 2`

## 📚 Usage

### MCP Server

Start the server for Claude Desktop or other MCP clients:

```bash
# Start with stdio transport (default for Claude Desktop)
export TRANSPORT="stdio"
pixi run serve

# Start with HTTP transport
export TRANSPORT="streamable-http"
pixi run serve
```

### Primary MCP Tools

#### `generate_cv_from_markdown` ⭐

Complete end-to-end pipeline: Markdown → LaTeX → PDF → Style → Final PDF

**Parameters:**
- `markdown_content` (required): Markdown CV content
- `output_filename` (optional): Custom output filename
- `enable_style_improvement` (default: true): Enable style phase
- `max_compile_attempts` (default: 3): Max compilation retries
- `max_style_iterations` (default: 1): Max style iterations
- `num_style_variants` (default: 1): Number of variants per iteration
- `enable_quality_validation` (default: None): Judge enabled if variants >= 2

**Example:**
```json
{
  "markdown_content": "# John Doe\n## Experience...",
  "num_style_variants": 2,
  "max_style_iterations": 3
}
```

#### `compile_and_improve_style` ⭐

Compile existing LaTeX and improve styling: LaTeX → PDF → Style → Final PDF

**Parameters:**
- `tex_filename` (required): Name of .tex file
- `output_filename` (optional): Custom output filename
- `max_compile_attempts` (default: 3): Max compilation retries
- `max_style_iterations` (default: 1): Max style iterations
- `num_style_variants` (default: 1): Number of variants
- `enable_quality_validation` (default: None): Auto-enabled if variants >= 2

### CLI Commands

#### Primary Workflows

```bash
# Fast mode (1 variant, 1 iteration, no judge)
pixi run generate-cv-fast

# Quality mode (2 variants, judge picks best)
pixi run generate-cv-quality

# Iterative mode (3 iterations, 2 variants, judge-driven)
pixi run generate-cv-iterative

# Compile and improve existing LaTeX
pixi run compile-and-improve
```

#### Debug/Test Workflows (Individual Phases)

```bash
# Check LaTeX installation
pixi run check-latex

# Phase 1: Markdown → LaTeX only
pixi run convert-markdown

# Phase 2: LaTeX → PDF only (with error fixing)
pixi run compile-latex

# Phase 3: PDF → Styled LaTeX only
pixi run fix-style
```

#### Custom Commands

```bash
# Custom workflow with specific parameters
python -m cv_writer_mcp generate-cv-from-markdown input/cv.md \
  --output my_cv.pdf \
  --max-style-iter 3 \
  --variants 2 \
  --quality

# Disable style improvement (just convert and compile)
python -m cv_writer_mcp generate-cv-from-markdown input/cv.md \
  --no-enable-style

# Force quality judge on single variant
python -m cv_writer_mcp generate-cv-from-markdown input/cv.md \
  --variants 1 \
  --quality
```

## ⚙️ Configuration

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional
TRANSPORT=stdio                  # "stdio" or "streamable-http"
HOST=localhost
PORT=8000
OUTPUT_DIR=./output
TEMP_DIR=./temp
LATEX_TIMEOUT=180
LOG_LEVEL=INFO
```

## 🧪 Development

### Setup

```bash
# Install development dependencies
pixi install

# Run all checks (format, lint, type-check, test)
pixi run ci
```

### Development Commands

```bash
# Format code
pixi run format

# Lint code
pixi run lint

# Type checking
pixi run type-check

# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov
```

### Project Structure

```
cv-writer-mcp/
├── src/cv_writer_mcp/
│   ├── orchestration/         # End-to-end pipeline orchestrator
│   │   ├── pipeline_orchestrator.py
│   │   └── models.py
│   ├── compilation/            # LaTeX compilation with error fixing
│   │   ├── latex_expert.py
│   │   ├── compiler_agent.py
│   │   └── error_agent.py
│   ├── conversion/             # Markdown to LaTeX conversion
│   │   └── md2latex_agent.py
│   ├── style/                  # PDF style improvement
│   │   ├── pdf_style_coordinator.py
│   │   ├── page_capture_agent.py
│   │   ├── formatting_agent.py
│   │   └── quality_agent.py    # LLM-as-a-judge
│   ├── main.py                 # MCP server and CLI entry point
│   └── models.py               # Shared data models
├── context/                    # LaTeX templates
├── input/                      # Sample input files
├── output/                     # Generated files
└── tests/                      # Test suite
```

## 💡 How It Works

### Multi-Variant Style Improvement

1. **Capture & Analyze**: PageCaptureAgent uses Playwright to capture PDF pages and identify visual issues
2. **Generate Variants**: FormattingAgent creates N different style approaches in parallel
   - Variant 1: Conservative (safe, minimal changes)
   - Variant 2: Aggressive (bold formatting, space optimization)
   - Variant 3+: Balanced approaches
3. **Compile Each Variant**: Each variant goes through compile-fix-compile loop until success
4. **Judge Evaluation**: StyleQualityAgent compares all variants and selects the best
5. **Iterate**: If score is "needs_improvement", repeat with judge feedback

### Quality Criteria

The judge evaluates variants on:
- **Spacing Efficiency** (35%): Compact without crowding
- **Visual Consistency** (25%): Uniform formatting throughout
- **Readability** (25%): Clear hierarchy, no redundancy
- **Layout Quality** (15%): Margins, alignment, balance

## 🔍 Troubleshooting

### LaTeX Not Found

```bash
# Check installation
pixi run check-latex

# Verify LaTeX is in PATH
which pdflatex

# Install LaTeX (see Installation section)
```

### Compilation Errors

The system automatically fixes most LaTeX errors through the compile-fix-compile loop. If issues persist:

1. Check the error logs in console output
2. Review the generated `.tex` file in `./output/`
3. Try increasing `max_compile_attempts`

### Style Improvement Issues

If style improvement fails:

1. Ensure Playwright browsers are installed: `pixi run playwright install`
2. Check that PDF was generated successfully in Phase 2
3. Try with `--variants 1 --no-quality` to disable judge

### Cost Management

To reduce API costs:

```bash
# Use fast mode (default)
pixi run generate-cv-fast

# Disable style improvement entirely
python -m cv_writer_mcp generate-cv-from-markdown input/cv.md --no-enable-style

# Single variant, no judge
python -m cv_writer_mcp generate-cv-from-markdown input/cv.md --variants 1
```

## 📊 Performance

| Configuration | LLM Calls | Time | Quality |
|---------------|-----------|------|---------|
| Fast (default) | ~3-4 | ~30s | Good |
| Quality (2 variants) | ~6-8 | ~60s | Better |
| Iterative (3×2) | ~15-25 | ~2-3min | Best |

*Times are approximate and depend on CV complexity and API latency*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run checks: `pixi run ci`
5. Commit: `git commit -m "Add feature"`
6. Push and submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Francisco Perez-Sorrosal**
- Email: fperezsorrosal@gmail.com
- GitHub: [@francisco-perez-sorrosal](https://github.com/francisco-perez-sorrosal)

## 🙏 Acknowledgments

Built with:
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io)
