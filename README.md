# CV Writer MCP Server

An AI-powered Model Context Protocol (MCP) server that generates professional CVs from markdown with intelligent style optimization. Built with OpenAI Agents SDK and FastMCP framework.

## 🚀 Features

### Complete End-to-End Pipeline
- **Markdown → LaTeX → PDF → Style Optimization → Final PDF**
- Multi-variant style generation with AI quality judge
- Automatic LaTeX error fixing with retry logic
- Iterative quality improvement with feedback loops
- **Organized file naming**: `iter1_var2.tex`, `iter1_var2_refined.pdf` (iteration 1, variant 2, refined)

### AI-Powered Intelligence
- **OpenAI Agents SDK**: Multiple specialized agents for conversion, compilation, and styling
- **Quality Judge**: LLM-as-a-judge pattern for variant evaluation with scientific scoring
- **Smart Defaults**: Cost-aware configuration (fast by default, quality on demand)
- **True Parallel Processing**: Each variant progresses through ALL steps independently (generation → compilation → validation → refinement)
- **Enhanced Logging**: Clear file identification in judge comparisons

### MCP Integration
- **Primary Tools**: `md_to_latex`, `compile_and_improve_style`
- **Debug Tools**: Individual phase tools for testing
- **Transport Flexibility**: stdio (Claude Desktop) and HTTP support
- **Resource Serving**: PDFs and LaTeX files via MCP resources

### MCPB Bundle Support
- **Portable Distribution**: Bundle server + dependencies into single .mcpb file
- **Easy Installation**: Drag-and-drop installation in Claude Desktop
- **Self-Contained**: All Python dependencies bundled in lib/ directory
- **Automated Build**: Makefile and pixi tasks with `uv` for fast bundling

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

## 📦 MCPB Bundle Installation (Recommended)

For the easiest installation experience:

1. Download `cv-writer-mcp.mcpb` from releases
2. Open Claude Desktop settings
3. Go to "MCP Servers" tab
4. Click "Install from file"
5. Select `cv-writer-mcp.mcpb`
6. Set `OPENAI_API_KEY` in environment variables

That's it! The server is ready to use.

### Building Your Own Bundle

Requires `uv` to be installed (ultra-fast Python package manager):

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build complete MCPB bundle
make build-mcpb

# Or use individual steps
pixi run python-bundle       # Build wheel with uv
pixi run update-mcpb-deps    # Export dependencies with uv
pixi run mcp-bundle          # Install to lib/ with uv
pixi run pack                # Create .mcpb file
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

Phase 3: Style Improvement (LOOP #2: true parallel processing)
  └─ PDFStyleCoordinator
      FOR iteration in 1..max_iterations:
        ├─ Step 1: Capture & Analyze PDF (VisualCriticAgent)
        ├─ Step 2: Generate N variants IN PARALLEL
        │   FOR each variant (parallel execution):
        │     ├─ FormattingAgent: translate critiques → LaTeX fixes
        │     ├─ LOOP #1: compile-fix-compile (nested!)
        │     ├─ Visual Validation: detect critical regressions
        │     ├─ Refinement: fix critical issues (Branch Judge)
        │     └─ Final variant ready with validation metadata
        ├─ Step 3: Main Quality Judge evaluates all completed variants
        └─ Decision: pass/needs_improvement/fail → stop or continue
```

### Smart Defaults

**Auto-Enables Judge**: When `num_variants >= 2` (needed to pick best)
**Auto-Disables Judge**: When `num_variants = 1` (nothing to compare)
**Cost-Aware**: Default is 1 variant, 1 iteration (fast & cheap)
**Quality-Aware**: Easy to enable quality mode with `--variants 2`

### Key Architectural Improvements

**🚀 True Parallel Processing**: Each variant now progresses through its complete lifecycle independently:
- Generation → Compilation → Validation → Refinement (if needed)
- No sequential bottlenecks between processing phases
- Maximum resource utilization and speed

**📁 Clear File Naming**: Organized, descriptive naming convention:
- `iter1_var2.tex` instead of `i1v2.tex`
- `iter1_var2_refined.pdf` instead of `i1v2r.pdf`
- Easy to understand file relationships and versions

**🔧 Enhanced Error Handling**: Robust pipeline that continues processing even if individual variants fail:
- Graceful handling of compilation failures
- Null-safe PDF path processing
- Individual variant failures don't stop the entire pipeline

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

#### `md_to_latex` ⭐

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
│   │   ├── error_agent.py
│   │   ├── tools.py
│   │   ├── models.py
│   │   └── configs/
│   │       ├── compiler_agent.yaml
│   │       └── error_agent.yaml
│   ├── conversion/             # Markdown to LaTeX conversion
│   │   ├── md2latex_agent.py
│   │   ├── models.py
│   │   └── configs/
│   │       └── md2latex_agent.yaml
│   ├── style/                  # PDF style improvement
│   │   ├── pdf_style_coordinator.py
│   │   ├── visual_critic_agent.py    # Design quality critic
│   │   ├── formatting_agent.py       # LaTeX implementation
│   │   ├── quality_agent.py          # LLM-as-a-judge
│   │   ├── pdf_computer.py           # Screenshot capture
│   │   ├── tools.py
│   │   ├── models.py
│   │   └── configs/
│   │       ├── formatting_agent.yaml
│   │       ├── quality_agent.yaml
│   │       └── visual_critic_agent.yaml
│   ├── main.py                 # MCP server and CLI entry point
│   ├── models.py               # Shared data models
│   ├── utils.py                # Utility functions
│   └── logger.py               # Logging configuration
├── context/                    # LaTeX templates
├── input/                      # Sample input files
├── output/                     # Generated files
└── tests/                      # Test suite
```

## 💡 How It Works

### Multi-Variant Style Improvement

1. **Screenshot Capture**: Utility function uses Playwright to convert PDF pages to PNG images
2. **Visual Critique**: VisualCriticAgent analyzes screenshots and identifies design quality issues
   - Evaluates spacing, consistency, readability, layout
   - Describes problems in design language (not code)
   - Suggests WHAT should improve (goals, not implementation)
3. **Generate Variants in Parallel**: Each variant progresses through its complete lifecycle independently:
   - **Variant 1**: Conservative approach (safe, minimal changes)
   - **Variant 2**: Aggressive approach (bold formatting, space optimization)
   - **Variant 3+**: Balanced approaches
4. **Parallel Processing Pipeline**: Each variant simultaneously executes:
   - FormattingAgent: translates critiques → LaTeX fixes
   - Compilation: compile-fix-compile loop until success
   - Visual Validation: detect critical regressions
   - Refinement: fix critical issues using Branch Judge (if needed)
5. **Main Judge Evaluation**: StyleQualityAgent compares all completed variants and selects the best
6. **Iterate**: If score is "needs_improvement", repeat with judge feedback

### Quality Criteria & Scoring

The judge evaluates variants using a scientific scoring methodology:

#### Four Quality Dimensions (Weighted)
- **Design Coherence (30%)**: Unified, intentional design system
- **Spacing Efficiency (25%)**: Effective vertical space utilization  
- **Visual Consistency (25%)**: Uniform formatting across similar elements
- **Readability (20%)**: Easy information scanning and navigation

#### Quality Thresholds
- **"pass"**: Overall score ≥ 0.75 AND all metrics ≥ 0.65
- **"needs_improvement"**: 0.55 ≤ score < 0.75 OR any metric < 0.65 but ≥ 0.45
- **"fail"**: Score < 0.55 OR any metric < 0.45

#### Iteration Control
- **Early Termination**: System stops when judge returns "pass" score
- **Maximum Iterations**: Respects `max_iterations` parameter
- **Judge Feedback**: Each iteration uses feedback from previous evaluation

### File Naming System

The system uses clear, descriptive naming conventions:

```
Base variants:     iter1_var1.tex, iter1_var2.pdf           (iteration 1, variant 1/2)
Refined variants:  iter1_var1_refined.tex, iter1_var2_refined.pdf  (iteration 1, variant 1/2, refined)
Backup files:      iter1_var1_backup_20251006_145832.tex    (organized backups with timestamps)
```

**Benefits of New Naming:**
- **Clear Structure**: Easy to understand iteration and variant relationships
- **Linear Progression**: Files are organized in chronological order
- **Version Tracking**: Refined versions clearly distinguished from base variants
- **Backup Organization**: Timestamped backups with descriptive prefixes

### Enhanced Judge Logging

The system provides clear visibility into judge decisions:

```
⚖️  MAIN JUDGE: Comparing 2 variants (Iteration 1)
📄 Original PDF: schwab_cv_iterative.pdf
📊 Variants to compare:
  📄 Variant 1 (original): iter1_var1.pdf
  📄 Variant 2 (refined): iter1_var2_refined.pdf
──────────────────────────────────────────────────────────────────────

🏆 MAIN JUDGE RESULT: Selected Variant 2 (refined)
📄 Winning file: iter1_var2_refined.pdf
📊 Score: pass
──────────────────────────────────────────────────────────────────────
```

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

### Iteration Stopping Early

If iterations stop before reaching `max_iterations`:

1. **Check Judge Scores**: Look for "✅ Quality criteria met! Stopping iterations."
2. **Judge is Too Lenient**: The quality judge may be giving "pass" scores too easily
3. **Current Issue**: Judge returns "pass" even when issues remain, causing early termination
4. **Workaround**: Use `--variants 1` to disable judge, or modify quality thresholds

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

### Parallel Processing Benefits

The new parallel architecture provides significant performance improvements:

| Configuration | LLM Calls | Time | Quality | Parallel Benefits |
|---------------|-----------|------|---------|-------------------|
| Fast (default) | ~3-4 | ~30s | Good | Single variant processing |
| Quality (2 variants) | ~6-8 | ~45s | Better | **~25% faster** - variants process independently |
| Iterative (3×2) | ~15-25 | ~90s | Best | **~50% faster** - true parallel pipeline |

### Speed Improvements

**Before (Sequential)**: Variants → Wait for all → Validation → Refinement → Judge
**After (Parallel)**: Each variant: Generation → Compilation → Validation → Refinement → Ready

**Key Performance Gains:**
- **Parallel Validation**: No waiting for all variants to complete
- **Independent Refinement**: Each variant refines independently
- **Faster Iterations**: Reduced overall pipeline latency
- **Better Resource Utilization**: CPU and I/O operations happen concurrently

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