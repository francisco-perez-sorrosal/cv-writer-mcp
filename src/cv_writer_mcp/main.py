"""Main entry point for CV Writer MCP Server."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import typer
from dotenv import load_dotenv
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .compilation import LaTeXExpert
from .compilation.models import CompileLaTeXRequest, CompileLaTeXResponse, LaTeXEngine
from .conversion import MD2LaTeXAgent
from .conversion.models import MarkdownToLaTeXRequest, MarkdownToLaTeXResponse
from .logger import LogConfig, LogLevel, configure_logger
from .models import CompletionStatus, HealthStatusResponse, ServerConfig
from .orchestration import CVPipelineOrchestrator
from .orchestration.models import CVGenerationResponse
from .style import PDFStyleCoordinator
from .utils import read_text_file

# Load environment variables
load_dotenv()

# Global variables for MCP server
mcp: FastMCP | None = None
cv_converter: MD2LaTeXAgent | None = None
latex_expert: LaTeXExpert | None = None
style_coordinator: PDFStyleCoordinator | None = None
orchestrator: CVPipelineOrchestrator | None = None
config: ServerConfig | None = None

app = typer.Typer(
    name="cv-writer-mcp",
    help="Converts a markdown CV content to LaTeX and compiles it to PDF",
    rich_markup_mode="rich",
)
console = Console()

# Configure transport and statelessness
trspt: Literal["stdio", "streamable-http"] = "stdio"
stateless_http = False

match os.environ.get("TRANSPORT", trspt):
    case "streamable-http":
        trspt = "streamable-http"
        stateless_http = True
    case _:
        trspt = "stdio"
        stateless_http = False


# ============================================================================
# SHARED HELPER FUNCTIONS (reduce duplication between MCP tools and CLI)
# ============================================================================


def _initialize_pipeline_orchestrator(config: ServerConfig) -> CVPipelineOrchestrator:
    """Initialize all pipeline components and return orchestrator.

    Used by both MCP tools and CLI commands to avoid duplication.

    Args:
        config: Server configuration

    Returns:
        Initialized CVPipelineOrchestrator with all components
    """
    md2latex = MD2LaTeXAgent()
    latex_expert = LaTeXExpert(config=config)
    style_coordinator = PDFStyleCoordinator()
    return CVPipelineOrchestrator(
        md2latex_agent=md2latex,
        latex_expert=latex_expert,
        style_coordinator=style_coordinator,
        config=config,
    )


def _validate_prerequisites(
    check_openai: bool = True,
    check_latex: bool = True,
    config: ServerConfig | None = None,
) -> None:
    """Validate prerequisites for CV generation.

    Args:
        check_openai: Whether to check for OpenAI API key
        check_latex: Whether to check LaTeX installation
        config: Server config (required if check_latex=True)

    Raises:
        ValueError: If prerequisites are not met
    """
    if check_openai and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    if check_latex:
        if not config:
            raise ValueError("Config required for LaTeX installation check")
        latex_expert = LaTeXExpert(config=config)
        if not latex_expert.check_latex_installation():
            raise ValueError("LaTeX is not installed or not accessible")


def create_config(debug: bool = False) -> ServerConfig:
    """Create server configuration from environment variables with optional debug mode.

    Args:
        debug: Whether to enable debug mode

    Returns:
        Configured ServerConfig instance
    """
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))

    # If BASE_URL is explicitly set, use it; otherwise derive from host and port
    base_url = os.getenv("BASE_URL")
    if not base_url:
        base_url = f"http://{host}:{port}"

    # Determine debug and log level settings
    debug_mode = debug or os.getenv("DEBUG", "false").lower() == "true"
    log_level = (
        LogLevel.DEBUG if debug_mode else LogLevel[os.getenv("LOG_LEVEL", "INFO")]
    )

    return ServerConfig(
        host=host,
        port=port,
        base_url=base_url,
        debug=debug_mode,
        log_level=log_level,
        output_dir=Path(os.getenv("OUTPUT_DIR", "./output")),
        temp_dir=Path(os.getenv("TEMP_DIR", "./temp")),
        max_file_size=int(os.getenv("MAX_FILE_SIZE", "10485760")),
        latex_timeout=int(os.getenv("LATEX_TIMEOUT", "180")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        templates_dir=Path(os.getenv("TEMPLATES_DIR", "./context")),
    )


def setup_logging(log_level: LogLevel) -> None:
    """Configure logging for the application.

    Args:
        log_level: Log level to use
    """
    log_config = LogConfig(
        level=log_level,
        log_file=Path("server.log") if log_level != LogLevel.DEBUG else None,
        console_output=True,
    )
    configure_logger(log_config)


def setup_mcp_server(
    server_config: ServerConfig, host: str = "localhost", port: int = 8000
) -> None:
    """Set up the FastMCP server with tools and HTTP endpoints."""
    global mcp, cv_converter, latex_expert, style_coordinator, orchestrator, config

    config = server_config
    cv_converter = MD2LaTeXAgent(api_key=config.openai_api_key)
    latex_expert = LaTeXExpert(config)
    style_coordinator = PDFStyleCoordinator(api_key=config.openai_api_key)
    orchestrator = CVPipelineOrchestrator(
        md2latex_agent=cv_converter,
        latex_expert=latex_expert,
        style_coordinator=style_coordinator,
        config=config,
    )
    mcp = FastMCP("cv-writer-mcp", host=host, port=port)

    # ========================================================================
    # PRIMARY END-TO-END MCP TOOLS
    # ========================================================================

    @mcp.tool(structured_output=True)
    async def generate_cv_from_markdown(
        markdown_content: str = Field(..., description="Markdown CV content"),
        output_filename: str | None = Field(None, description="Custom output filename"),
        enable_style_improvement: bool = Field(
            True, description="Enable style improvement phase"
        ),
        max_compile_attempts: int = Field(
            3, description="Max compilation retry attempts"
        ),
        max_style_iterations: int = Field(
            1, description="Max style improvement iterations"
        ),
        num_style_variants: int = Field(
            1, description="Number of parallel style variants per iteration"
        ),
        enable_quality_validation: bool | None = Field(
            None,
            description="Enable quality judge (None=auto: enabled if num_variants>=2)",
        ),
    ) -> CVGenerationResponse:
        """Complete CV generation: Markdown → LaTeX → PDF → Style → Final PDF.

        This is the PRIMARY TOOL for end-to-end CV generation.

        Smart Defaults (fast & cheap):
        - Single style iteration (max_style_iterations=1)
        - Single variant (num_style_variants=1)
        - Judge disabled by default when num_variants=1 (save cost)
        - Judge auto-enabled when num_variants>=2 (needed to pick best)

        Usage Examples:
        - Fast mode (default): Use defaults → ~3-4 LLM calls
        - Quality mode: Set num_variants=2 → ~6-8 LLM calls (judge auto-enabled)
        - Iterative mode: Set max_style_iterations=3, num_variants=2 → ~15-25 LLM calls

        Args:
            markdown_content: Markdown CV content to convert
            output_filename: Custom output filename (optional)
            enable_style_improvement: Enable style improvement phase (default: True)
            max_compile_attempts: Max compilation attempts (default: 3)
            max_style_iterations: Max style iterations (default: 1 for speed)
            num_style_variants: Number of variants per iteration (default: 1 for cost)
            enable_quality_validation: Enable quality judge (None=auto based on num_variants)

        Returns:
            CVGenerationResponse with status, URLs, and diagnostics
        """
        if not orchestrator:
            return CVGenerationResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                tex_url=None,
                message="Server not initialized",
                diagnostics_summary=None,
            )

        try:
            result = await orchestrator.generate_cv_from_markdown(
                markdown_content=markdown_content,
                output_filename=output_filename,
                enable_style_improvement=enable_style_improvement,
                max_compile_attempts=max_compile_attempts,
                max_style_iterations=max_style_iterations,
                num_style_variants=num_style_variants,
                enable_quality_validation=enable_quality_validation,
            )

            return orchestrator.to_response(result)

        except Exception as e:
            logger.error(f"Error in generate_cv_from_markdown: {e}")
            return CVGenerationResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                tex_url=None,
                message=f"Pipeline failed: {str(e)}",
                diagnostics_summary=None,
            )

    @mcp.tool(structured_output=True)
    async def compile_and_improve_style(
        tex_filename: str = Field(
            ..., description="LaTeX filename to compile and improve"
        ),
        output_filename: str | None = Field(None, description="Custom output filename"),
        max_compile_attempts: int = Field(3, description="Max compilation attempts"),
        max_style_iterations: int = Field(1, description="Max style iterations"),
        num_style_variants: int = Field(1, description="Number of style variants"),
        enable_quality_validation: bool | None = Field(
            None, description="Enable quality judge (None=auto)"
        ),
    ) -> CVGenerationResponse:
        """Compile LaTeX and improve styling: LaTeX → PDF → Style → Final PDF.

        Same smart defaults as generate_cv_from_markdown.
        Judge auto-enabled when num_variants >= 2.

        Args:
            tex_filename: Name of the .tex file to compile
            output_filename: Custom output filename (optional)
            max_compile_attempts: Max compilation attempts (default: 3)
            max_style_iterations: Max style iterations (default: 1)
            num_style_variants: Number of variants (default: 1)
            enable_quality_validation: Enable quality judge (None=auto)

        Returns:
            CVGenerationResponse with status, URLs, and diagnostics
        """
        if not orchestrator:
            return CVGenerationResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                tex_url=None,
                message="Server not initialized",
                diagnostics_summary=None,
            )

        try:
            result = await orchestrator.compile_and_improve_style(
                tex_filename=tex_filename,
                output_filename=output_filename,
                max_compile_attempts=max_compile_attempts,
                max_style_iterations=max_style_iterations,
                num_style_variants=num_style_variants,
                enable_quality_validation=enable_quality_validation,
            )

            return orchestrator.to_response(result)

        except Exception as e:
            logger.error(f"Error in compile_and_improve_style: {e}")
            return CVGenerationResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                tex_url=None,
                message=f"Pipeline failed: {str(e)}",
                diagnostics_summary=None,
            )

    # ========================================================================
    # DEBUG/TEST MCP TOOLS (Individual Phases)
    # ========================================================================

    # MCP Tools
    @mcp.tool(structured_output=True)
    async def markdown_to_latex(
        markdown_content: str, output_filename: str | None = None
    ) -> MarkdownToLaTeXResponse:
        """Convert markdown CV content to LaTeX file using OpenAI agent.

        Args:
            markdown_content: Markdown content of the CV
            output_filename: Custom output filename for .tex file (optional)

        Returns:
            Structured response with conversion status, LaTeX file URL, and metadata
        """
        if not cv_converter:
            return MarkdownToLaTeXResponse(
                status=CompletionStatus.FAILED,
                tex_url=None,
                message="Server not initialized",
            )

        try:
            # Create request
            request = MarkdownToLaTeXRequest(
                markdown_content=markdown_content, output_filename=output_filename
            )

            # Convert markdown to LaTeX
            return await cv_converter.convert(request)

        except Exception as e:
            logger.error(f"Error in markdown_to_latex: {e}")

            return MarkdownToLaTeXResponse(
                status=CompletionStatus.FAILED,
                tex_url=None,
                message=f"Error processing markdown to LaTeX conversion request: {str(e)}",
            )

    @mcp.tool(structured_output=True)
    async def compile_latex_to_pdf(
        tex_filename: str = Field(..., description="Name of the .tex file to compile"),
        output_filename: str = Field(
            "", description="Custom output filename for PDF (optional)"
        ),
        latex_engine: LaTeXEngine = Field(
            LaTeXEngine.PDFLATEX, description="LaTeX engine to use"
        ),
    ) -> CompileLaTeXResponse:
        """Compile LaTeX file to PDF using intelligent agents.

        Args:
            tex_filename: Name of the .tex file to compile
            output_filename: Custom output filename for PDF (optional)
            latex_engine: LaTeX engine to use (currently only pdflatex supported)

        Returns:
            Structured response with compilation status, PDF URL, and metadata
        """
        if not latex_expert:
            return CompileLaTeXResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                message="Server not initialized",
            )

        # Create request object
        request = CompileLaTeXRequest(
            tex_filename=tex_filename,
            output_filename=output_filename,
            latex_engine=latex_engine,
            max_attempts=3,
            user_instructions="",
        )

        # Use the LaTeXExpert's async compilation method
        return await latex_expert.compile_latex_file(request)

    @mcp.tool()
    async def check_latex_installation(
        engine: str = Field(
            LaTeXEngine.PDFLATEX.value, description="LaTeX engine to check"
        )
    ) -> str:
        """Check if LaTeX is installed and accessible.

        Args:
            engine: LaTeX engine to check (currently only pdflatex supported)

        Returns:
            Installation status
        """
        if not latex_expert:
            return "Error: Server not initialized"

        try:
            if engine != LaTeXEngine.PDFLATEX.value:
                return f"Error: Currently only '{LaTeXEngine.PDFLATEX.value}' engine is supported. Requested: '{engine}'"

            latex_engine = LaTeXEngine.PDFLATEX
            is_installed = latex_expert.check_latex_installation(latex_engine)

            if is_installed:
                return f"✅ LaTeX engine '{engine}' is installed and accessible"
            else:
                return f"❌ LaTeX engine '{engine}' is not installed/accessible."

        except Exception as e:
            logger.error(f"Error in check_latex_installation: {e}")
            return f"Error checking LaTeX installation: {str(e)}"

    @mcp.tool()
    async def health_check() -> str:
        """Check the health status of the CV Writer MCP server."""
        health_response = HealthStatusResponse(
            status="healthy",
            service="cv-writer-mcp",
            timestamp=datetime.now().isoformat(),
            version="0.1.0",
        )

        return health_response.model_dump_json(indent=2)

    @mcp.resource("cv-writer://pdf/{filename}", mime_type="application/pdf")
    async def serve_pdf(filename: str) -> bytes:
        """Serve generated PDF files as MCP resource."""
        if not config:
            raise RuntimeError("Server not initialized")

        pdf_path = config.output_dir / filename

        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF {filename} not found")

        return pdf_path.read_bytes()

    @mcp.resource("cv-writer://tex/{filename}", mime_type="text/plain")
    async def serve_tex(filename: str) -> str:
        """Serve generated LaTeX files as MCP resource."""
        if not config:
            raise RuntimeError("Server not initialized")

        tex_path = config.output_dir / filename

        return read_text_file(tex_path, "LaTeX file", ".tex")


### CLI Commands to test the MCP Server functionality ###


@app.command()
def start_mcps(
    debug: bool = typer.Option(False, "--debug", help="Run in debug/development mode"),
    host: str = typer.Option(
        "localhost", "--host", help="Host to bind the MCP Server to"
    ),
    port: int = typer.Option(8000, "--port", help="Port to bind the MCP Server to"),
) -> None:
    """Start the simplified CV Writer MCP Server."""

    # Create and configure server configuration
    config = create_config(debug)

    # Configure logging
    setup_logging(config.log_level)

    # Log startup information
    logger.info("🚀 Starting CV Writer MCP Server")
    logger.info(f"📁 Output directory: {config.output_dir}")
    logger.info(f"🌐 Transport: {trspt}")
    if trspt == "streamable-http":
        logger.info(f"🌐 Server: http://{host}:{port}/mcp")

    # Display startup banner
    banner_text = Text()
    banner_text.append("CV Writer MCP Server", style="bold blue")
    banner_text.append("\n\n", style="default")
    banner_text.append("✨ Convert markdown CV to LaTeX\n", style="green")
    banner_text.append(
        "🔧 MCP Tools: markdown_to_latex, compile_latex_to_pdf, check_latex_installation, health_check\n",
        style="yellow",
    )
    banner_text.append(
        "🌐 MCP Resources: cv-writer://pdf/{filename}, cv-writer://tex/{filename}\n",
        style="cyan",
    )
    console.print(Panel(banner_text, title="Welcome", border_style="blue"))

    # Set up the FastMCP server
    setup_mcp_server(config, host=host, port=port)

    # Check LaTeX installation
    if not latex_expert or not latex_expert.check_latex_installation():
        console.print(
            "[yellow]⚠️  Warning: LaTeX is not installed. Install LaTeX to use this service.[/yellow]"
        )
    else:
        console.print(
            "[green]✅ LaTeX installation verified[/green]",
        )

    # Check OpenAI API key
    if not config.openai_api_key:
        console.print(
            "[yellow]⚠️  Warning: OPENAI_API_KEY not set. Please set your OpenAI API key.[/yellow]"
        )
    else:
        console.print("[green]✅ OpenAI API key found[/green]")

    # Start the server
    try:
        if not mcp:
            console.print("[red]❌ Failed to initialize MCP server[/red]")
            return

        mcp.run(transport=trspt)
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Shutting down CV Writer MCP Server[/yellow]")


@app.command()
def check_latex() -> None:
    """Check LaTeX installation (PDFLATEX engine)."""
    config = create_config()
    latex_compiler = LaTeXExpert(config=config)

    console.print("[blue]Checking LaTeX installation...[/blue]")

    # For now, only check PDFLATEX (keeping scaffolding for future engines)
    engine = LaTeXEngine.PDFLATEX
    is_installed = latex_compiler.check_latex_installation(engine)
    console.print(f"{'✅' if is_installed else '❌'} {engine}")


@app.command()
def convert_markdown(
    markdown_file: str = typer.Argument(
        ..., help="Path to the markdown file to convert"
    ),
    output_file: str = typer.Option(
        "", "--output", "-o", help="Custom output filename for .tex file"
    ),
    template: str = typer.Option(
        "moderncv_template.tex", "--template", "-t", help="LaTeX template to use"
    ),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Convert a markdown CV file to LaTeX from the command line."""

    # Create and configure server configuration
    config = create_config(debug)

    # Configure logging
    setup_logging(config.log_level)

    console.print(f"[blue]Converting markdown file: {markdown_file}[/blue]")
    console.print(f"[blue]Using template: {template}[/blue]")

    # Check if the input file exists
    md_path = Path(markdown_file)
    if not md_path.exists():
        console.print(
            f"[red]❌ Error: Markdown file not found at '{markdown_file}'[/red]"
        )
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not config.openai_api_key:
        console.print(
            "[red]❌ Error: OPENAI_API_KEY environment variable is required[/red]"
        )
        console.print(
            "[yellow]Set it with: export OPENAI_API_KEY='your-api-key-here'[/yellow]"
        )
        raise typer.Exit(1)

    console.print("[green]✅ Input file verified[/green]")
    console.print("[green]✅ OpenAI API key found[/green]")

    # Create converter with specified template
    cv_converter = MD2LaTeXAgent(api_key=config.openai_api_key, template_name=template)

    # Show conversion method
    console.print(
        "[blue]🤖 Using OpenAI agents for markdown to LaTeX conversion[/blue]"
    )

    # Convert the markdown file
    try:
        # Read markdown content
        markdown_content = md_path.read_text()

        # Handle asyncio loop for CLI context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create request object
        request = MarkdownToLaTeXRequest(
            markdown_content=markdown_content,
            output_filename=output_file or "",
        )

        response = loop.run_until_complete(cv_converter.convert(request))

        if response.status.value == "success":
            console.print("[green]✅ Successfully converted markdown to LaTeX[/green]")
            console.print(f"[blue]📄 LaTeX URL: {response.tex_url}[/blue]")
            if response.message:
                console.print(f"[blue]📊 Details: {response.message}[/blue]")
        else:
            console.print(f"[red]❌ Conversion failed: {response.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error during conversion: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command()
def compile_latex(
    tex_file: str = typer.Argument(..., help="Path to the .tex file to compile"),
    output_file: str = typer.Option(
        "", "--output", "-o", help="Custom output filename for PDF"
    ),
    latex_engine: str = typer.Option(
        "pdflatex", "--engine", "-e", help="LaTeX engine to use"
    ),
    max_attempts: int = typer.Option(
        5,
        "--max-attempts",
        "-m",
        help="Maximum number of attempts to compile the LaTeX file",
    ),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Compile a LaTeX file to PDF from the command line."""

    # Create and configure server configuration
    config = create_config(debug)

    # Configure logging
    setup_logging(config.log_level)

    console.print(f"[blue]Compiling LaTeX file: {tex_file}[/blue]")

    # Check if the input file exists and is a .tex file
    tex_path = Path(tex_file)
    if not tex_path.exists() or tex_path.suffix.lower() != ".tex":
        console.print(f"[red]❌ Error: No .tex file found at '{tex_file}'[/red]")
        raise typer.Exit(1)

    # Create LaTeX compiler
    latex_expert = LaTeXExpert(config=config)

    # Check LaTeX installation first
    if not latex_expert.check_latex_installation(LaTeXEngine(latex_engine)):
        console.print("[red]❌ Error: LaTeX is not installed or not accessible[/red]")
        raise typer.Exit(1)

    console.print("[green]✅ LaTeX installation verified[/green]")

    # Show compilation method
    console.print("[blue]🔧 Using intelligent agents for LaTeX compilation[/blue]")

    # Compile the LaTeX file
    try:
        # Handle asyncio loop for CLI context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Create request object
        request = CompileLaTeXRequest(
            tex_filename=tex_path.name,  # Use just the filename, not the full path
            output_filename=output_file
            or "",  # Convert None to empty string for lazy initialization
            latex_engine=LaTeXEngine(latex_engine),
            max_attempts=max_attempts,
            user_instructions="",
        )

        response = loop.run_until_complete(latex_expert.compile_latex_file(request))

        if response.status.value == "success":
            console.print("[green]✅ Successfully compiled LaTeX to PDF[/green]")
            console.print(f"[blue]📄 PDF URL: {response.pdf_url}[/blue]")
            if response.message:
                console.print(f"[blue]📊 Metadata: {response.message}[/blue]")
        else:
            console.print(f"[red]❌ Compilation failed: {response.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error during compilation: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command()
def fix_style(
    pdf_file: str = typer.Option(
        "./output/to_improve_style.pdf",
        "--pdf",
        "-p",
        help="Path to the PDF file to analyze",
    ),
    tex_file: str = typer.Option(
        "./output/to_improve_style.tex",
        "--tex",
        "-t",
        help="Path to the LaTeX source file",
    ),
    output_file: str = typer.Option(
        "improved.tex", "--output", "-o", help="Output filename for improved LaTeX file"
    ),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Analyze PDF layout and improve LaTeX formatting using AI agent with browser automation."""

    # Create and configure server configuration
    config = create_config(debug)

    # Configure logging
    setup_logging(config.log_level)

    console.print(f"[blue]🔍 Analyzing PDF layout: {pdf_file}[/blue]")
    console.print(f"[blue]📝 Source LaTeX: {tex_file}[/blue]")

    # Check if input files exist
    pdf_path = Path(pdf_file)
    tex_path = Path(tex_file)

    if not pdf_path.exists():
        console.print(f"[red]❌ Error: PDF file not found at '{pdf_file}'[/red]")
        raise typer.Exit(1)

    if not tex_path.exists():
        console.print(f"[red]❌ Error: LaTeX file not found at '{tex_file}'[/red]")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]❌ Error: OPENAI_API_KEY environment variable is required[/red]"
        )
        console.print(
            "[yellow]Set it with: export OPENAI_API_KEY='your-api-key-here'[/yellow]"
        )
        raise typer.Exit(1)

    console.print("[green]✅ Input files verified[/green]")
    console.print("[green]✅ OpenAI API key found[/green]")

    # Show analysis method
    console.print(
        "[blue]🤖 Using specialized AI agents for PDF analysis and LaTeX improvement[/blue]"
    )
    console.print(
        "[yellow]📱 A browser window will open to display the PDF for analysis[/yellow]"
    )

    # Run the analysis
    try:
        # Handle asyncio loop for CLI context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Initialize the PDF style coordinator
        coordinator = PDFStyleCoordinator()
        console.print("[green]✅ PDF Style Coordinator initialized[/green]")

        console.print("[blue]🔍 Starting PDF analysis and LaTeX improvement...[/blue]")

        # Call improve_with_variants directly (no compilation mode)
        result = loop.run_until_complete(
            coordinator.improve_with_variants(
                initial_pdf_path=pdf_path,
                initial_tex_path=tex_path,
                latex_expert=None,  # NO COMPILATION
                max_iterations=1,  # Single iteration
                num_variants=1,  # Single variant
                max_compile_attempts=0,  # Not used
                enable_quality_validation=False,  # No judge without compilation
                output_dir=Path("./output"),
            )
        )

        if result.status.value == "success":
            # Handle custom output filename if specified
            if output_file:
                output_filename = (
                    output_file
                    if output_file.endswith(".tex")
                    else f"{output_file}.tex"
                )
                final_path = Path("./output") / output_filename
                final_path.parent.mkdir(parents=True, exist_ok=True)

                if result.best_variant_tex_path:
                    final_path.write_text(
                        result.best_variant_tex_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                    improved_tex_url = f"cv-writer://tex/{output_filename}"
                else:
                    console.print("[red]❌ No LaTeX variant was generated[/red]")
                    raise typer.Exit(1)
            else:
                # Use the default variant path
                if result.best_variant_tex_path:
                    improved_tex_url = (
                        f"cv-writer://tex/{result.best_variant_tex_path.name}"
                    )
                else:
                    console.print("[red]❌ No LaTeX variant was generated[/red]")
                    raise typer.Exit(1)

            console.print("[green]✅ Analysis completed successfully![/green]")
            console.print(f"[blue]📋 Status: {result.message}[/blue]")
            console.print(f"[blue]🔗 Improved LaTeX file: {improved_tex_url}[/blue]")
            console.print("\n[green]📊 The coordinator has:[/green]")
            console.print("   • Captured and analyzed PDF pages visually")
            console.print("   • Identified specific visual formatting issues")
            console.print("   • Implemented targeted LaTeX improvements")
            console.print("   • Generated an improved version of your CV")
            console.print(
                "\n[blue]💡 You can now compile the improved LaTeX file to see the visual enhancements![/blue]"
            )
        else:
            console.print(f"[red]❌ Analysis failed: {result.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error during analysis: {e}[/red]")
        console.print("\n[yellow]🔧 Troubleshooting tips:[/yellow]")
        console.print("   • Ensure Playwright is installed: pixi add playwright")
        console.print("   • Install browser binaries: pixi run playwright install")
        console.print(
            "   • Check that your PDF and LaTeX files exist and are accessible"
        )
        console.print("   • Verify your OpenAI API key is valid")
        raise typer.Exit(1) from e


@app.command(name="generate-cv-from-markdown")
def generate_cv_from_markdown_cli(
    markdown_file: str = typer.Argument(
        ..., help="Path to the markdown file to convert"
    ),
    output: str = typer.Option(
        "", "--output", "-o", help="Custom output filename for final PDF"
    ),
    variants: int = typer.Option(
        1,
        "--variants",
        "-v",
        help="Number of style variants to generate per iteration (default: 1)",
    ),
    max_style_iter: int = typer.Option(
        1,
        "--max-style-iter",
        "-i",
        help="Maximum number of style improvement iterations (default: 1)",
    ),
    max_compile_attempts: int = typer.Option(
        3, "--max-compile", "-c", help="Maximum compilation attempts (default: 3)"
    ),
    no_enable_style: bool = typer.Option(
        False, "--no-enable-style", help="Disable style improvement phase"
    ),
    quality: bool = typer.Option(
        False, "--quality", help="Force enable quality judge even with 1 variant"
    ),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Complete end-to-end CV generation: Markdown → LaTeX → PDF → Style → Final PDF.

    This is the PRIMARY COMMAND for complete CV generation with all phases.

    Usage Examples:
    - Fast mode (default):     pixi run generate-cv-fast
    - Quality mode:            --variants 2 (judge auto-enabled)
    - Iterative mode:          --max-style-iter 3 --variants 2
    """
    # Create config and setup logging
    config = create_config(debug)
    setup_logging(config.log_level)

    console.print("[bold blue]🚀 Starting Complete CV Generation Pipeline[/bold blue]")
    console.print(f"[blue]📄 Input: {markdown_file}[/blue]")

    # Check if markdown file exists and read content
    md_path = Path(markdown_file)
    if not md_path.exists():
        console.print(
            f"[red]❌ Error: Markdown file not found at '{markdown_file}'[/red]"
        )
        raise typer.Exit(1)

    try:
        markdown_content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[red]❌ Error reading markdown file: {e}[/red]")
        raise typer.Exit(1) from e

    # Validate prerequisites using shared helper
    try:
        _validate_prerequisites(check_openai=True, check_latex=True, config=config)
    except ValueError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1) from e

    # Initialize pipeline using shared helper
    console.print("[blue]🔧 Initializing pipeline components...[/blue]")
    orchestrator = _initialize_pipeline_orchestrator(config)
    console.print("[green]✅ All components initialized[/green]")

    # Run the pipeline
    try:
        # Handle asyncio loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Determine quality validation setting
        enable_quality = True if quality else None  # Auto by default

        result = loop.run_until_complete(
            orchestrator.generate_cv_from_markdown(
                markdown_content=markdown_content,
                output_filename=output,
                enable_style_improvement=not no_enable_style,
                max_compile_attempts=max_compile_attempts,
                max_style_iterations=max_style_iter,
                num_style_variants=variants,
                enable_quality_validation=enable_quality,
            )
        )

        if result.status == CompletionStatus.SUCCESS:
            console.print("[green]✅ CV generation completed successfully![/green]")
            if result.final_pdf_url:
                console.print(f"[blue]📄 Final PDF: {result.final_pdf_url}[/blue]")
            if result.final_tex_url:
                console.print(f"[blue]📝 Final LaTeX: {result.final_tex_url}[/blue]")
            console.print(f"[blue]⏱️  Total time: {result.total_time:.2f}s[/blue]")

            # Build diagnostics summary from result (not response)
            parts = []
            if result.conversion_time:
                parts.append(f"Conversion: {result.conversion_time:.2f}s")
            parts.append(
                f"Compilation: {result.compilation_diagnostics.successful_compilations} success, "
                f"{result.compilation_diagnostics.failed_compilations} failed"
            )
            if result.style_diagnostics:
                parts.append(
                    f"Style: {result.style_diagnostics.iterations_completed} iterations, "
                    f"{result.style_diagnostics.total_variants_generated} variants"
                )
            console.print(f"[blue]📊 Diagnostics: {' | '.join(parts)}[/blue]")
        else:
            console.print(f"[red]❌ Pipeline failed: {result.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error during pipeline execution: {e}[/red]")
        raise typer.Exit(1) from e


@app.command(name="compile-and-improve-style")
def compile_and_improve_style_cli(
    tex_file: str = typer.Argument(..., help="LaTeX file to compile and improve"),
    output: str = typer.Option(
        "", "--output", "-o", help="Custom output filename for final PDF"
    ),
    variants: int = typer.Option(
        1, "--variants", "-v", help="Number of style variants (default: 1)"
    ),
    max_style_iter: int = typer.Option(
        1, "--max-style-iter", "-i", help="Max style iterations (default: 1)"
    ),
    max_compile_attempts: int = typer.Option(
        3, "--max-compile", "-c", help="Max compilation attempts (default: 3)"
    ),
    quality: bool = typer.Option(False, "--quality", help="Force enable quality judge"),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Compile LaTeX and improve styling: LaTeX → PDF → Style → Final PDF.

    Takes an existing LaTeX file, compiles it, and improves the styling.
    """
    # Create config and setup logging
    config = create_config(debug)
    setup_logging(config.log_level)

    console.print("[bold blue]🚀 Starting Compile & Improve Pipeline[/bold blue]")
    console.print(f"[blue]📄 Input: {tex_file}[/blue]")

    # Check if tex file exists
    tex_path = Path(tex_file)
    if not tex_path.exists():
        console.print(f"[red]❌ Error: LaTeX file not found at '{tex_file}'[/red]")
        raise typer.Exit(1)

    # Validate prerequisites using shared helper
    try:
        _validate_prerequisites(check_openai=True, check_latex=True, config=config)
    except ValueError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1) from e

    # Initialize pipeline using shared helper
    console.print("[blue]🔧 Initializing pipeline components...[/blue]")
    orchestrator = _initialize_pipeline_orchestrator(config)
    console.print("[green]✅ All components initialized[/green]")

    # Run the pipeline
    try:
        # Handle asyncio loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        enable_quality = True if quality else None

        result = loop.run_until_complete(
            orchestrator.compile_and_improve_style(
                tex_filename=tex_path.name,
                output_filename=output,
                max_compile_attempts=max_compile_attempts,
                max_style_iterations=max_style_iter,
                num_style_variants=variants,
                enable_quality_validation=enable_quality,
            )
        )

        if result.status == CompletionStatus.SUCCESS:
            console.print("[green]✅ Compile & improve completed![/green]")
            if result.final_pdf_url:
                console.print(f"[blue]📄 Final PDF: {result.final_pdf_url}[/blue]")
            console.print(f"[blue]⏱️  Total time: {result.total_time:.2f}s[/blue]")

            # Build diagnostics summary
            parts = []
            parts.append(
                f"Compilation: {result.compilation_diagnostics.successful_compilations} success, "
                f"{result.compilation_diagnostics.failed_compilations} failed"
            )
            if result.style_diagnostics:
                parts.append(
                    f"Style: {result.style_diagnostics.iterations_completed} iterations, "
                    f"{result.style_diagnostics.total_variants_generated} variants"
                )
            console.print(f"[blue]📊 Diagnostics: {' | '.join(parts)}[/blue]")
        else:
            console.print(f"[red]❌ Pipeline failed: {result.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
