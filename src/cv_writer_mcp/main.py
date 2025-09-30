"""Main entry point for CV Writer MCP Server."""

import asyncio
import os
import sys
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

from .md2latex_agent import MD2LaTeXAgent
from .latex_expert import LaTeXExpert
from .pdf_style_coordinator import PDFStyleCoordinator
from .utils import read_text_file
from .logger import LogConfig, LogLevel, configure_logger
from .models import (
    CompileLaTeXRequest,
    CompileLaTeXResponse,
    CompletionStatus,
    HealthStatusResponse,
    LaTeXEngine,
    MarkdownToLaTeXRequest,
    MarkdownToLaTeXResponse,
    PDFAnalysisRequest,
    PDFAnalysisResponse,
    ServerConfig,
)

# Load environment variables
load_dotenv()

# Global variables for MCP server
mcp: FastMCP | None = None
cv_converter: MD2LaTeXAgent | None = None
latex_expert: LaTeXExpert | None = None
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
    global mcp, cv_converter, latex_expert, config

    config = server_config
    cv_converter = MD2LaTeXAgent(api_key=config.openai_api_key)
    latex_expert = LaTeXExpert(config)
    mcp = FastMCP("cv-writer-mcp", host=host, port=port)

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
        output_filename: str = Field("", description="Custom output filename for PDF (optional)"
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

    # Log environment information for debugging
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"Environment variables: TRANSPORT={os.environ.get('TRANSPORT')}, HOST={os.environ.get('HOST')}, PORT={os.environ.get('PORT')}"
    )
    logger.info("🚀 Starting Simplified CV Writer MCP Server")
    logger.info(f"📁 Output directory: {config.output_dir}")
    logger.info(f"🌐 Base URL: {config.base_url}")

    if trspt == "stdio":
        logger.info("Using stdio transport")
    elif trspt == "streamable-http":
        logger.info(f"Using HTTP transport - server: http://{host}:{port}/mcp")

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
    markdown_file: str = typer.Argument(..., help="Path to the markdown file to convert"),
    output_file: str = typer.Option("", "--output", "-o", help="Custom output filename for .tex file"),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Convert a markdown CV file to LaTeX from the command line."""

    # Create and configure server configuration
    config = create_config(debug)

    # Configure logging
    setup_logging(config.log_level)

    console.print(f"[blue]Converting markdown file: {markdown_file}[/blue]")

    # Check if the input file exists
    md_path = Path(markdown_file)
    if not md_path.exists():
        console.print(f"[red]❌ Error: Markdown file not found at '{markdown_file}'[/red]")
        raise typer.Exit(1)

    # Check for OpenAI API key
    if not config.openai_api_key:
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable is required[/red]")
        console.print("[yellow]Set it with: export OPENAI_API_KEY='your-api-key-here'[/yellow]")
        raise typer.Exit(1)

    console.print("[green]✅ Input file verified[/green]")
    console.print("[green]✅ OpenAI API key found[/green]")

    # Create converter
    cv_converter = MD2LaTeXAgent(api_key=config.openai_api_key)

    # Show conversion method
    console.print("[blue]🤖 Using OpenAI agents for markdown to LaTeX conversion[/blue]")

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
    output_file: str = typer.Option("", "--output", "-o", help="Custom output filename for PDF"),
    latex_engine: str = typer.Option(
        "pdflatex", "--engine", "-e", help="LaTeX engine to use"
    ),
    max_attempts: int = typer.Option(5, "--max-attempts", "-m", help="Maximum number of attempts to compile the LaTeX file"),
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
        "--pdf", "-p", 
        help="Path to the PDF file to analyze"
    ),
    tex_file: str = typer.Option(
        "./output/to_improve_style.tex", 
        "--tex", "-t", 
        help="Path to the LaTeX source file"
    ),
    output_file: str = typer.Option(
        "improved.tex", 
        "--output", "-o", 
        help="Output filename for improved LaTeX file"
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
        console.print("[red]❌ Error: OPENAI_API_KEY environment variable is required[/red]")
        console.print("[yellow]Set it with: export OPENAI_API_KEY='your-api-key-here'[/yellow]")
        raise typer.Exit(1)

    console.print("[green]✅ Input files verified[/green]")
    console.print("[green]✅ OpenAI API key found[/green]")

    # Show analysis method
    console.print("[blue]🤖 Using specialized AI agents for PDF analysis and LaTeX improvement[/blue]")
    console.print("[yellow]📱 A browser window will open to display the PDF for analysis[/yellow]")

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

        # Create analysis request
        request = PDFAnalysisRequest(
            pdf_file_path=str(pdf_path.absolute()),
            tex_file_path=str(tex_path.absolute()),
            output_filename=output_file
        )

        console.print("[blue]🔍 Starting PDF analysis and LaTeX improvement...[/blue]")
        
        # Run the analysis
        response = loop.run_until_complete(coordinator.analyze_and_improve(request))

        if response.status.value == "success":
            console.print("[green]✅ Analysis completed successfully![/green]")
            console.print(f"[blue]📋 Status: {response.message}[/blue]")
            console.print(f"[blue]🔗 Improved LaTeX file: {response.improved_tex_url}[/blue]")
            console.print("\n[green]📊 The coordinator has:[/green]")
            console.print("   • Captured and analyzed PDF pages visually")
            console.print("   • Identified specific visual formatting issues")
            console.print("   • Implemented targeted LaTeX improvements")
            console.print("   • Generated an improved version of your CV")
            console.print("\n[blue]💡 You can now compile the improved LaTeX file to see the visual enhancements![/blue]")
        else:
            console.print(f"[red]❌ Analysis failed: {response.message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error during analysis: {e}[/red]")
        console.print("\n[yellow]🔧 Troubleshooting tips:[/yellow]")
        console.print("   • Ensure Playwright is installed: pixi add playwright")
        console.print("   • Install browser binaries: pixi run playwright install")
        console.print("   • Check that your PDF and LaTeX files exist and are accessible")
        console.print("   • Verify your OpenAI API key is valid")
        raise typer.Exit(1) from e




if __name__ == "__main__":
    app()
