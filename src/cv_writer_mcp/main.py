"""Simplified main entry point for CV Writer MCP Server."""

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger
from mcp.server.fastmcp import FastMCP
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .cv_converter import CVConverter
from .latex_compiler import LaTeXCompiler
from .logger import LogConfig, LogLevel, configure_logger
from .models import (
    CompileLaTeXRequest,
    CompileLaTeXResponse,
    LaTeXEngine,
    MarkdownToLaTeXRequest,
    MarkdownToLaTeXResponse,
    ServerConfig,
)

# Load environment variables
load_dotenv()

# Global variables for MCP server
mcp: FastMCP | None = None
cv_converter: CVConverter | None = None
latex_compiler: LaTeXCompiler | None = None
config: ServerConfig | None = None

app = typer.Typer(
    name="cv-writer-mcp",
    help="Converts a markdown CV content to LaTeX and compiles it to PDF",
    rich_markup_mode="rich",
)
console = Console()

# Configure transport and statelessness
trspt = "stdio"
stateless_http = False
match os.environ.get("TRANSPORT", trspt):
    case "streamable-http":
        trspt = "streamable-http"
        stateless_http = True
    case _:
        trspt = "stdio"
        stateless_http = False


def create_config() -> ServerConfig:
    """Create server configuration from environment variables."""
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))

    # If BASE_URL is explicitly set, use it; otherwise derive from host and port
    base_url = os.getenv("BASE_URL")
    if not base_url:
        base_url = f"http://{host}:{port}"

    return ServerConfig(
        host=host,
        port=port,
        base_url=base_url,
        debug=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        output_dir=Path(os.getenv("OUTPUT_DIR", "./output")),
        temp_dir=Path(os.getenv("TEMP_DIR", "./temp")),
        max_file_size=int(os.getenv("MAX_FILE_SIZE", "10485760")),
        latex_timeout=int(os.getenv("LATEX_TIMEOUT", "180")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        templates_dir=Path(os.getenv("TEMPLATES_DIR", "./context")),
    )


def setup_config_with_debug(debug: bool = False) -> ServerConfig:
    """Create and configure server configuration with optional debug mode.

    Args:
        debug: Whether to enable debug mode

    Returns:
        Configured ServerConfig instance
    """
    config = create_config()
    if debug:
        config.debug = True
        config.log_level = "DEBUG"
    return config


def setup_logging(config: ServerConfig, debug: bool = False) -> None:
    """Configure logging for the application.

    Args:
        config: Server configuration
        debug: Whether to run in debug mode
    """
    log_config = LogConfig(
        level=LogLevel(config.log_level),
        log_file=Path("server.log") if not debug else None,
        console_output=True,
    )
    configure_logger(log_config)


def _compile_latex_file(
    tex_filename: str,
    output_filename: str | None = None,
    latex_engine: str = "pdflatex",
    compiler: LaTeXCompiler | None = None,
) -> CompileLaTeXResponse:
    """Common LaTeX compilation logic using intelligent agents.

    Args:
        tex_filename: Name of the .tex file to compile
        output_filename: Custom output filename for PDF (optional)
        latex_engine: LaTeX engine to use (currently only pdflatex supported)
        compiler: LaTeX compiler instance (optional, will use global if not provided)

    Returns:
        Compilation response with PDF URL or error message
    """
    if not compiler:
        from .models import ConversionStatus

        return CompileLaTeXResponse(
            status=ConversionStatus.FAILED,
            pdf_url=None,
            error_message="LaTeX compiler not available",
        )

    try:
        # For now, only support PDFLATEX (keeping scaffolding for future engines)
        if latex_engine != "pdflatex":
            from .models import ConversionStatus

            return CompileLaTeXResponse(
                status=ConversionStatus.FAILED,
                pdf_url=None,
                error_message=f"Currently only 'pdflatex' engine is supported. Requested: '{latex_engine}'",
            )

        engine = LaTeXEngine.PDFLATEX

        # Create request
        request = CompileLaTeXRequest(
            tex_filename=tex_filename,
            output_filename=output_filename,
            latex_engine=engine,
        )

        # Compile LaTeX to PDF using intelligent agents
        # Note: This is a sync function, so we need to handle the async call
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(compiler.compile_from_request(request))

    except Exception as e:
        logger.error(f"Error in LaTeX compilation: {e}")
        from .models import ConversionStatus

        return CompileLaTeXResponse(
            status=ConversionStatus.FAILED,
            pdf_url=None,
            error_message=f"Error processing LaTeX to PDF compilation request: {str(e)}",
        )


def setup_mcp_server(
    server_config: ServerConfig, host: str = "localhost", port: int = 8000
) -> None:
    """Set up the FastMCP server with tools and HTTP endpoints."""
    global mcp, cv_converter, latex_compiler, config

    config = server_config
    cv_converter = CVConverter(config)
    latex_compiler = LaTeXCompiler(timeout=config.latex_timeout, config=config)
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
            from .models import ConversionStatus

            return MarkdownToLaTeXResponse(
                status=ConversionStatus.FAILED,
                tex_url=None,
                error_message="Server not initialized",
            )

        try:
            # Create request
            request = MarkdownToLaTeXRequest(
                markdown_content=markdown_content, output_filename=output_filename
            )

            # Convert markdown to LaTeX
            response = await cv_converter.convert_markdown_to_latex(request)
            return response

        except Exception as e:
            logger.error(f"Error in markdown_to_latex: {e}")
            from .models import ConversionStatus

            return MarkdownToLaTeXResponse(
                status=ConversionStatus.FAILED,
                tex_url=None,
                error_message=f"Error processing markdown to LaTeX conversion request: {str(e)}",
            )

    @mcp.tool(structured_output=True)
    async def compile_latex_to_pdf(
        tex_filename: str,
        output_filename: str | None = None,
        latex_engine: str = "pdflatex",
    ) -> CompileLaTeXResponse:
        """Compile LaTeX file to PDF using intelligent agents.

        Args:
            tex_filename: Name of the .tex file to compile
            output_filename: Custom output filename for PDF (optional)
            latex_engine: LaTeX engine to use (currently only pdflatex supported)

        Returns:
            Structured response with compilation status, PDF URL, and metadata
        """
        if not latex_compiler:
            from .models import ConversionStatus

            return CompileLaTeXResponse(
                status=ConversionStatus.FAILED,
                pdf_url=None,
                error_message="Server not initialized",
            )

        # Use the common compilation logic
        return _compile_latex_file(
            tex_filename=tex_filename,
            output_filename=output_filename,
            latex_engine=latex_engine,
            compiler=latex_compiler,
        )

    @mcp.tool()
    async def check_latex_installation(engine: str = "pdflatex") -> str:
        """Check if LaTeX is installed and accessible.

        Args:
            engine: LaTeX engine to check (currently only pdflatex supported)

        Returns:
            Installation status
        """
        if not latex_compiler:
            return "Error: Server not initialized"

        try:
            # For now, only support PDFLATEX (keeping scaffolding for future engines)
            if engine != "pdflatex":
                return f"Error: Currently only 'pdflatex' engine is supported. Requested: '{engine}'"

            latex_engine = LaTeXEngine.PDFLATEX
            is_installed = latex_compiler.check_latex_installation(latex_engine)

            if is_installed:
                return f"✅ LaTeX engine '{engine}' is installed and accessible"
            else:
                return f"❌ LaTeX engine '{engine}' is not installed or not accessible. Please install LaTeX to use this service."

        except Exception as e:
            logger.error(f"Error in check_latex_installation: {e}")
            return f"Error checking LaTeX installation: {str(e)}"

    @mcp.tool()
    async def health_check() -> str:
        """Check the health status of the CV Writer MCP server."""
        from datetime import datetime

        from .models import HealthStatusResponse

        health_response = HealthStatusResponse(
            status="healthy",
            service="cv-writer-mcp-simplified",
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

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF {filename} not found")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError("File is not a PDF")

        return pdf_path.read_bytes()

    @mcp.resource("cv-writer://tex/{filename}", mime_type="text/plain")
    async def serve_tex(filename: str) -> str:
        """Serve generated LaTeX files as MCP resource."""
        if not config:
            raise RuntimeError("Server not initialized")

        tex_path = config.output_dir / filename

        if not tex_path.exists():
            raise FileNotFoundError(f"LaTeX file {filename} not found")

        if not tex_path.suffix.lower() == ".tex":
            raise ValueError("File is not a LaTeX file")

        return tex_path.read_text(encoding="utf-8")


@app.command()
def start(
    debug: bool = typer.Option(False, "--debug", help="Run in debug/development mode"),
    host: str = typer.Option(
        "localhost", "--host", help="Host to bind the MCP Server to"
    ),
    port: int = typer.Option(8000, "--port", help="Port to bind the MCP Server to"),
) -> None:
    """Start the simplified CV Writer MCP Server."""

    # Create and configure server configuration
    server_config = setup_config_with_debug(debug)

    # Configure logging
    setup_logging(server_config, debug)

    # Log environment information for debugging
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(
        f"Environment variables: TRANSPORT={os.environ.get('TRANSPORT')}, HOST={os.environ.get('HOST')}, PORT={os.environ.get('PORT')}"
    )

    logger.info("🚀 Starting Simplified CV Writer MCP Server")
    logger.info(f"📁 Output directory: {server_config.output_dir}")
    logger.info(f"🌐 Base URL: {server_config.base_url}")

    # Log transport configuration
    logger.info(
        f"Starting CV Writer MCP server with {trspt} transport ({host}:{port}) and stateless_http={stateless_http}..."
    )

    # Additional pre-flight checks
    if trspt == "stdio":
        logger.info(
            "Using stdio transport - suitable for local Claude Desktop integration"
        )
    elif trspt == "streamable-http":
        logger.info(
            f"Using HTTP transport - server will be accessible at http://{host}:{port}/mcp"
        )

    # Display startup banner only for HTTP transport
    # For stdio transport, we must not output anything to stdout except JSON-RPC
    if trspt == "streamable-http":
        banner_text = Text()
        banner_text.append("CV Writer MCP Server", style="bold blue")
        banner_text.append("\n\n", style="default")
        banner_text.append(
            "✨ Convert markdown CV content to LaTeX using OpenAI\n", style="green"
        )
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
    setup_mcp_server(server_config, host=host, port=port)

    # Check LaTeX installation
    if not latex_compiler or not latex_compiler.check_latex_installation():
        logger.warning("⚠️  LaTeX is not installed. PDF compilation will fail.")
        if trspt == "streamable-http":
            console.print(
                "[yellow]⚠️  Warning: LaTeX is not installed. Please install LaTeX to use this service.[/yellow]"
            )
    else:
        logger.info("✅ LaTeX installation verified")
        if trspt == "streamable-http":
            console.print("[green]✅ LaTeX installation verified[/green]")

    # Check OpenAI API key
    if not server_config.openai_api_key:
        logger.warning(
            "⚠️  OpenAI API key not found. Markdown to LaTeX conversion will fail."
        )
        if trspt == "streamable-http":
            console.print(
                "[yellow]⚠️  Warning: OPENAI_API_KEY not set. Please set your OpenAI API key.[/yellow]"
            )
    else:
        logger.info("✅ OpenAI API key found")
        if trspt == "streamable-http":
            console.print("[green]✅ OpenAI API key found[/green]")

    # Start the server
    try:
        if not mcp:
            logger.error("Failed to initialize MCP server")
            return

        if trspt == "streamable-http":
            mcp.run(transport="streamable-http")
        else:
            mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("👋 Shutting down CV Writer FastMCP Server")
        if trspt == "streamable-http":
            console.print(
                "\n[yellow]👋 Shutting down CV Writer FastMCP Server[/yellow]"
            )


@app.command()
def check_latex() -> None:
    """Check LaTeX installation (PDFLATEX engine)."""
    config = setup_config_with_debug()
    latex_compiler = LaTeXCompiler(timeout=config.latex_timeout, config=config)

    console.print("[blue]Checking LaTeX installation...[/blue]")

    # For now, only check PDFLATEX (keeping scaffolding for future engines)
    engine = LaTeXEngine.PDFLATEX
    is_installed = latex_compiler.check_latex_installation(engine)
    status = "✅" if is_installed else "❌"
    console.print(f"{status} {engine}")


@app.command()
def compile_latex(
    tex_file: str = typer.Argument(..., help="Path to the .tex file to compile"),
    output_file: str = typer.Option(
        None, "--output", "-o", help="Custom output filename for PDF"
    ),
    latex_engine: str = typer.Option(
        "pdflatex", "--engine", "-e", help="LaTeX engine to use"
    ),
    debug: bool = typer.Option(False, "--debug", help="Run in debug mode"),
) -> None:
    """Compile a LaTeX file to PDF from the command line."""

    # Create and configure server configuration
    config = setup_config_with_debug(debug)

    # Configure logging
    setup_logging(config, debug)

    console.print(f"[blue]Compiling LaTeX file: {tex_file}[/blue]")

    # Check if the input file exists
    tex_path = Path(tex_file)
    if not tex_path.exists():
        console.print(f"[red]❌ Error: LaTeX file '{tex_file}' not found[/red]")
        raise typer.Exit(1)

    if not tex_path.suffix.lower() == ".tex":
        console.print(
            f"[red]❌ Error: File '{tex_file}' is not a LaTeX file (.tex)[/red]"
        )
        raise typer.Exit(1)

    # Create LaTeX compiler
    latex_compiler = LaTeXCompiler(timeout=config.latex_timeout, config=config)

    # Check LaTeX installation first
    if not latex_compiler.check_latex_installation():
        console.print("[red]❌ Error: LaTeX is not installed or not accessible[/red]")
        console.print("[yellow]Please install LaTeX to use this service.[/yellow]")
        raise typer.Exit(1)

    console.print("[green]✅ LaTeX installation verified[/green]")

    # Show compilation method
    console.print("[blue]🔧 Using intelligent agents for LaTeX compilation[/blue]")

    # Compile the LaTeX file
    try:
        response = _compile_latex_file(
            tex_filename=tex_path.name,  # Use just the filename, not the full path
            output_filename=output_file,
            latex_engine=latex_engine,
            compiler=latex_compiler,
        )

        if response.status.value == "success":
            console.print("[green]✅ Successfully compiled LaTeX to PDF[/green]")
            console.print(f"[blue]📄 PDF URL: {response.pdf_url}[/blue]")
            if response.metadata:
                console.print(f"[blue]📊 Metadata: {response.metadata}[/blue]")
        else:
            console.print(f"[red]❌ Compilation failed: {response.error_message}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error during compilation: {str(e)}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
