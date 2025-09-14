"""Simplified CV conversion functionality."""

from loguru import logger

from .md2latex_agent import MD2LaTeXAgent
from .models import (
    ConversionStatus,
    MarkdownToLaTeXRequest,
    MarkdownToLaTeXResponse,
    ServerConfig,
)


class CVConverter:
    """CV conversion service without job management complexity."""

    def __init__(self, config: ServerConfig):
        """Initialize the CV converter.

        Args:
            config: Server configuration
        """
        self.config = config

        # Initialize markdown to LaTeX agent
        try:
            self.md2latex_agent: MD2LaTeXAgent | None = MD2LaTeXAgent(
                api_key=config.openai_api_key
            )
            logger.info("Markdown to LaTeX agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize markdown to LaTeX agent: {e}")
            self.md2latex_agent = None

    async def convert_markdown_to_latex(
        self, request: MarkdownToLaTeXRequest
    ) -> MarkdownToLaTeXResponse:
        """Convert markdown CV content to LaTeX file.

        Args:
            request: Markdown to LaTeX conversion request

        Returns:
            Conversion response with LaTeX file URL
        """
        if not self.md2latex_agent:
            return MarkdownToLaTeXResponse(
                status=ConversionStatus.FAILED,
                tex_url=None,
                message="Markdown to LaTeX agent not initialized. Please check your API key.",
            )

        try:
            logger.info("Starting markdown to LaTeX conversion")
            return await self.md2latex_agent.convert(request)

        except Exception as e:
            logger.error(f"Unexpected error in markdown to LaTeX conversion: {e}")
            return MarkdownToLaTeXResponse(
                status=ConversionStatus.FAILED,
                tex_url=None,
                message=f"Unexpected error: {str(e)}",
            )
