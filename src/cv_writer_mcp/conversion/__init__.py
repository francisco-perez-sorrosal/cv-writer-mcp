"""Conversion package for document format conversion.

This package contains agents, models, and tools for document conversion:
- MD2LaTeXAgent: Converts Markdown CV content to LaTeX format
- Models: Request/Response models and conversion-specific types

Public API:
    from cv_writer_mcp.conversion import MD2LaTeXAgent
    from cv_writer_mcp.conversion.models import MarkdownToLaTeXRequest, MarkdownToLaTeXResponse
"""

from .md2latex_agent import MD2LaTeXAgent

__all__ = [
    "MD2LaTeXAgent",
]
