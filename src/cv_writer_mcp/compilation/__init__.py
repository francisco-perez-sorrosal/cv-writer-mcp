"""Compilation package for LaTeX to PDF conversion.

This package contains all compilation-related agents, models, and tools:
- CompilationAgent: Handles LaTeX compilation using AI agents
- CompilationErrorAgent: Analyzes and fixes LaTeX compilation errors
- LaTeXExpert: Orchestrates compilation with intelligent error fixing
- Models: Request/Response models and compilation-specific types
- Tools: latex2pdf_tool for compilation

Public API:
    from cv_writer_mcp.compilation import LaTeXExpert, CompilationAgent, CompilationErrorAgent
    from cv_writer_mcp.compilation.models import CompileLaTeXRequest, CompileLaTeXResponse
    from cv_writer_mcp.compilation.tools import latex2pdf_tool
"""

from .compiler_agent import CompilationAgent
from .error_agent import CompilationErrorAgent
from .latex_expert import LaTeXExpert

__all__ = [
    "CompilationAgent",
    "CompilationErrorAgent",
    "LaTeXExpert",
]
