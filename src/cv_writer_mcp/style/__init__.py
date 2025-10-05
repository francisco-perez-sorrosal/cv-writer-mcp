"""Style package for PDF style analysis and formatting.

This package contains agents, models, and tools for style analysis:
- VisualCriticAgent: Critiques document screenshots and identifies design quality issues
- FormattingAgent: Implements visual formatting improvements based on analysis
- StyleQualityAgent: LLM-as-a-judge for evaluating variant quality
- PDFStyleCoordinator: Coordinates multi-variant style improvement with quality evaluation
- Models: Style-specific types and diagnostics
- Tools: pdf_computer_use_tool for page capture

Public API:
    from cv_writer_mcp.style import VisualCriticAgent, FormattingAgent, PDFStyleCoordinator
    from cv_writer_mcp.style.tools import pdf_computer_use_tool
"""

from .formatting_agent import FormattingAgent
from .pdf_style_coordinator import PDFStyleCoordinator
from .visual_critic_agent import VisualCriticAgent

__all__ = [
    "VisualCriticAgent",
    "FormattingAgent",
    "PDFStyleCoordinator",
]
