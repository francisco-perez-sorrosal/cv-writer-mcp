"""Style package for PDF style analysis and formatting.

This package contains agents, models, and tools for style analysis:
- PageCaptureAgent: Captures PDF pages and analyzes visual formatting issues
- FormattingAgent: Implements visual formatting improvements based on analysis
- StyleQualityAgent: LLM-as-a-judge for evaluating variant quality
- PDFStyleCoordinator: Coordinates multi-variant style improvement with quality evaluation
- Models: Style-specific types and diagnostics
- Tools: pdf_computer_use_tool for page capture

Public API:
    from cv_writer_mcp.style import PageCaptureAgent, FormattingAgent, PDFStyleCoordinator
    from cv_writer_mcp.style.tools import pdf_computer_use_tool
"""

from .formatting_agent import FormattingAgent
from .page_capture_agent import PageCaptureAgent
from .pdf_style_coordinator import PDFStyleCoordinator

__all__ = [
    "PageCaptureAgent",
    "FormattingAgent",
    "PDFStyleCoordinator",
]
