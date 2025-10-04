"""Models for CV generation pipeline orchestration."""

from pydantic import BaseModel, Field

from ..compilation.models import CompilationDiagnostics
from ..models import CompletionStatus
from ..style.models import StyleDiagnostics


class CVGenerationResult(BaseModel):
    """Complete end-to-end pipeline result with all diagnostics."""

    status: CompletionStatus = Field(
        ..., description="Overall status of the CV generation pipeline"
    )
    final_pdf_url: str | None = Field(
        None, description="Resource URI to access the final PDF"
    )
    final_tex_url: str | None = Field(
        None, description="Resource URI to access the final LaTeX source"
    )

    # Phase diagnostics
    conversion_time: float | None = Field(
        None, description="Time taken for markdown to LaTeX conversion (seconds)"
    )
    compilation_diagnostics: CompilationDiagnostics = Field(
        ..., description="Diagnostics from compilation phase"
    )
    style_diagnostics: StyleDiagnostics | None = Field(
        None, description="Diagnostics from style improvement phase (if enabled)"
    )

    total_time: float = Field(
        ..., description="Total pipeline execution time (seconds)"
    )
    message: str = Field(..., description="Human-readable status message")


class CVGenerationResponse(BaseModel):
    """MCP tool response for CV generation (simplified for API)."""

    status: CompletionStatus = Field(..., description="Overall status")
    pdf_url: str | None = Field(None, description="PDF resource URI")
    tex_url: str | None = Field(None, description="LaTeX resource URI")
    message: str = Field(..., description="Status message")
    diagnostics_summary: str | None = Field(
        None, description="Summary of diagnostics (optional)"
    )
