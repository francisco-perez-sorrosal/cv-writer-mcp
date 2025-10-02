"""Style package models."""

from pydantic import BaseModel, Field, field_validator

from ..models import CompletionStatus


class PDFAnalysisRequest(BaseModel):
    """Request model for PDF analysis and LaTeX improvement."""

    pdf_file_path: str = Field(..., description="Path to the PDF file to analyze")
    tex_file_path: str = Field(..., description="Path to the LaTeX source file")
    output_filename: str | None = Field(
        None, description="Custom output filename for improved .tex file"
    )

    @field_validator("pdf_file_path", "tex_file_path")
    @classmethod
    def validate_file_paths(cls, v: str) -> str:
        """Validate file paths are not empty."""
        if not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()


class PDFAnalysisResponse(BaseModel):
    """Response model for PDF analysis and LaTeX improvement."""

    status: CompletionStatus
    improved_tex_url: str | None = Field(
        None, description="Resource URI to access the improved LaTeX file"
    )
    message: str | None = Field(
        None, description="Status message with analysis details, improvements made, or error information"
    )


class PageCaptureRequest(BaseModel):
    """Request model for page capture agent."""

    pdf_file_path: str = Field(
        ..., description="Path to the PDF file to capture and analyze"
    )
    num_pages: int | None = Field(
        default=None, description="Maximum number of pages to capture (None for auto-detect)"
    )


class PageCaptureResponse(BaseModel):
    """Response model for page capture agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the page capture and analysis was successful"
    )
    pages_analyzed: int = Field(
        default=0, description="Number of pages that were captured and analyzed"
    )
    visual_issues: list[str] = Field(
        default_factory=list, description="List of all visual formatting issues found across all pages"
    )
    suggested_fixes: list[str] = Field(
        default_factory=list, description="Detailed LaTeX fixes suggested for each identified issue"
    )
    analysis_summary: str = Field(
        default="", description="Overall assessment and summary of the CV formatting analysis"
    )
    message: str | None = Field(
        None, description="Status message with analysis details or error information"
    )


class PageCaptureOutput(BaseModel):
    """Structured output from the page capture agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the page capture and analysis was successful"
    )
    pages_analyzed: int = Field(
        ..., description="Number of pages that were captured and analyzed"
    )
    visual_issues: list[str] = Field(
        ..., description="List of all visual formatting issues found across all pages"
    )
    suggested_fixes: list[str] = Field(
        ..., description="Detailed LaTeX fixes suggested for each identified issue"
    )
    analysis_summary: str = Field(
        ..., description="Overall assessment and summary of the CV formatting analysis"
    )


class FormattingOutput(BaseModel):
    """Structured output from the formatting agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the formatting improvements were successfully implemented"
    )
    fixes_applied: list[str] = Field(
        ..., description="List of specific formatting improvements that were implemented"
    )
    improved_latex_content: str = Field(
        ..., description="The improved LaTeX file content with all formatting improvements applied"
    )
    implementation_notes: str = Field(
        ..., description="Detailed explanation of changes made and implementation notes"
    )


class PDFAnalysisOutput(BaseModel):
    """Structured output from the PDF analysis agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the analysis and improvement process was successful"
    )
    visual_issues_detected: list[str] = Field(
        ..., description="List of visual issues detected in the PDF layout"
    )
    improvements_made: list[str] = Field(
        ..., description="List of improvements applied to the LaTeX code"
    )
    improved_latex_content: str = Field(
        ..., description="The improved LaTeX file content with all fixes applied"
    )
    analysis_notes: str = Field(
        ..., description="Detailed notes about the analysis process and recommendations"
    )
    confidence_score: float = Field(
        ..., description="Confidence score for the analysis and improvements (0.0 to 1.0)"
    )
