"""Conversion package models."""

from pydantic import BaseModel, Field, field_validator

from ..models import CompletionStatus


class LaTeXOutput(BaseModel):
    """Structured output for LaTeX conversion."""

    latex_content: str = Field(
        description="The converted LaTeX content ready for insertion into the template"
    )
    conversion_notes: str = Field(
        description="Notes about the conversion process and any important considerations"
    )
    recommendations: str = Field(
        description="Actionable recommendations for the user to improve the effectiveness of the CV"
    )


class MarkdownToLaTeXRequest(BaseModel):
    """Request model for markdown to LaTeX conversion."""

    markdown_content: str = Field(..., description="Markdown content of the CV")
    output_filename: str | None = Field(
        None, description="Custom output filename for .tex file"
    )

    @field_validator("markdown_content")
    @classmethod
    def validate_markdown_content(cls, v: str) -> str:
        """Validate markdown content is not empty."""
        if not v.strip():
            raise ValueError("Markdown content cannot be empty")
        return v.strip()


class MarkdownToLaTeXResponse(BaseModel):
    """Response model for markdown to LaTeX conversion."""

    status: CompletionStatus
    tex_url: str | None = Field(
        None, description="Resource URI to access the generated LaTeX file"
    )
    message: str | None = Field(
        None,
        description="Status message with conversion notes, error details, or other information",
    )
