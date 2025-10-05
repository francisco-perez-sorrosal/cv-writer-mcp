"""Style package models."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..models import CompletionStatus


class VisualCriticRequest(BaseModel):
    """Request model for visual critic agent."""

    pdf_file_path: str = Field(
        ...,
        description="Path to the PDF file to convert to screenshots for visual critique",
    )
    num_pages: int | None = Field(
        default=None,
        description="Maximum number of pages to analyze (None for all pages)",
    )


class VisualCriticResponse(BaseModel):
    """Response model for visual critic agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the visual critique was successful"
    )
    pages_analyzed: int = Field(
        default=0, description="Number of screenshots analyzed by the visual critic"
    )
    visual_issues: list[str] = Field(
        default_factory=list,
        description="List of all visual formatting issues found across all pages",
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description="High-level improvement suggestions (goals, not implementation details)",
    )
    analysis_summary: str = Field(
        default="",
        description="Overall visual quality assessment and critique summary",
    )
    message: str | None = Field(
        None, description="Status message with analysis details or error information"
    )


class VisualCriticOutput(BaseModel):
    """Structured output from the visual critic agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the visual critique was successful"
    )
    pages_analyzed: int = Field(
        ..., description="Number of screenshots analyzed by the visual critic"
    )
    visual_issues: list[str] = Field(
        ..., description="List of all visual formatting issues found across all pages"
    )
    suggested_fixes: list[str] = Field(
        ...,
        description="High-level improvement suggestions (goals, not implementation details)",
    )
    analysis_summary: str = Field(
        ..., description="Overall visual quality assessment and critique summary"
    )


class FormattingOutput(BaseModel):
    """Structured output from the formatting agent."""

    status: CompletionStatus = Field(
        ...,
        description="Whether the formatting improvements were successfully implemented",
    )
    fixes_applied: list[str] = Field(
        ...,
        description="List of specific formatting improvements that were implemented",
    )
    improved_latex_content: str = Field(
        ...,
        description="The improved LaTeX file content with all formatting improvements applied",
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
        ...,
        description="Confidence score for the analysis and improvements (0.0 to 1.0)",
    )


# ============================================================================
# Multi-Variant Style Improvement Models
# ============================================================================


class SingleVariantEvaluationOutput(BaseModel):
    """Judge evaluation of a single variant (when num_variants=1 but judge is enabled)."""

    score: Literal["pass", "needs_improvement", "fail"] = Field(
        ..., description="Quality score for the variant"
    )
    feedback: str = Field(
        ..., description="Specific feedback for improvements (for next iteration)"
    )
    quality_metrics: dict[str, float] = Field(
        ...,
        description="Detailed quality metrics: spacing, consistency, readability, layout",
    )


class VariantEvaluationOutput(BaseModel):
    """Judge evaluation of multiple variants (when num_variants >= 2)."""

    best_variant_id: int = Field(
        ..., description="ID of the best variant among all evaluated"
    )
    score: Literal["pass", "needs_improvement", "fail"] = Field(
        ..., description="Quality score for the best variant"
    )
    feedback: str = Field(
        ..., description="Specific feedback for improvements (for next iteration)"
    )
    quality_metrics: dict[str, float] = Field(
        ..., description="Quality metrics for the best variant"
    )
    comparison_summary: str = Field(
        ..., description="Explanation of why best_variant was selected"
    )
    all_variant_scores: dict[int, dict[str, float]] = Field(
        ..., description="Detailed scores for all variants evaluated"
    )


class VariantResult(BaseModel):
    """Result of a single variant generation and compilation.

    Note: pdf_path and compilation_time are optional to support no-compilation mode
    (when only LaTeX generation is needed without PDF compilation).
    """

    variant_id: int = Field(..., description="Unique identifier for this variant")
    pdf_path: Path | None = Field(
        None, description="Path to the compiled PDF (None if compilation skipped)"
    )
    tex_path: Path = Field(..., description="Path to the LaTeX source")
    compilation_time: float | None = Field(
        None,
        description="Time taken to compile in seconds (None if compilation skipped)",
    )


class EvaluationResult(BaseModel):
    """Internal result from quality evaluation (used by coordinator)."""

    best_variant_id: int = Field(..., description="ID of the selected best variant")
    best_variant: VariantResult = Field(..., description="The best variant result")
    score: str = Field(..., description="Quality score")
    feedback: str = Field(..., description="Feedback for next iteration")


class StyleDiagnostics(BaseModel):
    """Diagnostics tracking for style improvement process."""

    iterations_completed: int = Field(
        ..., description="Number of style iterations completed"
    )
    variants_per_iteration: int = Field(
        ..., description="Number of variants generated per iteration"
    )
    total_variants_generated: int = Field(
        ..., description="Total number of variants generated across all iterations"
    )
    successful_variants_per_iteration: list[int] = Field(
        ..., description="Number of successfully compiled variants per iteration"
    )
    failed_variants_per_iteration: list[int] = Field(
        ..., description="Number of failed variants per iteration"
    )
    quality_validation_enabled: bool = Field(
        ..., description="Whether quality judge validation was enabled"
    )
    quality_scores: list[str] = Field(
        ..., description="Judge scores per iteration (if validation enabled)"
    )
    best_variant_per_iteration: list[int] = Field(
        ..., description="Best variant ID selected in each iteration"
    )


class StyleIterationResult(BaseModel):
    """Result from multi-iteration, multi-variant style improvement."""

    status: CompletionStatus = Field(
        ..., description="Overall status of style improvement process"
    )
    best_variant_pdf_path: Path | None = Field(
        None, description="Path to the best variant PDF (final result)"
    )
    best_variant_tex_path: Path | None = Field(
        None, description="Path to the best variant LaTeX source (final result)"
    )
    iterations_completed: int = Field(..., description="Number of iterations completed")
    quality_enabled: bool = Field(
        ..., description="Whether quality validation was enabled"
    )
    final_score: str | None = Field(
        None, description="Final quality score from judge (if enabled)"
    )
    diagnostics: StyleDiagnostics = Field(
        ..., description="Detailed diagnostics from the improvement process"
    )
    message: str = Field(..., description="Human-readable status message")
