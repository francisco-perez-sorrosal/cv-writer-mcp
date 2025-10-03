"""Compilation package models."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..models import CompletionStatus


class LaTeXEngine(str, Enum):
    """LaTeX engine enumeration."""

    PDFLATEX = "pdflatex"
    # TODO Add other engines here if necessary


class CompileLaTeXRequest(BaseModel):
    """Request model for LaTeX to PDF compilation."""

    tex_filename: str = Field(..., description="Name of the .tex file to compile")
    output_filename: str = Field(
        default_factory=lambda: "", description="Custom output filename for PDF"
    )
    latex_engine: LaTeXEngine = Field(
        LaTeXEngine.PDFLATEX, description="LaTeX engine to use"
    )
    max_attempts: int = Field(3, description="Maximum number of compilation attempts")
    user_instructions: str = Field(
        "", description="Optional additional instructions for the agents"
    )

    @field_validator("tex_filename")
    @classmethod
    def validate_tex_filename(cls, v: str) -> str:
        """Validate tex filename is not empty and has .tex extension."""
        if not v.strip():
            raise ValueError("TeX filename cannot be empty")
        v = v.strip()
        if not v.endswith(".tex"):
            v += ".tex"
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set default output filename if not provided."""
        if not self.output_filename:
            # Generate output filename from tex filename
            self.output_filename = f"{self.tex_filename.replace('.tex', '')}.pdf"

        # Ensure output filename has .pdf extension
        if not self.output_filename.endswith(".pdf"):
            self.output_filename += ".pdf"


class CompileLaTeXResponse(BaseModel):
    """Response model for LaTeX to PDF compilation."""

    status: CompletionStatus
    pdf_url: str | None = Field(
        None, description="Resource URI to access the generated PDF"
    )
    message: str | None = Field(
        None, description="Status message with compilation details, error information, or other details"
    )


class OrchestrationResult(BaseModel):
    """Result of LaTeX compilation."""

    status: CompletionStatus
    output_path: Path | None = None
    log_output: str = ""
    message: str | None = None
    compilation_time: float = 0.0
    errors_found: list[str] | None = None
    exit_code: int = 0

    def __str__(self) -> str:
        lines = [
            "LaTeX Compilation Result:",
            f"  Status: {self.status.value}",
            f"  Compilation Time: {self.compilation_time:.2f} seconds",
            f"  Exit Code: {self.exit_code}",
            f"  Message: {self.message}",
            f"  Output Path: {self.output_path}",
        ]
        if self.errors_found:
            lines.append(f"  Errors Found: {len(self.errors_found)}")
            for i, error in enumerate(self.errors_found[:3], 1):
                error_preview = error[:100] + "..." if len(error) > 100 else error
                lines.append(f"    {i}. {error_preview}")
            if len(self.errors_found) > 3:
                lines.append(f"    ... and {len(self.errors_found) - 3} more")
        if self.log_output:
            log_preview = (
                self.log_output[:200] + "..."
                if len(self.log_output) > 200
                else self.log_output
            )
            lines.append(f"  Log Output: {log_preview}")
        return "\n".join(lines)


class CompilerAgentOutput(BaseModel):
    """Structured output from the LaTeX compilation agent."""

    status: CompletionStatus = Field(
        ..., description='Compilation status: "success" (exit code 0) or "failure" (exit code > 0)'
    )
    compilation_time: float = Field(
        ..., description="Total duration in seconds"
    )
    compilation_summary: str = Field(
        ..., description="What happened during compilation (general outcome, warnings if present)"
    )
    errors_found: list[str] | None = Field(
        None, description="List of error messages from .log if failed, null if successful"
    )
    output_path: str | None = Field(
        None, description="Full PDF path if successful, null if failed"
    )

    def __str__(self) -> str:
        lines = [
            "Compilation Agent Output:",
            f"  Status: {self.status.value}",
            f"  Compilation Time: {self.compilation_time:.2f} seconds",
            f"  Summary: {self.compilation_summary}",
        ]
        if self.output_path:
            lines.append(f"  Output Path: {self.output_path}")
        if self.errors_found:
            lines.append(f"  Errors Found: {len(self.errors_found)}")
            for i, error in enumerate(self.errors_found[:3], 1):
                error_preview = error[:100] + "..." if len(error) > 100 else error
                lines.append(f"    {i}. {error_preview}")
            if len(self.errors_found) > 3:
                lines.append(f"    ... and {len(self.errors_found) - 3} more")
        return "\n".join(lines)


class ErrorFix(BaseModel):
    """Represents a specific error fix applied to LaTeX code."""

    error_type: str = Field(..., description="Type of error that was fixed")
    error_description: str = Field(..., description="Description of the error")
    fix_applied: str = Field(..., description="The fix that was applied")
    line_number: int | None = Field(
        None, description="Line number where fix was applied (if applicable)"
    )
    confidence: float = Field(
        1.0, description="Confidence level of the fix (0.0 to 1.0)"
    )

    def __str__(self) -> str:
        lines = [
            "Error Fix:",
            f"  Type: {self.error_type}",
            f"  Description: {self.error_description}",
            f"  Fix Applied: {self.fix_applied}",
            f"  Line Number: {self.line_number}",
            f"  Confidence: {self.confidence:.2f}",
        ]
        return "\n".join(lines)


class CompilationErrorOutput(BaseModel):
    """Structured output from the compilation error agent."""

    status: CompletionStatus = Field(
        ..., description="Whether the error fixing process was successful"
    )
    fixes_applied: list[ErrorFix] = Field(
        ..., description="List of fixes that were applied"
    )
    file_modified: bool = Field(
        ..., description="Whether the LaTeX file was actually modified"
    )
    total_fixes: int = Field(..., description="Total number of fixes applied")
    message: str | None = Field(
        None, description="Status message with explanation of what was fixed, error details, or other information"
    )
    remaining_issues: list[str] = Field(
        default_factory=list, description="Any remaining issues that couldn't be fixed"
    )
    corrected_content: str = Field(
        "", description="The corrected LaTeX file content with all fixes applied"
    )

    def __str__(self) -> str:
        """String representation of compilation error agent output."""
        lines = [
            "Error Fixing Agent Output:",
            f"  Status: {self.status.value}",
            f"  File Modified: {self.file_modified}",
            f"  Total Fixes: {self.total_fixes}",
            f"  Message: {self.message}",
        ]
        if self.fixes_applied:
            lines.append("  Fixes Applied:")
            for i, fix in enumerate(self.fixes_applied, 1):
                lines.append(
                    f"    {i}. {fix.error_type} (Line {fix.line_number}): {fix.fix_applied}"
                )
        if self.remaining_issues:
            lines.append("  Remaining Issues:")
            for issue in self.remaining_issues:
                lines.append(f"    - {issue}")
        return "\n".join(lines)


class CompilationDiagnostics(BaseModel):
    """Enhanced diagnostics tracking for LaTeX compilation process."""

    total_attempts: int = Field(
        default=0, description="Total number of compilation attempts"
    )
    successful_compilations: int = Field(
        default=0, description="Number of successful compilations"
    )
    failed_compilations: int = Field(
        default=0, description="Number of failed compilations"
    )
    parsing_failures: int = Field(
        default=0, description="Number of agent response parsing failures"
    )
    error_fixing_attempts: int = Field(
        default=0, description="Number of error fixing attempts"
    )
    successful_fixes: int = Field(
        default=0, description="Number of successful error fixes"
    )

    def increment(self, attribute: str, amount: int = 1) -> None:
        """Increment a counter attribute by the specified amount.

        Args:
            attribute: The name of the attribute to increment
            amount: The amount to increment by (default: 1)

        Raises:
            AttributeError: If the attribute doesn't exist
            ValueError: If the attribute is not an integer
        """
        if not hasattr(self, attribute):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attribute}'"
            )

        current_value = getattr(self, attribute)
        if not isinstance(current_value, int):
            raise ValueError(
                f"Attribute '{attribute}' is not an integer (got {type(current_value).__name__})"
            )

        setattr(self, attribute, current_value + amount)

    def get_compilation_success_rate(self) -> float:
        """Calculate compilation success rate as percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_compilations / self.total_attempts) * 100

    def get_fix_success_rate(self) -> float:
        """Calculate fix success rate as percentage."""
        if self.error_fixing_attempts == 0:
            return 0.0
        return (self.successful_fixes / self.error_fixing_attempts) * 100

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_attempts = 0
        self.successful_compilations = 0
        self.failed_compilations = 0
        self.parsing_failures = 0
        self.error_fixing_attempts = 0
        self.successful_fixes = 0

    def __str__(self) -> str:
        """String representation of compilation diagnostics."""
        lines = [
            "📊 COMPILATION DIAGNOSTICS SUMMARY:",
            f"   Total Attempts: {self.total_attempts}",
            f"   Successful Compilations: {self.successful_compilations}",
            f"   Failed Compilations: {self.failed_compilations}",
            f"   Parsing Failures: {self.parsing_failures}",
            f"   Error Fixing Attempts: {self.error_fixing_attempts}",
            f"   Successful Fixes: {self.successful_fixes}",
        ]

        if self.total_attempts > 0:
            lines.append(
                f"   Compilation Success Rate: {self.get_compilation_success_rate():.2f}%"
            )

        if self.error_fixing_attempts > 0:
            lines.append(f"   Fix Success Rate: {self.get_fix_success_rate():.2f}%")

        return "\n".join(lines)
