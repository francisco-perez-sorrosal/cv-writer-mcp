"""Simplified Pydantic models for CV Writer MCP Server."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cv_writer_mcp.logger import LogLevel



class CompletionStatus(str, Enum):
    """Generic completion status enumeration."""

    SUCCESS = "success"
    FAILED = "failed"


class LaTeXOutput(BaseModel):
    """Structured output for LaTeX conversion."""

    latex_content: str = Field(
        description="The converted LaTeX content ready for insertion into the template"
    )
    conversion_notes: str = Field(
        description="Notes about the conversion process and any important considerations"
    )


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


class LaTeXEngine(str, Enum):
    """LaTeX engine enumeration."""

    PDFLATEX = "pdflatex"
    # TODO Add other engines here if necessary


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
        None, description="Status message with conversion notes, error details, or other information"
    )


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

    def __str__(self) -> str:
        lines = [
            "LaTeX Compilation Result:",
            f"  Status: {self.status.value}",
            f"  Compilation Time: {self.compilation_time:.2f} seconds",
            f"  Message: {self.message}",
            f"  Output Path: {self.output_path}",
        ]
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

    status: CompletionStatus = Field(..., description="Whether the compilation was successful")
    compilation_time: float = Field(
        ..., description="Time taken for compilation in seconds"
    )
    message: str | None = Field(
        None, description="Status message with compilation details or error information"
    )
    log_summary: str = Field("", description="Summary of compilation log output")
    engine_used: str = Field(..., description="LaTeX engine that was used")
    output_path: str = Field(..., description="Path where the PDF was generated")

    def __str__(self) -> str:
        lines = [
            "Compilation Agent Output:",
            f"  Status: {self.status.value}",
            f"  Compilation Time: {self.compilation_time:.2f} seconds",
            f"  Message: {self.message}",
            f"  Engine Used: {self.engine_used}",
            f"  Output Path: {self.output_path}",
        ]
        if self.log_summary:
            log_preview = (
                self.log_summary[:300] + "..."
                if len(self.log_summary) > 300
                else self.log_summary
            )
            lines.append(f"  Log Summary: {log_preview}")
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


class HealthStatusResponse(BaseModel):
    """Response model for health check."""

    status: str = "healthy"
    service: str = "cv-writer-mcp"
    timestamp: str | None = None
    version: str | None = None


class FileOperationType(str, Enum):
    """File operation type enumeration."""

    READ = "read"
    WRITE = "write"
    REPLACE_LINE = "replace_line"


class FileOperationResult(BaseModel):
    """Simplified result model for file operations."""

    success: bool = Field(..., description="Whether the operation was successful")
    operation: FileOperationType = Field(..., description="Type of operation performed")
    file_path: str = Field(..., description="Path to the file that was operated on")
    
    # Success fields
    content: str | None = Field(default=None, description="File content (for read operations)")
    message: str | None = Field(default=None, description="Success message")
    line_count: int | None = Field(default=None, description="Number of lines in the file")
    
    # Error fields
    error: str | None = Field(default=None, description="Error message if operation failed")

    @classmethod
    def success_read(cls, file_path: str, content: str, line_count: int) -> "FileOperationResult":
        """Create a successful read operation result."""
        return cls(
            success=True,
            operation=FileOperationType.READ,
            file_path=file_path,
            content=content,
            line_count=line_count,
        )

    @classmethod
    def error_file_not_found(cls, file_path: str, operation: FileOperationType) -> "FileOperationResult":
        """Create a file not found error result."""
        return cls(
            success=False,
            operation=operation,
            file_path=file_path,
            error=f"File not found: {file_path}",
        )

    @classmethod
    def error_exception(cls, file_path: str, operation: FileOperationType, exception: Exception) -> "FileOperationResult":
        """Create a tool exception error result."""
        return cls(
            success=False,
            operation=operation,
            file_path=file_path,
            error=f"Tool error: {str(exception)}",
        )

    def to_json(self) -> str:
        """Convert the result to JSON string."""
        return self.model_dump_json()

    def __str__(self) -> str:
        """String representation of file operation result."""
        status = "SUCCESS" if self.success else "ERROR"
        if self.success:
            details = []
            if self.content is not None:
                content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
                details.append(f"Content: {content_preview}")
            if self.message:
                details.append(f"Message: {self.message}")
            if self.line_count is not None:
                details.append(f"Lines: {self.line_count}")
            details_str = "; ".join(details) if details else "Operation completed"
        else:
            details_str = f"Error: {self.error}"
        
        return f"FileOperationResult({status}) - {self.operation.value} {self.file_path}: {details_str}"


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


class ServerConfig(BaseModel):
    """Server configuration model."""

    host: str = "localhost"
    port: int = 8000
    base_url: str = "http://localhost:8000"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    output_dir: Path = Path("./output")
    temp_dir: Path = Path("./temp")
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    latex_timeout: int = 30
    openai_api_key: str | None = None
    templates_dir: Path = Path("./context")

    @field_validator("output_dir", "temp_dir", "templates_dir")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to ensure base_url consistency."""
        # If base_url is still the default, update it based on host and port
        if self.base_url == "http://localhost:8000" and (
            self.host != "localhost" or self.port != 8000
        ):
            self.base_url = f"http://{self.host}:{self.port}"

    def get_base_url(self) -> str:
        """Get the base URL, ensuring it reflects current host and port."""
        return f"http://{self.host}:{self.port}"


def get_output_type_class(output_type_name: str):
    """Get the actual output type class from the string name.
    
    Args:
        output_type_name: String name of the output type from config
        
    Returns:
        The actual class object for the output type
        
    Raises:
        ValueError: If the output type name is not recognized
    """
    # Centralized mapping of output type names to actual classes
    output_type_mapping = {
        "LaTeXOutput": LaTeXOutput,
        "CompilerAgentOutput": CompilerAgentOutput,
        "CompilationErrorOutput": CompilationErrorOutput,
        "PageCaptureOutput": PageCaptureOutput,
        "FormattingOutput": FormattingOutput,
        "PDFAnalysisOutput": PDFAnalysisOutput,
    }
    
    if output_type_name not in output_type_mapping:
        raise ValueError(f"Unknown output type: {output_type_name}. Available types: {list(output_type_mapping.keys())}")
    
    return output_type_mapping[output_type_name]
