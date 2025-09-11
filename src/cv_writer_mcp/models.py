"""Simplified Pydantic models for CV Writer MCP Server."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from cv_writer_mcp.logger import LogLevel


class ConversionStatus(str, Enum):
    """Conversion status enumeration."""

    SUCCESS = "success"
    FAILED = "failed"


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

    status: ConversionStatus
    tex_url: str | None = Field(
        None, description="Resource URI to access the generated LaTeX file"
    )
    error_message: str | None = Field(
        None, description="Error message if conversion failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CompileLaTeXRequest(BaseModel):
    """Request model for LaTeX to PDF compilation."""

    tex_filename: str = Field(..., description="Name of the .tex file to compile")
    output_filename: str | None = Field(
        None, description="Custom output filename for PDF"
    )
    latex_engine: LaTeXEngine = Field(
        LaTeXEngine.PDFLATEX, description="LaTeX engine to use"
    )
    use_agent: bool = Field(
        True, description="Whether to use AI agents for compilation"
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


class CompileLaTeXResponse(BaseModel):
    """Response model for LaTeX to PDF compilation."""

    status: ConversionStatus

    pdf_url: str | None = Field(
        None, description="Resource URI to access the generated PDF"
    )
    error_message: str | None = Field(
        None, description="Error message if compilation failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class LaTeXCompilationResult(BaseModel):
    """Result of LaTeX compilation."""

    success: bool
    output_path: Path | None = None
    log_output: str = ""
    error_message: str | None = None
    compilation_time: float = 0.0

    def __str__(self) -> str:
        """String representation of LaTeX compilation result."""
        lines = [
            "LaTeX Compilation Result:",
            f"  Success: {self.success}",
            f"  Compilation Time: {self.compilation_time:.2f} seconds",
            f"  Error Message: {self.error_message}",
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


class CompilationAgentOutput(BaseModel):
    """Structured output from the LaTeX compilation agent."""

    success: bool = Field(..., description="Whether the compilation was successful")
    compilation_time: float = Field(
        ..., description="Time taken for compilation in seconds"
    )
    error_message: str | None = Field(
        None, description="Error message if compilation failed"
    )
    log_summary: str = Field("", description="Summary of compilation log output")
    engine_used: str = Field(..., description="LaTeX engine that was used")
    output_path: str = Field(..., description="Path where the PDF was generated")

    def __str__(self) -> str:
        """String representation of compilation agent output."""
        lines = [
            "Compilation Agent Output:",
            f"  Success: {self.success}",
            f"  Compilation Time: {self.compilation_time:.2f} seconds",
            f"  Error Message: {self.error_message}",
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
        """String representation of error fix."""
        lines = [
            "Error Fix:",
            f"  Type: {self.error_type}",
            f"  Description: {self.error_description}",
            f"  Fix Applied: {self.fix_applied}",
            f"  Line Number: {self.line_number}",
            f"  Confidence: {self.confidence:.2f}",
        ]
        return "\n".join(lines)


class ErrorFixingAgentOutput(BaseModel):
    """Structured output from the LaTeX error fixing agent."""

    fixes_applied: list[ErrorFix] = Field(
        ..., description="List of fixes that were applied"
    )
    file_modified: bool = Field(
        ..., description="Whether the LaTeX file was actually modified"
    )
    total_fixes: int = Field(..., description="Total number of fixes applied")
    success: bool = Field(
        ..., description="Whether the error fixing process was successful"
    )
    explanation: str = Field("", description="Explanation of what was fixed and why")
    remaining_issues: list[str] = Field(
        default_factory=list, description="Any remaining issues that couldn't be fixed"
    )
    corrected_content: str = Field(
        "", description="The corrected LaTeX file content with all fixes applied"
    )

    def __str__(self) -> str:
        """String representation of error fixing agent output."""
        lines = [
            "Error Fixing Agent Output:",
            f"  Success: {self.success}",
            f"  File Modified: {self.file_modified}",
            f"  Total Fixes: {self.total_fixes}",
            f"  Explanation: {self.explanation}",
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
    """Unified result model for file editing operations.

    This model elegantly handles all possible return cases from file editing operations
    by using optional fields and a success flag. It can represent both successful
    operations and various error conditions with appropriate metadata.
    """

    # Core status
    success: bool = Field(..., description="Whether the operation was successful")

    # Operation details
    operation: FileOperationType = Field(..., description="Type of operation performed")
    file_path: str = Field(..., description="Path to the file that was operated on")

    # Success-specific fields (only populated when success=True)
    content: str | None = Field(
        default=None, description="File content (for read operations)"
    )
    message: str | None = Field(
        default=None, description="Success message describing what was accomplished"
    )
    line_count: int | None = Field(
        default=None, description="Number of lines in the file"
    )
    file_size: int | None = Field(default=None, description="Size of the file in bytes")
    verification: str | None = Field(
        default=None, description="Verification details for write operations"
    )
    analysis: dict[str, Any] | None = Field(
        default=None, description="Additional analysis data (e.g., brace analysis)"
    )

    # Error-specific fields (only populated when success=False)
    error: str | None = Field(
        default=None, description="Error message describing what went wrong"
    )
    file_exists: bool | None = Field(
        default=None, description="Whether the file exists (for error cases)"
    )

    # Additional metadata
    line_number: int | None = Field(
        default=None, description="Line number for replace_line operations"
    )
    total_lines: int | None = Field(
        default=None, description="Total number of lines in file (for range validation)"
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that success/error fields are used appropriately."""
        if self.success:
            # For successful operations, we should have either content (read) or message (write/replace)
            if not (self.content is not None or self.message is not None):
                raise ValueError(
                    "Successful operations must have either content or message"
                )
        else:
            # For failed operations, we should have an error message
            if not self.error:
                raise ValueError("Failed operations must have an error message")

    @classmethod
    def create_success_read(
        cls, file_path: str, content: str, line_count: int
    ) -> "FileOperationResult":
        """Create a successful read operation result."""
        return cls(
            success=True,
            operation=FileOperationType.READ,
            file_path=file_path,
            content=content,
            line_count=line_count,
        )

    @classmethod
    def create_success_write(
        cls, file_path: str, message: str, line_count: int, file_size: int
    ) -> "FileOperationResult":
        """Create a successful write operation result."""
        return cls(
            success=True,
            operation=FileOperationType.WRITE,
            file_path=file_path,
            message=message,
            line_count=line_count,
            file_size=file_size,
            verification="File exists and has content",
        )

    @classmethod
    def create_success_replace_line(
        cls, file_path: str, line_number: int, line_count: int, file_size: int
    ) -> "FileOperationResult":
        """Create a successful replace_line operation result."""
        return cls(
            success=True,
            operation=FileOperationType.REPLACE_LINE,
            file_path=file_path,
            message=f"Line {line_number} replaced successfully",
            line_count=line_count,
            file_size=file_size,
            line_number=line_number,
            verification="File exists and has content",
        )

    @classmethod
    def create_error_file_not_found(
        cls, file_path: str, operation: FileOperationType
    ) -> "FileOperationResult":
        """Create a file not found error result."""
        return cls(
            success=False,
            operation=operation,
            file_path=file_path,
            error=f"File not found: {file_path}",
            file_exists=False,
        )

    @classmethod
    def create_error_write_verification_failed(
        cls,
        file_path: str,
        operation: FileOperationType,
        file_exists: bool,
        file_size: int,
    ) -> "FileOperationResult":
        """Create a write verification failed error result."""
        return cls(
            success=False,
            operation=operation,
            file_path=file_path,
            error=f"File write verification failed: {file_path}",
            file_exists=file_exists,
            file_size=file_size,
        )

    @classmethod
    def create_error_line_out_of_range(
        cls, file_path: str, line_number: int, total_lines: int
    ) -> "FileOperationResult":
        """Create a line number out of range error result."""
        return cls(
            success=False,
            operation=FileOperationType.REPLACE_LINE,
            file_path=file_path,
            error=f"Line number {line_number} out of range (1-{total_lines})",
            line_number=line_number,
            total_lines=total_lines,
        )

    @classmethod
    def create_error_unknown_action(
        cls, file_path: str, action: str
    ) -> "FileOperationResult":
        """Create an unknown action error result."""
        return cls(
            success=False,
            operation=FileOperationType.READ,  # Default, will be overridden
            file_path=file_path,
            error=f"Unknown action: {action}. Use 'read', 'write', or 'replace_line'",
        )

    @classmethod
    def create_error_tool_exception(
        cls, file_path: str, operation: FileOperationType, exception: Exception
    ) -> "FileOperationResult":
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
        if self.success:
            status = "SUCCESS"
            details = []
            if self.content is not None:
                content_preview = (
                    self.content[:100] + "..."
                    if len(self.content) > 100
                    else self.content
                )
                details.append(f"Content: {content_preview}")
            if self.message:
                details.append(f"Message: {self.message}")
            if self.line_count is not None:
                details.append(f"Lines: {self.line_count}")
            if self.file_size is not None:
                details.append(f"Size: {self.file_size} bytes")
        else:
            status = "ERROR"
            details = [f"Error: {self.error}"]
            if self.file_exists is not None:
                details.append(f"File exists: {self.file_exists}")

        return f"FileOperationResult({status}) - {self.operation.value} {self.file_path}: {'; '.join(details)}"


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
