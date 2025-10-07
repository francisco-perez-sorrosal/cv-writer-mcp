"""Compilation error agent functionality.

This module contains the compilation error agent and related utilities for analyzing
and fixing LaTeX compilation errors using AI agents.
"""

from pathlib import Path
from typing import Any

from agents import Agent, Runner
from loguru import logger

from ..models import CompletionStatus, ServerConfig, create_agent_from_config
from ..utils import (
    create_versioned_file,
    get_next_version_number,
    load_agent_config,
    read_text_file,
)
from .models import (
    CompilationDiagnostics,
    CompilationErrorOutput,
    OrchestrationResult,
)


class CompilationErrorAgent:
    """Handles LaTeX compilation error fixing using AI agents."""

    def __init__(
        self,
        diagnostics: CompilationDiagnostics | None = None,
        server_config: ServerConfig | None = None,
    ):
        """Initialize the error fixing agent.

        Args:
            diagnostics: Optional diagnostics tracker for monitoring fixing attempts
            config: Server configuration for accessing templates and guides
        """
        self.diagnostics = diagnostics or CompilationDiagnostics()
        self.server_config = server_config
        self.agent_config = load_agent_config("error_agent.yaml")

    def _log_agent_diagnostics(self, stage: str, data: dict[str, Any]) -> None:
        """Enhanced diagnostic logging for agent communication and parsing."""
        logger.info(f"🔍 DIAGNOSTIC [{stage}]: {data}")

    def _validate_agent_response(
        self, response: Any, expected_type: str
    ) -> dict[str, Any]:
        """Validate and analyze agent response structure."""
        validation_result = {
            "is_valid": False,
            "response_type": str(type(response)),
            "has_final_output": hasattr(response, "final_output"),
            "final_output_type": None,
            "has_tool_calls": hasattr(response, "tool_calls"),
            "has_messages": hasattr(response, "messages"),
            "raw_content": (
                str(response)[:500] + "..."
                if len(str(response)) > 500
                else str(response)
            ),
        }

        if hasattr(response, "final_output"):
            validation_result["final_output_type"] = str(type(response.final_output))
            validation_result["is_valid"] = True

        self._log_agent_diagnostics(
            f"RESPONSE_VALIDATION_{expected_type}", validation_result
        )
        return validation_result

    def _validate_corrected_content(self, content: str) -> dict[str, Any]:
        """Validate corrected LaTeX content for basic structural integrity."""
        if not content:
            return {
                "is_valid": False,
                "content_length": 0,
                "has_document_class": False,
                "has_begin_document": False,
                "has_end_document": False,
                "validation_errors": ["Content is empty"],
            }

        validation_result: dict[str, Any] = {
            "is_valid": True,
            "content_length": len(content),
            "has_document_class": "\\documentclass" in content,
            "has_begin_document": "\\begin{document}" in content,
            "has_end_document": "\\end{document}" in content,
            "validation_errors": [],
        }

        # Basic LaTeX structure validation
        if not validation_result["has_document_class"]:
            validation_result["validation_errors"].append("Missing \\documentclass")
            validation_result["is_valid"] = False

        if not validation_result["has_begin_document"]:
            validation_result["validation_errors"].append("Missing \\begin{document}")
            validation_result["is_valid"] = False

        if not validation_result["has_end_document"]:
            validation_result["validation_errors"].append("Missing \\end{document}")
            validation_result["is_valid"] = False

        # Check for balanced braces (basic check)
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            validation_result["validation_errors"].append(
                f"Unbalanced braces: {open_braces} open, {close_braces} close"
            )
            validation_result["is_valid"] = False

        validation_result["brace_balance"] = {
            "open_braces": open_braces,
            "close_braces": close_braces,
            "balanced": open_braces == close_braces,
        }

        return validation_result

    def build_fixing_prompt(
        self,
        latex_content: str,
        tex_file_path: Path,
        errors_found: list[str],
        exit_code: int,
    ) -> str:
        """Build the error fixing prompt with all necessary components.

        Args:
            latex_content: Raw LaTeX content to be fixed
            tex_file_path: Path to the .tex file being fixed
            errors_found: List of detailed error messages from compilation
            exit_code: Compilation exit code

        Returns:
            Formatted error fixing prompt string
        """
        # Format the prompt template with error information
        # Note: Values are inserted as-is by format(), no escaping needed
        error_fixing_prompt = self.agent_config["prompt_template"].format(
            latex_content=latex_content,
            tex_file_path=tex_file_path,
            error_count=len(errors_found),
            exit_code=exit_code,
            error_log="\n".join(errors_found),
        )

        return error_fixing_prompt

    def create_error_agent(self) -> Agent:
        """Create an agent specialized in fixing LaTeX compilation errors.

        This agent analyzes LaTeX compilation errors and fixes common errors
        in the LaTeX source code using the ModernCV user guide for reference.

        Returns:
            An Agent configured for LaTeX error fixing
        """
        # Read the ModernCV user guide
        # Go up from src/cv_writer_mcp/compilation/ to project root
        base_path = Path(__file__).parent.parent.parent.parent
        moderncv_guide = read_text_file(
            base_path / "context" / "latex" / "moderncv_userguide.txt",
            "ModernCV user guide",
            ".txt",
        )

        # Get the error fixing agent instructions from YAML configuration
        # Note: Values are inserted as-is by format(), no escaping needed
        # We've escaped literal braces in the YAML template itself
        error_fixing_instructions = self.agent_config["instructions"].format(
            moderncv_guide=moderncv_guide
        )

        # Create agent using centralized helper with safe defaults
        return create_agent_from_config(
            agent_config=self.agent_config,
            instructions=error_fixing_instructions,
        )

    def parse_error_agent_output(
        self, error_fixing_result: Any
    ) -> CompilationErrorOutput:
        """Enhanced parsing of compilation error agent output with detailed diagnostics."""
        self.diagnostics.increment("error_fixing_attempts")

        # Validate the response structure first
        validation = self._validate_agent_response(error_fixing_result, "ERROR_FIXING")

        try:
            # Check if we have a properly structured response
            if validation["has_final_output"] and isinstance(
                error_fixing_result.final_output, CompilationErrorOutput
            ):
                fixing_output = error_fixing_result.final_output
                logger.info("✅ Successfully parsed CompilationErrorOutput directly")

                # Validate the content
                content_validation = self._validate_corrected_content(
                    fixing_output.corrected_content
                )
                self._log_agent_diagnostics(
                    "ERROR_FIXING_CONTENT_VALIDATION", content_validation
                )

                if (
                    fixing_output.status == CompletionStatus.SUCCESS
                    and fixing_output.corrected_content
                ):
                    self.diagnostics.increment("successful_fixes")

                return fixing_output

            # Fallback parsing - try to extract from final_output
            logger.info("🔄 Using fallback parsing for error fixing agent output")
            fixing_output = error_fixing_result.final_output

            # Log the fallback result
            self._log_agent_diagnostics(
                "ERROR_FIXING_FALLBACK",
                {
                    "output_type": str(type(fixing_output)),
                    "has_success_attr": hasattr(fixing_output, "success"),
                    "has_corrected_content": hasattr(
                        fixing_output, "corrected_content"
                    ),
                    "content_preview": (
                        str(fixing_output)[:300] + "..."
                        if len(str(fixing_output)) > 300
                        else str(fixing_output)
                    ),
                },
            )

            return fixing_output  # type: ignore

        except Exception as e:
            logger.error(f"❌ Error fixing agent parsing failed: {e}")
            self._log_agent_diagnostics(
                "ERROR_FIXING_PARSING_EXCEPTION",
                {"error_type": type(e).__name__, "error": str(e)},
            )

            # Return a failed CompilationErrorOutput
            return CompilationErrorOutput(
                status=CompletionStatus.FAILED,
                corrected_content="",
                total_fixes=0,
                fixes_applied=[],
                file_modified=False,
                message=f"Failed to parse error fixing agent output: {str(e)}",
            )

    async def fix_errors(
        self,
        tex_file_path: Path,
        compilation_result: "OrchestrationResult",
    ) -> tuple[CompilationErrorOutput, Path | None]:
        """Fix LaTeX errors using the error fixing agent.

        Args:
            tex_file_path: Path to the .tex file to fix
            compilation_result: Compilation result containing errors and exit code

        Returns:
            Tuple of (fixing result, path to corrected file if successful)
        """
        logger.info("Starting LaTeX error fixing with compilation error agent")

        # Validate we have errors to fix
        if not compilation_result.errors_found:
            logger.warning("No errors found in compilation result")
            return (
                CompilationErrorOutput(
                    status=CompletionStatus.FAILED,
                    corrected_content="",
                    total_fixes=0,
                    fixes_applied=[],
                    file_modified=False,
                    message="No errors to fix",
                ),
                None,
            )

        error_agent = self.create_error_agent()

        # Read the LaTeX file content
        try:
            latex_content = read_text_file(tex_file_path, "LaTeX file", ".tex")
        except Exception as e:
            logger.error(f"Failed to read LaTeX file content: {e}")
            return (
                CompilationErrorOutput(
                    status=CompletionStatus.FAILED,
                    corrected_content="",
                    total_fixes=0,
                    fixes_applied=[],
                    file_modified=False,
                    message=f"Failed to read LaTeX file: {str(e)}",
                ),
                None,
            )

        # Build error fixing prompt using errors from compilation result
        error_fixing_prompt = self.build_fixing_prompt(
            latex_content=latex_content,
            tex_file_path=tex_file_path,
            errors_found=compilation_result.errors_found,
            exit_code=compilation_result.exit_code,
        )

        try:
            logger.info("Calling compilation error agent...")
            error_fixing_result = await Runner.run(error_agent, error_fixing_prompt)

            # Parse the agent output
            fixing_output = self.parse_error_agent_output(error_fixing_result)

            if (
                fixing_output.status == CompletionStatus.SUCCESS
                and fixing_output.corrected_content
            ):
                logger.info(
                    f"Error fixing completed: {fixing_output.total_fixes} fixes applied"
                )

                # Create versioned file with the corrected content
                next_version = get_next_version_number(tex_file_path)
                versioned_file = create_versioned_file(tex_file_path, next_version)
                versioned_file.write_text(
                    fixing_output.corrected_content, encoding="utf-8"
                )

                logger.info(
                    f"Successfully saved corrected content to: {versioned_file.name} (v{next_version})"
                )
                return fixing_output, versioned_file
            else:
                logger.warning("Error fixing did not modify the file or failed")
                return fixing_output, None

        except Exception as e:
            logger.error(f"Error fixing failed: {e}")
            return (
                CompilationErrorOutput(
                    status=CompletionStatus.FAILED,
                    corrected_content="",
                    total_fixes=0,
                    fixes_applied=[],
                    file_modified=False,
                    message=f"Error fixing failed: {str(e)}",
                ),
                None,
            )
