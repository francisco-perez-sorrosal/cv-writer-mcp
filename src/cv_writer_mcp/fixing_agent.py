"""LaTeX error fixing agent functionality.

This module contains the error fixing agent and related utilities for analyzing
and fixing LaTeX compilation errors using AI agents.
"""

import re
from pathlib import Path
from typing import Any

from agents import Agent, Runner
from loguru import logger

from .models import (
    CompilationDiagnostics,
    ErrorFixingAgentOutput,
    ServerConfig,
)
from .utils import create_timestamped_version, load_agent_config, read_text_file


class ErrorFixingAgent:
    """Handles LaTeX error fixing using AI agents."""

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
        self.agent_config = load_agent_config("error_fixing_agent.yaml")

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

    def _extract_focused_error_context(
        self, tex_file_path: Path, compilation_output: str
    ) -> dict[str, Any]:
        """Extract focused error context from LaTeX logs with validation and analysis."""
        log_analysis: dict[str, Any] = {
            "log_file_found": False,
            "log_size": 0,
            "error_count": 0,
            "warning_count": 0,
            "critical_errors": [],
            "focused_errors": [],
            "raw_log_excerpt": "",
            "compilation_output_excerpt": "",
            "validation_status": "unknown",
        }

        # First, analyze the compilation output for immediate errors
        if compilation_output:
            log_analysis["compilation_output_excerpt"] = (
                compilation_output[:500] + "..."
                if len(compilation_output) > 500
                else compilation_output
            )

            # Look for common LaTeX error patterns in compilation output
            error_patterns = [
                r"! (.+)",  # LaTeX errors starting with !
                r"Error: (.+)",  # General errors
                r"Fatal error (.+)",  # Fatal errors
                r"Missing (.+)",  # Missing components
                r"Undefined (.+)",  # Undefined references
            ]

            for pattern in error_patterns:
                matches = re.findall(
                    pattern, compilation_output, re.MULTILINE | re.IGNORECASE
                )
                for match in matches:
                    log_analysis["critical_errors"].append(f"Compilation: {match}")
                    log_analysis["error_count"] += 1

        # Try to find and analyze the log file
        possible_log_files = [
            tex_file_path.with_suffix(".log"),
            tex_file_path.parent / f"{tex_file_path.stem}.log",
            Path("output") / f"{tex_file_path.stem}.log",
        ]

        log_file = None
        for log_path in possible_log_files:
            if log_path.exists():
                log_file = log_path
                log_analysis["log_file_found"] = True
                log_analysis["log_size"] = log_path.stat().st_size
                break

        if not log_file:
            log_analysis["validation_status"] = "no_log_file"
            self._log_agent_diagnostics("LOG_EXTRACTION_STATUS", log_analysis)
            return log_analysis

        try:
            with open(log_file, encoding="utf-8", errors="ignore") as f:
                log_content = f.read()

            # Extract focused error sections
            focused_errors = self._extract_latex_errors_from_log(log_content)
            log_analysis["focused_errors"] = focused_errors
            log_analysis["error_count"] += len(focused_errors)

            # Count warnings
            warning_count = len(re.findall(r"Warning:", log_content, re.IGNORECASE))
            log_analysis["warning_count"] = warning_count

            # Get relevant log excerpt (from errors or end of file)
            if focused_errors:
                # Find the first error location in the log
                first_error = focused_errors[0]["raw_error"]
                error_start = log_content.find(first_error)
                if error_start != -1:
                    # Extract context around the error
                    context_start = max(0, error_start - 200)
                    context_end = min(len(log_content), error_start + 800)
                    log_analysis["raw_log_excerpt"] = log_content[
                        context_start:context_end
                    ]
                else:
                    # Fallback to tail of log
                    log_analysis["raw_log_excerpt"] = (
                        log_content[-1000:] if len(log_content) > 1000 else log_content
                    )
            else:
                # No specific errors found, use tail of log
                log_analysis["raw_log_excerpt"] = (
                    log_content[-1000:] if len(log_content) > 1000 else log_content
                )

            log_analysis["validation_status"] = "success"

        except Exception as e:
            log_analysis["validation_status"] = f"read_error: {str(e)}"
            logger.error(f"Failed to read log file {log_file}: {e}")

        # Log the analysis results
        self._log_agent_diagnostics("LOG_EXTRACTION_ANALYSIS", log_analysis)

        return log_analysis

    def _extract_latex_errors_from_log(self, log_content: str) -> list[dict[str, Any]]:
        """Extract specific LaTeX errors from log content with context."""
        errors = []

        # Common LaTeX error patterns
        error_patterns = [
            {
                "pattern": r"! (.+)",
                "type": "latex_error",
                "description": "LaTeX compilation error",
            },
            {
                "pattern": r"Missing (.+?) before (.+)",
                "type": "missing_element",
                "description": "Missing required element",
            },
            {
                "pattern": r"Undefined control sequence (.+)",
                "type": "undefined_command",
                "description": "Undefined LaTeX command",
            },
            {
                "pattern": r"Package (.+?) Error: (.+)",
                "type": "package_error",
                "description": "LaTeX package error",
            },
            {
                "pattern": r"(.+?) on line (\d+)",
                "type": "line_specific_error",
                "description": "Error with line number",
            },
        ]

        for error_pattern in error_patterns:
            matches = re.finditer(
                error_pattern["pattern"], log_content, re.MULTILINE | re.IGNORECASE
            )
            for match in matches:
                error_info = {
                    "type": error_pattern["type"],
                    "description": error_pattern["description"],
                    "raw_error": match.group(0),
                    "error_text": match.group(1) if match.groups() else match.group(0),
                    "line_number": None,
                    "context": "",
                }

                # Try to extract line number if available
                line_match = re.search(r"line (\d+)", match.group(0))
                if line_match:
                    error_info["line_number"] = int(line_match.group(1))

                # Extract context around the error
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(log_content), match.end() + 200)
                error_info["context"] = log_content[start_pos:end_pos]

                errors.append(error_info)

        # Remove duplicates and sort by importance
        unique_errors = []
        seen_errors = set()

        for error in errors:
            error_key = f"{error['type']}:{error['error_text']}"
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                unique_errors.append(error)

        # Sort by error type priority
        priority_order = [
            "latex_error",
            "package_error",
            "undefined_command",
            "missing_element",
            "line_specific_error",
        ]
        unique_errors.sort(
            key=lambda x: (
                priority_order.index(x["type"]) if x["type"] in priority_order else 999
            )
        )

        return unique_errors[:5]  # Limit to top 5 most relevant errors

    def _prepare_numbered_latex_for_agent(
        self, latex_content: str
    ) -> tuple[str, dict[int, str]]:
        """Prepare line-numbered LaTeX content and create line mapping."""
        lines = latex_content.splitlines()
        line_mapping: dict[int, str] = {}  # For reverse processing
        numbered_lines = []

        for i, line in enumerate(lines, 1):
            padded_num = f"{i:03d}"  # e.g., "001", "045"
            numbered_line = f"{padded_num}: {line}"
            numbered_lines.append(numbered_line)
            line_mapping[i] = line  # Store original content

        return "\n".join(numbered_lines), line_mapping

    def _extract_corrected_content_from_agent(self, agent_response: str) -> str:
        """Extract and process corrected LaTeX content from agent response."""
        # Agent should return content still with line numbers for validation
        response_lines = agent_response.splitlines()
        corrected_lines = []

        for line in response_lines:
            # Match line number pattern: "045: content"
            line_match = re.match(r"^(\d{3}):\s*(.*)$", line)
            if line_match:
                int(line_match.group(1))
                content = line_match.group(2)
                corrected_lines.append(content)
            else:
                # Handle lines without numbers (fallback for malformed response)
                corrected_lines.append(line)

        return "\n".join(corrected_lines)

    def _create_latex_error_guide(self, error_context: dict[str, Any]) -> str:
        """Create LaTeX error guide using actual log format patterns."""
        guide_text = "\n<LATEX_ERROR_GUIDE>\n"

        # Add specific error patterns found in logs
        focused_errors = error_context.get("focused_errors", [])
        critical_errors = error_context.get("critical_errors", [])

        if focused_errors or critical_errors:
            guide_text += "LaTeX errors follow this format:\n"
            guide_text += "! [Error Type]\n"
            guide_text += "<inserted text>\n"
            guide_text += "                [suggested fix]\n"
            guide_text += "l.[line_number] [problematic content]\n\n"

            guide_text += "SPECIFIC ERRORS IN YOUR FILE:\n"

            # Extract actual error line references from log
            log_content = error_context.get("raw_log_excerpt", "")
            log_line_patterns = re.findall(r"l\.(\d+)\s+(.+)", log_content)

            for line_num, problematic_content in log_line_patterns[
                :5
            ]:  # Limit to top 5
                guide_text += f"Line {line_num}: {problematic_content.strip()}\n"

            guide_text += "\nFocus your fixes on these exact lines and their surrounding context.\n"

        guide_text += "</LATEX_ERROR_GUIDE>\n"
        return guide_text

    def _analyze_latex_log_errors(self, log_content: str) -> dict[str, Any]:
        """Analyze LaTeX log for specific error patterns and line references."""
        error_analysis: dict[str, Any] = {
            "line_specific_errors": [],
            "error_types": set(),
            "problematic_lines": {},
        }

        # Pattern for LaTeX line errors: "l.45 content"
        line_error_pattern = r"l\.(\d+)\s+(.+)"
        line_errors = re.findall(line_error_pattern, log_content, re.MULTILINE)

        for line_num, content in line_errors:
            error_analysis["line_specific_errors"].append(
                {
                    "line": int(line_num),
                    "content": content.strip(),
                    "context_needed": True,
                }
            )
            error_analysis["problematic_lines"][int(line_num)] = content.strip()

        # Pattern for error types: "! Error message"
        error_type_pattern = r"!\s+([^\n]+)"
        error_types = re.findall(error_type_pattern, log_content, re.MULTILINE)
        error_analysis["error_types"] = set(error_types)

        return error_analysis

    def build_fixing_prompt(
        self,
        latex_content: str,
        tex_file_path: Path,
        error_context: dict[str, Any],
        log_content: str,
    ) -> str:
        """Build the error fixing prompt with all necessary components.
        
        Args:
            latex_content: Raw LaTeX content to be fixed
            tex_file_path: Path to the .tex file being fixed
            error_context: Error context extracted from compilation logs
            log_content: Full log content for analysis
            
        Returns:
            Formatted error fixing prompt string
        """
        # Prepare line-numbered LaTeX content for precise error fixing
        numbered_latex_content, line_mapping = self._prepare_numbered_latex_for_agent(
            latex_content
        )
        
        # Create LaTeX error guide
        latex_error_guide = self._create_latex_error_guide(error_context)
        
        # Analyze LaTeX log errors
        log_error_analysis = self._analyze_latex_log_errors(log_content)
        
        # Build focused errors text
        focused_errors_text = ""
        if error_context["focused_errors"]:
            for i, error in enumerate(error_context["focused_errors"], 1):
                focused_errors_text += (
                    f"{i}. {error['description']}: {error['error_text']}"
                )
                if error["line_number"]:
                    focused_errors_text += f" (Line {error['line_number']})"
                focused_errors_text += f"\n   Context: {error['context'][:200]}...\n"

        # Build critical errors text
        critical_errors_text = ""
        if error_context["critical_errors"]:
            for error in error_context["critical_errors"]:
                critical_errors_text += f"- {error}\n"

        # Format the prompt template with all variables
        error_fixing_prompt = self.agent_config["prompt_template"].format(
            latex_error_guide=latex_error_guide,
            numbered_latex_content=numbered_latex_content,
            tex_file_path=tex_file_path,
            error_count=error_context['error_count'],
            warning_count=error_context['warning_count'],
            validation_status=error_context['validation_status'],
            line_specific_errors_count=len(log_error_analysis['line_specific_errors']),
            focused_errors_text=focused_errors_text,
            critical_errors_text=critical_errors_text,
            log_content=log_content
        )
        
        return error_fixing_prompt

    def create_error_fixing_agent(self) -> Agent:
        """Create an agent specialized in fixing LaTeX compilation errors.

        This agent analyzes LaTeX compilation logs and fixes common errors
        in the LaTeX source code using the ModernCV user guide and template for reference.

        Returns:
            An Agent configured for LaTeX error fixing
        """
        # Read the ModernCV user guide and template
        moderncv_guide = read_text_file(Path("context/latex") / "moderncv_userguide.txt", "ModernCV user guide", ".txt")
        moderncv_template = read_text_file(Path("context/latex") / "moderncv_template.tex", "ModernCV template", ".tex")

        # Get the error fixing agent instructions from YAML configuration
        error_fixing_instructions = self.agent_config["instructions"].format(
            moderncv_guide=moderncv_guide,
            moderncv_template=moderncv_template
        )

        return Agent(
            name="LaTeX Error Fixing Agent",
            instructions=error_fixing_instructions,
            tools=[],  # No tools needed - content passed in prompt
            model="gpt-5-mini",
            output_type=ErrorFixingAgentOutput,
        )

    def parse_error_fixing_agent_output(
        self, error_fixing_result: Any
    ) -> ErrorFixingAgentOutput:
        """Enhanced parsing of error fixing agent output with detailed diagnostics."""
        self.diagnostics.increment("error_fixing_attempts")

        # Validate the response structure first
        validation = self._validate_agent_response(error_fixing_result, "ERROR_FIXING")

        try:
            # Check if we have a properly structured response
            if validation["has_final_output"] and isinstance(
                error_fixing_result.final_output, ErrorFixingAgentOutput
            ):
                fixing_output = error_fixing_result.final_output
                logger.info("✅ Successfully parsed ErrorFixingAgentOutput directly")

                # Validate the content
                content_validation = self._validate_corrected_content(
                    fixing_output.corrected_content
                )
                self._log_agent_diagnostics(
                    "ERROR_FIXING_CONTENT_VALIDATION", content_validation
                )

                if fixing_output.success and fixing_output.corrected_content:
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

            # Return a failed ErrorFixingAgentOutput
            return ErrorFixingAgentOutput(
                success=False,
                corrected_content="",
                total_fixes=0,
                fixes_applied=[],
                file_modified=False,
                explanation=f"Failed to parse error fixing agent output: {str(e)}",
            )

    async def fix_errors(
        self,
        tex_file_path: Path,
        error_context: dict[str, Any],
        log_content: str,
    ) -> tuple[ErrorFixingAgentOutput, Path | None]:
        """Fix LaTeX errors using the error fixing agent.

        Args:
            tex_file_path: Path to the .tex file to fix
            error_context: Error context extracted from compilation logs
            log_content: Full log content for analysis

        Returns:
            Tuple of (fixing result, path to corrected file if successful)
        """
        logger.info("Starting LaTeX error fixing with error fixing agent")

        error_fixing_agent = self.create_error_fixing_agent()

        # Read the LaTeX file content
        try:
            latex_content = read_text_file(tex_file_path, "LaTeX file", ".tex")
        except Exception as e:
            logger.error(f"Failed to read LaTeX file content: {e}")
            return (
                ErrorFixingAgentOutput(
                    success=False,
                    corrected_content="",
                    total_fixes=0,
                    fixes_applied=[],
                    file_modified=False,
                    explanation=f"Failed to read LaTeX file: {str(e)}",
                ),
                None,
            )

        # Build enhanced error fixing prompt with line numbers
        error_fixing_prompt = self.build_fixing_prompt(
            latex_content=latex_content,
            tex_file_path=tex_file_path,
            error_context=error_context,
            log_content=log_content
        )

        try:
            logger.info("Calling error fixing agent with line-numbered content...")
            error_fixing_result = await Runner.run(
                error_fixing_agent, error_fixing_prompt
            )

            # Parse the agent output
            fixing_output = self.parse_error_fixing_agent_output(error_fixing_result)

            if fixing_output.success and fixing_output.corrected_content:
                logger.info(
                    f"Error fixing completed: {fixing_output.total_fixes} fixes applied"
                )

                # Extract clean LaTeX content by removing line numbers from agent response
                clean_corrected_content = self._extract_corrected_content_from_agent(
                    fixing_output.corrected_content
                )

                # Create timestamped version with the corrected content
                timestamped_version = create_timestamped_version(tex_file_path)
                timestamped_version.write_text(
                    clean_corrected_content, encoding="utf-8"
                )

                logger.info(
                    f"Successfully saved corrected content to: {timestamped_version}"
                )
                return fixing_output, timestamped_version
            else:
                logger.warning("Error fixing did not modify the file or failed")
                return fixing_output, None

        except Exception as e:
            logger.error(f"Error fixing failed: {e}")
            return (
                ErrorFixingAgentOutput(
                    success=False,
                    corrected_content="",
                    total_fixes=0,
                    fixes_applied=[],
                    file_modified=False,
                    explanation=f"Error fixing failed: {str(e)}",
                ),
                None,
            )
