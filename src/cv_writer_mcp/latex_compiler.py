"""LaTeX compilation functionality."""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import Agent, Runner, function_tool
from loguru import logger

from .models import (
    CompilationAgentOutput,
    CompilationDiagnostics,
    CompileLaTeXRequest,
    CompileLaTeXResponse,
    ConversionStatus,
    ErrorFixingAgentOutput,
    LaTeXCompilationResult,
    LaTeXEngine,
    ServerConfig,
)


class LaTeXCompiler:
    """Compiles LaTeX documents to PDF."""

    def __init__(self, timeout: int = 30, config: ServerConfig | None = None):
        """Initialize the LaTeX compiler.

        Args:
            timeout: Compilation timeout in seconds
            config: Server configuration (optional, for high-level compilation)
        """
        self.timeout = timeout
        self.config = config

        # Enhanced diagnostics tracking using Pydantic model
        self._compilation_diagnostics = CompilationDiagnostics()

    def _log_agent_diagnostics(self, stage: str, data: dict[str, Any]) -> None:
        """Enhanced diagnostic logging for agent communication and parsing."""
        logger.info(f"🔍 DIAGNOSTIC [{stage}]: {json.dumps(data, indent=2)}")

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

    def _parse_compilation_agent_output(
        self, compilation_result: Any, engine: LaTeXEngine
    ) -> CompilationAgentOutput:
        """Enhanced parsing of compilation agent output with detailed diagnostics."""
        self._compilation_diagnostics.increment("total_attempts")

        # Validate the response structure first
        validation = self._validate_agent_response(compilation_result, "COMPILATION")

        try:
            # Check if we have a properly structured response
            if validation["has_final_output"] and isinstance(
                compilation_result.final_output, CompilationAgentOutput
            ):
                agent_output = compilation_result.final_output
                logger.info("✅ Successfully parsed CompilationAgentOutput directly")
                return agent_output

            # Try JSON parsing approach
            logger.info("🔄 Attempting JSON parsing of compilation result")
            output_text = (
                str(compilation_result.final_output)
                if hasattr(compilation_result, "final_output")
                else str(compilation_result)
            )

            # Log what we're trying to parse
            self._log_agent_diagnostics(
                "JSON_PARSING_INPUT",
                {
                    "content_length": len(output_text),
                    "content_preview": (
                        output_text[:200] + "..."
                        if len(output_text) > 200
                        else output_text
                    ),
                    "search_for_json": True,
                },
            )

            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                self._log_agent_diagnostics(
                    "EXTRACTED_JSON",
                    {"json_content": json_str, "json_length": len(json_str)},
                )

                result_data = json.loads(json_str)
                agent_output = CompilationAgentOutput(
                    success=result_data.get("success", False),
                    compilation_time=result_data.get("compilation_time", 0.0),
                    error_message=result_data.get("error_message"),
                    log_summary=result_data.get("log_summary", ""),
                    engine_used=result_data.get("engine_used", engine.value),
                    output_path=result_data.get("output_path", ""),
                )

                logger.info("✅ Successfully parsed JSON to CompilationAgentOutput")
                self._log_agent_diagnostics(
                    "JSON_PARSING_SUCCESS",
                    {
                        "success": agent_output.success,
                        "compilation_time": agent_output.compilation_time,
                        "has_error_message": bool(agent_output.error_message),
                        "log_summary_length": len(agent_output.log_summary or ""),
                    },
                )
                return agent_output

            # Fallback parsing failed
            logger.warning("❌ JSON parsing failed - no valid JSON found")
            self._compilation_diagnostics.increment("parsing_failures")

            agent_output = CompilationAgentOutput(
                success=False,
                compilation_time=0.0,
                error_message="Failed to parse compilation result - no valid JSON found",
                log_summary=output_text,
                engine_used=engine.value,
                output_path="",
            )

            self._log_agent_diagnostics(
                "PARSING_FALLBACK",
                {"reason": "no_valid_json", "fallback_log_length": len(output_text)},
            )

            return agent_output

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            self._compilation_diagnostics.increment("parsing_failures")

            self._log_agent_diagnostics(
                "JSON_DECODE_ERROR",
                {
                    "error": str(e),
                    "error_position": getattr(e, "pos", None),
                    "error_line": getattr(e, "lineno", None),
                },
            )

            agent_output = CompilationAgentOutput(
                success=False,
                compilation_time=0.0,
                error_message=f"JSON parsing failed: {str(e)}",
                log_summary=(
                    str(compilation_result.final_output)
                    if hasattr(compilation_result, "final_output")
                    else str(compilation_result)
                ),
                engine_used=engine.value,
                output_path="",
            )
            return agent_output

        except Exception as e:
            logger.error(f"❌ Compilation agent parsing failed with exception: {e}")
            self._compilation_diagnostics.increment("parsing_failures")

            self._log_agent_diagnostics(
                "PARSING_EXCEPTION", {"error_type": type(e).__name__, "error": str(e)}
            )

            agent_output = CompilationAgentOutput(
                success=False,
                compilation_time=0.0,
                error_message=f"Parsing exception: {str(e)}",
                log_summary=str(compilation_result),
                engine_used=engine.value,
                output_path="",
            )
            return agent_output

    def _parse_error_fixing_agent_output(
        self, error_fixing_result: Any
    ) -> ErrorFixingAgentOutput:
        """Enhanced parsing of error fixing agent output with detailed diagnostics."""
        self._compilation_diagnostics.increment("error_fixing_attempts")

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
                    self._compilation_diagnostics.increment("successful_fixes")

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

    def _extract_log_tail(self, tex_file_path: Path) -> str:
        """Extract log content from the '[Loading MPS to PDF converter' line to the end.

        Args:
            tex_file_path: Path to the .tex file

        Returns:
            String containing the log content from the MPS converter line to the end, or error message
        """
        # Try multiple possible log file locations
        possible_log_files = [
            tex_file_path.with_suffix(".log"),  # Same directory as .tex
            tex_file_path.parent
            / f"{tex_file_path.stem}.log",  # Explicit parent directory
            Path("output") / f"{tex_file_path.stem}.log",  # Output directory
        ]

        log_file = None
        for log_path in possible_log_files:
            if log_path.exists():
                log_file = log_path
                break

        if not log_file:
            return "No log file found in any expected location"

        try:
            with open(log_file, encoding="utf-8", errors="ignore") as f:
                all_lines = f.readlines()

                # Look for the line that starts with "[Loading MPS to PDF converter"
                mps_converter_line_index = None
                for i, line in enumerate(all_lines):
                    if line.strip().startswith("[Loading MPS to PDF converter"):
                        mps_converter_line_index = i
                        break

                if mps_converter_line_index is not None:
                    # Extract from the MPS converter line to the end
                    tail_lines = all_lines[mps_converter_line_index:]
                    logger.info(
                        f"Found MPS converter line at index {mps_converter_line_index}, extracting {len(tail_lines)} lines"
                    )
                else:
                    # Fallback to entire log file if MPS converter line not found
                    logger.warning(
                        "MPS converter line not found, using entire log file"
                    )
                    tail_lines = all_lines

                return "".join(tail_lines)
        except Exception as e:
            return f"Error reading log file {log_file}: {str(e)}"

    def _check_pdf_generation_from_log(self, tex_file_path: Path) -> bool:
        """Check if PDF was successfully generated by analyzing the log file.

        Args:
            tex_file_path: Path to the .tex file

        Returns:
            True if PDF was generated successfully, False otherwise
        """
        # Try multiple possible log file locations
        possible_log_files = [
            tex_file_path.with_suffix(".log"),  # Same directory as .tex
            tex_file_path.parent
            / f"{tex_file_path.stem}.log",  # Explicit parent directory
            Path("output") / f"{tex_file_path.stem}.log",  # Output directory
        ]

        log_file = None
        for log_path in possible_log_files:
            if log_path.exists():
                log_file = log_path
                break

        if not log_file:
            logger.warning("No log file found to check PDF generation status")
            return False

        try:
            with open(log_file, encoding="utf-8", errors="ignore") as f:
                log_content = f.read()

            # Check for fatal error that prevents PDF generation
            if "Fatal error occurred, no output PDF file produced!" in log_content:
                logger.info(
                    "Log indicates: Fatal error occurred, no output PDF file produced!"
                )
                return False

            # Check for successful compilation indicators
            if "Output written on" in log_content and ".pdf" in log_content:
                logger.info("Log indicates: Output written on [filename].pdf")
                return True

            # If no clear indicators, assume failure
            logger.info("Log does not contain clear PDF generation indicators")
            return False

        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {str(e)}")
            return False

    def _show_tex_file_changes(
        self, tex_file_path: Path, error_fixing_result: ErrorFixingAgentOutput
    ) -> None:
        """Show details of changes made to the .tex file by the error fixing agent.

        Args:
            tex_file_path: Path to the .tex file
            error_fixing_result: Result from the error fixing agent
        """
        if not error_fixing_result.success or not error_fixing_result.corrected_content:
            logger.info("No corrected content was provided by the error fixing agent")
            return

        logger.info(
            f"Error fixing agent provided corrected content for {tex_file_path}:"
        )
        logger.info(f"{error_fixing_result}")
        logger.info(
            f"Corrected content length: {len(error_fixing_result.corrected_content)} characters"
        )

    def _create_timestamped_version(self, tex_file_path: Path) -> Path:
        """Create a timestamped backup version of the .tex file for tracking changes.

        Args:
            tex_file_path: Path to the current .tex file (may already be timestamped)

        Returns:
            Path to the new timestamped version
        """
        # Generate formatted timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract the original filename stem (remove any existing timestamps)
        # Pattern: originalname_YYYYMMDD_HHMMSS -> originalname
        original_stem = tex_file_path.stem
        # Remove any existing timestamp pattern (_YYYYMMDD_HHMMSS)
        original_stem = re.sub(r"_\d{8}_\d{6}$", "", original_stem)

        suffix = tex_file_path.suffix
        new_filename = f"{original_stem}_{timestamp}{suffix}"

        # Create new path in the same directory
        timestamped_path = tex_file_path.parent / new_filename

        try:
            # Copy the current content to the new timestamped file
            with open(tex_file_path, encoding="utf-8") as source:
                content = source.read()

            with open(timestamped_path, "w", encoding="utf-8") as target:
                target.write(content)

            logger.info(f"Created timestamped version: {timestamped_path.name}")
            return timestamped_path

        except Exception as e:
            logger.error(f"Failed to create timestamped version: {str(e)}")
            return tex_file_path  # Return original path if backup fails

    def _create_compile_latex_tool(self) -> Any:
        """Create a function tool for LaTeX compilation that works with all models."""

        async def compile_latex_tool(
            command: str, tex_file_path: str, output_dir: str
        ) -> str:
            """Compile LaTeX file using shell command.

            Args:
                command: The LaTeX compilation command to execute
                tex_file_path: Path to the .tex file
                output_dir: Directory where PDF should be generated

            Returns:
                JSON string with compilation result
            """
            try:
                import json
                import time

                start_time = time.time()

                # Execute the command
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                stdout, _ = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )

                compilation_time = time.time() - start_time
                output_text = stdout.decode("utf-8", errors="ignore")

                # Check if PDF was created using both file existence and log analysis
                pdf_path = Path(output_dir) / f"{Path(tex_file_path).stem}.pdf"
                pdf_exists = pdf_path.exists() and pdf_path.stat().st_size > 0

                # Also check the log file for PDF generation status
                pdf_generated_from_log = self._check_pdf_generation_from_log(
                    Path(tex_file_path)
                )

                # Success if PDF exists AND log does not indicate fatal error
                # If log shows "Fatal error occurred, no output PDF file produced!", then it's a failure
                compilation_success = pdf_exists and pdf_generated_from_log

                logger.info(f"PDF file exists: {pdf_exists}")
                logger.info(
                    f"PDF generation confirmed by log: {pdf_generated_from_log}"
                )
                logger.info(f"Compilation success: {compilation_success}")

                result = {
                    "success": compilation_success,
                    "compilation_time": compilation_time,
                    "error_message": (
                        None
                        if compilation_success
                        else f"LaTeX compilation failed (exit code: {process.returncode})"
                    ),
                    "log_summary": output_text,
                    "engine_used": command.split()[0],
                    "output_path": str(pdf_path) if pdf_exists else "",
                    "return_code": process.returncode,
                }

                return json.dumps(result)

            except TimeoutError:
                return json.dumps(
                    {
                        "success": False,
                        "error_message": f"LaTeX compilation timed out after {self.timeout} seconds",
                        "compilation_time": 0.0,
                        "return_code": 1,
                    }
                )
            except Exception as e:
                return json.dumps(
                    {
                        "success": False,
                        "error_message": f"Tool error: {str(e)}",
                        "compilation_time": 0.0,
                        "return_code": 1,
                    }
                )

        return function_tool(compile_latex_tool)

    def check_latex_installation(
        self, engine: LaTeXEngine = LaTeXEngine.PDFLATEX
    ) -> bool:
        """Check if LaTeX is installed and accessible.

        Args:
            engine: LaTeX engine to check

        Returns:
            True if LaTeX is available, False otherwise
        """
        try:
            import subprocess

            result = subprocess.run(
                [engine.value, "--version"], capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def compile_from_request(
        self, request: CompileLaTeXRequest
    ) -> CompileLaTeXResponse:
        """Compile LaTeX file to PDF using Pydantic request model.

        Args:
            request: LaTeX to PDF compilation request

        Returns:
            Compilation response with PDF URL or error message
        """
        if not self.config:
            return CompileLaTeXResponse(
                status=ConversionStatus.FAILED,
                pdf_url=None,
                error_message="Server configuration not provided to LaTeX compiler",
            )

        try:
            logger.info(f"Starting LaTeX to PDF compilation for {request.tex_filename}")

            # Check if LaTeX file exists
            tex_path = self.config.output_dir / request.tex_filename
            if not tex_path.exists():
                return CompileLaTeXResponse(
                    status=ConversionStatus.FAILED,
                    pdf_url=None,
                    error_message=f"LaTeX file not found: {request.tex_filename}",
                )

            # Determine output filename
            output_filename = (
                request.output_filename
                or f"{request.tex_filename.replace('.tex', '')}.pdf"
            )
            if not output_filename.endswith(".pdf"):
                output_filename += ".pdf"

            output_path = self.config.output_dir / output_filename

            # Use intelligent agent for LaTeX compilation
            logger.info("Using intelligent agent for LaTeX compilation")
            compilation_result = await self.compile_with_agent(
                tex_path, output_path, request.latex_engine
            )

            # Add compilation metadata
            metadata = {
                "tex_filename": request.tex_filename,
                "output_filename": output_filename,
                "latex_engine": request.latex_engine.value,
                "compilation_time": compilation_result.compilation_time,
                "compilation_method": "agent",
            }

            if not compilation_result.success:
                return CompileLaTeXResponse(
                    status=ConversionStatus.FAILED,
                    pdf_url=None,
                    error_message=f"LaTeX compilation failed: {compilation_result.error_message}",
                    metadata=metadata,
                )

            # Generate PDF resource URI
            pdf_url = f"cv-writer://pdf/{output_filename}"

            logger.info(
                f"LaTeX to PDF compilation completed successfully: {output_filename}"
            )

            return CompileLaTeXResponse(
                status=ConversionStatus.SUCCESS,
                pdf_url=pdf_url,
                error_message=None,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Unexpected error in LaTeX to PDF compilation: {e}")
            return CompileLaTeXResponse(
                status=ConversionStatus.FAILED,
                pdf_url=None,
                error_message=f"Unexpected error: {str(e)}",
            )

    def _create_file_reading_tool(self) -> Any:
        """Create a function tool for reading LaTeX file content only."""
        from .models import FileOperationResult, FileOperationType

        async def file_reading_tool(file_path: str) -> str:
            """Read LaTeX file content from a given path.

            Args:
                file_path: Path to the .tex file

            Returns:
                JSON string containing the file content or error
            """
            try:
                file_path_obj = Path(file_path)

                if not file_path_obj.exists():
                    result = FileOperationResult.create_error_file_not_found(
                        file_path, FileOperationType.READ
                    )
                    return result.to_json()

                file_content = file_path_obj.read_text(encoding="utf-8")
                result = FileOperationResult.create_success_read(
                    file_path, file_content, len(file_content.splitlines())
                )
                return result.to_json()

            except Exception as e:
                result = FileOperationResult.create_error_tool_exception(
                    file_path, FileOperationType.READ, e
                )
                return result.to_json()

        return function_tool(file_reading_tool)

    def _read_moderncv_user_guide(self) -> str:
        """Read the moderncv user guide from the input directory.

        Returns:
            Content of the moderncv user guide as a string
        """
        try:
            # Try to find the moderncv user guide in the context/latex directory
            latex_dir = Path("context") / "latex"
            if self.config and self.config.templates_dir:
                latex_dir = self.config.templates_dir

            user_guide_path = latex_dir / "moderncv_userguide.txt"

            if not user_guide_path.exists():
                logger.warning(f"ModernCV user guide not found at {user_guide_path}")
                return "ModernCV user guide not available"

            with open(user_guide_path, encoding="utf-8") as f:
                content = f.read()
                logger.info(
                    f"Successfully loaded ModernCV user guide ({len(content)} characters)"
                )
                return content

        except Exception as e:
            logger.error(f"Error reading ModernCV user guide: {e}")
            return "Error loading ModernCV user guide"

    def _read_moderncv_template(self) -> str:
        """Read the moderncv template from the context directory.

        Returns:
            Content of the moderncv template as a string
        """
        try:
            # Try to find the moderncv template in the context directory
            templates_dir = Path("context/latex")
            if self.config and self.config.templates_dir:
                templates_dir = self.config.templates_dir / "latex"

            template_path = templates_dir / "moderncv_template.tex"

            if not template_path.exists():
                logger.warning(f"ModernCV template not found at {template_path}")
                return "ModernCV template not available"

            with open(template_path, encoding="utf-8") as f:
                content = f.read()
                logger.info(
                    f"Successfully loaded ModernCV template ({len(content)} characters)"
                )
                return content

        except Exception as e:
            logger.error(f"Error reading ModernCV template: {e}")
            return "Error loading ModernCV template"

    def _create_error_fixing_agent(self) -> Agent:
        """Create an agent specialized in fixing LaTeX compilation errors.

        This agent analyzes LaTeX compilation logs and fixes common errors
        in the LaTeX source code using the ModernCV user guide and template for reference.

        Returns:
            An Agent configured for LaTeX error fixing (no file reading tool needed)
        """
        # Read the ModernCV user guide and template
        moderncv_guide = self._read_moderncv_user_guide()
        moderncv_template = self._read_moderncv_template()

        # Define the error fixing agent instructions
        error_fixing_instructions = f"""
You are an expert in fixing LaTeX compilation errors and an expert in using the ModernCV LaTeX package. As a guide, follow the process and rules below adapting them when required to the specific LaTeX file you are fixing:

<MODERNCV_REFERENCE>
{moderncv_guide}
</MODERNCV_REFERENCE>

<MODERNCV_TEMPLATE>
{moderncv_template}
</MODERNCV_TEMPLATE>

<PROCESS>
1. Analyze the LaTeX file content provided in the context to understand the current LaTeX document structure
2. Identify root cause or causes of compilation errors based on the error log provided
3. Analyze the ModernCV user guide for proper command syntax
4. Use the ModernCV template as a working reference for correct LaTeX structure and command usage
5. Apply all the necessary minimal syntax fixes to the LaTeX file content
6. Return the corrected LaTeX file content with all fixes applied
</PROCESS>

<RULES>
- Fix only structural syntax issues
- Preserve all original content and formatting
- Use ModernCV guide for proper command syntax when required. Otherwise, use your own LaTeX knowledge.
- Reference the ModernCV template to see working examples of proper LaTeX structure and command usage
- Return the corrected LaTeX file content with all fixes applied
- Focus on specific errors mentioned in the context provided
- The LaTeX file content will be provided in the context, no need to read files
</RULES>
"""

        return Agent(
            name="LaTeX Error Fixing Agent",
            instructions=error_fixing_instructions,
            tools=[],  # No tools needed - content passed in prompt
            model="gpt-4.1-mini",
            output_type=ErrorFixingAgentOutput,
        )

    def create_compilation_agent(self) -> Agent:
        """Create an agent that can control the LaTeX compilation process.

        This agent can intelligently handle LaTeX compilation tasks using shell commands,
        including choosing the appropriate LaTeX engine and managing compilation results.

        Returns:
            An Agent configured with function tool for LaTeX compilation
        """
        return Agent(
            name="LaTeX Compilation Agent",
            instructions=(
                "You are a specialized LaTeX compilation agent. Your role is to:"
                "\n1. Compile LaTeX files to PDF using shell commands"
                "\n2. Choose the right LaTeX engine based on the document requirements"
                "\n3. Execute LaTeX compilation commands using the compile_latex_tool"
                "\n4. Monitor compilation results and provide detailed feedback"
                "\n\nWhen compiling:"
                "\n- Use pdflatex for standard documents"
                "\n- Use xelatex for documents with Unicode or special fonts"
                "\n- Use lualatex for advanced Lua scripting features"
                "\n- Always use the compile_latex_tool to execute LaTeX commands"
                "\n- Check exit codes to determine success/failure"
                "\n- Analyze compilation output for errors and warnings"
                "\n\nIMPORTANT: You must use the compile_latex_tool to execute LaTeX compilation commands."
                "Do not attempt to compile manually. Always use the tool and report the result."
                "\n\nProvide a clear explanation of what happened during compilation, including:"
                "\n- Whether the compilation was successful (exit code 0)"
                "\n- How long it took"
                "\n- Any errors encountered and their nature"
                "\n- A summary of the compilation output highlighting key information"
            ),
            tools=[self._create_compile_latex_tool()],
            model="gpt-4.1-mini",
            output_type=CompilationAgentOutput,
        )

    async def _orchestrate_compilation(
        self,
        tex_file_path: Path,
        output_path: Path,
        engine: LaTeXEngine,
        max_attempts: int = 3,
        user_instructions: str | None = None,
    ) -> LaTeXCompilationResult:
        """Orchestrate the compilation process with error fixing.

        This function coordinates between the compilation agent and error fixing agent
        to achieve successful LaTeX compilation through iterative error fixing.

        Args:
            tex_file_path: Path to the .tex file to compile
            output_path: Path where the PDF should be saved
            engine: LaTeX engine to use
            max_attempts: Maximum number of compilation attempts
            user_instructions: Optional additional instructions for the agents

        Returns:
            Final compilation result
        """
        compilation_agent = self.create_compilation_agent()
        error_fixing_agent = self._create_error_fixing_agent()

        current_tex_path = tex_file_path  # Current file being compiled
        final_result = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"🔄 COMPILATION ATTEMPT {attempt}/{max_attempts}")
            logger.info(f"📄 COMPILING FILE: {current_tex_path}")

            # Step 1: Compile with compilation agent using shell commands
            tex_file_abs = current_tex_path.absolute()
            output_dir_abs = output_path.parent.absolute()

            compilation_prompt = f"""
            Please compile the LaTeX file using the compile_latex_tool with these parameters:

            command: "{engine} -interaction=nonstopmode -output-directory {output_dir_abs} {tex_file_abs}"
            tex_file_path: "{tex_file_abs}"
            output_dir: "{output_dir_abs}"

            This will compile the LaTeX file {tex_file_abs} and output the PDF to {output_dir_abs}.

            After execution, please:
            1. Check if the compilation was successful (exit code 0)
            2. Report any errors if the compilation failed
            3. Verify that the PDF file was created
            4. Provide a summary of the compilation process
            """

            if user_instructions:
                compilation_prompt += f"\nAdditional instructions: {user_instructions}"

            try:
                compilation_result = await Runner.run(
                    compilation_agent, compilation_prompt
                )

                # Use enhanced parsing method
                agent_output = self._parse_compilation_agent_output(
                    compilation_result, engine
                )

                # Simplified success check - rely on agent's determination
                if agent_output.success:
                    self._compilation_diagnostics.increment("successful_compilations")
                    logger.info(f"Compilation successful on attempt {attempt}")

                    # Use the actual PDF path from the compilation result if available
                    actual_output_path = output_path
                    if (
                        hasattr(agent_output, "output_path")
                        and agent_output.output_path
                    ):
                        actual_output_path = Path(agent_output.output_path)
                        logger.info(
                            f"Using actual PDF path from compilation result: {actual_output_path}"
                        )

                    # Create final successful result
                    final_result = LaTeXCompilationResult(
                        success=True,
                        compilation_time=agent_output.compilation_time,
                        log_output=agent_output.log_summary,
                        output_path=actual_output_path,
                    )
                    break
                else:
                    # Track failed compilation
                    self._compilation_diagnostics.increment("failed_compilations")
                    # Log complete CompilationAgentOutput information using string representation
                    logger.error(f"Compilation failed on attempt {attempt}")
                    logger.error(f"{agent_output}")

                    if attempt < max_attempts:
                        # Step 2: Fix errors with error fixing agent
                        logger.info(
                            f"Attempting to fix errors with error fixing agent (attempt {attempt})"
                        )

                        # Extract focused error context with validation
                        error_context = self._extract_focused_error_context(
                            current_tex_path, agent_output.log_summary
                        )
                        logger.info(
                            f"Extracted focused error context: {error_context['error_count']} errors, {error_context['warning_count']} warnings"
                        )

                        # Use the validated log excerpt or fallback to basic extraction
                        if (
                            error_context["validation_status"] == "success"
                            and error_context["raw_log_excerpt"]
                        ):
                            log_content = error_context["raw_log_excerpt"]
                        else:
                            # Fallback to original method if enhanced extraction fails
                            log_content = self._extract_log_tail(current_tex_path)
                            logger.warning("Using fallback log extraction method")

                        # Read the LaTeX file content to pass directly in the prompt
                        try:
                            latex_content = Path(current_tex_path).read_text(
                                encoding="utf-8"
                            )
                            logger.info(
                                f"Successfully read LaTeX file content ({len(latex_content)} characters)"
                            )
                        except Exception as e:
                            logger.error(f"Failed to read LaTeX file content: {e}")
                            latex_content = f"Error reading LaTeX file: {e}"

                        # Prepare line-numbered LaTeX content for precise error fixing
                        numbered_latex_content, line_mapping = (
                            self._prepare_numbered_latex_for_agent(latex_content)
                        )
                        latex_error_guide = self._create_latex_error_guide(
                            error_context
                        )
                        log_error_analysis = self._analyze_latex_log_errors(log_content)

                        # Build enhanced error fixing prompt with line numbers
                        focused_errors_text = ""
                        if error_context["focused_errors"]:
                            focused_errors_text = "\n<FOCUSED_ERRORS>\n"
                            for i, error in enumerate(
                                error_context["focused_errors"], 1
                            ):
                                focused_errors_text += f"{i}. {error['description']}: {error['error_text']}"
                                if error["line_number"]:
                                    focused_errors_text += (
                                        f" (Line {error['line_number']})"
                                    )
                                focused_errors_text += (
                                    f"\n   Context: {error['context'][:200]}...\n"
                                )
                            focused_errors_text += "</FOCUSED_ERRORS>\n"

                        critical_errors_text = ""
                        if error_context["critical_errors"]:
                            critical_errors_text = "\n<CRITICAL_ERRORS>\n"
                            for error in error_context["critical_errors"]:
                                critical_errors_text += f"- {error}\n"
                            critical_errors_text += "</CRITICAL_ERRORS>\n"

                        error_fixing_prompt = f"""
Fix the following LaTeX compilation errors by analyzing the line-numbered content:

{latex_error_guide}

<LATEX_FILE_CONTENT_WITH_LINE_NUMBERS>
{numbered_latex_content}
</LATEX_FILE_CONTENT_WITH_LINE_NUMBERS>

<ERROR_ANALYSIS>
Target File: {current_tex_path}
Compilation Status: {agent_output.error_message}
Total Errors Found: {error_context['error_count']}
Warnings: {error_context['warning_count']}
Log Status: {error_context['validation_status']}
Line-Specific Errors Found: {len(log_error_analysis['line_specific_errors'])}
</ERROR_ANALYSIS>
{focused_errors_text}
{critical_errors_text}
<FULL_LOG_EXCERPT>
{log_content}
</FULL_LOG_EXCERPT>

CRITICAL INSTRUCTIONS:
1. The content above shows LINE NUMBERS (e.g., "045: content")
2. When log shows "l.45 \\end{{itemize}}}}", look at line 045 in the numbered content
3. Fix the exact issues at those specific lines and surrounding context
4. IMPORTANT: Return your corrected content WITH the same line number format
5. Do not remove the line numbers - I will process them after
6. Focus on brace matching, environment nesting, and command syntax
7. Pay special attention to the problematic lines identified in the error guide

Return the corrected content maintaining the "XXX: content" line number format.
"""

                        # Show the enhanced context that will be sent to the fixing agent
                        logger.info("=" * 80)
                        logger.info(
                            "ENHANCED LINE-NUMBERED CONTEXT FOR ERROR FIXING AGENT:"
                        )
                        logger.info("=" * 80)
                        logger.info(
                            f"ORIGINAL LATEX CONTENT LENGTH: {len(latex_content)} characters"
                        )
                        logger.info(
                            f"NUMBERED LATEX CONTENT LENGTH: {len(numbered_latex_content)} characters"
                        )
                        logger.info(f"LINE MAPPING ENTRIES: {len(line_mapping)}")
                        logger.info(
                            f"ERROR ANALYSIS: {error_context['error_count']} errors, {error_context['warning_count']} warnings"
                        )
                        logger.info(
                            f"LOG VALIDATION STATUS: {error_context['validation_status']}"
                        )
                        logger.info(
                            f"FOCUSED ERRORS COUNT: {len(error_context['focused_errors'])}"
                        )
                        logger.info(
                            f"CRITICAL ERRORS COUNT: {len(error_context['critical_errors'])}"
                        )
                        logger.info(
                            f"LINE-SPECIFIC ERRORS: {len(log_error_analysis['line_specific_errors'])}"
                        )
                        logger.info(f"CURRENT COMPILATION TARGET: {current_tex_path}")
                        logger.info(
                            f"LOG CONTENT LENGTH: {len(log_content)} characters"
                        )
                        logger.info("=" * 80)
                        logger.info(
                            "CALLING ERROR FIXING AGENT WITH LINE-NUMBERED CONTENT..."
                        )
                        logger.info("=" * 80)

                        try:
                            logger.info(
                                f"Error fixing agent prompt: {error_fixing_prompt}"
                            )
                            error_fixing_result = await Runner.run(
                                error_fixing_agent, error_fixing_prompt
                            )

                            logger.info("=" * 80)
                            logger.info("ERROR FIXING AGENT RESPONSE RECEIVED")
                            logger.info("=" * 80)
                            logger.info(f"Raw result type: {type(error_fixing_result)}")
                            logger.info(f"Raw result: {error_fixing_result}")

                            # Use enhanced parsing method
                            fixing_output = self._parse_error_fixing_agent_output(
                                error_fixing_result
                            )

                            # Log error fixing agent output using string representation
                            logger.info("Error fixing agent response:")
                            logger.info(f"{fixing_output}")

                            # Show where changes were made in the .tex file
                            self._show_tex_file_changes(current_tex_path, fixing_output)

                            if (
                                fixing_output.success
                                and fixing_output.corrected_content
                            ):
                                logger.info(
                                    f"Error fixing completed: {fixing_output.total_fixes} fixes applied"
                                )

                                # Log details of each fix using string representation
                                for fix in fixing_output.fixes_applied:
                                    logger.info(f"  - {fix}")

                                # Create new timestamped version with the corrected content
                                try:
                                    # Extract clean LaTeX content by removing line numbers from agent response
                                    logger.info(
                                        "Processing line-numbered agent response to extract clean LaTeX content..."
                                    )
                                    clean_corrected_content = (
                                        self._extract_corrected_content_from_agent(
                                            fixing_output.corrected_content
                                        )
                                    )
                                    logger.info(
                                        "Line-numbered content processing complete:"
                                    )
                                    logger.info(
                                        f"  Original length: {len(fixing_output.corrected_content)} characters"
                                    )
                                    logger.info(
                                        f"  Clean length: {len(clean_corrected_content)} characters"
                                    )
                                    logger.info(
                                        f"  Line mapping entries: {len(line_mapping)}"
                                    )

                                    # Create timestamped version based on current file (which may be a previous timestamped version)
                                    timestamped_version = (
                                        self._create_timestamped_version(
                                            current_tex_path
                                        )
                                    )
                                    timestamped_version.write_text(
                                        clean_corrected_content, encoding="utf-8"
                                    )
                                    logger.info(
                                        f"Successfully saved corrected content to: {timestamped_version}"
                                    )

                                    # Verify that the timestamped file was actually saved and exists
                                    if (
                                        timestamped_version.exists()
                                        and timestamped_version.stat().st_size > 0
                                    ):
                                        logger.info(
                                            f"File verification successful: {timestamped_version} exists and has content ({timestamped_version.stat().st_size} bytes)"
                                        )

                                        # Update current_tex_path to use the new timestamped version for next compilation attempt
                                        current_tex_path = timestamped_version
                                        logger.info(
                                            f"🔄 TARGET UPDATED: Next compilation will use: {current_tex_path}"
                                        )

                                        logger.info(
                                            "Proceeding to next compilation attempt"
                                        )
                                    else:
                                        logger.error(
                                            f"File verification failed: {timestamped_version} does not exist or is empty"
                                        )
                                        break

                                except Exception as e:
                                    logger.error(
                                        f"Failed to save corrected content: {str(e)}"
                                    )
                                    break
                            else:
                                logger.warning(
                                    "Error fixing did not modify the file or failed"
                                )

                            # Check if the file still exists
                            if not current_tex_path.exists():
                                logger.error(
                                    "Error fixing failed - file not found after fixing attempt"
                                )
                                break

                        except Exception as e:
                            logger.error(f"Error fixing failed: {e}")
                            break
                    else:
                        # Final attempt failed - log complete result information using string representation
                        logger.error(f"All {max_attempts} compilation attempts failed")
                        logger.error("Final CompilationAgentOutput details:")
                        logger.error(f"{agent_output}")

                        final_result = LaTeXCompilationResult(
                            success=False,
                            error_message=f"Compilation failed after {max_attempts} attempts. Last error: {agent_output.error_message}",
                            compilation_time=agent_output.compilation_time,
                            log_output=agent_output.log_summary,
                        )

            except Exception as e:
                logger.error(
                    f"Compilation attempt {attempt} failed with exception: {e}"
                )
                logger.error(f"Exception {type(e).__name__} details: {str(e)}")

                final_result = LaTeXCompilationResult(
                    success=False,
                    error_message=f"Compilation failed after {attempt}/{max_attempts} attempts. Last exception: {str(e)}",
                    compilation_time=0.0,
                )

        if final_result is None:
            # This shouldn't happen, but just in case
            logger.error("Orchestration failed - no result generated")
            final_result = LaTeXCompilationResult(
                success=False,
                error_message="Orchestration failed - no result generated",
                compilation_time=0.0,
            )

        # Log final result summary using string representation
        logger.info("Orchestration completed. Final result:")
        logger.info(f"{final_result}")

        # Log comprehensive diagnostics summary
        logger.info(str(self._compilation_diagnostics))

        return final_result

    async def compile_with_agent(
        self,
        tex_file_path: Path,
        output_path: Path,
        engine: LaTeXEngine = LaTeXEngine.PDFLATEX,
        user_instructions: str | None = None,
        max_attempts: int = 5,
    ) -> LaTeXCompilationResult:
        """Compile LaTeX using intelligent agent orchestration with error fixing.

        This method uses a two-agent system:
        1. Compilation Agent: Handles the actual LaTeX compilation
        2. Error Fixing Agent: Analyzes and fixes LaTeX errors when compilation fails

        The process iterates up to max_attempts times, with error fixing between failed attempts.

        Args:
            tex_file_path: Path to the .tex file to compile
            output_path: Path where the PDF should be saved
            engine: LaTeX engine to use
            user_instructions: Optional additional instructions for the agents
            max_attempts: Maximum number of compilation attempts (default: 3)

        Returns:
            Final compilation result
        """
        logger.info(
            f"Starting orchestrated LaTeX compilation with {max_attempts} max attempts"
        )

        # Use the orchestration system
        compilation_result = await self._orchestrate_compilation(
            tex_file_path=tex_file_path,
            output_path=output_path,
            engine=engine,
            max_attempts=max_attempts,
            user_instructions=user_instructions,
        )

        logger.info(
            f"Orchestrated compilation completed. Success: {compilation_result.success}"
        )

        return compilation_result
