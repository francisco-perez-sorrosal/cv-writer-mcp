"""LaTeX compilation agent functionality.

This module contains the compilation agent and related utilities for handling
LaTeX compilation tasks using AI agents.
"""

import json
from pathlib import Path
from typing import Any

from agents import Agent, Runner
from loguru import logger

from .models import (
    CompilerAgentOutput,
    CompilationDiagnostics,
    OrchestrationResult,
    LaTeXEngine,
)
from .tools import latex2pdf_tool


class CompilationAgent:
    """Handles LaTeX compilation using AI agents."""

    def __init__(
        self, timeout: int = 30, diagnostics: CompilationDiagnostics | None = None
    ):
        """Initialize the compilation agent.

        Args:
            timeout: Compilation timeout in seconds
            diagnostics: Optional diagnostics tracker for monitoring compilation attempts
        """
        self.timeout = timeout
        self.diagnostics = diagnostics or CompilationDiagnostics()

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

    def create_compiler_agent(self) -> Agent:
        """Create an agent that can control the LaTeX compilation process.

        This agent can intelligently handle LaTeX compilation tasks using shell commands,
        including choosing the appropriate LaTeX engine and managing compilation results.

        Returns:
            An Agent configured with function tool for LaTeX compilation
        """
        return Agent(
            name="LaTeX Compiler Agent",
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
            tools=[latex2pdf_tool],
            model="gpt-4.1-mini",
            output_type=CompilerAgentOutput,
        )

    def parse_compiler_agent_output(
        self, compilation_result: Any, engine: LaTeXEngine
    ) -> CompilerAgentOutput:
        """Enhanced parsing of compilation agent output with detailed diagnostics."""
        self.diagnostics.increment("total_attempts")

        # Validate the response structure first
        validation = self._validate_agent_response(compilation_result, "COMPILATION")

        try:
            # Check if we have a properly structured response
            if validation["has_final_output"] and isinstance(
                compilation_result.final_output, CompilerAgentOutput
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
                agent_output = CompilerAgentOutput(
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
            self.diagnostics.increment("parsing_failures")

            agent_output = CompilerAgentOutput(
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
            self.diagnostics.increment("parsing_failures")

            self._log_agent_diagnostics(
                "JSON_DECODE_ERROR",
                {
                    "error": str(e),
                    "error_position": getattr(e, "pos", None),
                    "error_line": getattr(e, "lineno", None),
                },
            )

            agent_output = CompilerAgentOutput(
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
            self.diagnostics.increment("parsing_failures")

            self._log_agent_diagnostics(
                "PARSING_EXCEPTION", {"error_type": type(e).__name__, "error": str(e)}
            )

            agent_output = CompilerAgentOutput(
                success=False,
                compilation_time=0.0,
                error_message=f"Parsing exception: {str(e)}",
                log_summary=str(compilation_result),
                engine_used=engine.value,
                output_path="",
            )
            return agent_output

    async def compile_latex(
        self,
        tex_file_path: Path,
        output_path: Path,
        engine: LaTeXEngine = LaTeXEngine.PDFLATEX,
        user_instructions: str | None = None,
    ) -> OrchestrationResult:
        """Compile LaTeX using the compilation agent.

        Args:
            tex_file_path: Path to the .tex file to compile
            output_path: Path where the PDF should be saved
            engine: LaTeX engine to use
            user_instructions: Optional additional instructions for the agent

        Returns:
            Compilation result
        """
        logger.info("Starting LaTeX compilation with compilation agent")

        compilation_agent = self.create_compiler_agent()

        # Prepare compilation prompt
        tex_file_abs = tex_file_path.absolute()
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
            compilation_result = await Runner.run(compilation_agent, compilation_prompt)

            # Parse the agent output
            agent_output = self.parse_compiler_agent_output(
                compilation_result, engine
            )

            if agent_output.success:
                self.diagnostics.increment("successful_compilations")

                # Use the actual PDF path from the compilation result if available
                actual_output_path = output_path
                if hasattr(agent_output, "output_path") and agent_output.output_path:
                    actual_output_path = Path(agent_output.output_path)

                return OrchestrationResult(
                    success=True,
                    compilation_time=agent_output.compilation_time,
                    log_output=agent_output.log_summary,
                    output_path=actual_output_path,
                )
            else:
                # Track failed compilation
                self.diagnostics.increment("failed_compilations")

                return OrchestrationResult(
                    success=False,
                    error_message=agent_output.error_message,
                    compilation_time=agent_output.compilation_time,
                    log_output=agent_output.log_summary,
                )

        except Exception as e:
            self.diagnostics.increment("failed_compilations")
            logger.error(f"Compilation failed with exception: {e}")
            return OrchestrationResult(
                success=False,
                error_message=f"Compilation failed with exception: {str(e)}",
                compilation_time=0.0,
            )
