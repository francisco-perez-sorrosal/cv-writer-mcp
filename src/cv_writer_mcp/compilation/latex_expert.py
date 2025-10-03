"""LaTeX compilation functionality."""

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from ..models import CompletionStatus, ServerConfig
from .compiler_agent import CompilationAgent
from .error_agent import CompilationErrorAgent
from .models import (
    CompilationDiagnostics,
    CompileLaTeXRequest,
    CompileLaTeXResponse,
    LaTeXEngine,
    OrchestrationResult,
)


class LaTeXExpert:
    """Compiles LaTeX documents to PDF."""

    def __init__(self, config: ServerConfig):
        """Initialize the LaTeX compiler.

        Args:
            timeout: Compilation timeout in seconds
            config: Server configuration (optional, for high-level compilation)
        """
        self.timeout = config.latex_timeout
        self.config = config

        # Enhanced diagnostics tracking using Pydantic model
        self._compilation_diagnostics = CompilationDiagnostics()

        # Initialize agents
        self._compilation_agent = CompilationAgent(
            timeout=self.timeout, diagnostics=self._compilation_diagnostics
        )
        self._fixing_agent = CompilationErrorAgent(
            diagnostics=self._compilation_diagnostics, server_config=config
        )

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
            result = subprocess.run(
                [engine.value, "--version"], capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def compile_latex_file(
        self, request: CompileLaTeXRequest
    ) -> CompileLaTeXResponse:
        """Compile LaTeX file to PDF using intelligent agents.

        Args:
            request: LaTeX to PDF compilation request

        Returns:
            Compilation response with PDF URL or error message
        """
        if not self.config:
            return CompileLaTeXResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                message="Server configuration not provided to LaTeX compiler",
            )

        try:
            logger.info(f"Starting LaTeX to PDF compilation for {request.tex_filename}")

            # Check if LaTeX file exists
            tex_path = self.config.output_dir / request.tex_filename
            if not tex_path.exists():
                return CompileLaTeXResponse(
                    status=CompletionStatus.FAILED,
                    pdf_url=None,
                    message=f"LaTeX file not found: {request.tex_filename}",
                )

            # Use the output filename from the request (already processed in model_post_init)
            output_filename = request.output_filename
            output_path = self.config.output_dir / output_filename

            # Orchestrate LaTeX compilation with agents
            compilation_result = await self.orchestrate_compilation(
                tex_file_path=tex_path,
                output_path=output_path,
                engine=request.latex_engine,
                max_attempts=request.max_attempts,
                user_instructions=request.user_instructions,
            )

            if compilation_result.status != CompletionStatus.SUCCESS:
                return CompileLaTeXResponse(
                    status=CompletionStatus.FAILED,
                    pdf_url=None,
                    message=f"LaTeX compilation failed: {compilation_result.message}",
                )

            logger.info(f"LaTeX -> PDF conversion completed: {output_filename}")

            # Generate PDF resource URI
            return CompileLaTeXResponse(
                status=CompletionStatus.SUCCESS,
                pdf_url=f"cv-writer://pdf/{output_filename}",
                message=f"Successfully compiled {request.tex_filename} to {output_filename}",
            )

        except Exception as e:
            logger.error(f"Unexpected error in LaTeX to PDF compilation: {e}")
            return CompileLaTeXResponse(
                status=CompletionStatus.FAILED,
                pdf_url=None,
                message=f"Unexpected error: {str(e)}",
            )

    async def orchestrate_compilation(
        self,
        tex_file_path: Path,
        output_path: Path,
        engine: LaTeXEngine,
        max_attempts: int = 3,
        user_instructions: str | None = None,
    ) -> OrchestrationResult:
        """Compile LaTeX using intelligent agent orchestration with error fixing.

        This method uses a two-agent system:
        1. Compilation Agent: Handles the actual LaTeX compilation
        2. Error Fixing Agent: Analyzes and fixes LaTeX errors when compilation fails

        Coordinates between the compilation agent and error fixing agent
        to achieve successful LaTeX compilation through iterative error fixing.
        The process iterates up to max_attempts times, with error fixing between failed attempts.

        Args:
            tex_file_path: Path to the .tex file to compile
            output_path: Path where the PDF should be saved
            engine: LaTeX engine to use
            max_attempts: Maximum number of compilation attempts
            user_instructions: Optional additional instructions for the agents

        Returns:
            Final compilation result
        """
        # Error fixing agent is already initialized in __init__

        logger.info(
            f"Starting LaTeX -> PDF commpilation orchestration ({max_attempts} max attempts)"
        )

        current_tex_path = tex_file_path  # Current file being compiled
        final_result = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"🔄 COMPILATION ATTEMPT {attempt}/{max_attempts}")
            logger.info(f"📄 COMPILING FILE: {current_tex_path}")

            try:
                # Step 1: Compile with compilation agent using shell commands
                compilation_result = await self._compilation_agent.compile_latex(
                    current_tex_path, output_path, engine, user_instructions
                )

                # Success check - rely on agent's determination
                if compilation_result.status == CompletionStatus.SUCCESS:
                    logger.info(f"Compilation successful (Attempt {attempt}")

                    # Create final successful result
                    final_result = OrchestrationResult(
                        status=CompletionStatus.SUCCESS,
                        compilation_time=compilation_result.compilation_time,
                        log_output=compilation_result.log_output,
                        output_path=output_path,
                    )
                    break
                else:
                    # Failed compilation: - Log the result (counters already incremented by compiler_agent)
                    # Log complete compilation result information using string representation
                    logger.warning("--------------------------------")
                    logger.warning(f"Compilation failed (Attempt {attempt})")
                    logger.warning(f"{compilation_result}")
                    logger.warning("--------------------------------")

                    if attempt < max_attempts:
                        # Step 2: Fix errors with error fixing agent
                        logger.warning(
                            f"Attempt to fix errors with fixing agent (Attempt {attempt})"
                        )

                        try:
                            # Use the error fixing agent to fix errors
                            # Pass compilation_result directly which contains errors_found and exit_code
                            fixing_output, corrected_file_path = (
                                await self._fixing_agent.fix_errors(
                                    current_tex_path, compilation_result
                                )
                            )

                            if fixing_output.status == CompletionStatus.SUCCESS and corrected_file_path:
                                logger.info(
                                    f"Error fixing completed: {fixing_output.total_fixes} fixes applied"
                                )

                                # Log details of each fix
                                for fix in fixing_output.fixes_applied:
                                    logger.info(f"  - {fix}")

                                # Update current_tex_path to use the corrected file for next compilation attempt
                                current_tex_path = corrected_file_path
                                logger.info(
                                    f"🔄 TARGET UPDATED: Next compilation will use: {current_tex_path}"
                                )
                                logger.info("Proceeding to next compilation attempt")
                            else:
                                logger.warning(
                                    "Error fixing did not modify the file or failed"
                                )
                                break

                        except Exception as e:
                            logger.error(f"Error fixing failed: {e}")
                            break
                    else:
                        # Final attempt failed - log complete result information using string representation
                        logger.error(f"All {max_attempts} compilation attempts failed")
                        logger.error("Final compilation result details:")
                        logger.error(f"{compilation_result}")

                        final_result = OrchestrationResult(
                            status=CompletionStatus.FAILED,
                            message=f"Compilation failed after {max_attempts} attempts. Last error: {compilation_result.message}",
                            compilation_time=compilation_result.compilation_time,
                            log_output=compilation_result.log_output,
                        )

            except Exception as e:
                logger.error(f"Attempt {attempt} failed with exception: {e}")
                logger.error(f"Exception {type(e).__name__} details: {str(e)}")

                final_result = OrchestrationResult(
                    status=CompletionStatus.FAILED,
                    message=f"Compilation Failed after {attempt}/{max_attempts} attempts. Last exception: {str(e)}",
                    compilation_time=0.0,
                )

        if final_result is None:
            # This shouldn't happen, but just in case
            logger.error("Orchestration failed - no result generated")
            final_result = OrchestrationResult(
                status=CompletionStatus.FAILED,
                message="Orchestration failed - no result generated",
                compilation_time=0.0,
            )

        # Log final result summary using string representation
        logger.info("Orchestration completed. Final result:")
        logger.info(f"{final_result}")

        # Log comprehensive diagnostics summary
        logger.info(str(self._compilation_diagnostics))

        return final_result
