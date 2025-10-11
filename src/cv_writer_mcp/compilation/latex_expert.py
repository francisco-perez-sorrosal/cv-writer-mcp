"""LaTeX compilation functionality."""

import json
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from ..models import CompletionStatus, ServerConfig
from ..utils import (
    PeriodicProgressTicker,
    ProgressCallback,
    ProgressMapper,
    create_error_version,
)
from .compiler_agent import CompilationAgent
from .error_agent import CompilationErrorAgent
from .models import (
    CompilationDiagnostics,
    CompilationErrorOutput,
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
        self.timeout = config.latex_timeout_seconds
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

    async def _compile_step(
        self,
        tex_file_path: Path,
        output_path: Path,
        engine: LaTeXEngine,
        attempt: int,
        max_attempts: int,
        user_instructions: str | None = None,
        progress_callback: ProgressCallback = None,
    ) -> OrchestrationResult:
        """Execute a single compilation step.

        Args:
            tex_file_path: Path to .tex file to compile
            output_path: Path where PDF should be saved
            engine: LaTeX engine to use
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum number of attempts
            user_instructions: Optional user instructions
            progress_callback: Optional callback for progress reporting (0-100)

        Returns:
            OrchestrationResult with compilation outcome
        """
        logger.info("")
        logger.info("┌" + "─" * 68 + "┐")
        logger.info(f"│ 🔄 COMPILATION ATTEMPT {attempt}/{max_attempts}")
        logger.info(f"│ 📄 Input TEX: {tex_file_path.name}")
        logger.info(f"│ 📁 Output dir: {output_path.parent}")
        logger.info("└" + "─" * 68 + "┘")

        try:
            compilation_result = await self._compilation_agent.compile_latex(
                tex_file_path, output_path, engine, user_instructions, progress_callback=progress_callback
            )

            if compilation_result.status == CompletionStatus.SUCCESS:
                # Derive actual PDF path from the TEX file that was compiled
                # (since pdflatex creates PDF with same stem as TEX file)
                actual_pdf_path = output_path.parent / f"{tex_file_path.stem}.pdf"

                logger.info(f"✅ Compilation successful (Attempt {attempt})")
                logger.info(f"   📄 PDF created: {actual_pdf_path.name}")

                return OrchestrationResult(
                    status=CompletionStatus.SUCCESS,
                    compilation_time=compilation_result.compilation_time,
                    log_output=compilation_result.log_output,
                    output_path=actual_pdf_path,
                    errors_found=None,
                    exit_code=0,
                )
            else:
                logger.warning(f"❌ Compilation failed (Attempt {attempt})")
                logger.warning(f"{compilation_result}")
                return compilation_result

        except Exception as e:
            logger.error(f"❌ Compilation exception (Attempt {attempt}): {e}")
            return OrchestrationResult(
                status=CompletionStatus.FAILED,
                message=f"Compilation exception: {str(e)}",
                compilation_time=0.0,
                errors_found=[str(e)],
                exit_code=1,
            )

    async def _fix_step(
        self,
        tex_file_path: Path,
        compilation_result: OrchestrationResult,
        attempt: int,
        progress_callback: ProgressCallback = None,
    ) -> tuple[CompilationErrorOutput, Path | None]:
        """Execute a single error fixing step.

        Args:
            tex_file_path: Path to .tex file to fix
            compilation_result: Failed compilation result with errors
            attempt: Current attempt number (1-indexed)
            progress_callback: Optional callback for progress reporting (0-100)

        Returns:
            Tuple of (fixing output, corrected file path or None)
        """
        logger.info("─" * 70)
        logger.info(f"🔧 FIXING ERRORS (After Attempt {attempt})")
        logger.info("─" * 70)

        try:
            fixing_output, corrected_file_path = await self._fixing_agent.fix_errors(
                tex_file_path, compilation_result, progress_callback=progress_callback
            )

            if fixing_output.status == CompletionStatus.SUCCESS and corrected_file_path:
                logger.info(
                    f"✅ Fixed {fixing_output.total_fixes} error(s) → {corrected_file_path}"
                )
                for fix in fixing_output.fixes_applied:
                    logger.debug(f"  - {fix}")
            else:
                logger.warning("❌ Error fixing failed or made no changes")

            return fixing_output, corrected_file_path

        except Exception as e:
            logger.error(f"❌ Error fixing exception: {e}")
            return (
                CompilationErrorOutput(
                    status=CompletionStatus.FAILED,
                    corrected_content="",
                    total_fixes=0,
                    fixes_applied=[],
                    file_modified=False,
                    message=f"Fixing exception: {str(e)}",
                ),
                None,
            )

    def _create_failure_result(
        self, last_result: OrchestrationResult, reason: str
    ) -> OrchestrationResult:
        """Create a final failure result.

        Args:
            last_result: The last compilation result
            reason: Reason for final failure

        Returns:
            OrchestrationResult indicating final failure
        """
        return OrchestrationResult(
            status=CompletionStatus.FAILED,
            message=f"{reason}. Last error: {last_result.message}",
            compilation_time=last_result.compilation_time,
            log_output=last_result.log_output,
            errors_found=last_result.errors_found,
            exit_code=last_result.exit_code,
        )

    async def orchestrate_compilation(
        self,
        tex_file_path: Path,
        output_path: Path,
        engine: LaTeXEngine,
        max_attempts: int = 3,
        user_instructions: str | None = None,
        progress_callback: ProgressCallback = None,
    ) -> OrchestrationResult:
        """Compile LaTeX using intelligent agent orchestration with error fixing.

        This method uses a two-agent system:
        1. Compilation Agent: Handles the actual LaTeX compilation
        2. Error Fixing Agent: Analyzes and fixes LaTeX errors when compilation fails

        Uses deterministic flow pattern: compile → (if fail) → fix → compile
        The process iterates up to max_attempts times, with error fixing between failed attempts.

        Args:
            tex_file_path: Path to the .tex file to compile
            output_path: Path where the PDF should be saved
            engine: LaTeX engine to use
            max_attempts: Maximum number of compilation attempts
            user_instructions: Optional additional instructions for the agents
            progress_callback: Optional callback for progress reporting (0-100)

        Returns:
            Final compilation result
        """
        # Report start
        if progress_callback:
            await progress_callback(0)

        logger.info("=" * 70)
        logger.info("🚀 STARTING LATEX COMPILATION ORCHESTRATION")
        logger.info(f"   Max attempts: {max_attempts}")
        logger.info(f"   Initial file: {tex_file_path.name}")
        logger.info("=" * 70)

        current_file = tex_file_path

        for attempt in range(1, max_attempts + 1):
            # Create progress mapper for this attempt
            # Each attempt gets an equal portion of 0-100% range
            attempt_start = ((attempt - 1) * 100) // max_attempts
            attempt_end = (attempt * 100) // max_attempts
            attempt_progress = ProgressMapper(progress_callback, attempt_start, attempt_end)
            
            # Report start of attempt
            await attempt_progress.report(0)
            
            # Step 1: Attempt compilation (0-60% of attempt range)
            # Create progress mapper for compilation (10-60% of attempt)
            compile_progress = attempt_progress.create_sub_mapper(10, 60)
            
            compile_result = await self._compile_step(
                current_file,
                output_path,
                engine,
                attempt,
                max_attempts,
                user_instructions,
                progress_callback=compile_progress.report,
            )
            
            await attempt_progress.report(60)

            # Gate: Success? Done.
            if compile_result.status == CompletionStatus.SUCCESS:
                await attempt_progress.report(100)
                # Set final file paths (may be timestamped versions from error fixing)
                compile_result.final_tex_path = current_file
                # Derive PDF path from the final TEX file
                final_pdf_path = current_file.parent / f"{current_file.stem}.pdf"
                if final_pdf_path.exists():
                    compile_result.final_pdf_path = final_pdf_path

                logger.info("")
                logger.info("=" * 70)
                logger.info("✅ COMPILATION ORCHESTRATION SUCCEEDED")
                logger.info(f"   📄 Final TEX: {current_file.name}")
                if compile_result.final_pdf_path:
                    logger.info(
                        f"   📄 Final PDF: {compile_result.final_pdf_path.name}"
                    )
                logger.info(str(self._compilation_diagnostics))
                logger.info("=" * 70)
                return compile_result

            # Step 2: Try to fix errors (if retries remain) (60-95% of attempt range)
            if attempt < max_attempts:
                # Create progress mapper for error fixing (65-95% of attempt)
                fixing_progress = attempt_progress.create_sub_mapper(65, 95)
                
                _, corrected_file = await self._fix_step(
                    current_file, compile_result, attempt, progress_callback=fixing_progress.report
                )

                # Gate: Fixed? Use corrected file for next compile
                if corrected_file:
                    await attempt_progress.report(100)
                    current_file = corrected_file
                    logger.info(f"🔄 Retrying with fixed file: {current_file}")
                    continue
                else:
                    # Cannot fix, fail immediately
                    result = self._create_failure_result(
                        compile_result, "Error fixing failed"
                    )

                    # Create error version of the failed file
                    error_file_path = create_error_version(current_file)
                    if current_file.exists():
                        # Copy the failed file to error version
                        error_file_path.write_text(
                            current_file.read_text(encoding="utf-8"), encoding="utf-8"
                        )
                        logger.info(f"📄 Created error version: {error_file_path.name}")

                    # Set final file paths to error versions
                    result.final_tex_path = error_file_path
                    result.final_pdf_path = None  # No PDF for error versions

                    logger.error(f"❌ Orchestration failed: {result.message}")
                    logger.info(str(self._compilation_diagnostics))
                    return result

        # Gate: All attempts exhausted
        result = self._create_failure_result(
            compile_result, f"Failed after {max_attempts} attempts"
        )

        # Create error version of the failed file
        error_file_path = create_error_version(current_file)
        if current_file.exists():
            # Copy the failed file to error version
            error_file_path.write_text(
                current_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
            logger.info(f"📄 Created error version: {error_file_path.name}")

        # Set final file paths to error versions
        result.final_tex_path = error_file_path
        result.final_pdf_path = None  # No PDF for error versions

        logger.error(f"❌ Orchestration failed: {result.message}")
        logger.info(str(self._compilation_diagnostics))
        return result
