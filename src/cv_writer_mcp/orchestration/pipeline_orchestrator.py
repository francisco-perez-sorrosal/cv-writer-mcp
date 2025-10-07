"""End-to-end CV generation pipeline orchestrator."""

import time
from pathlib import Path

from loguru import logger

from ..compilation.latex_expert import LaTeXExpert
from ..compilation.models import CompilationDiagnostics, LaTeXEngine
from ..conversion.md2latex_agent import MD2LaTeXAgent
from ..conversion.models import MarkdownToLaTeXRequest
from ..models import CompletionStatus, ServerConfig
from ..style.pdf_style_coordinator import PDFStyleCoordinator
from .models import CVGenerationResponse, CVGenerationResult


class CVPipelineOrchestrator:
    """Orchestrates complete end-to-end CV generation pipeline.

    This orchestrator coordinates three main phases:
    1. Conversion: Markdown → LaTeX (MD2LaTeXAgent)
    2. Initial Compilation: LaTeX → PDF with error fixing (LaTeXExpert)
    3. Style Improvement: PDF → Styled PDF with N variants and quality judge (PDFStyleCoordinator)
    """

    def __init__(
        self,
        md2latex_agent: MD2LaTeXAgent,
        latex_expert: LaTeXExpert,
        style_coordinator: PDFStyleCoordinator,
        config: ServerConfig,
    ):
        """Initialize the pipeline orchestrator.

        Args:
            md2latex_agent: Agent for markdown to LaTeX conversion
            latex_expert: Expert for LaTeX compilation and error fixing
            style_coordinator: Coordinator for PDF style improvement
            config: Server configuration
        """
        self.md2latex = md2latex_agent
        self.latex_expert = latex_expert
        self.style_coordinator = style_coordinator
        self.config = config

    async def generate_cv_from_markdown(
        self,
        markdown_content: str,
        output_filename: str | None = None,
        enable_style_improvement: bool = True,
        max_compile_attempts: int = 3,
        max_style_iterations: int = 1,
        num_style_variants: int = 1,
        enable_quality_validation: bool | None = None,
    ) -> CVGenerationResult:
        """Complete end-to-end pipeline: MD → LaTeX → PDF → Style → Final PDF.

        Args:
            markdown_content: Markdown CV content
            output_filename: Custom output filename (optional)
            enable_style_improvement: Enable style improvement phase (default: True)
            max_compile_attempts: Max compilation attempts (default: 3)
            max_style_iterations: Max style iterations (default: 1)
            num_style_variants: Number of variants per iteration (default: 1)
            enable_quality_validation: Enable quality judge (None=auto)

        Returns:
            CVGenerationResult with all diagnostics
        """
        start_time = time.time()

        try:
            logger.info("=" * 80)
            logger.info("🚀 STARTING COMPLETE CV GENERATION PIPELINE")
            logger.info("=" * 80)

            # ================================================================
            # PHASE 1: Markdown → LaTeX Conversion
            # ================================================================
            logger.info("")
            logger.info("█" * 80)
            logger.info("📝 PHASE 1: MARKDOWN → LATEX CONVERSION")
            logger.info("█" * 80)
            conversion_start = time.time()

            md_request = MarkdownToLaTeXRequest(
                markdown_content=markdown_content,
                output_filename=output_filename or "",
            )
            conversion_result = await self.md2latex.convert(md_request)

            if conversion_result.status != CompletionStatus.SUCCESS:
                return self._create_failure_result(
                    "Markdown to LaTeX conversion failed",
                    conversion_result.message or "Unknown error",
                    total_time=time.time() - start_time,
                )

            conversion_time = time.time() - conversion_start
            logger.info(f"✅ Conversion completed in {conversion_time:.2f}s")

            # Extract file paths
            if not conversion_result.tex_url:
                return self._create_failure_result(
                    "Markdown to LaTeX conversion failed",
                    "No LaTeX file URL returned",
                    conversion_time=conversion_time,
                    total_time=time.time() - start_time,
                )

            tex_filename = Path(conversion_result.tex_url.split("/")[-1])
            tex_path = self.config.output_dir / tex_filename

            # ================================================================
            # PHASE 2: Initial Compilation (with compile-fix-compile loop)
            # ================================================================
            logger.info("")
            logger.info("█" * 80)
            logger.info(
                f"🔨 PHASE 2: INITIAL COMPILATION (up to {max_compile_attempts} attempts)"
            )
            logger.info("█" * 80)

            initial_pdf_filename = tex_filename.stem + ".pdf"
            initial_pdf_path = self.config.output_dir / initial_pdf_filename

            compile_result = await self.latex_expert.orchestrate_compilation(
                tex_file_path=tex_path,
                output_path=initial_pdf_path,
                engine=LaTeXEngine.PDFLATEX,
                max_attempts=max_compile_attempts,
            )

            if compile_result.status != CompletionStatus.SUCCESS:
                return self._create_failure_result(
                    "Initial compilation failed",
                    compile_result.message or "Unknown compilation error",
                    conversion_time=conversion_time,
                    compilation_diagnostics=self.latex_expert._compilation_diagnostics,
                    total_time=time.time() - start_time,
                )

            # Get actual compiled PDF path (may differ from requested if fixing was applied)
            actual_compiled_pdf = compile_result.output_path or initial_pdf_path
            actual_compiled_tex = (
                actual_compiled_pdf.parent / f"{actual_compiled_pdf.stem}.tex"
            )

            logger.info(
                f"✅ Compilation completed in {compile_result.compilation_time:.2f}s"
            )
            logger.info(f"   📄 PDF: {actual_compiled_pdf.name}")
            logger.info(f"   📝 TEX: {actual_compiled_tex.name}")

            # ================================================================
            # PHASE 3: Style Improvement (optional, with N variants)
            # ================================================================
            style_diagnostics = None
            final_pdf_path = actual_compiled_pdf
            final_tex_path = actual_compiled_tex

            if enable_style_improvement:
                logger.info("")
                logger.info("█" * 80)
                logger.info(
                    f"🎨 PHASE 3: STYLE IMPROVEMENT ({num_style_variants} variant(s), {max_style_iterations} iteration(s))"
                )
                logger.info("█" * 80)

                style_result = await self.style_coordinator.improve_with_variants(
                    initial_pdf_path=actual_compiled_pdf,
                    initial_tex_path=actual_compiled_tex,
                    latex_expert=self.latex_expert,
                    max_iterations=max_style_iterations,
                    num_variants=num_style_variants,
                    max_compile_attempts=max_compile_attempts,
                    enable_quality_validation=enable_quality_validation,
                    output_dir=self.config.output_dir / "style_variants",
                )

                if style_result.status == CompletionStatus.SUCCESS:
                    if (
                        style_result.best_variant_pdf_path
                        and style_result.best_variant_tex_path
                    ):
                        final_pdf_path = style_result.best_variant_pdf_path
                        final_tex_path = style_result.best_variant_tex_path
                        style_diagnostics = style_result.diagnostics
                        logger.info("✅ Style improvement completed")
                    else:
                        logger.warning("⚠️  Style improvement returned no variant paths")
                        logger.warning("   Using initial compilation result")
                else:
                    logger.warning(
                        f"⚠️  Style improvement failed: {style_result.message}"
                    )
                    logger.warning("   Using initial compilation result")
            else:
                logger.info("⏭️  PHASE 3: Style improvement skipped (disabled)")

            # ================================================================
            # PIPELINE COMPLETE
            # ================================================================
            total_time = time.time() - start_time

            logger.info("")
            logger.info("█" * 80)
            logger.info("🎉 CV GENERATION PIPELINE COMPLETED!")
            logger.info("█" * 80)
            logger.info(f"📊 Total time: {total_time:.2f}s")
            logger.info(f"📄 Final PDF: {final_pdf_path.name}")
            logger.info(f"📝 Final LaTeX: {final_tex_path.name}")
            logger.info("█" * 80)

            return CVGenerationResult(
                status=CompletionStatus.SUCCESS,
                final_pdf_url=f"cv-writer://pdf/{final_pdf_path.name}",
                final_tex_url=f"cv-writer://tex/{final_tex_path.name}",
                conversion_time=conversion_time,
                compilation_diagnostics=self.latex_expert._compilation_diagnostics,
                style_diagnostics=style_diagnostics,
                total_time=total_time,
                message=f"Successfully generated CV: {final_pdf_path.name}",
            )

        except Exception as e:
            logger.error(f"❌ Pipeline failed with exception: {e}")
            return self._create_failure_result(
                "Pipeline exception",
                str(e),
                total_time=time.time() - start_time,
            )

    async def compile_and_improve_style(
        self,
        tex_filename: str,
        output_filename: str | None = None,
        max_compile_attempts: int = 3,
        max_style_iterations: int = 1,
        num_style_variants: int = 1,
        enable_quality_validation: bool | None = None,
    ) -> CVGenerationResult:
        """Compile LaTeX and improve styling: LaTeX → PDF → Style → Final PDF.

        Args:
            tex_filename: Name of the .tex file to compile
            output_filename: Custom output filename (optional)
            max_compile_attempts: Max compilation attempts (default: 3)
            max_style_iterations: Max style iterations (default: 1)
            num_style_variants: Number of variants (default: 1)
            enable_quality_validation: Enable quality judge (None=auto)

        Returns:
            CVGenerationResult with diagnostics
        """
        start_time = time.time()

        try:
            logger.info("=" * 80)
            logger.info("🚀 STARTING COMPILE & IMPROVE PIPELINE")
            logger.info("=" * 80)

            # Locate tex file
            tex_path = self.config.output_dir / tex_filename
            if not tex_path.exists():
                return self._create_failure_result(
                    "LaTeX file not found",
                    f"File not found: {tex_filename}",
                    total_time=time.time() - start_time,
                )

            # ================================================================
            # PHASE 1: Initial Compilation
            # ================================================================
            logger.info("")
            logger.info("█" * 80)
            logger.info("🔨 PHASE 1: Initial compilation")
            logger.info("█" * 80)

            output_name = output_filename or (tex_path.stem + ".pdf")
            if not output_name.endswith(".pdf"):
                output_name += ".pdf"

            initial_pdf_path = self.config.output_dir / output_name

            compile_result = await self.latex_expert.orchestrate_compilation(
                tex_file_path=tex_path,
                output_path=initial_pdf_path,
                engine=LaTeXEngine.PDFLATEX,
                max_attempts=max_compile_attempts,
            )

            if compile_result.status != CompletionStatus.SUCCESS:
                return self._create_failure_result(
                    "Initial compilation failed",
                    compile_result.message or "Unknown compilation error",
                    compilation_diagnostics=self.latex_expert._compilation_diagnostics,
                    total_time=time.time() - start_time,
                )

            # Get actual compiled PDF path (may differ from requested if fixing was applied)
            actual_compiled_pdf = compile_result.output_path or initial_pdf_path
            actual_compiled_tex = (
                actual_compiled_pdf.parent / f"{actual_compiled_pdf.stem}.tex"
            )

            logger.info("✅ Compilation completed")
            logger.info(f"   📄 PDF: {actual_compiled_pdf.name}")
            logger.info(f"   📝 TEX: {actual_compiled_tex.name}")

            # ================================================================
            # PHASE 2: Style Improvement
            # ================================================================
            logger.info("")
            logger.info("█" * 80)
            logger.info("🎨 PHASE 2: Style improvement")
            logger.info("█" * 80)

            style_result = await self.style_coordinator.improve_with_variants(
                initial_pdf_path=actual_compiled_pdf,
                initial_tex_path=actual_compiled_tex,
                latex_expert=self.latex_expert,
                max_iterations=max_style_iterations,
                num_variants=num_style_variants,
                max_compile_attempts=max_compile_attempts,
                enable_quality_validation=enable_quality_validation,
                output_dir=self.config.output_dir / "style_variants",
            )

            if style_result.status != CompletionStatus.SUCCESS:
                logger.warning(f"⚠️  Style improvement failed: {style_result.message}")
                logger.warning("   Using initial compilation result")

                final_pdf_path = actual_compiled_pdf
                final_tex_path = actual_compiled_tex
                style_diagnostics = style_result.diagnostics
            else:
                if (
                    style_result.best_variant_pdf_path
                    and style_result.best_variant_tex_path
                ):
                    final_pdf_path = style_result.best_variant_pdf_path
                    final_tex_path = style_result.best_variant_tex_path
                    style_diagnostics = style_result.diagnostics
                    logger.info("✅ Style improvement completed")
                else:
                    logger.warning("⚠️  Style improvement returned no variant paths")
                    logger.warning("   Using initial compilation result")
                    final_pdf_path = actual_compiled_pdf
                    final_tex_path = actual_compiled_tex
                    style_diagnostics = style_result.diagnostics

            # ================================================================
            # PIPELINE COMPLETE
            # ================================================================
            total_time = time.time() - start_time

            logger.info("")
            logger.info("█" * 80)
            logger.info("🎉 COMPILE & IMPROVE PIPELINE COMPLETED!")
            logger.info("█" * 80)
            logger.info(f"📊 Total time: {total_time:.2f}s")
            logger.info(f"📄 Final PDF: {final_pdf_path}")
            logger.info("█" * 80)

            return CVGenerationResult(
                status=CompletionStatus.SUCCESS,
                final_pdf_url=f"cv-writer://pdf/{final_pdf_path.name}",
                final_tex_url=f"cv-writer://tex/{final_tex_path.name}",
                conversion_time=None,
                compilation_diagnostics=self.latex_expert._compilation_diagnostics,
                style_diagnostics=style_diagnostics,
                total_time=total_time,
                message=f"Successfully compiled and improved: {final_pdf_path.name}",
            )

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return self._create_failure_result(
                "Pipeline exception",
                str(e),
                total_time=time.time() - start_time,
            )

    def _create_failure_result(
        self,
        stage: str,
        error_message: str,
        conversion_time: float | None = None,
        compilation_diagnostics: CompilationDiagnostics | None = None,
        total_time: float = 0.0,
    ) -> CVGenerationResult:
        """Create failure result with available diagnostics.

        Args:
            stage: Pipeline stage where failure occurred
            error_message: Error message
            conversion_time: Conversion time if available
            compilation_diagnostics: Compilation diagnostics if available
            total_time: Total time elapsed

        Returns:
            CVGenerationResult indicating failure
        """
        from ..compilation.models import CompilationDiagnostics

        return CVGenerationResult(
            status=CompletionStatus.FAILED,
            final_pdf_url=None,
            final_tex_url=None,
            conversion_time=conversion_time,
            compilation_diagnostics=compilation_diagnostics or CompilationDiagnostics(),
            style_diagnostics=None,
            total_time=total_time,
            message=f"{stage}: {error_message}",
        )

    def to_response(self, result: CVGenerationResult) -> CVGenerationResponse:
        """Convert CVGenerationResult to CVGenerationResponse for MCP API.

        Args:
            result: Full pipeline result with diagnostics

        Returns:
            Simplified response for MCP API
        """
        # Create diagnostics summary
        diag_summary = None
        if result.compilation_diagnostics or result.style_diagnostics:
            parts = []
            if result.conversion_time:
                parts.append(f"Conversion: {result.conversion_time:.2f}s")
            if result.compilation_diagnostics:
                parts.append(
                    f"Compilation: {result.compilation_diagnostics.successful_compilations} success, "
                    f"{result.compilation_diagnostics.failed_compilations} failed"
                )
            if result.style_diagnostics:
                parts.append(
                    f"Style: {result.style_diagnostics.iterations_completed} iterations, "
                    f"{result.style_diagnostics.total_variants_generated} variants generated"
                )
            diag_summary = " | ".join(parts)

        return CVGenerationResponse(
            status=result.status,
            pdf_url=result.final_pdf_url,
            tex_url=result.final_tex_url,
            message=result.message,
            diagnostics_summary=diag_summary,
        )
