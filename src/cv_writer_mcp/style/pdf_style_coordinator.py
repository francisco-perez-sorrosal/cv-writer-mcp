"""Coordinator for PDF style analysis using two specialized agents."""

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..compilation.latex_expert import LaTeXExpert

from ..models import CompletionStatus
from .formatting_agent import FormattingAgent
from .models import (
    EvaluationResult,
    StyleDiagnostics,
    StyleIterationResult,
    VariantResult,
)
from .quality_agent import StyleQualityAgent
from .visual_critic_agent import VisualCriticAgent, VisualCriticRequest


class PDFStyleCoordinator:
    """Coordinates PDF style improvement with multi-variant generation and quality evaluation."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.model = model or "gpt-5-mini"

        # Initialize all agents
        self.visual_critic = VisualCriticAgent(api_key=self.api_key, model=self.model)
        self.formatting_agent = FormattingAgent(api_key=self.api_key, model=self.model)
        self.quality_agent = StyleQualityAgent(api_key=self.api_key, model=self.model)

    # ============================================================================
    # Multi-Variant Style Improvement with Iteration and Quality Evaluation
    # ============================================================================

    async def improve_with_variants(
        self,
        initial_pdf_path: Path,
        initial_tex_path: Path,
        latex_expert: "LaTeXExpert | None" = None,
        max_iterations: int = 1,
        num_variants: int = 1,
        max_compile_attempts: int = 5,
        enable_quality_validation: bool | None = None,
        output_dir: Path | None = None,
    ) -> StyleIterationResult:
        """Multi-iteration style improvement with N-variant strategy and quality judge.

        Supports two modes:
        1. With compilation (latex_expert provided): Generate variants → Compile → Judge
        2. Without compilation (latex_expert=None): Generate LaTeX variants only

        Args:
            initial_pdf_path: Path to the initial PDF to improve
            initial_tex_path: Path to the initial LaTeX source
            latex_expert: LaTeXExpert instance for recompilation (None=skip compilation)
            max_iterations: Maximum number of style iterations (default: 1)
            num_variants: Number of variants to generate per iteration (default: 1)
            max_compile_attempts: Max compilation attempts per variant (default: 5, ignored if no compilation)
            enable_quality_validation: Enable quality judge (None=auto: enabled if num_variants>=2)
            output_dir: Directory for variant outputs (default: ./output/style_variants)

        Returns:
            StyleIterationResult with best variant and diagnostics
        """
        # Smart default: enable judge if num_variants >= 2
        if enable_quality_validation is None:
            enable_quality_validation = num_variants >= 2

        # Validate: Quality judge requires compilation (needs PDFs to compare)
        if enable_quality_validation and latex_expert is None:
            raise ValueError(
                "Quality validation requires compilation. "
                "Provide latex_expert or disable quality validation."
            )

        # Validate configuration
        self._validate_configuration(
            max_iterations, num_variants, enable_quality_validation
        )

        # Setup output directory
        output_dir = output_dir or Path("./output/style_variants")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize diagnostics
        diagnostics = StyleDiagnostics(
            iterations_completed=0,
            variants_per_iteration=num_variants,
            total_variants_generated=0,
            successful_variants_per_iteration=[],
            failed_variants_per_iteration=[],
            quality_validation_enabled=enable_quality_validation,
            quality_scores=[],
            best_variant_per_iteration=[],
        )

        # Initialize state
        current_pdf = initial_pdf_path
        current_tex_content = initial_tex_path.read_text(encoding="utf-8")
        iteration_feedback: str | None = None  # None for first iteration, set by judge feedback
        best_result = None

        logger.info("🎨 Starting multi-variant style improvement")
        logger.info(
            f"   Iterations: {max_iterations}, Variants per iteration: {num_variants}"
        )
        logger.info(
            f"   Quality validation: {'enabled' if enable_quality_validation else 'disabled'}"
        )

        # OUTER LOOP: Style iterations
        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🎨 Style Iteration {iteration}/{max_iterations}")
            logger.info(f"{'='*60}")

            # Step 1: Capture & Analyze current PDF
            capture_result = await self._capture_and_analyze_pdf(current_pdf)
            if capture_result.status != CompletionStatus.SUCCESS:
                logger.error(f"Page capture failed: {capture_result.message}")
                break

            logger.info(
                f"📊 Analysis: {len(capture_result.visual_issues)} issues found"
            )

            # Step 2: Generate N variants in parallel
            logger.info(f"🔧 Generating {num_variants} variant(s) in parallel...")
            variants = await self._generate_variants_parallel(
                num_variants=num_variants,
                current_tex_content=current_tex_content,
                suggested_fixes=capture_result.suggested_fixes,
                visual_analysis=capture_result.analysis_summary,
                iteration_feedback=iteration_feedback,
                latex_expert=latex_expert,
                max_compile_attempts=max_compile_attempts,
                output_dir=output_dir,
                iteration=iteration,
            )

            # Step 3: Filter successful variants
            successful_variants = [v for v in variants if v is not None]
            num_failed = num_variants - len(successful_variants)

            diagnostics.successful_variants_per_iteration.append(
                len(successful_variants)
            )
            diagnostics.failed_variants_per_iteration.append(num_failed)
            diagnostics.total_variants_generated += num_variants

            logger.info(
                f"✅ {len(successful_variants)}/{num_variants} variant(s) compiled successfully"
            )

            if not successful_variants:
                logger.error(f"❌ All {num_variants} variants failed to compile")
                if best_result:
                    logger.info("Using previous iteration's best result")
                    break
                else:
                    return self._create_failure_result(
                        f"All variants failed in iteration {iteration}",
                        diagnostics,
                        iteration,
                    )

            # Step 4: Quality Evaluation
            eval_result = await self._evaluate_quality(
                enable_quality_validation=enable_quality_validation,
                num_variants=num_variants,
                original_pdf=current_pdf,
                successful_variants=successful_variants,
                improvement_goals=capture_result.visual_issues,
                iteration=iteration,
            )

            diagnostics.quality_scores.append(eval_result.score)
            diagnostics.best_variant_per_iteration.append(eval_result.best_variant_id)

            # Update best result
            best_result = eval_result.best_variant
            iteration_feedback = eval_result.feedback

            logger.info(f"🏆 Best variant: {eval_result.best_variant_id}")
            logger.info(f"📊 Judge score: {eval_result.score}")

            # Step 5: Decision
            if eval_result.score == "pass":
                logger.info("✅ Quality criteria met! Stopping iterations.")
                diagnostics.iterations_completed = iteration
                break
            elif eval_result.score == "needs_improvement":
                if iteration < max_iterations:
                    logger.info(
                        f"📈 Continuing to iteration {iteration + 1} with judge feedback"
                    )
                    current_pdf = best_result.pdf_path
                    current_tex_content = best_result.tex_path.read_text(
                        encoding="utf-8"
                    )
                    diagnostics.iterations_completed = iteration
                else:
                    logger.info("Max iterations reached")
                    diagnostics.iterations_completed = iteration
                    break
            else:  # "fail" or "unknown"
                logger.warning("❌ Quality criteria not met. Using best available.")
                diagnostics.iterations_completed = iteration
                break

        # Return final result
        if best_result:
            logger.info(f"\n{'='*60}")
            logger.info("🎉 Style improvement completed!")
            logger.info(f"   Iterations: {diagnostics.iterations_completed}")
            logger.info(f"   Best variant: {best_result.variant_id}")
            logger.info(f"   Final PDF: {best_result.pdf_path}")
            logger.info(f"{'='*60}\n")

            return StyleIterationResult(
                status=CompletionStatus.SUCCESS,
                best_variant_pdf_path=best_result.pdf_path,
                best_variant_tex_path=best_result.tex_path,
                iterations_completed=diagnostics.iterations_completed,
                quality_enabled=enable_quality_validation,
                final_score=(
                    diagnostics.quality_scores[-1]
                    if diagnostics.quality_scores
                    else None
                ),
                diagnostics=diagnostics,
                message=f"Completed {diagnostics.iterations_completed} iteration(s), selected variant {best_result.variant_id}",
            )
        else:
            return self._create_failure_result(
                "No successful variants generated",
                diagnostics,
                0,
            )

    def _validate_configuration(
        self, max_iterations: int, num_variants: int, enable_quality_validation: bool
    ) -> None:
        """Validate configuration and warn about unusual combinations."""
        if max_iterations > 1 and num_variants == 1 and not enable_quality_validation:
            logger.warning(
                "⚠️  Multiple iterations with single variant and no judge: "
                "Iterations won't improve without judge feedback. "
                "Consider num_variants=2 or enable_quality_validation=True"
            )

        if num_variants > 3:
            logger.warning(
                f"⚠️  Generating {num_variants} variants may be slow and costly. "
                "Consider num_variants=2 for most use cases."
            )

    async def _capture_and_analyze_pdf(self, pdf_path: Path):
        """Capture and analyze PDF pages."""
        capture_request = VisualCriticRequest(
            pdf_file_path=str(pdf_path.absolute()),
            num_pages=None,
        )
        return await self.visual_critic.critique(capture_request)

    async def _generate_variants_parallel(
        self,
        num_variants: int,
        current_tex_content: str,
        suggested_fixes: list[str],
        visual_analysis: str,
        iteration_feedback: str | None,
        latex_expert,
        max_compile_attempts: int,
        output_dir: Path,
        iteration: int,
    ) -> list[VariantResult | None]:
        """Generate N variants in parallel using asyncio.gather."""
        tasks = [
            self._generate_single_variant(
                variant_id=vid,
                current_tex_content=current_tex_content,
                suggested_fixes=suggested_fixes,
                visual_analysis=visual_analysis,
                iteration_feedback=iteration_feedback,
                latex_expert=latex_expert,
                max_compile_attempts=max_compile_attempts,
                output_dir=output_dir,
                iteration=iteration,
            )
            for vid in range(1, num_variants + 1)
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _generate_single_variant(
        self,
        variant_id: int,
        current_tex_content: str,
        suggested_fixes: list[str],
        visual_analysis: str,
        iteration_feedback: str | None,
        latex_expert: "LaTeXExpert | None",
        max_compile_attempts: int,
        output_dir: Path,
        iteration: int,
    ) -> VariantResult | None:
        """Generate LaTeX variant and optionally compile it.

        Returns None if formatting fails or compilation fails (when requested).
        """
        try:
            logger.info(f"  Variant {variant_id}: Starting generation...")

            # Step 1: Generate variant LaTeX
            formatting_output = await self.formatting_agent.implement_fixes(
                latex_content=current_tex_content,
                visual_analysis_results=visual_analysis,
                suggested_fixes=suggested_fixes,
                variant_id=variant_id,
                iteration_feedback=iteration_feedback,
            )

            if formatting_output.status != CompletionStatus.SUCCESS:
                logger.error(f"  Variant {variant_id}: ❌ Formatting failed")
                return None

            # Save variant LaTeX
            variant_tex_path = output_dir / f"iter{iteration}_variant{variant_id}.tex"
            variant_tex_path.write_text(
                formatting_output.improved_latex_content, encoding="utf-8"
            )

            # Step 2: Compile variant (optional - only if latex_expert provided)
            if latex_expert:
                logger.info(
                    f"  Variant {variant_id}: Compiling (up to {max_compile_attempts} attempts)..."
                )
                variant_pdf_path = (
                    output_dir / f"iter{iteration}_variant{variant_id}.pdf"
                )

                # Import LaTeXEngine here to avoid circular imports
                from ..compilation.models import LaTeXEngine

                compile_result = await latex_expert.orchestrate_compilation(
                    tex_file_path=variant_tex_path,
                    output_path=variant_pdf_path,
                    engine=LaTeXEngine.PDFLATEX,
                    max_attempts=max_compile_attempts,
                )

                if compile_result.status != CompletionStatus.SUCCESS:
                    logger.error(
                        f"  Variant {variant_id}: ❌ Compilation failed after {max_compile_attempts} attempts"
                    )
                    return None

                logger.info(f"  Variant {variant_id}: ✅ Compiled successfully!")

                return VariantResult(
                    variant_id=variant_id,
                    pdf_path=variant_pdf_path,
                    tex_path=variant_tex_path,
                    compilation_time=compile_result.compilation_time,
                )
            else:
                # No compilation - just return LaTeX variant
                logger.info(
                    f"  Variant {variant_id}: ✅ LaTeX generated (compilation skipped)"
                )

                return VariantResult(
                    variant_id=variant_id,
                    pdf_path=None,  # No PDF when compilation skipped
                    tex_path=variant_tex_path,
                    compilation_time=None,  # No compilation time
                )

        except Exception as e:
            logger.error(f"  Variant {variant_id}: ❌ Exception: {e}")
            return None

    async def _evaluate_quality(
        self,
        enable_quality_validation: bool,
        num_variants: int,
        original_pdf: Path,
        successful_variants: list[VariantResult],
        improvement_goals: list[str],
        iteration: int,
    ) -> EvaluationResult:
        """Evaluate variant quality and select best."""
        if not enable_quality_validation:
            # No judge: pick first successful variant
            logger.info("  📋 No judge: using first successful variant")
            best = successful_variants[0]
            return EvaluationResult(
                best_variant_id=best.variant_id,
                best_variant=best,
                score="unknown",
                feedback="",
            )

        if num_variants >= 2 and len(successful_variants) >= 2:
            # Multi-variant evaluation
            logger.info(f"  ⚖️  Judge comparing {len(successful_variants)} variants...")
            variant_pdfs = [(v.variant_id, v.pdf_path) for v in successful_variants]

            judge_output = await self.quality_agent.evaluate_variants(
                original_pdf_path=original_pdf,
                variant_pdfs=variant_pdfs,
                improvement_goals=improvement_goals,
                iteration_number=iteration,
            )

            best = next(
                v
                for v in successful_variants
                if v.variant_id == judge_output.best_variant_id
            )

            return EvaluationResult(
                best_variant_id=judge_output.best_variant_id,
                best_variant=best,
                score=judge_output.score,
                feedback=judge_output.feedback,
            )
        else:
            # Single variant evaluation
            logger.info("  📊 Judge evaluating single variant...")
            best = successful_variants[0]

            single_judge_output = await self.quality_agent.evaluate_single_variant(
                original_pdf_path=original_pdf,
                improved_pdf_path=best.pdf_path,
                improvement_goals=improvement_goals,
            )

            return EvaluationResult(
                best_variant_id=best.variant_id,
                best_variant=best,
                score=single_judge_output.score,
                feedback=single_judge_output.feedback,
            )

    def _create_failure_result(
        self, message: str, diagnostics: StyleDiagnostics, iterations: int
    ) -> StyleIterationResult:
        """Create failure result with diagnostics."""
        diagnostics.iterations_completed = iterations
        return StyleIterationResult(
            status=CompletionStatus.FAILED,
            best_variant_pdf_path=None,
            best_variant_tex_path=None,
            iterations_completed=iterations,
            quality_enabled=diagnostics.quality_validation_enabled,
            final_score=None,
            diagnostics=diagnostics,
            message=message,
        )
