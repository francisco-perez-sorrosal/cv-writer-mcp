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
    VariantValidation,
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
        enable_variant_validation: bool = True,
        enable_variant_refinement: bool = True,
        output_dir: Path | None = None,
    ) -> StyleIterationResult:
        """Multi-iteration style improvement with N-variant strategy and quality judge.

        Supports two modes:
        1. With compilation (latex_expert provided): Generate variants → Compile → Validate → Judge
        2. Without compilation (latex_expert=None): Generate LaTeX variants only

        Args:
            initial_pdf_path: Path to the initial PDF to improve
            initial_tex_path: Path to the initial LaTeX source
            latex_expert: LaTeXExpert instance for recompilation (None=skip compilation)
            max_iterations: Maximum number of style iterations (default: 1)
            num_variants: Number of variants to generate per iteration (default: 1)
            max_compile_attempts: Max compilation attempts per variant (default: 5, ignored if no compilation)
            enable_quality_validation: Enable quality judge (None=auto: enabled if num_variants>=2)
            enable_variant_validation: Enable visual validation gate for variants (default: True)
            enable_variant_refinement: Enable refinement of variants with critical issues (default: True)
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

        logger.info("=" * 80)
        logger.info("🎨 STARTING MULTI-VARIANT STYLE IMPROVEMENT")
        logger.info(f"   Iterations: {max_iterations}, Variants per iteration: {num_variants}")
        logger.info(f"   Quality validation: {'enabled' if enable_quality_validation else 'disabled'}")
        logger.info("=" * 80)

        # OUTER LOOP: Style iterations
        for iteration in range(1, max_iterations + 1):
            logger.info("")
            logger.info("─" * 80)
            logger.info(f"🎨 STYLE ITERATION {iteration}/{max_iterations}")
            logger.info("─" * 80)

            # Step 1: Capture & Analyze current PDF
            if current_pdf is None:
                raise ValueError("Current PDF path is None - cannot proceed with style improvement")
            
            capture_result = await self._capture_and_analyze_pdf(current_pdf)
            if capture_result.status != CompletionStatus.SUCCESS:
                logger.error(f"Page capture failed: {capture_result.message}")
                break

            logger.info("")
            logger.info("┌" + "─" * 78 + "┐")
            logger.info(f"│ 📊 VISUAL ANALYSIS: {len(capture_result.visual_issues)} issues found")
            logger.info("└" + "─" * 78 + "┘")

            # Step 2: Generate N variants in parallel
            logger.info("")
            logger.info("┌" + "─" * 78 + "┐")
            logger.info(f"│ 🔧 GENERATING {num_variants} VARIANT(S) IN PARALLEL")
            logger.info("└" + "─" * 78 + "┘")
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
                original_issues=capture_result.visual_issues,
                enable_variant_validation=enable_variant_validation,
                enable_variant_refinement=enable_variant_refinement,
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

            # Step 4: Quality Evaluation (variants already include validation/refinement)
            if current_pdf is None:
                raise ValueError("Current PDF path is None - cannot evaluate quality")
            
            eval_result = await self._evaluate_quality(
                enable_quality_validation=enable_quality_validation,
                num_variants=num_variants,
                original_pdf=current_pdf,
                successful_variants=successful_variants,  # Variants already include validation/refinement
                improvement_goals=capture_result.visual_issues,
                iteration=iteration,
            )

            # Track diagnostics from variants (already computed during parallel processing)
            variants_with_issues = sum(
                1 for v in successful_variants if v.validation and v.validation.has_critical_regressions
            )
            variants_refined = sum(
                1 for v in successful_variants if v.validation and v.validation.was_refined
            )
            diagnostics.variants_validated_per_iteration.append(len(successful_variants))
            diagnostics.variants_with_issues_per_iteration.append(variants_with_issues)
            diagnostics.variants_refined_per_iteration.append(variants_refined)

            diagnostics.quality_scores.append(eval_result.score)
            diagnostics.best_variant_per_iteration.append(eval_result.best_variant_id)

            # Update best result
            best_result = eval_result.best_variant
            iteration_feedback = eval_result.feedback

            logger.info(
                f"🏆 Best variant: {eval_result.best_variant_id} ({best_result.version})"
            )
            logger.info(f"📊 Judge score: {eval_result.score}")

            # Display quality metrics
            if eval_result.quality_metrics:
                best_version = eval_result.best_variant.version
                logger.info(
                    f"📈 Quality Metrics (Best Variant {eval_result.best_variant_id} - {best_version}):"
                )
                for metric, score in eval_result.quality_metrics.items():
                    logger.info(f"     {metric}: {score:.2f}")

            # Display comparison reasoning (multi-variant only)
            if eval_result.comparison_summary:
                logger.info(f"")
                logger.info(f"🧠 Judge Reasoning:")
                # Split summary into lines for better readability
                for line in eval_result.comparison_summary.split('. '):
                    if line.strip():
                        logger.info(f"   {line.strip()}.")
                logger.info(f"")

            # Display all variant scores (multi-variant only)
            if eval_result.all_variant_scores:
                logger.info(f"📊 All Variant Scores:")
                for variant_id, scores in sorted(eval_result.all_variant_scores.items()):
                    # Calculate weighted score
                    weighted = (scores.get('design_coherence', 0) * 0.30 +
                               scores.get('spacing', 0) * 0.25 +
                               scores.get('consistency', 0) * 0.25 +
                               scores.get('readability', 0) * 0.20)
                    logger.info(f"   Variant {variant_id}: coherence={scores.get('design_coherence', 0):.2f}, "
                               f"spacing={scores.get('spacing', 0):.2f}, "
                               f"consistency={scores.get('consistency', 0):.2f}, "
                               f"readability={scores.get('readability', 0):.2f} "
                               f"→ weighted={weighted:.3f}")
                logger.info(f"")

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
            logger.info("")
            logger.info("█" * 80)
            logger.info("🎉 STYLE IMPROVEMENT COMPLETED")
            logger.info("█" * 80)
            logger.info(f"   Iterations: {diagnostics.iterations_completed}")
            logger.info(f"   Best variant: {best_result.variant_id} ({best_result.version})")
            if best_result.pdf_path:
                logger.info(f"   Final PDF: {best_result.pdf_path.name}")
            else:
                logger.info("   Final PDF: No PDF available")
            logger.info(f"   Final TEX: {best_result.tex_path.name}")
            logger.info("█" * 80)

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
        original_issues: list[str] | None = None,
        enable_variant_validation: bool = True,
        enable_variant_refinement: bool = True,
    ) -> list[VariantResult | None]:
        """Generate N variants in parallel using asyncio.gather.
        
        Each variant progresses through its complete lifecycle independently:
        Generation → Compilation → Validation → Refinement (if needed)
        """
        import time
        start_time = time.time()
        
        logger.info(f"🚀 Starting {num_variants} variants in PARALLEL...")
        
        # Create tasks for parallel execution
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
                original_issues=original_issues,
                enable_variant_validation=enable_variant_validation,
                enable_variant_refinement=enable_variant_refinement,
            )
            for vid in range(1, num_variants + 1)
        ]

        # Execute all variants in parallel
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful_count = sum(1 for r in results if r is not None)
        logger.info(f"✅ Parallel processing completed in {duration:.1f}s")
        logger.info(f"   {successful_count}/{num_variants} variants completed successfully")
        
        return results

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
        original_issues: list[str] | None = None,
        enable_variant_validation: bool = True,
        enable_variant_refinement: bool = True,
    ) -> VariantResult | None:
        """Generate LaTeX variant and optionally compile it.

        Returns None if formatting fails or compilation fails (when requested).
        """
        try:
            logger.info(f"🔧 Variant {variant_id}: Starting...")

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

            # Save variant LaTeX with organized naming
            variant_tex_path = output_dir / f"iter{iteration}_var{variant_id}.tex"
            variant_tex_path.write_text(
                formatting_output.improved_latex_content, encoding="utf-8"
            )

            # Step 2: Compile variant (optional - only if latex_expert provided)
            if latex_expert:
                variant_pdf_path = (
                    output_dir / f"iter{iteration}_var{variant_id}.pdf"
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

                # Create initial variant result
                variant_result = VariantResult(
                    variant_id=variant_id,
                    pdf_path=variant_pdf_path,
                    tex_path=variant_tex_path,
                    compilation_time=compile_result.compilation_time,
                    validation=None,
                    version="original",
                    parent_variant=None,
                )

                # Step 3: Visual Validation & Refinement (if enabled)
                if enable_variant_validation and original_issues:
                    # Visual validation
                    validation = await self._validate_variant(variant_result)
                    
                    # Identify critical regressions
                    has_critical, critical_issues = self._identify_critical_regressions(
                        validation.visual_issues, original_issues
                    )
                    
                    validation.has_critical_regressions = has_critical
                    validation.critical_issues = critical_issues
                    
                    # Refine if needed
                    if has_critical and enable_variant_refinement:
                        logger.warning(
                            f"  Variant {variant_id}: ⚠️  {len(critical_issues)} critical regression(s), refining..."
                        )
                        
                        refined_variant = await self._refine_variant(
                            variant_result,
                            validation,
                            latex_expert,
                            max_compile_attempts,
                            output_dir,
                            iteration,
                        )
                        
                        if refined_variant:
                            logger.info(f"  Variant {variant_id}: ✅ Refined successfully")
                            return refined_variant
                        else:
                            logger.warning(f"  Variant {variant_id}: ❌ Refinement failed, using original")
                    
                    # Attach validation metadata to variant
                    variant_result.validation = validation
                
                logger.info(f"🎉 Variant {variant_id}: Complete!")
                return variant_result
            else:
                # No compilation - just return LaTeX variant
                logger.info(f"  Variant {variant_id}: ✅ Complete! (LaTeX only)")

                return VariantResult(
                    variant_id=variant_id,
                    pdf_path=None,  # No PDF when compilation skipped
                    tex_path=variant_tex_path,
                    compilation_time=None,  # No compilation time
                    validation=None,
                    version="original",
                    parent_variant=None,
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
                quality_metrics=None,
                comparison_summary=None,
                all_variant_scores=None,
            )

        # Filter variants with valid PDF paths for quality evaluation
        variants_with_pdfs = [v for v in successful_variants if v.pdf_path is not None]
        
        if num_variants >= 2 and len(variants_with_pdfs) >= 2:
            # Multi-variant evaluation
            logger.info("")
            logger.info("┌" + "─" * 78 + "┐")
            logger.info(f"│ ⚖️  MAIN JUDGE: Comparing {len(variants_with_pdfs)} variants (Iteration {iteration})")
            logger.info(f"│ 📄 Original PDF: {original_pdf.name}")
            for v in variants_with_pdfs:
                if v.pdf_path:
                    logger.info(f"│   📄 Variant {v.variant_id} ({v.version}): {v.pdf_path.name}")
                else:
                    logger.info(f"│   📄 Variant {v.variant_id} ({v.version}): No PDF available")
            logger.info("└" + "─" * 78 + "┘")
            
            # Prepare variant PDFs for comparison: (id, path) tuples
            # Filter to ensure all paths are non-None
            variant_pdfs = [
                (v.variant_id, v.pdf_path) for v in variants_with_pdfs if v.pdf_path is not None
            ]

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

            logger.info("")
            logger.info("┌" + "─" * 78 + "┐")
            logger.info(f"│ 🏆 MAIN JUDGE RESULT: Selected Variant {best.variant_id} ({best.version})")
            if best.pdf_path:
                logger.info(f"│ 📄 Winning file: {best.pdf_path.name}")
            else:
                logger.info("│ 📄 Winning file: No PDF available")
            logger.info(f"│ 📊 Score: {judge_output.score}")
            logger.info("└" + "─" * 78 + "┘")

            return EvaluationResult(
                best_variant_id=judge_output.best_variant_id,
                best_variant=best,
                score=judge_output.score,
                feedback=judge_output.feedback,
                quality_metrics=judge_output.quality_metrics,
                comparison_summary=judge_output.comparison_summary,
                all_variant_scores=judge_output.all_variant_scores,
            )
        else:
            # Single variant evaluation
            # Use first variant with PDF, fallback to any variant
            best = variants_with_pdfs[0] if variants_with_pdfs else successful_variants[0]
            logger.info("")
            logger.info("┌" + "─" * 78 + "┐")
            logger.info(f"│ ⚖️  SINGLE VARIANT JUDGE: Evaluating variant {best.variant_id} (Iteration {iteration})")
            logger.info(f"│ 📄 Original PDF: {original_pdf.name}")
            if best.pdf_path:
                logger.info(f"│ 📄 Improved PDF: {best.pdf_path.name}")
            else:
                logger.info("│ 📄 Improved PDF: No PDF available (compilation skipped)")
            logger.info("└" + "─" * 78 + "┘")

            if best.pdf_path:
                single_judge_output = await self.quality_agent.evaluate_single_variant(
                    original_pdf_path=original_pdf,
                    improved_pdf_path=best.pdf_path,
                    improvement_goals=improvement_goals,
                )
            else:
                # No PDF available - return dummy result
                from .models import SingleVariantEvaluationOutput
                single_judge_output = SingleVariantEvaluationOutput(
                    score="needs_improvement",
                    quality_metrics={"design_coherence": 0.5, "spacing": 0.5, "consistency": 0.5, "readability": 0.5},
                    feedback="No PDF available for evaluation - compilation was skipped",
                )

            logger.info("")
            logger.info("┌" + "─" * 78 + "┐")
            logger.info(f"│ 🏆 SINGLE VARIANT JUDGE RESULT: Evaluated Variant {best.variant_id}")
            if best.pdf_path:
                logger.info(f"│ 📄 File: {best.pdf_path.name}")
            else:
                logger.info("│ 📄 File: No PDF available")
            logger.info(f"│ 📊 Score: {single_judge_output.score}")
            logger.info("└" + "─" * 78 + "┘")

            return EvaluationResult(
                best_variant_id=best.variant_id,
                best_variant=best,
                score=single_judge_output.score,
                feedback=single_judge_output.feedback,
                quality_metrics=single_judge_output.quality_metrics,
                comparison_summary=None,  # Single variant has no comparison
                all_variant_scores=None,  # Single variant has no comparison
            )

    async def _validate_variant(
        self, variant: VariantResult
    ) -> VariantValidation:
        """Visually validate a single variant PDF to detect issues.

        Args:
            variant: The variant to validate

        Returns:
            Validation result with issues found
        """
        if variant.pdf_path is None:
            logger.warning(
                f"  Variant {variant.variant_id}: No PDF to validate (compilation was skipped)"
            )
            return VariantValidation(
                variant_id=variant.variant_id,
                has_critical_regressions=False,
                visual_issues=[],
                critical_issues=[],
                refinement_successful=None,
            )

        # Critique variant PDF
        critique_result = await self.visual_critic.critique(
            VisualCriticRequest(pdf_file_path=str(variant.pdf_path))
        )

        return VariantValidation(
            variant_id=variant.variant_id,
            has_critical_regressions=False,  # Will be updated by caller
            visual_issues=critique_result.visual_issues,
            critical_issues=[],  # Will be updated by caller
            refinement_successful=None,
        )

    def _identify_critical_regressions(
        self, variant_issues: list[str], original_issues: list[str]
    ) -> tuple[bool, list[str]]:
        """Identify if variant has critical NEW issues (regressions).

        Compares variant issues against original issues to find regressions.
        A regression is a critical issue that wasn't in the original PDF.

        Args:
            variant_issues: Issues found in the variant
            original_issues: Issues from the original PDF

        Returns:
            Tuple of (has_critical_regressions, list_of_critical_issues)
        """
        # Critical issue keywords that indicate serious visual bugs
        critical_keywords = [
            "page jump",
            "page break",
            "awkward break",
            "content jump",
            "discontinuity",
            "overflow",
            "cut off",
            "truncated",
            "overlapping",
            "collision",
            "broken layout",
            "layout break",
            "missing content",
            "lost content",
            "unreadable",
            "illegible",
        ]

        critical_issues = []

        for issue in variant_issues:
            issue_lower = issue.lower()

            # Check if this issue is NEW (not in original)
            is_new = not any(
                self._semantic_similarity(issue_lower, orig.lower()) > 0.6
                for orig in original_issues
            )

            # Check if it's CRITICAL
            is_critical = any(keyword in issue_lower for keyword in critical_keywords)

            if is_new and is_critical:
                critical_issues.append(issue)

        return len(critical_issues) > 0, critical_issues

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple semantic similarity check using keyword overlap.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity: intersection over union
        return len(words1 & words2) / len(words1 | words2)

    async def _compare_variant_versions(
        self,
        original: VariantResult,
        refined: VariantResult,
        improvement_goals: list[str],
    ) -> VariantResult:
        """Compare original vs refined variant and return the better one.

        Uses the quality agent to determine which version is objectively better.
        The key question: did refinement fix critical issues without degrading
        overall quality?

        Args:
            original: Original variant
            refined: Refined variant (after fixing critical issues)
            improvement_goals: List of issues the refinement was targeting

        Returns:
            The better variant (either original or refined) with version label set
        """
        logger.info("")
        logger.info("┌" + "─" * 48 + "┐")
        logger.info(f"│ 🔍 BRANCH JUDGE: Comparing variant {original.variant_id}")
        if original.pdf_path:
            logger.info(f"│ 📄 Original: {original.pdf_path.name}")
        else:
            logger.info("│ 📄 Original: No PDF available")
        if refined.pdf_path:
            logger.info(f"│ 📄 Refined:  {refined.pdf_path.name}")
        else:
            logger.info("│ 📄 Refined:  No PDF available")
        logger.info("└" + "─" * 48 + "┘")

        try:
            # Only proceed if both variants have PDFs
            if not original.pdf_path or not refined.pdf_path:
                logger.warning("Cannot compare variants without PDFs")
                return original
            
            # Use temporary IDs for comparison: 1=original, 2=refined
            # This allows the judge to clearly distinguish between versions
            variant_pdfs = [
                (1, original.pdf_path),  # Version 1 = original
                (2, refined.pdf_path),   # Version 2 = refined
            ]

            # Use quality agent to compare the two versions
            evaluation = await self.quality_agent.evaluate_variants(
                original_pdf_path=original.pdf_path,  # Use original as baseline
                variant_pdfs=variant_pdfs,
                improvement_goals=improvement_goals,
                iteration_number=0,  # Branch comparison, not iteration-level
            )

            # Determine winner based on selected temporary ID
            if evaluation.best_variant_id == 1:
                # Judge selected version 1 (original)
                winner = original
                winner.version = "original"
                logger.info("")
                logger.info("┌" + "─" * 48 + "┐")
                logger.info(f"│ 🏆 BRANCH JUDGE RESULT: Selected ORIGINAL for variant {original.variant_id}")
                logger.info(f"│ 📄 Winning file: {original.pdf_path.name}")
                logger.info(f"│ 📊 Score: {evaluation.score}")
                logger.info("│ 💡 Refinement did not improve quality")
                logger.info("└" + "─" * 48 + "┘")
            else:
                # Judge selected version 2 (refined)
                winner = refined
                winner.version = "refined"
                logger.info("")
                logger.info("┌" + "─" * 48 + "┐")
                logger.info(f"│ 🏆 BRANCH JUDGE RESULT: Selected REFINED for variant {original.variant_id}")
                if refined.pdf_path:
                    logger.info(f"│ 📄 Winning file: {refined.pdf_path.name}")
                else:
                    logger.info("│ 📄 Winning file: No PDF available")
                logger.info(f"│ 📊 Score: {evaluation.score}")
                logger.info("│ 💡 Refinement successfully improved quality")
                logger.info("└" + "─" * 48 + "┘")

            return winner

        except Exception as e:
            logger.warning(
                f"    ⚠️  Variant {original.variant_id}: Branch comparison failed: {e}"
            )
            logger.warning(
                f"    Defaulting to REFINED version (attempted to fix critical issues)"
            )
            refined.version = "refined"
            return refined

    async def _refine_variant(
        self,
        variant: VariantResult,
        validation: VariantValidation,
        latex_expert,
        max_compile_attempts: int,
        output_dir: Path,
        iteration: int,
    ) -> VariantResult:
        """Refine a variant to fix critical regressions.

        Uses the Formatting Agent with a targeted refinement prompt to fix
        the critical issues detected in validation.

        Args:
            variant: Variant to refine
            validation: Validation result with critical issues
            latex_expert: LaTeX compilation expert
            max_compile_attempts: Max compilation attempts
            output_dir: Output directory for refined files
            iteration: Current iteration number

        Returns:
            Refined variant (or original if refinement fails)
        """
        from ..models import CompletionStatus

        try:
            logger.info(
                f"    🔧 Variant {variant.variant_id}: Refining {len(validation.critical_issues)} critical issue(s)..."
            )

            # Read current LaTeX content
            latex_content = variant.tex_path.read_text(encoding="utf-8")

            # Build refinement-specific visual analysis
            refinement_analysis = f"""⚠️  CRITICAL REFINEMENT TASK ⚠️

This variant (Variant {variant.variant_id}) was initially generated but visual analysis
detected the following CRITICAL issues that MUST be fixed:

{chr(10).join(f'- {issue}' for issue in validation.critical_issues)}

IMPORTANT: These issues were NOT in the original document - they were INTRODUCED
during the formatting improvements. Your task is to FIX these regressions while
preserving the other improvements you made.

Focus specifically on:
- Fixing page breaks/jumps (ensure content flows naturally across pages)
- Fixing spacing/overflow issues (prevent content from being cut off)
- Ensuring no content is overlapping or causing layout collisions
- Maintaining readability and professional appearance"""

            # Call formatting agent for refinement
            formatting_output = await self.formatting_agent.implement_fixes(
                latex_content=latex_content,
                visual_analysis_results=refinement_analysis,
                suggested_fixes=validation.critical_issues,
                variant_id=variant.variant_id,
                iteration_feedback=None,  # No judge feedback for refinement
            )

            if formatting_output.status != CompletionStatus.SUCCESS:
                logger.warning(
                    f"    Variant {variant.variant_id}: Refinement formatting failed"
                )
                return variant  # Return original

            # Save refined LaTeX with organized naming
            refined_tex_path = (
                output_dir / f"iter{iteration}_var{variant.variant_id}_refined.tex"
            )
            refined_tex_path.write_text(
                formatting_output.improved_latex_content, encoding="utf-8"
            )

            # Compile refined variant
            if latex_expert:
                logger.info(f"    Compiling refined variant {variant.variant_id}...")
                refined_pdf_path = (
                    output_dir
                    / f"iter{iteration}_var{variant.variant_id}_refined.pdf"
                )

                from ..compilation.models import LaTeXEngine

                compile_result = await latex_expert.orchestrate_compilation(
                    tex_file_path=refined_tex_path,
                    output_path=refined_pdf_path,
                    engine=LaTeXEngine.PDFLATEX,
                    max_attempts=max_compile_attempts,
                )

                if compile_result.status == CompletionStatus.SUCCESS:
                    logger.info(
                        f"    ✅ Variant {variant.variant_id} refined successfully"
                    )
                    # Create refined variant result
                    refined_variant = VariantResult(
                        variant_id=variant.variant_id,
                        pdf_path=refined_pdf_path,
                        tex_path=refined_tex_path,
                        compilation_time=compile_result.compilation_time,
                        validation=validation,
                        version="refined",
                        parent_variant=variant,
                    )
                    return refined_variant
                else:
                    logger.warning(
                        f"    Variant {variant.variant_id}: Refined version failed to compile"
                    )
                    return variant  # Return original
            else:
                # No compilation - just return with refined tex
                refined_variant = VariantResult(
                    variant_id=variant.variant_id,
                    pdf_path=None,
                    tex_path=refined_tex_path,
                    compilation_time=None,
                    validation=validation,
                    version="refined",
                    parent_variant=variant,
                )
                return refined_variant

        except Exception as e:
            logger.error(f"    Variant {variant.variant_id}: Refinement failed: {e}")
            return variant  # Return original on any error

    async def _validate_and_refine_variants(
        self,
        variants: list[VariantResult],
        original_issues: list[str],
        latex_expert,
        max_compile_attempts: int,
        output_dir: Path,
        iteration: int,
        enable_refinement: bool = True,
    ) -> list[VariantResult]:
        """Validate variants and optionally refine those with critical regressions.

        This is the main validation gate that:
        1. Visually validates each variant
        2. Identifies critical regressions (new bugs not in original)
        3. Attempts to refine problematic variants (if enabled)
        4. Returns variants with validation metadata attached

        Args:
            variants: Generated variants to validate
            original_issues: Issues from original PDF (to detect regressions)
            latex_expert: LaTeX compilation expert
            max_compile_attempts: Max compilation attempts for refinement
            output_dir: Output directory for refined files
            iteration: Current iteration number
            enable_refinement: Whether to attempt refinement of problematic variants

        Returns:
            Refined variants (with validation metadata attached)
        """
        refined_variants = []
        variants_with_issues = 0
        variants_refined = 0

        for variant in variants:
            logger.info(f"  Validating variant {variant.variant_id}...")

            # Step 1: Visual validation
            validation = await self._validate_variant(variant)

            # Step 2: Identify critical regressions
            has_critical, critical_issues = self._identify_critical_regressions(
                validation.visual_issues, original_issues
            )

            validation.has_critical_regressions = has_critical
            validation.critical_issues = critical_issues

            # Step 3: Refine if needed
            if has_critical:
                variants_with_issues += 1
                logger.warning(
                    f"  ⚠️  Variant {variant.variant_id} has {len(critical_issues)} critical regression(s)"
                )

                if enable_refinement:
                    variants_refined += 1
                    refined_variant = await self._refine_variant(
                        variant,
                        validation,
                        latex_expert,
                        max_compile_attempts,
                        output_dir,
                        iteration,
                    )

                    # Re-validate refined variant
                    revalidation = await self._validate_variant(refined_variant)
                    (
                        still_has_critical,
                        remaining_issues,
                    ) = self._identify_critical_regressions(
                        revalidation.visual_issues, original_issues
                    )

                    validation.was_refined = True
                    validation.refinement_successful = not still_has_critical

                    if still_has_critical:
                        logger.warning(
                            f"  ⚠️  Variant {variant.variant_id} still has {len(remaining_issues)} issue(s) after refinement"
                        )
                    else:
                        logger.info(
                            f"  ✅ Variant {variant.variant_id} successfully refined"
                        )

                    # NEW: Branch-level comparison between original and refined
                    # Use quality agent to determine which version is actually better
                    improvement_goals = (
                        validation.critical_issues if validation.critical_issues else []
                    )
                    best_version = await self._compare_variant_versions(
                        original=variant,
                        refined=refined_variant,
                        improvement_goals=improvement_goals,
                    )

                    # Use the winning version
                    best_version.validation = validation
                    refined_variants.append(best_version)
                else:
                    # Refinement disabled - just attach validation and mark as original
                    variant.version = "original"
                    variant.validation = validation
                    refined_variants.append(variant)
            else:
                # No critical issues - keep original version
                logger.info(f"  ✅ Variant {variant.variant_id} passed validation")
                variant.version = "original"
                variant.validation = validation
                refined_variants.append(variant)

        # Log summary
        logger.info("")
        logger.info(f"  Validation Summary:")
        logger.info(f"    Variants validated: {len(variants)}")
        logger.info(f"    Variants with critical issues: {variants_with_issues}")
        logger.info(f"    Variants refined: {variants_refined}")

        return refined_variants

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
