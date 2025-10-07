"""Style quality evaluation agent."""

import os
from pathlib import Path
from typing import cast

from agents import Agent, Runner
from loguru import logger

from ..models import create_agent_from_config
from ..utils import load_agent_config
from .models import SingleVariantEvaluationOutput, VariantEvaluationOutput


class StyleQualityAgent:
    """Judge agent that evaluates CV style quality and compares variants."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the style quality agent.

        Args:
            api_key: OpenAI API key (optional, uses env var if not provided)
            model: Model to use for evaluation (optional, uses config default)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.agent_config = load_agent_config("quality_agent.yaml")
        self.model = model or self.agent_config["agent_metadata"]["model"]

    def _create_agent(self, output_type: str) -> Agent:
        """Create quality evaluation agent with specified output type.

        Args:
            output_type: Name of the output type class to use

        Returns:
            Configured Agent instance
        """
        # Create a modified config with the desired output type
        config_with_output = self.agent_config.copy()
        config_with_output["agent_metadata"] = self.agent_config["agent_metadata"].copy()
        config_with_output["agent_metadata"]["output_type"] = output_type

        # Create agent using centralized helper with safe defaults
        return create_agent_from_config(
            agent_config=config_with_output,
            instructions=self.agent_config["instructions"],
            model=self.model,
        )

    async def evaluate_single_variant(
        self,
        original_pdf_path: Path,
        improved_pdf_path: Path,
        improvement_goals: list[str],
    ) -> SingleVariantEvaluationOutput:
        """Evaluate a single variant against quality criteria.

        Used when num_variants=1 but user forces quality check.

        Args:
            original_pdf_path: Path to the original PDF before improvements
            improved_pdf_path: Path to the improved PDF variant
            improvement_goals: List of visual issues that were targeted for improvement

        Returns:
            SingleVariantEvaluationOutput with score, feedback, and metrics
        """
        try:
            logger.info("")
            logger.info("┌" + "─" * 68 + "┐")
            logger.info("│ ⚖️  QUALITY JUDGE: Evaluating single variant")
            logger.info(f"│ 📄 Original PDF: {original_pdf_path.name}")
            logger.info(f"│ 📄 Improved PDF: {improved_pdf_path.name}")
            logger.info("└" + "─" * 68 + "┘")

            agent = self._create_agent("SingleVariantEvaluationOutput")

            # Build prompt from template
            prompt = self.agent_config["prompt_templates"]["single_variant"].format(
                original_pdf_path=str(original_pdf_path.absolute()),
                improved_pdf_path=str(improved_pdf_path.absolute()),
                improvement_goals="\n".join(f"- {goal}" for goal in improvement_goals),
            )

            result = await Runner.run(agent, prompt)
            evaluation = cast(SingleVariantEvaluationOutput, result.final_output)

            logger.info(f"  Score: {evaluation.score}")
            logger.info(f"  Metrics: {evaluation.quality_metrics}")

            return evaluation

        except Exception as e:
            logger.error(f"Single variant evaluation failed: {e}")
            # Return a conservative evaluation on error
            return SingleVariantEvaluationOutput(
                score="needs_improvement",
                feedback=f"Evaluation failed: {str(e)}. Please review manually.",
                quality_metrics={
                    "design_coherence": 0.5,
                    "spacing": 0.5,
                    "consistency": 0.5,
                    "readability": 0.5,
                },
            )

    async def evaluate_variants(
        self,
        original_pdf_path: Path,
        variant_pdfs: list[tuple[int, Path]],
        improvement_goals: list[str],
        iteration_number: int = 1,
    ) -> VariantEvaluationOutput:
        """Compare N variants (N >= 2) and select the best.

        Args:
            original_pdf_path: Path to the original PDF before improvements
            variant_pdfs: List of (variant_id, pdf_path) tuples for each variant
            improvement_goals: List of visual issues targeted for improvement
            iteration_number: Current iteration number (for context)

        Returns:
            VariantEvaluationOutput with best variant ID, score, comparison, and metrics
        """
        try:
            logger.info("")
            logger.info("─" * 70)
            logger.info(f"⚖️  QUALITY JUDGE: Comparing {len(variant_pdfs)} variants")
            logger.info(f"📄 Original PDF: {original_pdf_path.name}")
            logger.info("📊 Variants being compared:")
            
            # Build variant info string with version labels and clear file names
            # Handle both 2-tuple (vid, path) and 3-tuple (vid, path, version) formats
            variant_info_lines: list[str] = []
            for item in variant_pdfs:
                if len(item) == 3:
                    vid, path, version = item  # type: ignore
                    variant_info_lines.append(
                        f"  Variant {vid} ({version}): {path.name}"
                    )
                    logger.info(f"  📄 Variant {vid} ({version}): {path.name}")
                else:
                    vid, path = item  # type: ignore
                    variant_info_lines.append(f"  Variant {vid}: {path.name}")
                    logger.info(f"  📄 Variant {vid}: {path.name}")
            variant_info = "\n".join(variant_info_lines)
            logger.info("─" * 70)

            agent = self._create_agent("VariantEvaluationOutput")

            # Build prompt from template
            prompt = self.agent_config["prompt_templates"]["multi_variant"].format(
                num_variants=len(variant_pdfs),
                original_pdf_path=str(original_pdf_path.absolute()),
                variant_info=variant_info,
                improvement_goals="\n".join(f"- {goal}" for goal in improvement_goals),
                iteration_number=iteration_number,
            )

            result = await Runner.run(agent, prompt)
            evaluation = cast(VariantEvaluationOutput, result.final_output)

            logger.info(f"  Best variant: {evaluation.best_variant_id}")
            logger.info(f"  Score: {evaluation.score}")
            logger.info(f"  Reasoning: {evaluation.comparison_summary[:100]}...")

            return evaluation

        except Exception as e:
            logger.error(f"Multi-variant evaluation failed: {e}")
            # Return a conservative evaluation on error: pick first variant
            first_variant_id = variant_pdfs[0][0]

            return VariantEvaluationOutput(
                best_variant_id=first_variant_id,
                best_variant_version="original",  # Default to original version
                score="needs_improvement",
                feedback=f"Evaluation failed: {str(e)}. Defaulting to first variant.",
                quality_metrics={
                    "design_coherence": 0.5,
                    "spacing": 0.5,
                    "consistency": 0.5,
                    "readability": 0.5,
                },
                comparison_summary=f"Evaluation error occurred. Selected variant {first_variant_id} by default.",
                all_variant_scores={
                    vid: {
                        "design_coherence": 0.5,
                        "spacing": 0.5,
                        "consistency": 0.5,
                        "readability": 0.5,
                    }
                    for vid, _ in variant_pdfs
                },
            )
