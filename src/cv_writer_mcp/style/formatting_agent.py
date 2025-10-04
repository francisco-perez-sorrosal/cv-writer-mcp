"""Formatting agent for implementing visual formatting improvements."""

import os

from agents import Agent, Runner
from loguru import logger

from ..models import CompletionStatus
from ..utils import load_agent_config
from .models import FormattingOutput


class FormattingAgent:
    """Formatting agent that implements visual formatting improvements with variant support."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.agent_config = load_agent_config("formatting_agent.yaml")
        self.model = model or self.agent_config["agent_metadata"]["model"]

    def _get_variant_strategy(self, variant_id: int) -> str:
        """Get variant-specific strategy instructions.

        Args:
            variant_id: ID of the variant (1, 2, 3, ...)

        Returns:
            Strategy-specific instructions to append to base instructions
        """
        strategies = {
            1: """
VARIANT STRATEGY: Conservative Approach
────────────────────────────────────────
- Focus on spacing optimization and consistency
- Make minimal structural changes
- Prefer safe, proven LaTeX patterns
- Prioritize compilation safety
- Conservative use of vertical space reduction
""",
            2: """
VARIANT STRATEGY: Aggressive Approach
────────────────────────────────────────
- Bold formatting improvements and restructuring
- Maximize space efficiency
- More aggressive spacing reduction
- Willing to restructure sections for better layout
- Prioritize visual impact over safety
""",
            3: """
VARIANT STRATEGY: Balanced Approach
────────────────────────────────────────
- Balance between spacing and readability
- Moderate structural changes
- Mix conservative and aggressive techniques
- Focus on overall coherence
- Optimize for professional appearance
""",
        }

        return strategies.get(variant_id, strategies[3])  # Default to balanced

    def _create_agent(self, variant_id: int = 1) -> Agent:
        """Create formatting agent with variant-specific configuration.

        Args:
            variant_id: ID of the variant to configure strategy for

        Returns:
            Configured Agent instance
        """
        from ..models import get_output_type_class

        output_type_class = get_output_type_class(
            self.agent_config["agent_metadata"]["output_type"]
        )

        # Build variant-specific instructions
        base_instructions = self.agent_config["instructions"]
        variant_strategy = self._get_variant_strategy(variant_id)
        full_instructions = base_instructions + variant_strategy

        return Agent(
            name=f"{self.agent_config['agent_metadata']['name']}_v{variant_id}",
            instructions=full_instructions,
            tools=[],
            model=self.model,
            output_type=output_type_class,
        )

    def _build_prompt(
        self,
        latex_content: str,
        visual_analysis_results: str,
        suggested_fixes: list[str],
        variant_id: int,
        iteration_feedback: str,
    ) -> str:
        """Build prompt with all context including variant and feedback.

        Args:
            latex_content: Original LaTeX content
            visual_analysis_results: Analysis from page capture
            suggested_fixes: List of suggested fixes
            variant_id: Variant ID for context
            iteration_feedback: Feedback from previous iteration (if any)

        Returns:
            Complete prompt string
        """
        # Base prompt from template
        template = self.agent_config["prompt_template"]
        prompt = (
            template.replace("{latex_content}", latex_content)
            .replace("{visual_analysis_results}", visual_analysis_results)
            .replace("{suggested_fixes}", "\n".join(suggested_fixes))
        )

        # Add iteration feedback if available
        if iteration_feedback:
            feedback_section = f"""

FEEDBACK FROM PREVIOUS ITERATION:
═════════════════════════════════
{iteration_feedback}

Please incorporate this feedback into your improvements.
"""
            prompt += feedback_section

        # Add variant context
        variant_context = f"""

VARIANT CONTEXT:
════════════════
You are generating Variant {variant_id}. Apply your variant-specific strategy as instructed.
"""
        prompt += variant_context

        return prompt

    async def implement_fixes(
        self,
        latex_content: str,
        visual_analysis_results: str,
        suggested_fixes: list[str],
        variant_id: int = 1,
        iteration_feedback: str = "",
    ) -> FormattingOutput:
        """Implement formatting improvements with variant strategy.

        Args:
            latex_content: Original LaTeX content to improve
            visual_analysis_results: Analysis results from page capture agent
            suggested_fixes: List of suggested fixes from analysis
            variant_id: Variant ID for strategy differentiation (default: 1)
            iteration_feedback: Feedback from quality judge in previous iteration (default: "")

        Returns:
            FormattingOutput with improved LaTeX and metadata
        """
        try:
            logger.info(f"  🔧 Formatting agent generating variant {variant_id}")

            # Create variant-specific agent
            agent = self._create_agent(variant_id)

            # Build complete prompt
            prompt = self._build_prompt(
                latex_content,
                visual_analysis_results,
                suggested_fixes,
                variant_id,
                iteration_feedback,
            )

            # Run agent
            result = await Runner.run(agent, prompt)
            output = result.final_output

            if output.status == CompletionStatus.SUCCESS:
                logger.info(
                    f"  ✅ Variant {variant_id}: {len(output.fixes_applied)} fixes applied"
                )
            else:
                logger.warning(
                    f"  ⚠️  Variant {variant_id}: formatting returned non-success status"
                )

            return output

        except Exception as e:
            logger.error(f"  ❌ Variant {variant_id} formatting failed: {e}")
            return FormattingOutput(
                status=CompletionStatus.FAILED,
                fixes_applied=[],
                improved_latex_content=latex_content,
                implementation_notes=f"Formatting implementation failed: {str(e)}",
            )
