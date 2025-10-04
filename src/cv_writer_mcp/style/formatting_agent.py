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
            Strategy-specific instructions for the prompt
        """
        strategies = {
            1: """Strategy: CONSERVATIVE APPROACH
- Focus on spacing optimization and consistency fixes only
- Make minimal structural changes to preserve original organization
- Prefer safe, proven LaTeX patterns that ensure compilation safety
- Apply spacing reduction conservatively (only obvious excessive spacing)
- Prioritize maintaining document structure and readability""",
            2: """Strategy: AGGRESSIVE APPROACH
- Pursue bold formatting improvements and section restructuring
- Maximize space efficiency with substantial spacing reductions
- Willing to significantly restructure sections for optimal layout
- Apply aggressive spacing reduction throughout document
- Prioritize visual impact and space optimization""",
            3: """Strategy: BALANCED APPROACH (recommended)
- Balance spacing optimization with readability preservation
- Apply moderate structural changes where clearly beneficial
- Mix conservative patterns with selective aggressive improvements
- Focus on overall document coherence and professionalism
- Optimize for professional appearance while maintaining safety""",
        }

        return strategies.get(variant_id, strategies[3])  # Default to balanced

    def _create_agent(self, variant_id: int = 1) -> Agent:
        """Create formatting agent instance.

        Args:
            variant_id: ID of the variant (for naming purposes)

        Returns:
            Configured Agent instance
        """
        from ..models import get_output_type_class

        output_type_class = get_output_type_class(
            self.agent_config["agent_metadata"]["output_type"]
        )

        return Agent(
            name=f"{self.agent_config['agent_metadata']['name']}_v{variant_id}",
            instructions=self.agent_config["instructions"],
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
        """Build complete prompt with all context.

        Args:
            latex_content: Original LaTeX content
            visual_analysis_results: Analysis from page capture
            suggested_fixes: List of suggested fixes
            variant_id: Variant ID for strategy selection
            iteration_feedback: Feedback from previous iteration (if any)

        Returns:
            Complete prompt string with all placeholders replaced
        """
        template = self.agent_config["prompt_template"]

        # Get variant strategy
        variant_strategy = self._get_variant_strategy(variant_id)

        # Format iteration feedback
        if iteration_feedback:
            feedback_text = f"""Previous iteration feedback from quality judge:
{iteration_feedback}

Please incorporate this feedback to improve the formatting."""
        else:
            feedback_text = "No previous iteration feedback (first iteration)."

        # Format suggested fixes
        fixes_text = "\n".join(f"- {fix}" for fix in suggested_fixes)

        # Replace all placeholders
        prompt = (
            template.replace("{latex_content}", latex_content)
            .replace("{visual_analysis_results}", visual_analysis_results)
            .replace("{suggested_fixes}", fixes_text)
            .replace("{variant_strategy}", variant_strategy)
            .replace("{iteration_feedback}", feedback_text)
        )

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
