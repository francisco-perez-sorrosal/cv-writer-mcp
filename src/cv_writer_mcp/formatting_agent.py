"""Formatting agent for implementing visual formatting improvements."""

import os
from pathlib import Path

from loguru import logger
from agents import Agent, Runner

from .models import CompletionStatus, FormattingOutput
from .utils import load_agent_config


class FormattingAgent:
    """Formatting agent that implements visual formatting improvements based on analysis."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.agent_config = load_agent_config("formatting_agent.yaml")
        self.model = model or self.agent_config["agent_metadata"]["model"]

    def _create_agent(self) -> Agent:
        """Create formatting agent."""
        from .models import get_output_type_class
        
        output_type_class = get_output_type_class(
            self.agent_config["agent_metadata"]["output_type"]
        )
        
        return Agent(
            name=self.agent_config["agent_metadata"]["name"],
            instructions=self.agent_config["instructions"],
            tools=[],
            model=self.model,
            output_type=output_type_class,
        )

    async def implement_fixes(
        self,
        latex_content: str,
        visual_analysis_results: str,
        suggested_fixes: list[str]
    ) -> FormattingOutput:
        """Implement formatting improvements based on visual analysis results."""
        try:
            agent = self._create_agent()
            
            # Create prompt with manual replacement to avoid LaTeX brace conflicts
            template = self.agent_config["prompt_template"]
            prompt = (
                template.replace("{latex_content}", latex_content)
                .replace("{visual_analysis_results}", visual_analysis_results)
                .replace("{suggested_fixes}", "\n".join(suggested_fixes))
            )
            
            result = await Runner.run(agent, prompt)
            return result.final_output

        except Exception as e:
            logger.error(f"Formatting implementation failed: {e}")
            return FormattingOutput(
                status=CompletionStatus.FAILED,
                fixes_applied=[],
                improved_latex_content=latex_content,
                implementation_notes=f"Formatting implementation failed: {str(e)}",
            )
