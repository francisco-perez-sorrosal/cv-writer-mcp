"""Markdown to LaTeX conversion agent using OpenAI Agents SDK."""

import os
from pathlib import Path

from agents import Agent, Runner
from loguru import logger
from pydantic import BaseModel, Field

from .models import CompletionStatus, MarkdownToLaTeXRequest, MarkdownToLaTeXResponse, LaTeXOutput, get_output_type_class
from .utils import load_agent_config, read_text_file


class MD2LaTeXAgent:
    """Markdown to LaTeX conversion agent using OpenAI Agents SDK with moderncv template."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the markdown to LaTeX conversion agent.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for conversion (if None, will use model from agent config)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Load agent config first to get the model
        self.agent_config = load_agent_config("md2latex_agent.yaml")
        
        # Use provided model or fall back to config model
        self.model = model or self.agent_config["agent_metadata"]["model"]
        
        self._load_resources()
        self._create_agent()

    def _load_resources(self) -> None:
        """Load required template and documentation files."""
        base_path = Path(__file__).parent.parent.parent

        # Load LaTeX template
        template_path = base_path / "context" / "latex" / "moderncv_template.tex"
        self.latex_template = read_text_file(template_path, "LaTeX template", ".tex")
        logger.info(f"Loaded LaTeX template from {template_path}")

        # Load user guide
        userguide_path = base_path / "context" / "latex" / "moderncv_userguide.txt"
        self.userguide_content = read_text_file(userguide_path, "ModernCV user guide", ".txt")
        logger.info(f"Loaded moderncv user guide from {userguide_path}")

    def _create_agent(self) -> None:
        """Create the OpenAI agent with structured output."""
        # Get the agent instructions from YAML configuration
        agent_instructions = self.agent_config["instructions"].format(
            moderncv_guide=self.userguide_content,
            moderncv_template=self.latex_template
        )

        # Get output type class from centralized mapping
        output_type_class = get_output_type_class(
            self.agent_config["agent_metadata"]["output_type"]
        )
        
        self.agent = Agent(
            name=self.agent_config["agent_metadata"]["name"],
            instructions=agent_instructions,
            model=self.model,
            output_type=output_type_class,
        )
        logger.info("Created OpenAI agent with structured output")

    async def convert(self, request: MarkdownToLaTeXRequest) -> MarkdownToLaTeXResponse:
        """Convert markdown CV content to LaTeX using OpenAI Agents SDK.

        Args:
            request: Markdown to LaTeX conversion request

        Returns:
            Conversion response with LaTeX file URL or error message
        """
        try:
            logger.info("Starting markdown to LaTeX conversion using OpenAI Agents SDK")

            # Create the user prompt using YAML template
            user_prompt = self.agent_config["prompt_template"].format(
                markdown_content=request.markdown_content
            )

            # Run the agent with structured output
            result = await Runner.run(self.agent, user_prompt)

            # Extract structured output
            latex_output: LaTeXOutput = result.final_output

            # Validate the LaTeX content
            if not latex_output.latex_content or not latex_output.latex_content.strip():
                return MarkdownToLaTeXResponse(
                    status=CompletionStatus.FAILED,
                    tex_url=None,
                    message=f"Agent response does not contain valid LaTeX content. Conversion notes: {latex_output.conversion_notes}",
                )

            # Save the LaTeX file
            output_filename = request.output_filename or "cv_converted.tex"
            if not output_filename.endswith(".tex"):
                output_filename += ".tex"

            output_path = Path("./output") / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(latex_output.latex_content, encoding="utf-8")

            # Generate resource URI
            tex_url = f"cv-writer://tex/{output_filename}"

            logger.info(f"Successfully converted markdown to LaTeX: {output_filename}")

            return MarkdownToLaTeXResponse(
                status=CompletionStatus.SUCCESS,
                tex_url=tex_url,
                message=f"Successfully converted to {output_filename}. {latex_output.conversion_notes}",
            )

        except Exception as e:
            logger.error(f"Error in markdown to LaTeX conversion: {e}")
            return MarkdownToLaTeXResponse(
                status=ConversionStatus.FAILED,
                tex_url=None,
                message=f"Conversion failed: {str(e)}",
            )
