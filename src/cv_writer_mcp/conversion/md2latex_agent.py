"""Markdown to LaTeX conversion agent using OpenAI Agents SDK."""

import asyncio
import os
from pathlib import Path

from agents import Runner
from loguru import logger

from ..models import CompletionStatus, create_agent_from_config
from ..utils import (
    PeriodicProgressTicker,
    ProgressCallback,
    load_agent_config,
    read_text_file,
)
from .models import LaTeXOutput, MarkdownToLaTeXRequest, MarkdownToLaTeXResponse


class MD2LaTeXAgent:
    """Markdown to LaTeX conversion agent using OpenAI Agents SDK with moderncv template."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        template_name: str = "moderncv_template.tex",
    ):
        """Initialize the markdown to LaTeX conversion agent.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for conversion (if None, will use model from agent config)
            template_name: Name of the LaTeX template file (default: moderncv_template.tex)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.template_name = template_name

        # Load agent config first to get the model
        self.agent_config = load_agent_config("md2latex_agent.yaml")

        # Use provided model or fall back to config model
        self.model = model or self.agent_config["agent_metadata"]["model"]

        self._load_resources()
        self._create_agent()

    def _load_resources(self) -> None:
        """Load required template and documentation files."""
        # Go up from src/cv_writer_mcp/conversion/ to project root
        base_path = Path(__file__).parent.parent.parent.parent

        # Load LaTeX template
        template_path = base_path / "context" / "latex" / self.template_name
        self.latex_template = read_text_file(template_path, "LaTeX template", ".tex")
        logger.info(f"Loaded LaTeX template from {template_path}")

        # Load user guide
        userguide_path = base_path / "context" / "latex" / "moderncv_userguide.txt"
        self.userguide_content = read_text_file(
            userguide_path, "ModernCV user guide", ".txt"
        )
        logger.info(f"Loaded moderncv user guide from {userguide_path}")

        # Load personal information (source of truth for candidate data)
        personal_info_path = base_path / "context" / "latex" / "personal_info.txt"
        self.personal_info_content = read_text_file(
            personal_info_path, "Personal information", ".txt"
        )
        logger.info(f"Loaded personal information from {personal_info_path}")

    def _create_agent(self) -> None:
        """Create the OpenAI agent with structured output."""
        # Get the agent instructions from YAML configuration
        agent_instructions = self.agent_config["instructions"].format(
            moderncv_guide=self.userguide_content,
            moderncv_template=self.latex_template,
            personal_info=self.personal_info_content,
        )

        # Create agent using centralized helper with safe defaults
        self.agent = create_agent_from_config(
            agent_config=self.agent_config,
            instructions=agent_instructions,
            model=self.model,
        )
        logger.info("Created OpenAI agent with structured output")

    async def convert(
        self,
        request: MarkdownToLaTeXRequest,
        progress_callback: ProgressCallback = None,
    ) -> MarkdownToLaTeXResponse:
        """Convert markdown CV content to LaTeX using OpenAI Agents SDK.

        Args:
            request: Markdown to LaTeX conversion request
            progress_callback: Optional callback for progress reporting (0-100)

        Returns:
            Conversion response with LaTeX file URL or error message
        """
        try:
            # Report start
            if progress_callback:
                await progress_callback(0)

            logger.info("Starting markdown to LaTeX conversion using OpenAI Agents SDK")

            # Create the user prompt using YAML template
            user_prompt = self.agent_config["prompt_template"].format(
                markdown_content=request.markdown_content
            )

            if progress_callback:
                await progress_callback(10)

            # Run the agent with structured output
            # Use periodic ticker to prevent MCP timeout during long operation
            logger.debug("Running OpenAI agent (this may take 10-30 seconds)...")
            logger.info("🔄 Progress reporting enabled - updates every 10 seconds")
            
            async with PeriodicProgressTicker(
                progress_callback,
                start_percent=10,
                end_percent=85,
                interval_seconds=10.0,
                step_size=5,
            ):
                result = await Runner.run(self.agent, user_prompt)

            if progress_callback:
                await progress_callback(85)

            # Extract structured output
            latex_output: LaTeXOutput = result.final_output

            # Validate the LaTeX content
            if not latex_output.latex_content or not latex_output.latex_content.strip():
                return MarkdownToLaTeXResponse(
                    status=CompletionStatus.FAILED,
                    tex_url=None,
                    message=f"Agent response does not contain valid LaTeX content. Conversion notes: {latex_output.conversion_notes}",
                )

            if progress_callback:
                await progress_callback(90)

            # Save the LaTeX file
            output_filename = request.output_filename or "cv_converted.tex"
            if not output_filename.endswith(".tex"):
                output_filename += ".tex"

            output_path = Path(os.getenv("OUTPUT_DIR", "./output")) / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(latex_output.latex_content, encoding="utf-8")

            # Generate resource URI
            tex_url = f"cv-writer://tex/{output_filename}"

            logger.info(f"Successfully converted markdown to LaTeX: {output_filename}")

            if progress_callback:
                await progress_callback(100)

            return MarkdownToLaTeXResponse(
                status=CompletionStatus.SUCCESS,
                tex_url=tex_url,
                message=f"Successfully converted to {output_filename}. {latex_output.conversion_notes}",
            )

        except Exception as e:
            logger.error(f"Error in markdown to LaTeX conversion: {e}")
            return MarkdownToLaTeXResponse(
                status=CompletionStatus.FAILED,
                tex_url=None,
                message=f"Conversion failed: {str(e)}",
            )
