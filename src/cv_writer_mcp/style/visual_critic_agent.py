"""Visual critic agent for PDF design quality critique."""

import base64
import json
import os
from pathlib import Path

from agents import Agent, Runner
from loguru import logger

from ..models import CompletionStatus, create_agent_from_config
from ..utils import load_agent_config
from .models import VisualCriticRequest, VisualCriticResponse
from .tools import capture_pdf_screenshots


class VisualCriticAgent:
    """Visual design critic that analyzes document screenshots and identifies quality issues.

    This agent receives screenshot images and provides professional design critique
    focusing on spacing, consistency, readability, and layout quality. It does NOT
    implement fixes - that's the responsibility of the FormattingAgent downstream.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.agent_config = load_agent_config("visual_critic_agent.yaml")
        self.model = model or self.agent_config["agent_metadata"]["model"]

    def _create_agent(self) -> Agent:
        """Create visual critic agent."""
        # Create agent using centralized helper with safe defaults
        return create_agent_from_config(
            agent_config=self.agent_config,
            instructions=self.agent_config["instructions"],
            model=self.model,
        )

    async def critique(self, request: VisualCriticRequest) -> VisualCriticResponse:
        """Critique PDF visual quality by analyzing screenshot images.

        Args:
            request: Contains PDF path to screenshot and analyze

        Returns:
            Design critiques and quality issues identified in the document
        """
        try:
            pdf_path = Path(request.pdf_file_path)
            if not pdf_path.exists():
                return VisualCriticResponse(
                    status=CompletionStatus.FAILED,
                    message=f"PDF file not found: {pdf_path}",
                )

            # Step 1: Convert PDF to screenshot images (utility function)
            logger.info(f"📸 Capturing screenshots from PDF: {pdf_path}")
            tool_result = await capture_pdf_screenshots(str(pdf_path.absolute()))
            tool_data = json.loads(tool_result)

            if not tool_data.get("success"):
                return VisualCriticResponse(
                    status=CompletionStatus.FAILED,
                    message=f"Screenshot capture failed: {tool_data.get('error_message', 'Unknown error')}",
                )

            screenshot_paths = tool_data.get("screenshot_paths", [])
            if not screenshot_paths:
                return VisualCriticResponse(
                    status=CompletionStatus.FAILED,
                    message="No screenshots captured",
                )

            logger.info(f"✅ Captured {len(screenshot_paths)} screenshots")

            # Step 2: Prepare images for visual critique
            content = []

            # Add all images to content array
            for i, screenshot_path in enumerate(screenshot_paths, 1):
                screenshot_file = Path(screenshot_path)
                if screenshot_file.exists():
                    try:
                        with open(screenshot_file, "rb") as image_file:
                            image_data = image_file.read()

                            # Validate image data is not empty
                            if not image_data:
                                logger.warning(
                                    f"Screenshot {i} is empty, skipping: {screenshot_file}"
                                )
                                continue

                            base64_image = base64.b64encode(image_data).decode("utf-8")

                            # Validate base64 encoding succeeded
                            if not base64_image:
                                logger.warning(
                                    f"Failed to encode screenshot {i}, skipping: {screenshot_file}"
                                )
                                continue

                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to process screenshot {i} ({screenshot_file}): {e}"
                        )
                        continue

            # Validate we have at least one image to analyze
            if not content:
                return VisualCriticResponse(
                    status=CompletionStatus.FAILED,
                    message="No valid screenshots were successfully encoded for analysis",
                )

            # Add the analysis prompt to same content array
            analysis_prompt = self.agent_config["prompt_template"]

            content.append({"type": "input_text", "text": analysis_prompt})

            # Create single user message with all images and prompt
            messages = [{"role": "user", "content": content}]

            logger.info(
                f"🎨 Sending {len(screenshot_paths)} images to Visual Critic Agent for analysis..."
            )

            # Step 3: Run visual critique agent
            agent = self._create_agent()
            result = await Runner.run(agent, messages)  # type: ignore[arg-type]

            # Step 4: Extract design critiques from agent output
            if hasattr(result, "final_output") and result.final_output:
                output = result.final_output
                if output.status == CompletionStatus.SUCCESS:
                    visual_issues = getattr(output, "visual_issues", [])
                    suggested_fixes = getattr(output, "suggested_fixes", [])

                    # Log agent's findings
                    logger.info(
                        f"✅ Visual critique complete: {len(visual_issues)} design issues identified across {output.pages_analyzed} pages"
                    )

                    # Log each visual issue for debugging
                    if visual_issues:
                        logger.info(
                            f"Visual issues found ({len(visual_issues)} total):"
                        )
                        for i, issue in enumerate(visual_issues[:10], 1):
                            logger.info(f"  {i}. {issue}")
                        if len(visual_issues) > 10:
                            logger.info(
                                f"  ... and {len(visual_issues) - 10} more issues"
                            )

                    return VisualCriticResponse(
                        status=CompletionStatus.SUCCESS,
                        pages_analyzed=output.pages_analyzed,
                        visual_issues=visual_issues,
                        suggested_fixes=suggested_fixes,
                        analysis_summary=f"Analyzed {output.pages_analyzed} pages. Found {len(visual_issues)} visual issues.",
                        message=f"Successfully analyzed {output.pages_analyzed} pages.",
                    )

            return VisualCriticResponse(
                status=CompletionStatus.FAILED,
                message="Agent did not return structured output",
            )

        except Exception as e:
            logger.error(f"Visual critique failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return VisualCriticResponse(
                status=CompletionStatus.FAILED, message=f"Analysis failed: {str(e)}"
            )
