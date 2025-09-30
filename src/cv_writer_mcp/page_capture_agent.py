"""Page capture agent for PDF visual analysis."""

import json
import os
from pathlib import Path

from loguru import logger
from agents import Agent, Runner

from .models import CompletionStatus, PageCaptureRequest, PageCaptureResponse
from .tools import pdf_computer_use_tool
from .utils import load_agent_config


class PageCaptureAgent:
    """Page capture agent that captures PDF pages and analyzes visual issues."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.agent_config = load_agent_config("page_capture_agent.yaml")
        self.model = model or self.agent_config["agent_metadata"]["model"]

    def _create_agent(self) -> Agent:
        """Create page capture agent."""
        from .models import get_output_type_class
        
        output_type_class = get_output_type_class(
            self.agent_config["agent_metadata"]["output_type"]
        )
        
        return Agent(
            name=self.agent_config["agent_metadata"]["name"],
            instructions=self.agent_config["instructions"],
            tools=[pdf_computer_use_tool],
            model=self.model,
            output_type=output_type_class,
        )


    async def capture_and_analyze(self, request: PageCaptureRequest) -> PageCaptureResponse:
        """Capture PDF pages and analyze visual formatting issues."""
        try:
            pdf_path = Path(request.pdf_file_path)
            if not pdf_path.exists():
                return PageCaptureResponse(
                    status=CompletionStatus.FAILED, 
                    message=f"PDF file not found: {pdf_path}"
                )

            agent = self._create_agent()
            prompt = f"Capture and analyze all pages of the PDF file: {pdf_path}"
            result = await Runner.run(agent, prompt)
            
            # Extract results - try final output first, then tool results
            if hasattr(result, 'final_output') and result.final_output:
                output = result.final_output
                if output.status == CompletionStatus.SUCCESS:
                    return PageCaptureResponse(
                        status=CompletionStatus.SUCCESS,
                        pages_analyzed=output.pages_analyzed,
                        visual_issues=getattr(output, 'visual_issues', []),
                        suggested_fixes=getattr(output, 'suggested_fixes', []),
                        analysis_summary=f"Analyzed {output.pages_analyzed} pages. Found {len(getattr(output, 'visual_issues', []))} visual issues.",
                        message=f"Successfully analyzed {output.pages_analyzed} pages."
                    )
            
            # Fallback: extract from tool call results
            tool_data = self._extract_tool_results(result)
            if tool_data:
                return PageCaptureResponse(
                    status=CompletionStatus.SUCCESS,
                    pages_analyzed=tool_data.get('pages_captured', 0),
                    visual_issues=tool_data.get('visual_issues', []),
                    suggested_fixes=tool_data.get('suggested_fixes', []),
                    analysis_summary=f"Analyzed {tool_data.get('pages_captured', 0)} pages. Found {len(tool_data.get('visual_issues', []))} visual issues.",
                    message=f"Successfully analyzed {tool_data.get('pages_captured', 0)} pages."
                )

            return PageCaptureResponse(
                status=CompletionStatus.FAILED,
                message="No analysis results found"
            )

        except Exception as e:
            logger.error(f"Page capture failed: {e}")
            return PageCaptureResponse(
                status=CompletionStatus.FAILED,
                message=f"Analysis failed: {str(e)}"
            )
    
    def _extract_tool_results(self, result) -> dict | None:
        """Extract results from tool call messages."""
        if not hasattr(result, 'messages') or not result.messages:
            return None
            
        for message in result.messages:
            if hasattr(message, 'tool_call_results'):
                for tool_result in message.tool_call_results:
                    try:
                        data = json.loads(tool_result.content)
                        if data.get('success'):
                            return data
                    except json.JSONDecodeError:
                        continue
        return None
