"""Coordinator for PDF style analysis using two specialized agents."""

import asyncio
import os
from pathlib import Path

from loguru import logger

from .latex_fix_agent import LaTeXFixAgent
from .models import CompletionStatus, PDFAnalysisRequest, PDFAnalysisResponse
from .page_capture_agent import PageCaptureAgent, PageCaptureRequest
from .utils import read_text_file


class PDFStyleCoordinator:
    """Coordinates the PDF style analysis using page capture and LaTeX fix agents."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.model = model or "gpt-5-mini"
        
        # Initialize both agents
        self.page_capture_agent = PageCaptureAgent(api_key=self.api_key, model=self.model)
        self.latex_fix_agent = LaTeXFixAgent(api_key=self.api_key, model=self.model)

    async def analyze_and_improve(self, request: PDFAnalysisRequest) -> PDFAnalysisResponse:
        """Analyze PDF and improve LaTeX using two specialized agents.
        
        This coordinator:
        1. Uses PageCaptureAgent to capture pages and analyze visual issues
        2. Uses LaTeXFixAgent to implement the suggested fixes
        
        Args:
            request: PDF analysis request with file paths
            
        Returns:
            PDFAnalysisResponse with improved LaTeX content
        """
        try:
            logger.info("Starting coordinated PDF style analysis")

            # Validate file paths
            pdf_path = Path(request.pdf_file_path)
            tex_path = Path(request.tex_file_path)
            
            if not pdf_path.exists() or not tex_path.exists():
                missing_files = []
                if not pdf_path.exists():
                    missing_files.append(f"PDF file: {pdf_path}")
                if not tex_path.exists():
                    missing_files.append(f"LaTeX file: {tex_path}")
                return PDFAnalysisResponse(
                    status=CompletionStatus.FAILED,
                    improved_tex_url=None,
                    message=f"Files not found: {', '.join(missing_files)}",
                )

            # Step 1: Read LaTeX content
            latex_content = read_text_file(tex_path, "LaTeX source", ".tex")

            # Step 2: Capture pages and analyze visual issues
            logger.info("Step 1: Capturing pages and analyzing visual issues")
            capture_request = PageCaptureRequest(
                pdf_file_path=str(pdf_path.absolute()),
                num_pages=None  # Let the tool determine actual page count
            )
            
            capture_response = await self.page_capture_agent.capture_and_analyze(capture_request)
            
            if capture_response.status != CompletionStatus.SUCCESS:
                return PDFAnalysisResponse(
                    status=CompletionStatus.FAILED,
                    improved_tex_url=None,
                    message=f"Page capture failed: {capture_response.message}",
                )

            logger.info(f"Captured {capture_response.pages_analyzed} pages, found {len(capture_response.visual_issues)} issues")

            # Step 3: Implement LaTeX fixes
            logger.info("Step 2: Implementing LaTeX fixes")
            
            # Combine visual analysis results
            visual_analysis_summary = f"""
            PAGES ANALYZED: {capture_response.pages_analyzed}
            ANALYSIS SUMMARY: {capture_response.analysis_summary}
            
            VISUAL ISSUES FOUND:
            {chr(10).join(f"- {issue}" for issue in capture_response.visual_issues)}
            """
            
            fix_output = await self.latex_fix_agent.implement_fixes(
                latex_content=latex_content,
                visual_analysis_results=visual_analysis_summary,
                suggested_fixes=capture_response.suggested_fixes
            )

            if fix_output.status != CompletionStatus.SUCCESS:
                return PDFAnalysisResponse(
                    status=CompletionStatus.FAILED,
                    improved_tex_url=None,
                    message=f"LaTeX fix implementation failed: {fix_output.implementation_notes}",
                )

            # Step 4: Save improved LaTeX
            output_filename = request.output_filename or "improved.tex"
            if not output_filename.endswith(".tex"):
                output_filename += ".tex"

            output_path = Path("./output") / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(fix_output.improved_latex_content, encoding="utf-8")

            improved_tex_url = f"cv-writer://tex/{output_filename}"
            
            # Create comprehensive message
            message = f"""Successfully analyzed PDF and improved LaTeX: {output_filename}
            
Analysis Results:
- Pages analyzed: {capture_response.pages_analyzed}
- Visual issues found: {len(capture_response.visual_issues)}
- Fixes implemented: {len(fix_output.fixes_applied)}

Improvements made:
{chr(10).join(f"- {fix}" for fix in fix_output.fixes_applied[:5])}
{f"... and {len(fix_output.fixes_applied) - 5} more fixes" if len(fix_output.fixes_applied) > 5 else ""}
"""

            logger.info(f"Successfully completed coordinated PDF analysis: {output_filename}")

            return PDFAnalysisResponse(
                status=CompletionStatus.SUCCESS,
                improved_tex_url=improved_tex_url,
                message=message.strip(),
            )

        except Exception as e:
            logger.error(f"Error in coordinated PDF analysis: {e}")
            return PDFAnalysisResponse(
                status=CompletionStatus.FAILED,
                improved_tex_url=None,
                message=f"Coordinated analysis failed: {str(e)}",
            )
