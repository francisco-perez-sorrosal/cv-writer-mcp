"""Style package tools."""

import asyncio
import json
import os
import time
from pathlib import Path

from agents import function_tool
from loguru import logger

from .pdf_computer import PDFPlaywrightComputer


@function_tool
async def pdf_computer_use_tool(pdf_file_path: str) -> str:
    """Capture PDF pages using Computer Use with Playwright browser automation.

    This tool uses OpenAI's Computer Use capabilities through Playwright to open a PDF
    in a browser, navigate through pages, capture screenshots, and analyze visual issues.

    Args:
        pdf_file_path: Path to the PDF file to capture and analyze

    Returns:
        JSON string with analysis results and suggested fixes
    """
    try:
        start_time = time.time()
        logger.info(f"Starting Computer Use PDF page capture: {pdf_file_path}")

        # Initialize OpenAI client for visual analysis
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        page_analyses = []
        screenshot_paths = []

        # Create output directory
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use Computer Use with Playwright to capture PDF pages
        async with PDFPlaywrightComputer() as computer:
            try:
                # Capture all pages using browser automation
                pages_data = await computer.capture_all_pages(pdf_file_path)

                if not pages_data:
                    return json.dumps({
                        "success": False,
                        "error_message": "No pages were captured",
                        "capture_time": time.time() - start_time,
                        "pages_captured": 0,
                        "page_analyses": [],
                        "visual_issues": [],
                        "suggested_fixes": [],
                        "screenshot_paths": [],
                        "pdf_file": pdf_file_path,
                    })

                # Save screenshots and analyze each page
                timestamp = int(time.time())

                for page_data in pages_data:
                    page_num = page_data['page_number']
                    screenshot_bytes = page_data['screenshot_bytes']
                    screenshot_base64 = page_data['screenshot_base64']

                    # Save screenshot to file
                    image_path = output_dir / f"computer_use_{timestamp}-{page_num}.png"
                    image_path.write_bytes(screenshot_bytes)
                    screenshot_paths.append(str(image_path))

                    logger.info(f"Saved screenshot for page {page_num}: {image_path} ({len(screenshot_bytes)} bytes)")
                    logger.info(f"Screenshot base64 length: {len(screenshot_base64)} chars")
                    logger.info(f"Analyzing page {page_num} with Computer Use")

                    try:
                        # Validate screenshot data before sending
                        if len(screenshot_bytes) < 100:
                            logger.warning(f"Screenshot for page {page_num} seems too small ({len(screenshot_bytes)} bytes)")

                        # Analyze page using OpenAI Vision API (use full gpt-4o for better vision capabilities)
                        response = openai_client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": """You are a CV visual analysis expert. Analyze this PDF page screenshot and identify specific visual formatting issues.

CRITICAL RULES - Follow these formatting guidelines:
1. NEVER use standalone \\begin{itemize} as first-level entries in any section
2. ALWAYS prefer moderncv commands: \\cventry, \\cvitem, \\cvskill
3. Use \\begin{itemize} ONLY inside \\cventry or \\cvitem descriptions when meaningful
4. ALWAYS use compact formatting: [noitemsep,topsep=0pt,parsep=0pt,partopsep=0pt] for itemize
5. Create concise, descriptive labels (≤8 chars when possible)
6. Eliminate ALL redundancy between labels and content
7. Use consistent font sizes - no \\large, \\Large, \\small in regular content
8. Reduce spacing: use \\vspace{0.2em} or \\vspace{0.3em} instead of larger values

Examples:
- BAD: \\begin{itemize}\\item Python Programming\\item Java Development\\end{itemize}
- GOOD: \\cvitem{Code}{Python, Java Development}
- BAD: \\begin{itemize}\\item 2020-2023 Software Engineer at Company\\end{itemize}
- GOOD: \\cventry{2020-2023}{Software Engineer}{Company}{Location}{}{Led development projects...}
- BAD: \\cvitem{Languages}{\\textbf{Languages:} Spanish, English}
- GOOD: \\cvitem{Langs}{Spanish (Native), English (Fluent)}

Return your analysis as a concise list of specific issues with exact text quotes and LaTeX fixes."""
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"Analyze page {page_num} of this CV for visual formatting issues. For each problem, provide: 1) Description of issue 2) Exact text content affected 3) Specific LaTeX command to fix it 4) Location in document. Be specific and quote exact text."
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{screenshot_base64}",
                                                "detail": "high"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_completion_tokens=1500
                        )

                        analysis = response.choices[0].message.content
                        page_analyses.append({
                            "page_number": page_num,
                            "analysis": analysis,
                            "screenshot_path": str(image_path)
                        })

                        logger.info(f"Page {page_num} analysis completed with Computer Use")
                        logger.debug(f"Page {page_num} analysis results: {analysis}")

                        # Extract and log metadata about the analysis
                        try:
                            # Count issues found in the analysis text
                            analysis_lower = analysis.lower()
                            issue_count = analysis_lower.count('issue:') + analysis_lower.count('affected text:') + analysis_lower.count('fix:')
                            fix_count = analysis_lower.count('latex fix:') + analysis_lower.count('fix command:') + analysis_lower.count('latex:')

                            # Log detailed metadata for this page
                            logger.info(f"Page {page_num} metadata:")
                            logger.info(f"  - Screenshot size: {len(screenshot_bytes)} bytes")
                            logger.info(f"  - Base64 encoding size: {len(screenshot_base64)} characters")
                            logger.info(f"  - Analysis response length: {len(analysis)} characters")
                            logger.info(f"  - Issues identified: {issue_count}")
                            logger.info(f"  - LaTeX fixes suggested: {fix_count}")
                            logger.info(f"  - Screenshot saved to: {image_path}")

                            # Extract specific issues if possible
                            if 'issue:' in analysis_lower:
                                issues = []
                                lines = analysis.split('\n')
                                for line in lines:
                                    if 'issue:' in line.lower() or 'affected text:' in line.lower():
                                        issues.append(line.strip())

                                if issues:
                                    logger.info(f"  - Specific issues found on page {page_num}:")
                                    for i, issue in enumerate(issues[:5], 1):  # Log first 5 issues
                                        logger.info(f"    {i}. {issue}")
                                    if len(issues) > 5:
                                        logger.info(f"    ... and {len(issues) - 5} more issues")

                        except Exception as meta_e:
                            logger.warning(f"Could not extract analysis metadata for page {page_num}: {meta_e}")

                    except Exception as e:
                        logger.error(f"Error analyzing page {page_num} with Computer Use: {e}")
                        logger.error(f"Screenshot info - bytes: {len(screenshot_bytes)}, base64: {len(screenshot_base64)}, path: {image_path}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        page_analyses.append({
                            "page_number": page_num,
                            "analysis": f"Analysis failed: {str(e)}",
                            "screenshot_path": str(image_path)
                        })

            except Exception as e:
                logger.error(f"Computer Use PDF capture failed: {e}")
                return json.dumps({
                    "success": False,
                    "error_message": f"Computer Use PDF capture failed: {str(e)}",
                    "capture_time": time.time() - start_time,
                    "pages_captured": 0,
                    "page_analyses": [],
                    "visual_issues": [],
                    "suggested_fixes": [],
                    "screenshot_paths": [],
                    "pdf_file": pdf_file_path,
                })

        capture_time = time.time() - start_time
        logger.info(f"Completed Computer Use PDF analysis: {len(page_analyses)} pages in {capture_time:.2f}s")
        logger.info(f"Screenshots saved to: {screenshot_paths}")

        # Process results using the same extraction logic as the original tool
        def extract_issues_from_analysis(analysis_text: str, page_num: int) -> list[str]:
            """Extract individual issues from analysis text."""
            issues = []

            # Split by common issue markers
            lines = analysis_text.split('\n')
            current_issue = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Look for issue markers (### 1., ### 2., Issue 1:, etc.)
                if (line.startswith('###') and any(char.isdigit() for char in line[:20])) or \
                   (line.startswith('Issue') and ':' in line[:20]) or \
                   (line.startswith(f'{page_num}.') and len(line) < 100):
                    # Save previous issue if exists
                    if current_issue:
                        issue_text = ' '.join(current_issue).strip()
                        if len(issue_text) > 10:  # Skip very short issues
                            issues.append(f"Page {page_num}: {issue_text}")
                    # Start new issue
                    current_issue = [line]
                else:
                    # Continue current issue
                    current_issue.append(line)

            # Don't forget the last issue
            if current_issue:
                issue_text = ' '.join(current_issue).strip()
                if len(issue_text) > 10:
                    issues.append(f"Page {page_num}: {issue_text}")

            # Fallback: if no issues were parsed, treat the whole analysis as one issue
            if not issues and len(analysis_text.strip()) > 10:
                issues.append(f"Page {page_num}: {analysis_text.strip()}")

            return issues

        # Compile comprehensive analysis - extract individual issues from each page
        all_issues = []
        all_fixes = []

        for page_analysis in page_analyses:
            if page_analysis["analysis"] and not page_analysis["analysis"].startswith("Analysis failed"):
                # Extract individual issues from this page's analysis
                page_issues = extract_issues_from_analysis(
                    page_analysis["analysis"],
                    page_analysis["page_number"]
                )
                all_issues.extend(page_issues)
                all_fixes.append(page_analysis['analysis'])

        return json.dumps({
            "success": True,
            "capture_time": capture_time,
            "pages_captured": len(page_analyses),
            "actual_page_count": len(pages_data),
            "page_analyses": page_analyses,
            "visual_issues": all_issues,
            "suggested_fixes": all_fixes,
            "screenshot_paths": screenshot_paths,
            "pdf_file": pdf_file_path,
            "method": "computer_use_playwright"
        })

    except Exception as e:
        error_message = f"Computer Use PDF analysis failed: {str(e)}"
        logger.error(error_message)

        return json.dumps({
            "success": False,
            "error_message": error_message,
            "capture_time": time.time() - start_time if 'start_time' in locals() else 0.0,
            "pages_captured": 0,
            "page_analyses": [],
            "visual_issues": [],
            "suggested_fixes": [],
            "screenshot_paths": [],
            "pdf_file": pdf_file_path,
            "method": "computer_use_playwright"
        })
