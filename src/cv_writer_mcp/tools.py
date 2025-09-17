import asyncio
import base64
import json
import os
import time
from pathlib import Path

from agents import function_tool
from loguru import logger
from playwright.async_api import async_playwright


@function_tool
async def latex2pdf_tool(
    command: str, tex_file_path: str, output_dir: str, timeout: int = 30
) -> str:
    """Compile LaTeX file using shell command.

    Args:
        command: The LaTeX compilation command to execute
        tex_file_path: Path to the .tex file
        output_dir: Directory where PDF should be generated
        timeout: Timeout for the compilation

    Returns:
        JSON string with compilation result
    """
    try:
        start_time = time.time()

        # Execute the command        
        logger.info(f"Compile LaTeX command: {command}")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)

        compilation_time = time.time() - start_time
        
        output_text = stdout.decode("utf-8", errors="ignore")

        # Check if PDF was created using both file existence and log analysis
        pdf_path = Path(output_dir) / f"{Path(tex_file_path).stem}.pdf"
        compilation_success = pdf_path.exists() and pdf_path.stat().st_size > 0
        
        logger.info(f"Compilation success: {compilation_success}")

        return json.dumps({
            "success": compilation_success,
            "compilation_time": compilation_time,
            "error_message": (
                None
                if compilation_success
                else f"LaTeX compilation failed (exit code: {process.returncode})"
            ),
            "log_summary": output_text,
            "engine_used": command.split()[0],
            "output_path": str(pdf_path) if compilation_success else "",
            "return_code": process.returncode,
        })

    except Exception as e:
        error_message = f"Tool error: {str(e)}"
        if isinstance(e, TimeoutError) or isinstance(e, asyncio.TimeoutError):
            error_message = f"Tool error: LaTeX compilation timed out after {timeout} seconds"
        
        return json.dumps(
            {
                "success": False,
                "error_message": error_message,
                "compilation_time": 0.0,
                "return_code": 1,
            }
        )


@function_tool
async def capture_pdf_pages_tool(pdf_file_path: str) -> str:
    """Convert PDF pages to images and analyze them for visual issues.
    
    This tool converts PDF pages to PNG images using poppler-utils and performs
    immediate visual analysis using OpenAI Vision API to provide actionable LaTeX fixes.
    
    Args:
        pdf_file_path: Path to the PDF file to capture and analyze
        
    Returns:
        JSON string with analysis results and suggested fixes
    """
    try:
        import subprocess
        
        start_time = time.time()
        logger.info(f"Starting PDF page capture and analysis: {pdf_file_path}")
        
        # First, get actual page count from PDF
        try:
            result = subprocess.run(
                ["pdfinfo", pdf_file_path], 
                capture_output=True, 
                text=True, 
                check=True
            )
            pages_line = [line for line in result.stdout.split('\n') if line.startswith('Pages:')]
            actual_page_count = int(pages_line[0].split(':')[1].strip()) if pages_line else 4
            logger.info(f"PDF has {actual_page_count} pages")
        except Exception as e:
            logger.warning(f"Could not determine page count, defaulting to 4: {e}")
            actual_page_count = 4
        
        # Initialize OpenAI client for immediate analysis
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        page_analyses = []
        screenshot_paths = []
        
        # Create output directory
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PDF to images using pdftoppm (part of poppler-utils)
        timestamp = int(time.time())
        image_prefix = f"page_capture_{timestamp}"
        
        try:
            # Use pdftoppm to convert PDF pages to PNG images
            subprocess.run([
                "pdftoppm", 
                "-png", 
                "-r", "150",  # 150 DPI for good quality
                pdf_file_path,
                str(output_dir / image_prefix)
            ], check=True, capture_output=True)
            
            logger.info(f"Successfully converted PDF to {actual_page_count} PNG images")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return json.dumps({
                "success": False,
                "error_message": f"PDF to image conversion failed: {str(e)}",
                "capture_time": 0.0,
                "pages_captured": 0,
                "page_analyses": [],
                "visual_issues": [],
                "suggested_fixes": [],
                "screenshot_paths": [],
                "pdf_file": pdf_file_path,
            })
        
        # Analyze each generated image
        for page_num in range(1, actual_page_count + 1):
            try:
                # pdftoppm generates files with format: prefix-1.png, prefix-2.png, etc.
                image_path = output_dir / f"{image_prefix}-{page_num}.png"
                
                if not image_path.exists():
                    logger.warning(f"Image not found for page {page_num}: {image_path}")
                    continue
                
                logger.info(f"Analyzing page {page_num}: {image_path}")
                screenshot_paths.append(str(image_path))
                
                # Immediate visual analysis using OpenAI Vision
                try:
                    with open(image_path, "rb") as f:
                        screenshot_data = f.read()
                    
                    screenshot_base64 = base64.b64encode(screenshot_data).decode('utf-8')
                    
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # Fixed model name
                        messages=[
                            {
                                "role": "system",
                                "content": """You are a CV visual analysis expert. Analyze this PDF page screenshot and identify specific visual formatting issues.

For each problem you identify, provide:
1. A clear description of the issue
2. The specific content/text that is affected (quote exact text from the PDF)
3. The specific LaTeX/moderncv command(s) to fix it
4. The approximate location in the document

Focus on: text alignment, spacing, indentation, typography, layout, and professional appearance. Be extremely specific about which text content has issues to help locate it in the LaTeX source.

CRITICAL INDENTATION CONSISTENCY RULES - Pay special attention to:
- PROFESSIONAL EXPERIENCE: All role titles, companies, and descriptions within each time period MUST be indented at exactly the same level
- ADDITIONAL QUALIFICATIONS: All items within this section MUST have consistent left margins and indentation
- ADDITIONAL QUALIFICATIONS SUBSECTIONS: All subsection headers (e.g., "Certifications:", "Technical Skills:", "Languages:") MUST align perfectly, and their content must have consistent sub-indentation
- SECTION UNIFORMITY: All similar content types (roles, qualifications, skills, etc.) MUST align at the same indentation level across the entire document
- BULLET POINT CONSISTENCY: All bullet points within the same section MUST start at the same horizontal position
- MARGIN ALIGNMENT: Check that all subsections and their content maintain consistent left margins
- SUBSECTION CONTENT ALIGNMENT: Within Additional Qualifications, all content under each subsection must have identical indentation

Common LaTeX/moderncv commands to suggest:
- \\vspace{0.5em} - Small vertical spacing
- \\cvitem{label}{content} - Consistent item formatting (keep label ≤10-12 chars for aesthetics)
- \\cventry{year}{title}{company}{location}{grade}{description} - Experience entries
- \\section{title} - Main sections
- \\subsection{title} - Subsections within Additional Qualifications
- \\cvlistitem{content} - Consistent bullet items in Additional Qualifications
- \\textbf{text} - Bold text
- \\setlength{\\leftmargini}{1em} - Standardize bullet indentation
- \\setlength{\\leftmarginii}{2em} - Standardize nested bullet indentation
- \\begin{itemize}[leftmargin=1em] - Consistent bullet margins
- \\begin{itemize}[leftmargin=2em] - Nested bullet margins for subsections
- \\hspace{1em} - Manual horizontal alignment when needed
- \\cvitem{Subsection Title}{\\begin{itemize}[leftmargin=1em]\\item content\\end{itemize}} - Structured subsection format

CRITICAL LABEL LENGTH GUIDELINES:
- Use short labels (≤10-12 characters) for \\cvitem commands to maintain visual balance
- Good examples: "2024-2025", "PhD", "Skills", "Contact", "Address", "Email"
- Avoid long labels: "Professional Experience", "Additional Qualifications", "Work History"
- For long labels, leave the first parameter empty: \\cvitem{}{content}
- Years and periods work well: "2020-2023", "Jan 2024", "2024--2025", "2024"
- Short descriptors are ideal: "Phone", "Location", "Degree", "Company"

CRITICAL ANTI-REDUNDANCY REQUIREMENTS:
- Never duplicate information between labels and content in the same command
- Example BAD: \\cventry{2020-2023}{Software Engineer}{Company Name}{Location}{}{Software Engineer at Company Name...}
- Example GOOD: \\cventry{2020-2023}{Software Engineer}{Company Name}{Location}{}{Led development of...}
- Remove redundant words from content when they appear in labels
- For \\cvitem commands, avoid repeating label text in the content
- Example BAD: \\cvitem{Education}{Education: Bachelor's Degree in Computer Science}
- Example GOOD: \\cvitem{Education}{Bachelor's Degree in Computer Science}
- If label contains a word/phrase, remove it from the content
- If content starts with the same word as the label, remove it from content
- Preserve all unique information while eliminating duplication

CRITICAL TEXT COMPRESSION GUIDELINES:
- Compress long text to fit in labels (≤10-12 characters) while maintaining meaning
- Examples: "Development Tools" → "Dev Tools", "Programming Languages" → "Languages", "Work Experience" → "Experience"
- Use abbreviations that are commonly understood: "Tech", "Dev", "Mgmt", "Admin", "Coord"
- Preserve essential meaning while reducing character count
- Use industry-standard abbreviations when appropriate

CRITICAL MODERNCV COMMAND OPTIMIZATION:
Command Hierarchy (Best to Worst):
1. \\cventry - For structured entries with multiple fields (experience, education, projects, achievements)
2. \\cvitem - For labeled content items  
3. \\cvitemwithcomment - For items with additional comments
4. \\cvskill - For skills with proficiency levels
5. \\cvlanguage - For language skills
6. \\cvline - For simple labeled lines
7. \\begin{itemize} - Only as last resort when content cannot be structured

Examples:
- BAD: \\begin{itemize}\\item Python Programming\\item Java Development\\end{itemize}
- GOOD: \\cvitem{Programming}{Python, Java Development}
- BAD: \\begin{itemize}\\item 2020-2023 Software Engineer at Company\\end{itemize}
- GOOD: \\cventry{2020-2023}{Software Engineer}{Company}{Location}{}{Led development projects...}

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
                                            "url": f"data:image/png;base64,{screenshot_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=800
                    )
                    
                    analysis = response.choices[0].message.content
                    page_analyses.append({
                        "page_number": page_num,
                        "analysis": analysis,
                        "screenshot_path": str(image_path)
                    })
                    
                    logger.info(f"Page {page_num} analysis completed")
                    logger.debug(f"Page {page_num} analysis results: {analysis}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing page {page_num}: {e}")
                    page_analyses.append({
                        "page_number": page_num,
                        "analysis": f"Analysis failed: {str(e)}",
                        "screenshot_path": str(image_path)
                    })
                
            except Exception as e:
                logger.warning(f"Could not process page {page_num}: {e}")
                continue
        
        capture_time = time.time() - start_time
        logger.info(f"Completed PDF analysis: {len(page_analyses)} pages in {capture_time:.2f}s")
        
        # Compile comprehensive analysis
        all_issues = []
        all_fixes = []
        
        for page_analysis in page_analyses:
            if page_analysis["analysis"] and not page_analysis["analysis"].startswith("Analysis failed"):
                all_issues.append(f"Page {page_analysis['page_number']}: {page_analysis['analysis']}")
                all_fixes.append(page_analysis['analysis'])
        
        return json.dumps({
            "success": True,
            "capture_time": capture_time,
            "pages_captured": len(page_analyses),
            "actual_page_count": actual_page_count,
            "page_analyses": page_analyses,
            "visual_issues": all_issues,
            "suggested_fixes": all_fixes,
            "screenshot_paths": screenshot_paths,
            "pdf_file": pdf_file_path,
        })
        
    except Exception as e:
        error_message = f"PDF analysis failed: {str(e)}"
        logger.error(error_message)
        
        return json.dumps({
            "success": False,
            "error_message": error_message,
            "capture_time": 0.0,
            "pages_captured": 0,
            "page_analyses": [],
            "visual_issues": [],
            "suggested_fixes": [],
            "screenshot_paths": [],
            "pdf_file": pdf_file_path,
        })
