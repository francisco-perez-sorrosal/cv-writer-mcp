"""Markdown to LaTeX conversion agent using OpenAI Agents SDK."""

import os
from pathlib import Path

from agents import Agent, Runner
from loguru import logger
from pydantic import BaseModel, Field

from .models import ConversionStatus, MarkdownToLaTeXRequest, MarkdownToLaTeXResponse
from .utils import read_text_file


class LaTeXOutput(BaseModel):
    """Structured output for LaTeX conversion."""

    latex_content: str = Field(
        description="The converted LaTeX content ready for insertion into the template"
    )
    conversion_notes: str = Field(
        description="Notes about the conversion process and any important considerations"
    )


class MD2LaTeXAgent:
    """Markdown to LaTeX conversion agent using OpenAI Agents SDK with moderncv template."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-5-mini"):
        """Initialize the markdown to LaTeX conversion agent.

        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for conversion
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
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
        system_prompt = self._build_system_prompt()

        self.agent = Agent(
            name="MD2LaTeXAgent",
            instructions=system_prompt,
            model=self.model,
            output_type=LaTeXOutput,
        )
        logger.info("Created OpenAI agent with structured output")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the OpenAI agent."""
        return f"""You are an expert developer and editor of Markdown and LaTeX. You are specialized in professional CV creation using Markdown and the LaTeX moderncv package. Your task is to convert Markdown CV content into a complete, compilable LaTeX document using the moderncv package.

You must return a structured response with two fields:
1. latex_content: A complete, compilable LaTeX document ready for compilation translated from the Markdown content
2. conversion_notes: Notes about the conversion process and any important considerations

<conversion_guidelines>
CRITICAL REQUIREMENTS:
1. Return a COMPLETE LaTeX document that can be compiled directly to PDF
2. Remember to remove Markdown formatting from the Markdown content before converting it to LaTeX
3. Include ALL necessary LaTeX structure: documentclass, packages, personal data, document begin/end
4. Extract personal information from the markdown and set up proper moderncv personal data commands
5. Use moderncv commands and environments throughout the document
6. Convert Markdown formatting to appropriate moderncv commands
6. Handle lists, dates, and contact information using moderncv-specific macros
7. Maintain professional formatting and consistent structure
8. Use appropriate moderncv sections and entries for different content types
9. Apply appropriate moderncv commands for each content type
10. Do not forget any information from the Markdown content
11. Provide helpful conversion_notes about any assumptions or recommendations

REQUIRED LaTeX STRUCTURE:
- Document class: \\documentclass[11pt,a4paper,sans]{{moderncv}}
- Style and color: \\moderncvstyle{{classic}}, \\moderncvcolor{{blue}}
- Geometry: \\usepackage[scale=0.75]{{geometry}}
- Font encoding: \\usepackage[T1]{{fontenc}}, \\usepackage{{lmodern}}
- Language: \\usepackage[english]{{babel}}
- Symbols: \\usepackage{{amssymb}} (for \\checkmark and other symbols)
- Personal data commands: \\name{{first}}{{last}}, \\title{{title}}, \\address{{street}}{{city}}{{country}}, \\phone{{number}}, \\email{{email}}, \\homepage{{url}}
- Document structure: \\begin{{document}}, \\makecvtitle, content sections, \\end{{document}}

IMPORTANT FORMATTING RULES:
- Do NOT include \\photo command unless you have an actual image file
- Escape special LaTeX characters (\\ → \\textbackslash{{}}, {{ }} → \\{{ \\}}, $ → \\$, & → \\&, # → \\#, % → \\%, ^ → \\textasciicircum{{}}, _ → \\_,  ~ → \\textasciitilde{{}}) in literal text only, preserving existing LaTeX commands.
- Convert Unicode characters (✓, →, etc.) to LaTeX equivalents (\\checkmark, \\rightarrow, etc.)
- Use proper \\cventry syntax: \\cventry{{year}}{{title}}{{institution}}{{location}}{{grade}}{{description}}
- For \\cventry, keep descriptions simple and avoid complex nested structures
- If description is complex, use \\cvitem instead of \\cventry
- Ensure all braces are properly matched in all commands
- Use \\textbf{{}} for bold text, \\textit{{}} for italic text
- Convert markdown links to LaTeX format or remove them if not essential
- For complex job descriptions, break them into multiple \\cvitem entries
</conversion_guidelines>

This is the moderncv user guide:
<moderncv_documentation>
{self.userguide_content}
</moderncv_documentation>

You get inspiration from this moderncv template as a reference for structuring the CV:
<template_structure>
{self.latex_template}
</template_structure>

<output_latex_requirements>
For the latex_content field, return a COMPLETE LaTeX document that includes:
- Document class declaration with moderncv
- All necessary package imports
- Personal data setup (\\name, \\email, \\phone, \\address, etc.) extracted from markdown
- Document begin/end structure
- \\makecvtitle command
- All converted content sections
- Proper moderncv formatting throughout

The document must be ready to compile with pdflatex without any additional modifications.

For the conversion_notes field, provide any important notes about the conversion process, assumptions made, or recommendations for the user.
</output_latex_requirements>"""

    async def convert(self, request: MarkdownToLaTeXRequest) -> MarkdownToLaTeXResponse:
        """Convert markdown CV content to LaTeX using OpenAI Agents SDK.

        Args:
            request: Markdown to LaTeX conversion request

        Returns:
            Conversion response with LaTeX file URL or error message
        """
        try:
            logger.info("Starting markdown to LaTeX conversion using OpenAI Agents SDK")

            # Create the user prompt
            user_prompt = f"""Convert the following markdown CV content to a complete, compilable LaTeX document using the moderncv package. Extract personal information (name, email, phone, address) from the markdown and set up the proper moderncv personal data commands. Create a complete LaTeX document that can be compiled directly to PDF.

<markdown_content>
{request.markdown_content}
</markdown_content>"""

            # Run the agent with structured output
            result = await Runner.run(self.agent, user_prompt)

            # Extract structured output
            latex_output: LaTeXOutput = result.final_output

            # Validate the LaTeX content
            if not latex_output.latex_content or not latex_output.latex_content.strip():
                raise ValueError("Agent response does not contain valid LaTeX content")

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
                status=ConversionStatus.SUCCESS,
                tex_url=tex_url,
                metadata={
                    "output_filename": output_filename,
                    "model_used": self.model,
                    "conversion_notes": latex_output.conversion_notes,
                    "template_used": "moderncv",
                },
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Error in markdown to LaTeX conversion: {e}")
            return MarkdownToLaTeXResponse(
                status=ConversionStatus.FAILED,
                tex_url=None,
                error_message=f"Conversion failed: {str(e)}",
            )
