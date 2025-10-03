"""Compilation-specific tools for LaTeX to PDF conversion."""

import asyncio
import json
import time
from pathlib import Path

from agents import function_tool
from loguru import logger


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

        # Check if PDF was created (note: pdflatex can create PDF even with errors)
        pdf_path = Path(output_dir) / f"{Path(tex_file_path).stem}.pdf"
        pdf_created = pdf_path.exists() and pdf_path.stat().st_size > 0

        # True success means: PDF created AND exit code 0
        compilation_success = pdf_created and process.returncode == 0

        logger.info(
            f"LaTeX execution: PDF created={pdf_created}, "
            f"exit_code={process.returncode}, clean_success={compilation_success}"
        )

        # Read .log file if compilation failed - agent needs this for error extraction
        log_file_content = ""
        if process.returncode != 0:
            log_file_path = Path(output_dir) / f"{Path(tex_file_path).stem}.log"
            if log_file_path.exists():
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        log_file_content = f.read()
                    logger.info(f"Read .log file ({len(log_file_content)} chars) for error extraction")
                except Exception as e:
                    logger.warning(f"Could not read .log file: {e}")

        return json.dumps({
            "success": compilation_success,
            "compilation_time": compilation_time,
            "error_message": (
                None
                if compilation_success
                else f"LaTeX compilation had errors (exit code: {process.returncode})"
            ),
            "log_summary": output_text,
            "engine_used": command.split()[0],
            "output_path": str(pdf_path) if pdf_created else "",
            "return_code": process.returncode,
            "log_file_content": log_file_content,  # Agent can extract errors from this
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
