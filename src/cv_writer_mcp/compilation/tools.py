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
