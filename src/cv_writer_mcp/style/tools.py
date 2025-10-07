"""Style package tools."""

import json
import time
from pathlib import Path

from agents import function_tool
from loguru import logger

from .pdf_computer import PDFPlaywrightComputer


async def capture_pdf_screenshots(pdf_file_path: str) -> str:
    """Capture PDF pages as screenshots using Playwright browser automation.

    This is a regular async function that can be called directly.

    Args:
        pdf_file_path: Path to the PDF file to capture

    Returns:
        JSON string with screenshot paths and capture metadata
    """
    try:
        start_time = time.time()
        logger.info(f"Starting PDF screenshot capture: {pdf_file_path}")

        screenshot_paths = []

        # Create output directory
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use Playwright to capture PDF pages
        async with PDFPlaywrightComputer() as computer:
            try:
                # Capture all pages using browser automation
                pages_data = await computer.capture_all_pages(pdf_file_path)

                if not pages_data:
                    return json.dumps(
                        {
                            "success": False,
                            "error_message": "No pages were captured",
                            "capture_time": time.time() - start_time,
                            "pages_captured": 0,
                            "screenshot_paths": [],
                            "pdf_file": pdf_file_path,
                        }
                    )

                # Save screenshots
                timestamp = int(time.time())

                for page_data in pages_data:
                    page_num = page_data["page_number"]
                    screenshot_bytes = page_data["screenshot_bytes"]

                    # Save screenshot to file
                    image_path = output_dir / f"computer_use_{timestamp}-{page_num}.png"
                    image_path.write_bytes(screenshot_bytes)
                    screenshot_paths.append(str(image_path))

            except Exception as e:
                logger.error(f"Screenshot capture failed: {e}")
                return json.dumps(
                    {
                        "success": False,
                        "error_message": f"Screenshot capture failed: {str(e)}",
                        "capture_time": time.time() - start_time,
                        "pages_captured": 0,
                        "screenshot_paths": [],
                        "pdf_file": pdf_file_path,
                    }
                )

        capture_time = time.time() - start_time
        logger.info(
            f"✅ Screenshot capture complete: {len(screenshot_paths)} pages in {capture_time:.2f}s"
        )

        result_data = {
            "success": True,
            "capture_time": capture_time,
            "pages_captured": len(screenshot_paths),
            "screenshot_paths": screenshot_paths,
            "pdf_file": pdf_file_path,
            "method": "playwright_screenshot_capture",
        }

        return json.dumps(result_data)

    except Exception as e:
        error_message = f"PDF screenshot capture failed: {str(e)}"
        logger.error(error_message)

        return json.dumps(
            {
                "success": False,
                "error_message": error_message,
                "capture_time": (
                    time.time() - start_time if "start_time" in locals() else 0.0
                ),
                "pages_captured": 0,
                "screenshot_paths": [],
                "pdf_file": pdf_file_path,
                "method": "playwright_screenshot_capture",
            }
        )


@function_tool
async def pdf_computer_use_tool(pdf_file_path: str) -> str:
    """Capture PDF pages as screenshots using Playwright browser automation.

    This tool wrapper allows the screenshot capture to be used by agents.

    Args:
        pdf_file_path: Path to the PDF file to capture

    Returns:
        JSON string with screenshot paths and capture metadata
    """
    return await capture_pdf_screenshots(pdf_file_path)
