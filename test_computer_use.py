#!/usr/bin/env python3
"""Test script for PDF Computer Use functionality."""

import asyncio
from pathlib import Path
from src.cv_writer_mcp.pdf_computer import PDFPlaywrightComputer

async def test_pdf_capture():
    pdf_path = 'output/to_improve_style.pdf'
    
    async with PDFPlaywrightComputer() as computer:
        print(f'Testing PDF capture for: {pdf_path}')
        try:
            pages_data = await computer.capture_all_pages(pdf_path)
            print(f'Successfully captured {len(pages_data)} pages')
            for i, page in enumerate(pages_data, 1):
                print(f'Page {i}: Screenshot size = {len(page["screenshot_bytes"])} bytes')
        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pdf_capture())