"""PDF Computer Use implementation using Playwright for browser automation."""

import asyncio
import base64
from pathlib import Path
from typing import Any

from loguru import logger
from playwright.async_api import (
    Browser,
    Page,
    PlaywrightContextManager,
    async_playwright,
)
from pypdf import PdfReader


class PDFPlaywrightComputer:
    """Computer Use implementation for PDF handling via Playwright browser automation."""

    def __init__(self):
        self.playwright: PlaywrightContextManager | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.current_pdf_path: str | None = None

    async def launch_browser(self, headless: bool = False) -> None:
        """Launch Playwright browser for PDF viewing."""
        try:
            logger.info("Launching Playwright browser for PDF viewing")
            self.playwright = async_playwright()
            playwright_instance = await self.playwright.start()

            # Launch Chromium with proper PDF viewer support (headful mode required)
            self.browser = await playwright_instance.chromium.launch(
                headless=headless,
                args=[
                    "--allow-file-access-from-files",
                    "--disable-web-security",
                    "--no-sandbox",
                    "--disable-features=VizDisplayCompositor",
                    "--force-device-scale-factor=1",
                    # Enable Chrome's built-in PDF viewer
                    "--enable-pdf-viewer",
                    # Don't disable PDF extension - we want to use it
                    "--no-default-browser-check",
                    "--disable-translate",
                ],
            )

            # Create a simple browser context for PDF viewing
            context = await self.browser.new_context(
                viewport={"width": 1200, "height": 1600},  # Good for PDF viewing
            )

            self.page = await context.new_page()

            logger.info("Browser launched successfully")

        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            await self.cleanup()
            raise

    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """
        Get accurate page count from PDF file using pypdf.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages in the PDF
        """
        try:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            logger.info(f"PDF has {page_count} pages (detected via pypdf)")
            return page_count
        except Exception as e:
            logger.error(f"Failed to read PDF page count with pypdf: {e}")
            # Fallback to default if pypdf fails
            return 1

    async def open_pdf(self, pdf_path: str) -> int:
        """
        Open PDF file in browser and return page count.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages in the PDF
        """
        try:
            if not self.page:
                await self.launch_browser()

            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Get accurate page count using pypdf
            page_count = self._get_pdf_page_count(pdf_path)

            # Use Chrome's built-in PDF viewer directly
            file_url = pdf_file.resolve().as_uri()
            logger.info(f"Opening PDF directly in Chrome's PDF viewer: {file_url}")
            self.current_pdf_path = pdf_path

            # Navigate directly to PDF file - Chrome will open it in built-in viewer
            await self.page.goto(file_url, wait_until="networkidle", timeout=30000)

            # Wait for PDF to load
            await asyncio.sleep(5)

            logger.info(
                f"PDF opened successfully in Chrome PDF viewer with {page_count} pages"
            )
            return page_count

        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise

    async def _ensure_pdf_viewer_focus(self) -> None:
        """Ensure the PDF viewer has focus for keyboard navigation."""
        try:
            # Get viewport dimensions
            viewport = self.page.viewport_size
            center_x = viewport["width"] // 2
            center_y = viewport["height"] // 2

            # Click directly on the center coordinates to focus the PDF
            await self.page.mouse.click(center_x, center_y)
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"Could not ensure PDF focus: {e}")
            # Continue anyway - sometimes keyboard navigation works without explicit focus

    async def navigate_to_page(self, page_number: int) -> None:
        """Navigate to a specific page number."""
        try:
            if not self.page:
                raise RuntimeError("Browser not initialized")

            # Check if we're using PDF.js viewer or Chrome's built-in viewer
            current_url = self.page.url
            if "pdf.js" in current_url:
                # Use PDF.js specific navigation
                await self._navigate_pdfjs_page(page_number)
            else:
                # Use Chrome's built-in PDF viewer navigation
                await self._navigate_chrome_pdf_page(page_number)

        except Exception as e:
            logger.error(f"Failed to navigate to page {page_number}: {e}")
            raise

    async def _navigate_chrome_pdf_page(self, page_number: int) -> None:
        """Navigate to a specific page in Chrome's built-in PDF viewer."""
        try:
            # Ensure PDF viewer has focus first
            await self._ensure_pdf_viewer_focus()

            # Use keyboard navigation (Page Down from start)
            # Go to beginning first
            await self.page.keyboard.press("Home")
            await asyncio.sleep(1)

            # Navigate to the desired page (1-indexed) using Page Down
            for _i in range(page_number - 1):
                await self.page.keyboard.press("PageDown")
                await asyncio.sleep(1)  # Give Chrome PDF viewer time to respond

        except Exception as e:
            logger.error(f"Failed to navigate Chrome PDF to page {page_number}: {e}")
            raise

    async def _navigate_pdfjs_page(self, page_number: int) -> None:
        """Navigate to a specific page in PDF.js viewer."""
        try:
            # Method 1: Try using the page input field
            try:
                page_input = await self.page.wait_for_selector(
                    "#pageNumber", timeout=3000
                )
                if page_input:
                    await page_input.clear()
                    await page_input.fill(str(page_number))
                    await self.page.keyboard.press("Enter")
                    await asyncio.sleep(2)  # Wait for page to render
                    return
            except Exception as e:
                logger.debug(f"Page input navigation failed: {e}")

            # Method 2: Try JavaScript navigation
            try:
                await self.page.evaluate(
                    f"""
                    () => {{
                        if (window.PDFViewerApplication) {{
                            window.PDFViewerApplication.page = {page_number};
                        }}
                    }}
                """
                )
                await asyncio.sleep(2)
                return
            except Exception as e:
                logger.debug(f"JavaScript navigation failed: {e}")

            # Method 3: Fallback to keyboard navigation
            # Go to first page then navigate
            await self.page.keyboard.press("Home")
            await asyncio.sleep(1)

            # Navigate to the desired page
            for _i in range(page_number - 1):
                await self.page.keyboard.press("PageDown")
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to navigate PDF.js to page {page_number}: {e}")
            raise

    async def take_screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        try:
            if not self.page:
                raise RuntimeError("Browser not initialized")

            # Take viewport screenshot only (one page at a time, not full scrollable content)
            screenshot_bytes = await self.page.screenshot(type="png", full_page=False)

            return screenshot_bytes

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise

    async def capture_all_pages(self, pdf_path: str) -> list[dict[str, Any]]:
        """
        Capture all pages of a PDF as screenshots.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of page data with screenshots
        """
        try:
            # Open PDF and get page count
            page_count = await self.open_pdf(pdf_path)

            pages_data = []

            # Capture each page
            for page_num in range(1, page_count + 1):
                try:
                    logger.info(f"Capturing page {page_num}/{page_count}")

                    # Navigate to the page
                    await self.navigate_to_page(page_num)

                    # Take screenshot
                    screenshot_bytes = await self.take_screenshot()
                    screenshot_base64 = base64.b64encode(screenshot_bytes).decode(
                        "utf-8"
                    )

                    pages_data.append(
                        {
                            "page_number": page_num,
                            "screenshot_bytes": screenshot_bytes,
                            "screenshot_base64": screenshot_base64,
                        }
                    )

                    logger.debug(f"Captured page {page_num} successfully")

                except Exception as e:
                    logger.error(f"Failed to capture page {page_num}: {e}")
                    # Continue with other pages
                    continue

            logger.info(f"Successfully captured {len(pages_data)} pages")
            return pages_data

        except Exception as e:
            logger.error(f"Failed to capture PDF pages: {e}")
            raise
        finally:
            # Clean up browser resources
            await self.cleanup()

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None

            if self.playwright:
                await self.playwright.__aexit__(None, None, None)
                self.playwright = None

            self.page = None
            self.current_pdf_path = None

            logger.debug("Browser resources cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.launch_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
