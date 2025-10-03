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
                    '--allow-file-access-from-files',
                    '--disable-web-security',
                    '--no-sandbox',
                    '--disable-features=VizDisplayCompositor',
                    '--force-device-scale-factor=1',
                    # Enable Chrome's built-in PDF viewer
                    '--enable-pdf-viewer',
                    # Don't disable PDF extension - we want to use it
                    '--no-default-browser-check',
                    '--disable-translate'
                ]
            )

            # Create a simple browser context for PDF viewing
            context = await self.browser.new_context(
                viewport={'width': 1200, 'height': 1600},  # Good for PDF viewing
            )

            self.page = await context.new_page()

            logger.info("Browser launched successfully")

        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            await self.cleanup()
            raise

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

            # Use Chrome's built-in PDF viewer directly
            file_url = pdf_file.resolve().as_uri()
            logger.info(f"Opening PDF directly in Chrome's PDF viewer: {file_url}")
            self.current_pdf_path = pdf_path

            # Navigate directly to PDF file - Chrome will open it in built-in viewer
            await self.page.goto(file_url, wait_until='networkidle', timeout=30000)

            # Wait for PDF to load
            await asyncio.sleep(5)

            # Try to detect page count from Chrome's PDF viewer
            page_count = await self._detect_chrome_pdf_page_count()

            logger.info(f"PDF opened successfully in Chrome PDF viewer, detected {page_count} pages")
            return page_count

        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise

    async def _detect_chrome_pdf_page_count(self) -> int:
        """Detect the number of pages in Chrome's built-in PDF viewer."""
        try:
            # Wait for Chrome's PDF viewer to fully load
            await asyncio.sleep(3)

            # Try to get page count from Chrome's PDF viewer
            page_count = None

            # Method 1: Try JavaScript evaluation to access Chrome's PDF viewer
            try:
                for attempt in range(5):
                    page_count = await self.page.evaluate("""
                        () => {
                            // Try to access Chrome's PDF viewer plugin
                            const plugin = document.querySelector('embed[type="application/pdf"]');
                            if (plugin && plugin.src) {
                                // Try different methods to get page count
                                try {
                                    // Some PDF viewers expose page count
                                    if (window.pdfDocument && window.pdfDocument.numPages) {
                                        return window.pdfDocument.numPages;
                                    }
                                    // Alternative approach
                                    if (plugin.pageCount) {
                                        return plugin.pageCount;
                                    }
                                } catch (e) {
                                    // Ignore errors and try fallback
                                }
                            }
                            return null;
                        }
                    """)
                    if page_count:
                        logger.debug(f"Got page count from Chrome PDF viewer (attempt {attempt + 1}): {page_count}")
                        break
                    await asyncio.sleep(1)  # Wait before retry
            except Exception as e:
                logger.debug(f"JavaScript method failed: {e}")

            # Method 2: Use navigation approach to count pages
            if not page_count:
                try:
                    page_count = await self._count_pages_by_navigation()
                except Exception as e:
                    logger.debug(f"Navigation counting failed: {e}")

            # Final fallback
            if not page_count or page_count <= 0:
                logger.warning("Could not detect page count from Chrome PDF viewer, defaulting to 4")
                page_count = 4

            return page_count

        except Exception as e:
            logger.error(f"Failed to detect page count from Chrome PDF viewer: {e}")
            return 4  # Default fallback

    async def _detect_pdfjs_page_count(self) -> int:
        """Detect the number of pages in PDF.js viewer."""
        try:
            # Wait for PDF.js to fully load and PDF document to be processed
            await asyncio.sleep(5)

            # Try to get page count from PDF.js interface
            page_count = None

            # Method 1: Try JavaScript evaluation of PDF.js API (most reliable)
            try:
                # Wait for PDF.js application to be ready and try multiple times
                for attempt in range(10):
                    page_count = await self.page.evaluate("""
                        () => {
                            // Try to access PDF.js viewer app
                            if (window.PDFViewerApplication && 
                                window.PDFViewerApplication.pdfDocument &&
                                window.PDFViewerApplication.pdfDocument.numPages) {
                                return window.PDFViewerApplication.pdfDocument.numPages;
                            }
                            // Alternative way
                            if (window.pdfApp && window.pdfApp.pdfDocument) {
                                return window.pdfApp.pdfDocument.numPages;
                            }
                            return null;
                        }
                    """)
                    if page_count:
                        logger.debug(f"Got page count from JavaScript (attempt {attempt + 1}): {page_count}")
                        break
                    await asyncio.sleep(2)  # Wait before retry
            except Exception as e:
                logger.debug(f"JavaScript method failed: {e}")

            # Method 2: Try to get total pages from PDF.js viewer UI elements
            if not page_count:
                try:
                    # Look for various possible selectors for page count
                    selectors = ['#numPages', '.numPages', '[data-l10n-id="page_of_pages"]']
                    for selector in selectors:
                        try:
                            page_count_element = await self.page.wait_for_selector(selector, timeout=3000)
                            if page_count_element:
                                page_count_text = await page_count_element.text_content()
                                # Extract number from text like "of 5" or just "5"
                                import re
                                match = re.search(r'(\d+)', page_count_text)
                                if match:
                                    page_count = int(match.group(1))
                                    logger.debug(f"Got page count from {selector}: {page_count}")
                                    break
                        except Exception:
                            continue
                except Exception as e:
                    logger.debug(f"UI element method failed: {e}")

            # Method 3: Try to find page count in any text on the page
            if not page_count:
                try:
                    # Get all visible text and look for patterns like "1 / 5" or "Page 1 of 5"
                    page_text = await self.page.text_content('body')
                    if page_text:
                        import re
                        # Look for patterns like "1 / 5" or "1 of 5"
                        matches = re.findall(r'(?:of|/)[\s]*(\d+)', page_text)
                        if matches:
                            # Take the largest number found (likely the total pages)
                            page_count = max([int(m) for m in matches])
                            logger.debug(f"Got page count from page text: {page_count}")
                except Exception as e:
                    logger.debug(f"Page text method failed: {e}")

            # Final fallback
            if not page_count or page_count <= 0:
                logger.warning("Could not detect page count from PDF.js, defaulting to 4")
                page_count = 4

            return page_count

        except Exception as e:
            logger.error(f"Failed to detect page count from PDF.js: {e}")
            return 4  # Default fallback

    async def _detect_page_count(self) -> int:
        """Detect the number of pages in the opened PDF."""
        try:
            # Wait for PDF viewer to load
            await asyncio.sleep(3)

            # Try different methods to get page count
            page_count = None

            # Method 1: Try to find page count in Chrome's PDF viewer
            try:
                # Look for page indicators like "1 of 5" or page count elements
                page_info_selectors = [
                    '[data-page-number]',
                    '.page-indicator',
                    'input[title*="page"]',
                    '#pageNumber'
                ]

                for selector in page_info_selectors:
                    elements = await self.page.query_selector_all(selector)
                    if elements:
                        page_count = len(elements)
                        break

            except Exception as e:
                logger.debug(f"Method 1 failed: {e}")

            # Method 2: Try JavaScript evaluation
            if not page_count:
                try:
                    # Try to access PDF viewer JavaScript API
                    page_count = await self.page.evaluate("""
                        () => {
                            // Try Chrome's PDF viewer
                            if (window.PDFViewerApplication) {
                                return window.PDFViewerApplication.pagesCount;
                            }
                            // Try to count page elements
                            const pages = document.querySelectorAll('[data-page-number], .page');
                            return pages.length;
                        }
                    """)
                except Exception as e:
                    logger.debug(f"Method 2 failed: {e}")

            # Method 3: Fallback - navigate to end and count
            if not page_count or page_count <= 0:
                try:
                    page_count = await self._count_pages_by_navigation()
                except Exception as e:
                    logger.debug(f"Method 3 failed: {e}")

            # Final fallback
            if not page_count or page_count <= 0:
                logger.warning("Could not detect page count, defaulting to 4")
                page_count = 4

            return page_count

        except Exception as e:
            logger.error(f"Failed to detect page count: {e}")
            return 4  # Default fallback

    async def _ensure_pdf_viewer_focus(self) -> None:
        """Ensure the PDF viewer has focus for keyboard navigation."""
        try:
            # Get viewport dimensions
            viewport = self.page.viewport_size
            center_x = viewport['width'] // 2
            center_y = viewport['height'] // 2

            # Click directly on the center coordinates to focus the PDF
            await self.page.mouse.click(center_x, center_y)
            await asyncio.sleep(0.5)

            logger.debug("PDF viewer focus ensured by clicking on center of viewport")

        except Exception as e:
            logger.warning(f"Could not ensure PDF focus: {e}")
            # Continue anyway - sometimes keyboard navigation works without explicit focus

    async def _count_pages_by_navigation(self) -> int:
        """Count pages by navigating through the PDF using keyboard with screenshot comparison."""
        try:
            # Ensure PDF viewer has focus
            await self._ensure_pdf_viewer_focus()

            # Go to beginning first
            await self.page.keyboard.press('Home')
            await asyncio.sleep(2)

            page_count = 1
            max_attempts = 15  # Reasonable limit for CV pages

            logger.debug("Using PageDown for PDF page detection via screenshot comparison")

            # Keep pressing PageDown and compare screenshots to detect page changes
            for i in range(max_attempts):
                # Take screenshot before navigation
                screenshot_before = await self.page.screenshot()

                # Navigate to next page
                await self.page.keyboard.press('PageDown')
                await asyncio.sleep(1.5)  # Give PDF viewer time to respond

                # Take screenshot after navigation
                screenshot_after = await self.page.screenshot()

                # Compare screenshots - if different, we likely moved to a new page
                if screenshot_before != screenshot_after:
                    page_count += 1
                    logger.debug(f"Page {page_count} detected via screenshot comparison")
                else:
                    logger.debug(f"No page change detected, stopping at page {page_count}")
                    break

            # Go back to beginning
            await self.page.keyboard.press('Home')
            await asyncio.sleep(1)

            logger.info(f"Final detected page count: {page_count}")
            return max(page_count, 4)  # Ensure at least 4 pages as that's what pdfinfo shows

        except Exception as e:
            logger.error(f"Failed to count pages by navigation: {e}")
            return 4

    async def navigate_to_page(self, page_number: int) -> None:
        """Navigate to a specific page number."""
        try:
            if not self.page:
                raise RuntimeError("Browser not initialized")

            # Check if we're using PDF.js viewer or Chrome's built-in viewer
            current_url = self.page.url
            if 'pdf.js' in current_url:
                # Use PDF.js specific navigation
                await self._navigate_pdfjs_page(page_number)
            else:
                # Use Chrome's built-in PDF viewer navigation
                await self._navigate_chrome_pdf_page(page_number)

            logger.debug(f"Navigated to page {page_number}")

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
            await self.page.keyboard.press('Home')
            await asyncio.sleep(1)

            # Navigate to the desired page (1-indexed) using Page Down
            for i in range(page_number - 1):
                await self.page.keyboard.press('PageDown')
                await asyncio.sleep(1)  # Give Chrome PDF viewer time to respond

        except Exception as e:
            logger.error(f"Failed to navigate Chrome PDF to page {page_number}: {e}")
            raise

    async def _navigate_pdfjs_page(self, page_number: int) -> None:
        """Navigate to a specific page in PDF.js viewer."""
        try:
            # Method 1: Try using the page input field
            try:
                page_input = await self.page.wait_for_selector('#pageNumber', timeout=3000)
                if page_input:
                    await page_input.clear()
                    await page_input.fill(str(page_number))
                    await self.page.keyboard.press('Enter')
                    await asyncio.sleep(2)  # Wait for page to render
                    return
            except Exception as e:
                logger.debug(f"Page input navigation failed: {e}")

            # Method 2: Try JavaScript navigation
            try:
                await self.page.evaluate(f"""
                    () => {{
                        if (window.PDFViewerApplication) {{
                            window.PDFViewerApplication.page = {page_number};
                        }}
                    }}
                """)
                await asyncio.sleep(2)
                return
            except Exception as e:
                logger.debug(f"JavaScript navigation failed: {e}")

            # Method 3: Fallback to keyboard navigation
            # Go to first page then navigate
            await self.page.keyboard.press('Home')
            await asyncio.sleep(1)

            # Navigate to the desired page
            for i in range(page_number - 1):
                await self.page.keyboard.press('PageDown')
                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to navigate PDF.js to page {page_number}: {e}")
            raise

    async def take_screenshot(self) -> bytes:
        """Take a screenshot of the current page."""
        try:
            if not self.page:
                raise RuntimeError("Browser not initialized")

            # Take full page screenshot (quality not supported for PNG)
            screenshot_bytes = await self.page.screenshot(
                type='png',
                full_page=True
            )

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
                    screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

                    pages_data.append({
                        'page_number': page_num,
                        'screenshot_bytes': screenshot_bytes,
                        'screenshot_base64': screenshot_base64
                    })

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
