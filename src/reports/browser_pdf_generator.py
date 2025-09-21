"""
Browser-based PDF Generator for SRE Reports
Uses headless browser to generate PDFs that match HTML exactly
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
import pyppeteer


class BrowserPDFGenerator:
    """PDF generator using headless browser for exact HTML matching"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def create_pdf_from_html_content(self, html_content: str, output_path: str) -> bool:
        """
        Create PDF from HTML content using headless browser

        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Launch headless browser
            browser = await pyppeteer.launch({
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-first-run',
                    '--disable-default-apps',
                    '--disable-extensions'
                ]
            })

            try:
                # Create new page
                page = await browser.newPage()

                # Set viewport for consistent rendering
                await page.setViewport({'width': 1200, 'height': 800})

                # Inject print-specific CSS for better page breaks
                print_css = self._get_print_optimized_css()
                enhanced_html = self._inject_print_css(html_content, print_css)

                # Set HTML content
                await page.setContent(enhanced_html)

                # Wait for any dynamic content to load
                await asyncio.sleep(2)

                # Generate PDF with print-optimized settings
                pdf_options = {
                    'path': output_path,
                    'format': 'A4',
                    'printBackground': True,
                    'margin': {
                        'top': '0.5in',
                        'right': '0.5in',
                        'bottom': '0.5in',
                        'left': '0.5in'
                    },
                    'preferCSSPageSize': True
                }

                await page.pdf(pdf_options)

                self.logger.info(f"✅ Browser PDF generated successfully: {output_path}")
                return True

            finally:
                await browser.close()

        except Exception as e:
            self.logger.error(f"❌ Browser PDF generation failed: {e}")
            return False

    def create_pdf_from_html_sync(self, html_content: str, output_path: str) -> bool:
        """
        Synchronous wrapper for PDF generation

        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.create_pdf_from_html_content(html_content, output_path)
            )
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"❌ Sync PDF generation failed: {e}")
            return False

    async def create_pdf_from_file(self, html_file_path: str, output_path: str = None) -> str:
        """
        Create PDF from HTML file using browser

        Args:
            html_file_path: Path to HTML file
            output_path: Output PDF path (optional)

        Returns:
            str: Path to generated PDF file
        """
        if not output_path:
            html_path = Path(html_file_path)
            output_path = html_path.with_suffix('.pdf')

        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            success = await self.create_pdf_from_html_content(
                html_content,
                str(output_path)
            )

            if success:
                return str(output_path)
            else:
                return ""

        except Exception as e:
            self.logger.error(f"❌ Failed to create PDF from file {html_file_path}: {e}")
            return ""

    def test_browser_installation(self) -> bool:
        """Test if browser is properly installed and configured"""
        try:
            async def test():
                browser = await pyppeteer.launch({'headless': True})
                await browser.close()
                return True

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test())
            loop.close()

            if result:
                self.logger.info("✅ Browser installation test passed")
                return True
            else:
                self.logger.error("❌ Browser installation test failed")
                return False

        except Exception as e:
            self.logger.error(f"❌ Browser test failed: {e}")
            return False

    def _get_print_optimized_css(self) -> str:
        """Get CSS optimized for print with proper page break handling"""
        return """
        <style>
        @media print {
            /* CRITICAL: Prevent all metric cards from breaking */
            .metric-card {
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                page-break-before: auto !important;
                page-break-after: auto !important;
                margin-bottom: 20px !important;
                padding: 15px !important;
                border: 1px solid #ddd !important;
                background: white !important;
                min-height: 150px !important;
                display: block !important;
                width: 100% !important;
                float: none !important;
            }

            /* PRESERVE Executive Summary grid layout */
            .grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-5,
            .grid[class*="grid-cols-5"] {
                display: grid !important;
                grid-template-columns: repeat(5, 1fr) !important;
                gap: 1rem !important;
                margin-bottom: 2rem !important;
            }

            /* Executive summary metric cards - keep inline */
            .grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-5 > .metric-card,
            .grid[class*="grid-cols-5"] > .metric-card {
                display: block !important;
                width: auto !important;
                margin: 0 !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                min-height: 120px !important;
                padding: 12px !important;
            }

            /* Convert OTHER grids to vertical stack for better page breaks */
            .grid:not([class*="grid-cols-5"]),
            div[class*="grid"]:not([class*="grid-cols-5"]) {
                display: block !important;
                columns: unset !important;
                column-count: unset !important;
            }

            /* Force OTHER metrics to stack vertically */
            .grid:not([class*="grid-cols-5"]) > div,
            div[class*="grid"]:not([class*="grid-cols-5"]) > div {
                display: block !important;
                width: 100% !important;
                margin: 0 0 20px 0 !important;
                float: none !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }

            /* Specific targeting for detailed metrics section (non-executive summary) */
            div[class*="grid-cols"]:not([class*="grid-cols-5"]) {
                display: block !important;
                grid-template-columns: none !important;
            }

            div[class*="grid-cols"]:not([class*="grid-cols-5"]) > * {
                display: block !important;
                width: 100% !important;
                margin-bottom: 20px !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }

            /* Summary cards */
            .summary-card {
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                margin-bottom: 15px !important;
                padding: 10px !important;
                background: #f8f9fa !important;
                border: 1px solid #ddd !important;
            }

            /* Section headers should stay with content */
            h1, h2, h3, h4, h5, h6 {
                page-break-after: avoid !important;
                break-after: avoid !important;
                page-break-inside: avoid !important;
                margin-bottom: 15px !important;
                margin-top: 20px !important;
            }

            /* Section containers */
            .section {
                page-break-inside: auto !important;
                break-inside: auto !important;
                margin-bottom: 30px !important;
            }

            /* Ensure incident sections don't break */
            .incident-section,
            .llm-analysis,
            .ai-insight {
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                margin: 20px 0 !important;
                padding: 15px !important;
                background: white !important;
                border: 1px solid #ddd !important;
            }

            /* Chart containers */
            .chart-container {
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                margin: 25px 0 !important;
                text-align: center !important;
            }

            .chart-container img {
                max-width: 100% !important;
                page-break-inside: avoid !important;
            }

            /* Tables - keep headers with content */
            .metrics-table {
                page-break-inside: auto !important;
                break-inside: auto !important;
                width: 100% !important;
                border-collapse: collapse !important;
            }

            .metrics-table thead {
                display: table-header-group !important;
            }

            .metrics-table tr {
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }

            .metrics-table th,
            .metrics-table td {
                padding: 8px !important;
                border: 1px solid #ddd !important;
                font-size: 10pt !important;
            }

            /* Better spacing for print */
            body {
                line-height: 1.4 !important;
                font-size: 11pt !important;
                background: white !important;
                color: black !important;
            }

            /* Container adjustments */
            .container {
                padding: 15px !important;
                max-width: none !important;
                background: white !important;
                box-shadow: none !important;
            }

            /* Remove flexbox layouts that can cause issues */
            div[class*="flex"] {
                display: block !important;
            }

            /* Status indicators - make them simpler for print */
            .status-indicator {
                display: inline-block !important;
                width: 12px !important;
                height: 12px !important;
                border-radius: 50% !important;
                margin-right: 8px !important;
                page-break-inside: avoid !important;
            }

            /* Force consistent margins and avoid floating issues */
            * {
                box-sizing: border-box !important;
                float: none !important;
            }

            /* Better orphans and widows control */
            p, div, li {
                orphans: 3 !important;
                widows: 3 !important;
            }

            /* Hide interactive elements */
            .floating-menu,
            button,
            [onclick] {
                display: none !important;
            }

            /* Ensure proper text contrast */
            * {
                color: black !important;
                background-color: transparent !important;
            }

            .metric-card,
            .summary-card,
            .incident-section,
            .llm-analysis {
                background-color: white !important;
                border: 1px solid #ccc !important;
            }
        }
        </style>
        """

    def _inject_print_css(self, html_content: str, print_css: str) -> str:
        """Inject print-specific CSS into HTML content"""
        # Find the closing head tag and inject CSS before it
        if "</head>" in html_content:
            return html_content.replace("</head>", f"{print_css}</head>")
        elif "<head>" in html_content:
            # If there's a head tag but no closing, add before body
            return html_content.replace("<body", f"{print_css}<body")
        else:
            # If no head tag, add CSS at the beginning
            return f"{print_css}\n{html_content}"


if __name__ == "__main__":
    # Test browser PDF generation
    logging.basicConfig(level=logging.INFO)

    generator = BrowserPDFGenerator()

    # Test installation
    if generator.test_browser_installation():
        print("✅ Browser is properly installed and configured")
    else:
        print("❌ Browser installation has issues")

    # Test PDF generation with sample content
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browser PDF Test</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: Inter, Arial, sans-serif;
                margin: 20px;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 30px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
            }
            .metric-card {
                background: #2d3748;
                border: 1px solid #4a5568;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #10b981;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>SRE Browser PDF Test</h1>
                <p>Testing headless browser PDF generation with exact HTML rendering</p>
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Availability</h3>
                    <div class="metric-value">99.9%</div>
                </div>
                <div class="metric-card">
                    <h3>Latency</h3>
                    <div class="metric-value">120ms</div>
                </div>
                <div class="metric-card">
                    <h3>Error Rate</h3>
                    <div class="metric-value">0.1%</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Generate sample PDF
    sample_pdf_path = "reports/generated/browser_pdf_test.pdf"
    success = generator.create_pdf_from_html_sync(sample_html, sample_pdf_path)

    if success:
        print(f"✅ Sample browser PDF generated: {sample_pdf_path}")
    else:
        print("❌ Sample browser PDF generation failed")