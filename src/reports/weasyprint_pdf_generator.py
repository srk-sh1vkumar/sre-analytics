"""
WeasyPrint PDF Generator for SRE Reports
Generates high-quality PDF reports from HTML using WeasyPrint
"""

import logging
import weasyprint
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import base64
import os

class WeasyPrintPDFGenerator:
    """PDF generator using WeasyPrint for high-quality output"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_pdf_from_html(self, html_content: str, output_path: str,
                           base_url: str = None) -> bool:
        """
        Create PDF from HTML content using WeasyPrint

        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved
            base_url: Base URL for resolving relative links

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Create CSS for better PDF styling
            pdf_css = self._get_pdf_css()

            # Create WeasyPrint HTML object
            html_obj = weasyprint.HTML(
                string=html_content,
                base_url=base_url or f"file://{os.getcwd()}/"
            )

            # Create CSS object
            css_obj = weasyprint.CSS(string=pdf_css)

            # Generate PDF
            html_obj.write_pdf(output_path, stylesheets=[css_obj])

            self.logger.info(f"✅ PDF generated successfully: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"❌ PDF generation failed: {e}")
            return False

    def create_pdf_from_file(self, html_file_path: str, output_path: str = None) -> str:
        """
        Create PDF from HTML file

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

            success = self.create_pdf_from_html(
                html_content,
                str(output_path),
                base_url=f"file://{Path(html_file_path).parent.absolute()}/"
            )

            if success:
                return str(output_path)
            else:
                return ""

        except Exception as e:
            self.logger.error(f"❌ Failed to create PDF from file {html_file_path}: {e}")
            return ""

    def _get_pdf_css(self) -> str:
        """Get CSS optimized for PDF generation"""
        return """
        @page {
            size: A4;
            margin: 2cm;
            @top-center {
                content: "SRE Performance Report";
                font-family: Arial, sans-serif;
                font-size: 10pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-family: Arial, sans-serif;
                font-size: 10pt;
                color: #666;
            }
        }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #333;
        }

        .container {
            max-width: none;
            background: white;
            padding: 0;
            border-radius: 0;
            box-shadow: none;
            margin: 0;
        }

        .header h1 {
            font-size: 18pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }

        .section h2 {
            font-size: 14pt;
            color: #333;
            border-bottom: 2pt solid #007acc;
            padding-bottom: 5pt;
            margin-top: 20pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }

        .section h3 {
            font-size: 12pt;
            margin-top: 15pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15pt;
            margin: 15pt 0;
            page-break-inside: avoid;
        }

        .summary-card {
            background: #f8f9fa;
            padding: 10pt;
            border-radius: 5pt;
            border-left: 3pt solid #007acc;
            text-align: center;
            page-break-inside: avoid;
        }

        .summary-card .value {
            font-size: 16pt;
            font-weight: bold;
            color: #007acc;
        }

        .chart-container {
            margin: 15pt 0;
            text-align: center;
            page-break-inside: avoid;
        }

        .chart-container img {
            max-width: 100%;
            max-height: 300pt;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15pt 0;
            font-size: 9pt;
        }

        .metrics-table th,
        .metrics-table td {
            padding: 6pt;
            border: 1pt solid #ddd;
            text-align: left;
        }

        .metrics-table th {
            background-color: #f5f5f5;
            font-weight: bold;
        }

        .metrics-table tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .incident-section {
            margin: 20pt 0;
            padding: 15pt;
            background: #fff3cd;
            border-radius: 5pt;
            border-left: 4pt solid #ffc107;
            page-break-inside: avoid;
        }

        .incident-critical {
            background: #f8d7da;
            border-left-color: #dc3545;
        }

        .llm-analysis {
            background: #e7f3ff;
            padding: 12pt;
            border-radius: 5pt;
            border-left: 3pt solid #007acc;
            margin: 10pt 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }

        .recommendation-item {
            background: #f8f9fa;
            border: 1pt solid #ddd;
            border-radius: 5pt;
            padding: 10pt;
            margin: 8pt 0;
            border-left: 3pt solid #007acc;
            page-break-inside: avoid;
        }

        .status-compliant { color: #28a745; font-weight: bold; }
        .status-at-risk { color: #ffc107; font-weight: bold; }
        .status-breached { color: #dc3545; font-weight: bold; }

        /* Page break rules */
        .section {
            page-break-before: auto;
        }

        .trend-section {
            page-break-before: auto;
        }

        /* Hide elements that don't work well in PDF */
        .interactive-element {
            display: none;
        }

        /* Ensure good contrast for printing */
        a {
            color: #000 !important;
            text-decoration: underline;
        }
        """

    def add_metadata_to_pdf(self, pdf_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Add metadata to generated PDF (if possible with WeasyPrint)

        Args:
            pdf_path: Path to PDF file
            metadata: Metadata dictionary

        Returns:
            bool: True if successful
        """
        try:
            # WeasyPrint doesn't support metadata modification after generation
            # This would require pypdf or similar library
            self.logger.info("PDF metadata addition not implemented for WeasyPrint")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add metadata: {e}")
            return False

    def test_weasyprint_installation(self) -> bool:
        """Test if WeasyPrint is properly installed and configured"""
        try:
            # Test basic HTML to PDF conversion
            test_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test</title>
                <style>
                    body { font-family: Arial, sans-serif; }
                    h1 { color: #007acc; }
                </style>
            </head>
            <body>
                <h1>WeasyPrint Test</h1>
                <p>This is a test document to verify WeasyPrint installation.</p>
            </body>
            </html>
            """

            test_output = "test_weasyprint.pdf"
            success = self.create_pdf_from_html(test_html, test_output)

            if success and os.path.exists(test_output):
                os.remove(test_output)  # Clean up test file
                self.logger.info("✅ WeasyPrint installation test passed")
                return True
            else:
                self.logger.error("❌ WeasyPrint installation test failed")
                return False

        except Exception as e:
            self.logger.error(f"❌ WeasyPrint test failed: {e}")
            return False


def enhance_html_for_pdf(html_content: str) -> str:
    """
    Enhance HTML content for better PDF rendering

    Args:
        html_content: Original HTML content

    Returns:
        str: Enhanced HTML content optimized for PDF
    """
    # Add PDF-specific meta tags and styles
    pdf_enhancements = """
    <style>
        /* PDF-specific styles */
        @media print {
            .no-print { display: none !important; }
            .page-break { page-break-before: always; }
            .avoid-break { page-break-inside: avoid; }
        }

        /* Improve table rendering */
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }

        /* Better image handling */
        img { max-width: 100%; height: auto; }
    </style>
    """

    # Insert enhancements before closing head tag
    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{pdf_enhancements}</head>")
    else:
        # If no head tag, add it
        html_content = f"<head>{pdf_enhancements}</head>" + html_content

    return html_content


if __name__ == "__main__":
    # Test WeasyPrint PDF generation
    logging.basicConfig(level=logging.INFO)

    generator = WeasyPrintPDFGenerator()

    # Test installation
    if generator.test_weasyprint_installation():
        print("✅ WeasyPrint is properly installed and configured")
    else:
        print("❌ WeasyPrint installation has issues")

    # Test PDF generation with sample content
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SRE Report Sample</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Sample SRE Performance Report</h1>
                <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Total Services</h3>
                        <div class="value">5</div>
                    </div>
                    <div class="summary-card">
                        <h3>Compliance Rate</h3>
                        <div class="value">98.5%</div>
                    </div>
                    <div class="summary-card">
                        <h3>System Health</h3>
                        <div class="value">Good</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>SLO Metrics</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Service</th>
                            <th>Availability</th>
                            <th>Response Time</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Web Service</td>
                            <td>99.9%</td>
                            <td>120ms</td>
                            <td class="status-compliant">Compliant</td>
                        </tr>
                        <tr>
                            <td>API Service</td>
                            <td>99.8%</td>
                            <td>180ms</td>
                            <td class="status-compliant">Compliant</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    # Generate sample PDF
    sample_pdf_path = "reports/generated/weasyprint_sample.pdf"
    success = generator.create_pdf_from_html(sample_html, sample_pdf_path)

    if success:
        print(f"✅ Sample PDF generated: {sample_pdf_path}")
    else:
        print("❌ Sample PDF generation failed")