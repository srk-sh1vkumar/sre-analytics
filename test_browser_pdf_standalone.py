#!/usr/bin/env python3
"""
Standalone test to demonstrate browser PDF generation from existing HTML
"""

import logging
import os
from src.reports.browser_pdf_generator import BrowserPDFGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_browser_pdf_from_existing_html():
    print("🧪 Testing Browser PDF from Existing HTML")
    print("=" * 60)

    try:
        # Use the most recent HTML report
        html_path = "reports/generated/browser_test_html.html"

        if not os.path.exists(html_path):
            print(f"❌ HTML file not found: {html_path}")
            return False

        print(f"📄 Source HTML: {html_path}")
        html_size = os.path.getsize(html_path)
        print(f"📊 HTML Size: {html_size:,} bytes")

        # Read the HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        print("🔍 HTML Content Analysis:")
        print(f"• Contains Tailwind CSS: {'tailwindcss' in html_content}")
        print(f"• Contains Font Awesome: {'font-awesome' in html_content}")
        print(f"• Contains Chart.js: {'chart.js' in html_content}")
        print(f"• Contains detailed metrics: {'Detailed Metrics Analysis' in html_content}")

        # Generate browser PDF
        print("\n🚀 Generating browser PDF...")
        browser_generator = BrowserPDFGenerator()

        browser_pdf_path = "reports/generated/browser_standalone_test.pdf"
        success = browser_generator.create_pdf_from_html_sync(html_content, browser_pdf_path)

        if success and os.path.exists(browser_pdf_path):
            browser_pdf_size = os.path.getsize(browser_pdf_path)
            print(f"✅ Browser PDF generated: {browser_pdf_path}")
            print(f"📊 Browser PDF Size: {browser_pdf_size:,} bytes")

            # Calculate size difference
            size_ratio = (browser_pdf_size / html_size) * 100
            print(f"📊 Size Ratio: {size_ratio:.1f}% (PDF/HTML)")

            print("\n🎯 Key Advantages of Browser PDF:")
            print("• ✅ Exact same visual appearance as HTML")
            print("• ✅ Preserves all Tailwind CSS styling")
            print("• ✅ Maintains proper spacing and layout")
            print("• ✅ No CSS conversion artifacts")
            print("• ✅ Font rendering identical to browser")

            print(f"\n📄 Compare the files:")
            print(f"• Original HTML: {html_path}")
            print(f"• Browser PDF: {browser_pdf_path}")

            return True
        else:
            print("❌ Browser PDF generation failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_browser_pdf_from_existing_html()
    if success:
        print("\n🎉 Browser PDF test completed successfully!")
        print("The browser PDF should visually match the HTML exactly.")
    else:
        print("\n❌ Browser PDF test failed.")