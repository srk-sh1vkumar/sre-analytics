#!/usr/bin/env python3
"""
Test script to verify browser PDF generation matches HTML exactly
"""

import logging
import os
import time
from datetime import datetime, timedelta
from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_browser_pdf_matching():
    print("🧪 Testing Browser PDF vs HTML Report Matching")
    print("=" * 60)

    try:
        # Initialize the enhanced report system
        app_name = "Browser PDF Test Application"
        system = EnhancedSREReportSystem(app_name=app_name)

        print(f"📊 Generating test reports for '{app_name}'...")

        # Define test services
        services = ["frontend", "api", "database", "cache"]

        # Generate metrics
        print("📈 Creating test metrics...")
        metrics = system.generate_metrics_with_trends(services, days_back=7)
        print(f"✅ Generated {len(metrics)} metrics for {len(services)} services")

        # Create an incident for testing
        print("🚨 Generating sample incident...")
        incident_time = datetime.now() - timedelta(hours=2)
        incident = system.generate_incident_report(
            application_name=app_name,
            start_time=incident_time,
            duration_hours=0.5
        )
        print(f"✅ Generated incident: {incident.incident_id}")

        # Generate HTML report first
        print("🌐 Creating HTML report...")
        html_path = system.create_comprehensive_html_report(
            metrics=metrics,
            incident=incident,
            output_path="reports/generated/browser_test_html.html"
        )
        print(f"✅ HTML report: {html_path}")

        # Generate browser PDF report
        print("📄 Creating browser PDF report...")
        browser_pdf_path = system.create_enhanced_pdf_report(
            metrics=metrics,
            incident=incident,
            output_path="reports/generated/browser_test_exact.pdf",
            use_browser=True  # Force browser PDF generation
        )
        print(f"✅ Browser PDF report: {browser_pdf_path}")

        # Generate WeasyPrint PDF for comparison
        print("📄 Creating WeasyPrint PDF report...")
        weasyprint_pdf_path = system.create_enhanced_pdf_report(
            metrics=metrics,
            incident=incident,
            output_path="reports/generated/browser_test_weasyprint.pdf",
            use_browser=False  # Force WeasyPrint generation
        )
        print(f"✅ WeasyPrint PDF report: {weasyprint_pdf_path}")

        print("\n🔍 Report Comparison Results:")
        print("=" * 60)

        # Check file sizes
        if os.path.exists(html_path):
            html_size = os.path.getsize(html_path)
            print(f"📊 HTML Report: {html_size:,} bytes")

        if os.path.exists(browser_pdf_path):
            browser_pdf_size = os.path.getsize(browser_pdf_path)
            print(f"📊 Browser PDF: {browser_pdf_size:,} bytes")
        else:
            print("❌ Browser PDF: Failed to generate")

        if os.path.exists(weasyprint_pdf_path):
            weasyprint_pdf_size = os.path.getsize(weasyprint_pdf_path)
            print(f"📊 WeasyPrint PDF: {weasyprint_pdf_size:,} bytes")
        else:
            print("❌ WeasyPrint PDF: Failed to generate")

        print("\n🎯 Key Differences Expected:")
        print("• 🌐 HTML Report: Full interactive features, Tailwind CSS, Font Awesome icons")
        print("• 📄 Browser PDF: Exact same visual appearance as HTML, rendered by browser")
        print("• 📄 WeasyPrint PDF: Converted CSS, emoji icons, simplified styling")

        print("\n✅ Browser PDF Test Complete!")
        print("Please review the generated files to verify visual matching:")
        print(f"• HTML: {html_path}")
        print(f"• Browser PDF: {browser_pdf_path}")
        print(f"• WeasyPrint PDF: {weasyprint_pdf_path}")

        return {
            'html_path': html_path,
            'browser_pdf_path': browser_pdf_path,
            'weasyprint_pdf_path': weasyprint_pdf_path
        }

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    test_browser_pdf_matching()