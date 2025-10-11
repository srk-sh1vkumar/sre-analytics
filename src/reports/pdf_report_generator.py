"""
PDF Report Generator Module

Handles all PDF generation logic with multi-tier fallback strategy:
1. Browser PDF (Puppeteer) - exact HTML rendering (PRIMARY)
2. WeasyPrint - CSS-based rendering (FALLBACK)
3. ReportLab - basic PDF generation (LAST RESORT)
"""

import os
import re
import logging
import jinja2
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.config.app_config import get_config
from src.exceptions import (
    PDFGenerationError, FileWriteError, TemplateError
)

# Import data classes from extracted modules
from src.reports.llm_analyzer import SLOMetric, IncidentData

# Check availability of PDF generators
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage,
        Table, TableStyle, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import blue, green, red, orange, black
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import weasyprint
    from reports.weasyprint_pdf_generator import WeasyPrintPDFGenerator
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False

try:
    from .browser_pdf_generator import BrowserPDFGenerator
    BROWSER_PDF_AVAILABLE = True
except ImportError:
    BROWSER_PDF_AVAILABLE = False


class PDFReportGenerator:
    """
    Multi-tier PDF report generator with fallback strategy

    Attempts PDF generation in the following order:
    1. Browser PDF (Puppeteer) - Highest quality, exact HTML match
    2. WeasyPrint - CSS-based, good quality
    3. ReportLab - Basic formatting, always available
    """

    def __init__(self, app_name: str = "Application"):
        """
        Initialize PDF report generator

        Args:
            app_name: Application name for reports
        """
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)
        self._log_capabilities()

    def _log_capabilities(self) -> None:
        """Log available PDF generation capabilities"""
        self.logger.info("PDF Generation Capabilities:")
        self.logger.info(f"  🌐 Browser PDF (Puppeteer): {'✅ Available' if BROWSER_PDF_AVAILABLE else '❌ Not Available'}")
        self.logger.info(f"  📄 WeasyPrint: {'✅ Available' if WEASYPRINT_AVAILABLE else '❌ Not Available'}")
        self.logger.info(f"  📋 ReportLab: {'✅ Available' if REPORTLAB_AVAILABLE else '❌ Not Available'}")

    def create_enhanced_pdf(
        self,
        html_content: str,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None,
        use_browser: bool = True
    ) -> str:
        """
        Create PDF report using multi-tier fallback strategy

        Args:
            html_content: Rendered HTML content for PDF
            metrics: List of SLO metrics
            incident: Optional incident data
            output_path: Output file path
            use_browser: Whether to attempt browser PDF first

        Returns:
            str: Path to generated PDF file

        Raises:
            PDFGenerationError: If all PDF generation methods fail
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/enhanced_sre_report_{timestamp}.pdf"

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try Browser PDF first (PRIMARY)
        if BROWSER_PDF_AVAILABLE and use_browser:
            try:
                return self._create_browser_pdf(html_content, output_path)
            except Exception as e:
                self.logger.warning(f"⚠️ Browser PDF failed: {e}, falling back to WeasyPrint...")

        # Try WeasyPrint (FALLBACK)
        if WEASYPRINT_AVAILABLE:
            try:
                # Optimize HTML for WeasyPrint
                optimized_html = self._optimize_template_for_pdf(html_content)
                return self._create_weasyprint_pdf(optimized_html, output_path)
            except Exception as e:
                self.logger.warning(f"⚠️ WeasyPrint failed: {e}, falling back to ReportLab...")

        # Try ReportLab (LAST RESORT)
        if REPORTLAB_AVAILABLE:
            return self._create_reportlab_pdf(metrics, incident, output_path)

        # All methods failed
        raise PDFGenerationError(
            "All PDF generation methods failed",
            context={
                'browser_available': BROWSER_PDF_AVAILABLE,
                'weasyprint_available': WEASYPRINT_AVAILABLE,
                'reportlab_available': REPORTLAB_AVAILABLE
            }
        )

    def _create_browser_pdf(self, html_content: str, output_path: str) -> str:
        """
        Generate PDF using headless browser (Puppeteer)

        Args:
            html_content: HTML content to render
            output_path: Output PDF path

        Returns:
            str: Path to generated PDF

        Raises:
            PDFGenerationError: If browser PDF generation fails
        """
        self.logger.info("Generating PDF using headless browser...")

        browser_generator = BrowserPDFGenerator()
        success = browser_generator.create_pdf_from_html_sync(html_content, output_path)

        if success:
            self.logger.info(f"✅ Enhanced PDF generated with browser (PRIMARY): {output_path}")
            return output_path
        else:
            raise PDFGenerationError("Browser PDF generation returned failure status")

    def _create_weasyprint_pdf(self, html_content: str, output_path: str) -> str:
        """
        Generate PDF using WeasyPrint

        Args:
            html_content: HTML content to render
            output_path: Output PDF path

        Returns:
            str: Path to generated PDF

        Raises:
            PDFGenerationError: If WeasyPrint PDF generation fails
        """
        self.logger.info("Generating PDF using WeasyPrint...")

        # Setup WeasyPrint environment (macOS-specific paths if configured)
        self._setup_weasyprint_environment()

        pdf_generator = WeasyPrintPDFGenerator()
        success = pdf_generator.create_pdf_from_html(
            html_content,
            output_path,
            base_url=f"file://{Path().absolute()}/"
        )

        if success:
            self.logger.info(f"✅ Enhanced PDF generated with WeasyPrint (FALLBACK): {output_path}")
            return output_path
        else:
            raise PDFGenerationError("WeasyPrint PDF generation returned failure status")

    def _setup_weasyprint_environment(self) -> None:
        """Configure environment variables for WeasyPrint (macOS-specific)"""
        config = get_config()
        if config.system.pkg_config_path:
            os.environ['PKG_CONFIG_PATH'] = config.system.pkg_config_path
        if config.system.dyld_library_path:
            os.environ['DYLD_LIBRARY_PATH'] = config.system.dyld_library_path

    def _create_reportlab_pdf(
        self,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData],
        output_path: str
    ) -> str:
        """
        Generate basic PDF using ReportLab (last resort fallback)

        This method creates a simplified PDF when browser and WeasyPrint are unavailable.

        Args:
            metrics: List of SLO metrics
            incident: Optional incident data
            output_path: Output PDF path

        Returns:
            str: Path to generated PDF

        Raises:
            PDFGenerationError: If ReportLab PDF generation fails
        """
        if not REPORTLAB_AVAILABLE:
            raise PDFGenerationError("ReportLab is not available")

        try:
            self.logger.info("Generating PDF using ReportLab (LAST RESORT)...")

            # Import chart generator for visualizations
            from src.reports.chart_generator import ChartGenerator
            from src.reports.llm_analyzer import LLMAnalyzer

            chart_gen = ChartGenerator()
            llm_analyzer = LLMAnalyzer()

            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Generate trend charts for embedding
            self.logger.info("Generating trend charts for PDF embedding...")
            trend_chart_images = chart_gen.create_trend_visualizations(metrics, save_images=True)

            # Title page
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1,  # Center
            )
            story.append(Paragraph(f"{self.app_name}", title_style))
            story.append(Paragraph("Comprehensive SRE Performance Report", styles['Heading2']))
            story.append(Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles['Normal']
            ))
            story.append(Spacer(1, 25))

            # Executive Summary
            summary = self._create_summary_stats(metrics)
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            story.append(Spacer(1, 8))

            # Summary table
            summary_data = [
                ['Metric', 'Value'],
                ['Total Services Monitored', str(summary['total_services'])],
                ['Total SLO Metrics', str(summary['total_metrics'])],
                ['Compliant Metrics', f"{summary['compliant_count']} ({summary['compliance_percentage']:.1f}%)"],
                ['At Risk Metrics', str(summary['at_risk_count'])],
                ['Breached SLOs', str(summary['breached_count'])],
                ['Overall System Health', summary['health_status']]
            ]

            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), (0.95, 0.95, 0.95)),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))

            # SLO Metrics Details
            story.append(Paragraph("SLO Metrics Status Details", styles['Heading2']))
            story.append(Spacer(1, 8))

            # Create metrics table
            metrics_data = [['Service', 'Metric', 'Current', 'Target', 'Status', 'Error Budget']]
            for metric in metrics:
                metrics_data.append([
                    metric.service_name,
                    metric.metric_name.replace('_', ' ').title(),
                    f"{metric.current_value:.2f} {metric.unit}",
                    f"{metric.slo_target:.2f} {metric.unit}",
                    metric.status.title(),
                    f"{metric.error_budget_consumed:.1f}%"
                ])

            metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), (0.98, 0.98, 0.98)),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20))

            # Add trend visualizations
            if trend_chart_images:
                story.append(Paragraph("📊 Performance Trend Analysis", styles['Heading2']))
                story.append(Spacer(1, 8))
                story.append(Paragraph(
                    "The following charts show performance trends over the last 30 days.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 10))

                for chart_name, image_path in trend_chart_images.items():
                    try:
                        chart_title = f"{chart_name.replace('_', ' ').title()} Performance Trends"
                        story.append(Paragraph(chart_title, styles['Heading3']))
                        story.append(Spacer(1, 6))

                        chart_image = ReportLabImage(image_path, width=6.5*inch, height=4.5*inch)
                        story.append(chart_image)
                        story.append(Spacer(1, 10))
                    except Exception as img_error:
                        self.logger.warning(f"Failed to add chart {chart_name}: {img_error}")

            # AI-Powered Analysis
            story.append(Spacer(1, 15))
            story.append(Paragraph("🤖 AI-Powered Performance Analysis", styles['Heading2']))
            story.append(Spacer(1, 8))

            performance_analysis = llm_analyzer.analyze_performance_metrics(metrics, summary)
            analysis_paragraphs = performance_analysis.split('\n\n')
            for para in analysis_paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 10))

            # Recommendations
            story.append(Paragraph("Key Recommendations", styles['Heading3']))
            story.append(Spacer(1, 6))

            recommendations = self._generate_recommendations(summary)
            for rec in recommendations:
                story.append(Paragraph(rec, styles['Normal']))
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 15))

            # Incident information if available
            if incident:
                self._add_incident_to_pdf(story, incident, styles)

            # Footer
            self._add_pdf_footer(story, styles)

            # Build PDF
            doc.build(story)

            # Clean up temporary chart files
            for _, image_path in trend_chart_images.items():
                try:
                    os.unlink(image_path)
                except:
                    pass

            self.logger.info(f"✅ PDF generated with ReportLab (LAST RESORT): {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"ReportLab PDF generation failed: {e}")
            raise PDFGenerationError(
                f"ReportLab PDF generation failed: {str(e)}",
                context={'output_path': output_path}
            )

    def _create_summary_stats(self, metrics: List[SLOMetric]) -> Dict[str, Any]:
        """
        Create summary statistics from metrics

        Args:
            metrics: List of SLO metrics

        Returns:
            Dict containing summary statistics
        """
        total_metrics = len(metrics)
        compliant_count = sum(1 for m in metrics if m.status == 'compliant')
        at_risk_count = sum(1 for m in metrics if m.status == 'at_risk')
        breached_count = sum(1 for m in metrics if m.status == 'breached')

        compliance_percentage = (compliant_count / total_metrics * 100) if total_metrics > 0 else 0

        # Determine overall health
        if breached_count > 0:
            health_status = 'Unhealthy'
        elif at_risk_count > total_metrics * 0.3:
            health_status = 'Degraded'
        else:
            health_status = 'Healthy'

        # Calculate average error budget consumed
        avg_error_budget = sum(m.error_budget_consumed for m in metrics) / total_metrics if total_metrics > 0 else 0

        return {
            'total_services': len(set(m.service_name for m in metrics)),
            'total_metrics': total_metrics,
            'compliant_count': compliant_count,
            'at_risk_count': at_risk_count,
            'breached_count': breached_count,
            'compliance_percentage': compliance_percentage,
            'health_status': health_status,
            'avg_error_budget_consumed': avg_error_budget
        }

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on summary statistics"""
        recommendations = []
        if summary['breached_count'] > 0:
            recommendations.append(f"• URGENT: Address {summary['breached_count']} SLO breach(es) immediately")
        if summary['at_risk_count'] > 0:
            recommendations.append(f"• Monitor {summary['at_risk_count']} service(s) at risk of SLO breach")
        recommendations.extend([
            "• Review error budget consumption and implement proactive alerting",
            "• Analyze performance trends for capacity planning",
            "• Implement automated scaling based on performance metrics",
            "• Update incident response procedures based on latest analysis"
        ])
        return recommendations

    def _add_incident_to_pdf(self, story: List, incident: IncidentData, styles) -> None:
        """Add incident information to PDF story"""
        story.append(PageBreak())
        story.append(Paragraph("🚨 Incident Analysis Report", styles['Heading1']))
        story.append(Spacer(1, 20))

        # Incident details table
        incident_data = [
            ['Field', 'Value'],
            ['Incident ID', incident.incident_id],
            ['Application', incident.application_name],
            ['Severity', incident.severity],
            ['Start Time', incident.start_time.strftime('%Y-%m-%d %H:%M:%S')],
            ['End Time', incident.end_time.strftime('%Y-%m-%d %H:%M:%S') if incident.end_time else 'Ongoing'],
            ['Affected Services', ', '.join(incident.affected_services)]
        ]

        incident_table = Table(incident_data, colWidths=[2*inch, 4*inch])
        incident_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), red),
            ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), (1, 0.9, 0.9)),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        story.append(incident_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Root Cause Analysis", styles['Heading3']))
        story.append(Paragraph(incident.root_cause, styles['Normal']))
        story.append(Spacer(1, 15))

        if incident.llm_analysis:
            story.append(Paragraph("AI-Powered Analysis Summary", styles['Heading3']))
            llm_text = incident.llm_analysis[:1000]  # Truncate if too long
            for para in llm_text.split('\n\n'):
                if para.strip():
                    story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Spacer(1, 15))

    def _add_pdf_footer(self, story: List, styles) -> None:
        """Add footer to PDF"""
        story.append(Spacer(1, 20))
        story.append(Paragraph("Report Generation Details", styles['Heading3']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Generated by: Enhanced SRE Report System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            styles['Normal']
        ))

    def _optimize_template_for_pdf(self, html_content: str) -> str:
        """
        Optimize HTML template for PDF rendering

        Converts Tailwind classes to inline CSS and removes interactive elements.

        Args:
            html_content: HTML content to optimize

        Returns:
            str: PDF-optimized HTML content
        """
        # Tailwind to inline CSS conversions
        tailwind_conversions = [
            # Grid system
            (r'class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"',
             'style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15pt;"'),

            # Metric cards
            (r'class="metric-card p-6 rounded-xl"',
             'style="background: #2d3748; border: 1px solid #4a5568; padding: 15pt; border-radius: 8pt; margin-bottom: 10pt; page-break-inside: avoid;"'),

            # Status indicators
            (r'class="status-indicator ([^"]*) mr-3"',
             r'style="width: 12pt; height: 12pt; border-radius: 50%; margin-right: 8pt; display: inline-block; \1"'),
            (r'bg-green-500', 'background-color: #10b981;'),
            (r'bg-yellow-500', 'background-color: #f59e0b;'),
            (r'bg-red-500', 'background-color: #ef4444;'),

            # Text styling
            (r'class="text-lg font-semibold text-white"',
             'style="font-size: 12pt; font-weight: 600; color: #ffffff;"'),
            (r'class="text-3xl font-bold text-white"',
             'style="font-size: 18pt; font-weight: bold; color: #ffffff;"'),
            (r'class="text-sm text-slate-400 mb-1"',
             'style="font-size: 9pt; color: #94a3b8; margin-bottom: 3pt;"'),
        ]

        # Apply conversions
        for pattern, replacement in tailwind_conversions:
            html_content = re.sub(pattern, replacement, html_content, flags=re.IGNORECASE)

        # Icon conversions
        icon_conversions = [
            (r'<i class="fas fa-tachometer-alt[^"]*"></i>', '⚡'),
            (r'<i class="fas fa-brain[^"]*"></i>', '🧠'),
            (r'<i class="fas fa-[^"]*"></i>', '•'),
        ]

        for pattern, replacement in icon_conversions:
            html_content = re.sub(pattern, replacement, html_content, flags=re.IGNORECASE)

        # PDF optimizations (remove scripts, interactive elements)
        pdf_optimizations = [
            (r'<script.*?chart\.js.*?</script>', ''),
            (r'<script.*?tailwindcss.*?</script>', ''),
            (r'<link.*?font-awesome.*?>', ''),
            (r'<div class="floating-menu">.*?</div>', ''),
            (r'onclick="[^"]*"', ''),
            (r'<body[^>]*>',
             '<body style="font-family: Inter, Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #1f2937; background: white; margin: 0; padding: 20pt;">'),
        ]

        for pattern, replacement in pdf_optimizations:
            html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL | re.IGNORECASE)

        return html_content
