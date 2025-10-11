"""
Report Orchestrator Module

Coordinates the entire report generation process across multiple components.
This module serves as the main coordinator for:
- Performance metrics generation
- Trend analysis and visualization
- Incident report generation
- HTML report creation
- PDF report creation with multi-tier fallback
- JSON data export
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import jinja2

# Import configuration and constants
from src.config.app_config import get_config
from src.config.constants import DEFAULT_TREND_DAYS

# Import data classes
from src.reports.llm_analyzer import SLOMetric, IncidentData

# Import extracted component modules
from src.reports.llm_analyzer import LLMAnalyzer
from src.reports.incident_generator import IncidentGenerator
from src.reports.metrics_generator import MetricsGenerator
from src.reports.chart_generator import ChartGenerator
from src.reports.html_template_builder import HTMLTemplateBuilder
from src.reports.pdf_report_generator import PDFReportGenerator
from src.reports.configuration_loader import ConfigurationLoader

# Import exceptions
from src.exceptions import (
    ReportGenerationError, FileWriteError, TemplateError
)


class ReportOrchestrator:
    """
    Orchestrates the complete report generation process.

    Coordinates all sub-components to generate comprehensive SRE reports
    including HTML, PDF, and JSON formats with performance metrics,
    trend analysis, and incident reports.
    """

    def __init__(self, app_name: str = "Application", config_dir: str = "config"):
        """
        Initialize Report Orchestrator.

        Args:
            app_name: Application name for reports
            config_dir: Directory containing configuration files
        """
        self.app_name = app_name
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

        # Initialize all component modules
        self._initialize_components()

        self.logger.info(f"Report Orchestrator initialized for {app_name}")

    def _initialize_components(self) -> None:
        """Initialize all sub-component modules"""
        self.config_loader = ConfigurationLoader(str(self.config_dir))
        self.llm_analyzer = LLMAnalyzer()
        self.incident_generator = IncidentGenerator()
        self.metrics_generator = MetricsGenerator()
        self.chart_generator = ChartGenerator()
        self.html_template_builder = HTMLTemplateBuilder()
        self.pdf_generator = PDFReportGenerator(self.app_name)

        self.logger.debug("All report components initialized successfully")

    def generate_full_report_suite(
        self,
        application_name: Optional[str] = None,
        services: Optional[List[str]] = None,
        incident_time: Optional[datetime] = None,
        incident_duration: float = 1.0,
        output_dir: str = "reports/generated"
    ) -> Dict[str, str]:
        """
        Generate complete report suite with performance and incident analysis.

        Creates HTML, PDF, and JSON reports with comprehensive metrics,
        trend analysis, and optional incident analysis.

        Args:
            application_name: Override application name (uses default if None)
            services: List of services to analyze (None = all services)
            incident_time: When incident occurred (None = no incident analysis)
            incident_duration: Duration of incident in hours
            output_dir: Directory for generated reports

        Returns:
            Dict mapping report types to file paths:
                - 'html_report': Path to HTML report
                - 'pdf_report': Path to PDF report (if generated)
                - 'json_data': Path to JSON data export

        Raises:
            ReportGenerationError: If report generation fails
        """
        try:
            if not application_name:
                application_name = self.app_name

            self.logger.info(f"Generating comprehensive report suite for {application_name}")

            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Generate performance metrics with trends
            metrics = self.metrics_generator.generate_metrics_with_trends(
                services,
                days_back=DEFAULT_TREND_DAYS
            )

            if not metrics:
                self.logger.warning("No metrics generated, creating empty report")

            # Generate incident report if incident time provided
            incident = None
            if incident_time:
                incident = self.incident_generator.generate_incident_report(
                    application_name,
                    incident_time,
                    incident_duration
                )
                self.logger.info(f"Incident report generated: {incident.incident_id}")

            # Generate reports
            results = {}

            # HTML Report with trends and incident analysis
            html_path = self.create_html_report(
                metrics,
                incident,
                output_path=f"{output_dir}/comprehensive_sre_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            results['html_report'] = html_path

            # PDF Report with multi-tier fallback
            pdf_path = self.create_pdf_report(
                metrics,
                incident,
                output_path=f"{output_dir}/comprehensive_sre_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            if pdf_path:
                results['pdf_report'] = pdf_path

            # JSON Data Export
            json_path = self.export_json_data(
                metrics,
                incident,
                output_path=f"{output_dir}/comprehensive_sre_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            results['json_data'] = json_path

            self.logger.info(f"Full report suite generation completed: {len(results)} files generated")
            return results

        except Exception as e:
            self.logger.error(f"Report suite generation failed: {e}")
            raise ReportGenerationError(
                f"Failed to generate report suite: {str(e)}",
                context={
                    'application_name': application_name,
                    'services': services,
                    'has_incident': incident_time is not None
                }
            )

    def create_html_report(
        self,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive HTML report with trends and incident analysis.

        Args:
            metrics: List of SLO metrics with trend data
            incident: Optional incident data
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to generated HTML report

        Raises:
            ReportGenerationError: If HTML generation fails
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"reports/generated/comprehensive_sre_report_{timestamp}.html"

            self.logger.info(f"Creating HTML report: {output_path}")

            # Create trend visualizations (base64 for HTML embedding)
            trend_charts = self.chart_generator.create_trend_visualizations(
                metrics,
                save_images=False
            )

            # Generate LLM analysis for performance insights
            summary = self._create_summary_stats(metrics)
            llm_analysis = self.llm_analyzer.analyze_performance_metrics(metrics, summary)

            # Prepare template data
            template_data = {
                'app_name': self.app_name,
                'report_date': datetime.now().strftime("%B %d, %Y"),
                'report_time': datetime.now().strftime("%H:%M:%S UTC"),
                'metrics': metrics,
                'trend_charts': trend_charts,
                'incident': incident,
                'summary': summary,
                'has_incident': incident is not None,
                'llm_analysis': llm_analysis
            }

            # Load and render enhanced HTML template
            html_content = self._load_html_template()
            template = jinja2.Template(html_content)
            rendered_html = template.render(**template_data)

            # Save HTML file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rendered_html)

            self.logger.info(f"✅ HTML report saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"HTML report generation failed: {e}")
            raise ReportGenerationError(
                f"Failed to generate HTML report: {str(e)}",
                context={'output_path': output_path}
            )

    def create_pdf_report(
        self,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None,
        use_browser: bool = True
    ) -> str:
        """
        Create PDF report with multi-tier fallback strategy.

        Args:
            metrics: List of SLO metrics
            incident: Optional incident data
            output_path: Output file path (auto-generated if None)
            use_browser: Whether to use browser PDF as primary method

        Returns:
            Path to generated PDF report or empty string if failed

        Raises:
            ReportGenerationError: If all PDF generation methods fail
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"reports/generated/comprehensive_sre_report_{timestamp}.pdf"

            self.logger.info(f"Creating PDF report: {output_path}")

            # Prepare data for PDF generation
            summary = self._create_summary_stats(metrics)
            llm_analysis = self.llm_analyzer.analyze_performance_metrics(metrics, summary)
            trend_charts = self.chart_generator.create_trend_visualizations(
                metrics,
                save_images=False
            )

            # Prepare template data
            template_data = {
                'app_name': self.app_name,
                'report_date': datetime.now().strftime("%B %d, %Y"),
                'report_time': datetime.now().strftime("%H:%M:%S UTC"),
                'metrics': metrics,
                'trend_charts': trend_charts,
                'incident': incident,
                'summary': summary,
                'has_incident': incident is not None,
                'llm_analysis': llm_analysis
            }

            # Load HTML template for PDF conversion
            html_template_content = self._load_html_template()

            # Generate PDF using multi-tier fallback
            pdf_path = self.pdf_generator.create_enhanced_pdf(
                html_template_content=html_template_content,
                metrics=metrics,
                incident=incident,
                output_path=output_path,
                use_browser=use_browser,
                template_data=template_data
            )

            if pdf_path:
                self.logger.info(f"✅ PDF report saved to {pdf_path}")
            else:
                self.logger.warning("PDF generation returned no path")

            return pdf_path

        except Exception as e:
            self.logger.error(f"PDF report generation failed: {e}")
            # Don't raise - PDF is optional, return empty string
            return ""

    def export_json_data(
        self,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export all report data as JSON.

        Args:
            metrics: List of SLO metrics
            incident: Optional incident data
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to generated JSON file

        Raises:
            FileWriteError: If JSON export fails
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"reports/generated/comprehensive_sre_data_{timestamp}.json"

            self.logger.info(f"Exporting JSON data: {output_path}")

            # Convert dataclasses to dictionaries
            def convert_dataclass(obj):
                if hasattr(obj, '__dict__'):
                    return {k: convert_dataclass(v) for k, v in asdict(obj).items()}
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, list):
                    return [convert_dataclass(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_dataclass(v) for k, v in obj.items()}
                return obj

            # Create summary statistics
            summary = self._create_summary_stats(metrics)

            # Prepare JSON data structure
            data = {
                "report_metadata": {
                    "application_name": self.app_name,
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "Comprehensive SRE Report with Trends and Incident Analysis",
                    "data_period_days": DEFAULT_TREND_DAYS
                },
                "slo_metrics": [convert_dataclass(m) for m in metrics],
                "summary": summary,
                "incident": convert_dataclass(incident) if incident else None
            }

            # Write JSON file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"✅ JSON data exported to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            raise FileWriteError(
                f"Failed to export JSON data: {str(e)}",
                file_path=output_path
            )

    def _load_html_template(self) -> str:
        """
        Load enhanced HTML template.

        Tries to load from template file, falls back to builder if not found.

        Returns:
            HTML template content
        """
        template_path = Path("templates/enhanced_sre_template.html")

        # Try to load from templates directory
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.logger.debug(f"Loaded HTML template from {template_path}")
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Could not load template file: {e}")

        # Fallback to template builder
        self.logger.debug("Using HTMLTemplateBuilder for template generation")
        return self.html_template_builder.get_comprehensive_html_template()

    def _create_summary_stats(self, metrics: List[SLOMetric]) -> Dict[str, Any]:
        """
        Create summary statistics from metrics.

        Args:
            metrics: List of SLO metrics

        Returns:
            Dictionary containing summary statistics
        """
        if not metrics:
            return {
                'total_services': 0,
                'total_metrics': 0,
                'compliant_count': 0,
                'at_risk_count': 0,
                'breached_count': 0,
                'compliance_percentage': 0.0,
                'health_status': 'No Data',
                'avg_error_budget_consumed': 0.0
            }

        total_services = len(set(m.service_name for m in metrics))
        total_metrics = len(metrics)
        compliant_count = len([m for m in metrics if m.status == "compliant"])
        at_risk_count = len([m for m in metrics if m.status == "at_risk"])
        breached_count = len([m for m in metrics if m.status == "breached"])

        compliance_percentage = (compliant_count / total_metrics * 100) if total_metrics > 0 else 0

        # Determine overall health status
        if breached_count > 0:
            health_status = 'Critical'
        elif at_risk_count > total_metrics * 0.3:
            health_status = 'Warning'
        elif compliant_count == total_metrics:
            health_status = 'Excellent'
        else:
            health_status = 'Healthy'

        # Calculate average error budget consumed
        avg_error_budget = sum(m.error_budget_consumed for m in metrics) / total_metrics if total_metrics > 0 else 0

        return {
            'total_services': total_services,
            'total_metrics': total_metrics,
            'compliant_count': compliant_count,
            'at_risk_count': at_risk_count,
            'breached_count': breached_count,
            'compliance_percentage': compliance_percentage,
            'health_status': health_status,
            'avg_error_budget_consumed': avg_error_budget
        }

    def get_component_status(self) -> Dict[str, Any]:
        """
        Get status of all orchestrator components.

        Useful for debugging and system health checks.

        Returns:
            Dictionary with component status information
        """
        return {
            'orchestrator': {
                'app_name': self.app_name,
                'config_dir': str(self.config_dir),
                'initialized': True
            },
            'components': {
                'llm_analyzer': self.llm_analyzer is not None,
                'incident_generator': self.incident_generator is not None,
                'metrics_generator': self.metrics_generator is not None,
                'chart_generator': self.chart_generator is not None,
                'html_template_builder': self.html_template_builder is not None,
                'pdf_generator': self.pdf_generator is not None,
                'config_loader': self.config_loader is not None
            },
            'pdf_capabilities': {
                'browser_pdf': hasattr(self.pdf_generator, '_create_browser_pdf'),
                'weasyprint': hasattr(self.pdf_generator, '_create_weasyprint_pdf'),
                'reportlab': hasattr(self.pdf_generator, '_create_reportlab_pdf')
            }
        }
