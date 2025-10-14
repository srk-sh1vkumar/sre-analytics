"""
Enhanced SRE Report System - Refactored Facade

This module provides a backward-compatible API facade for the SRE report system.
All functionality has been extracted to specialized modules for better maintainability.

Main Components:
- ReportOrchestrator: Coordinates full report generation
- PDFReportGenerator: Multi-tier PDF generation with fallback
- HTMLTemplateBuilder: HTML template management
- MetricsGenerator: Performance metrics generation
- ChartGenerator: Trend visualization
- IncidentGenerator: Incident report generation
- LLMAnalyzer: AI-powered analysis
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.reports.chart_generator import ChartGenerator
from src.reports.configuration_loader import ConfigurationLoader
from src.reports.html_template_builder import HTMLTemplateBuilder
from src.reports.incident_generator import IncidentGenerator

# Import data classes
from src.reports.llm_analyzer import IncidentData, LLMAnalyzer, SLOMetric
from src.reports.metrics_generator import MetricsGenerator
from src.reports.pdf_report_generator import PDFReportGenerator

# Import all component modules
from src.reports.report_orchestrator import ReportOrchestrator


class EnhancedSREReportSystem:
    """
    Enhanced SRE Report System - Thin Facade

    This class maintains backward compatibility with the original API while
    delegating all functionality to specialized, focused modules.

    The refactored architecture provides:
    - Single Responsibility: Each module has one clear purpose
    - Maintainability: Smaller, focused files are easier to understand
    - Testability: Components can be tested independently
    - Reusability: Modules can be used standalone

    Example:
        >>> system = EnhancedSREReportSystem(app_name="MyApp")
        >>> reports = system.generate_full_report_suite(
        ...     application_name="MyApp",
        ...     services=["api", "db", "cache"]
        ... )
        >>> print(f"Generated: {reports['html_report']}")
    """

    def __init__(self, config_dir: str = "config", app_name: str = "Application"):
        """
        Initialize Enhanced SRE Report System.

        Args:
            config_dir: Directory containing configuration files
            app_name: Application name for reports
        """
        self.config_dir = Path(config_dir)
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)

        # Initialize all component modules
        self._initialize_components()

        self.logger.info(f"EnhancedSREReportSystem initialized for {app_name}")
        self._log_system_info()

    def _initialize_components(self) -> None:
        """Initialize all sub-component modules"""
        # Configuration loader
        self.config_loader = ConfigurationLoader(str(self.config_dir))

        # Core analysis and generation modules
        self.llm_analyzer = LLMAnalyzer()
        self.incident_generator = IncidentGenerator()
        self.metrics_generator = MetricsGenerator()
        self.chart_generator = ChartGenerator()

        # Template and report generation
        self.html_template_builder = HTMLTemplateBuilder()
        self.pdf_generator = PDFReportGenerator(self.app_name)

        # Main orchestrator
        self.orchestrator = ReportOrchestrator(self.app_name, str(self.config_dir))

        self.logger.debug("All components initialized successfully")

    def _log_system_info(self) -> None:
        """Log system initialization information"""
        self.logger.info("=" * 60)
        self.logger.info("Enhanced SRE Report System - Refactored Architecture")
        self.logger.info("=" * 60)
        self.logger.info(f"Application: {self.app_name}")
        self.logger.info(f"Config Directory: {self.config_dir}")
        self.logger.info("Components Initialized:")
        self.logger.info("  ✅ Report Orchestrator")
        self.logger.info("  ✅ PDF Generator (Multi-tier fallback)")
        self.logger.info("  ✅ HTML Template Builder")
        self.logger.info("  ✅ Metrics Generator")
        self.logger.info("  ✅ Chart Generator")
        self.logger.info("  ✅ Incident Generator")
        self.logger.info("  ✅ LLM Analyzer")
        self.logger.info("  ✅ Configuration Loader")
        self.logger.info("=" * 60)

    # ========================================================================
    # PUBLIC API - Backward Compatible Methods
    # ========================================================================

    def generate_full_report_suite(
        self,
        application_name: Optional[str] = None,
        services: Optional[List[str]] = None,
        incident_time: Optional[datetime] = None,
        incident_duration: float = 1.0,
    ) -> Dict[str, str]:
        """
        Generate complete report suite with performance and incident analysis.

        Creates HTML, PDF, and JSON reports with comprehensive metrics.

        Args:
            application_name: Override application name (uses default if None)
            services: List of services to analyze
            incident_time: When incident occurred (None = no incident analysis)
            incident_duration: Duration of incident in hours

        Returns:
            Dict mapping report types to file paths:
                - 'html_report': Path to HTML report
                - 'pdf_report': Path to PDF report (if generated)
                - 'json_data': Path to JSON data export

        Example:
            >>> reports = system.generate_full_report_suite(
            ...     application_name="E-Commerce",
            ...     services=["api", "db"]
            ... )
        """
        return self.orchestrator.generate_full_report_suite(
            application_name=application_name or self.app_name,
            services=services,
            incident_time=incident_time,
            incident_duration=incident_duration,
        )

    def create_comprehensive_html_report(
        self,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Create comprehensive HTML report with trends and incident analysis.

        Args:
            metrics: List of SLO metrics with trend data
            incident: Optional incident data
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to generated HTML report
        """
        return self.orchestrator.create_html_report(
            metrics=metrics, incident=incident, output_path=output_path
        )

    def create_enhanced_pdf_report(
        self,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None,
        use_browser: bool = True,
    ) -> str:
        """
        Create PDF report using enhanced template with multi-tier fallback.

        Priority order:
        1. Browser PDF (pyppeteer) - exact HTML rendering
        2. WeasyPrint - CSS-based rendering
        3. ReportLab - basic PDF

        Args:
            metrics: List of SLO metrics
            incident: Optional incident data
            output_path: Output file path (auto-generated if None)
            use_browser: Whether to use browser PDF as primary method

        Returns:
            Path to generated PDF file or empty string if failed
        """
        return self.orchestrator.create_pdf_report(
            metrics=metrics, incident=incident, output_path=output_path, use_browser=use_browser
        )

    def create_simple_pdf_report(
        self,
        html_path: str,
        metrics: List[SLOMetric],
        incident: Optional[IncidentData] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Create PDF report - delegates to enhanced PDF creation.

        Args:
            html_path: Path to HTML file (not used in new implementation)
            metrics: List of SLO metrics
            incident: Optional incident data
            output_path: Output file path

        Returns:
            Path to generated PDF file
        """
        # Backward compatibility: ignore html_path, use enhanced PDF generation
        return self.create_enhanced_pdf_report(
            metrics=metrics, incident=incident, output_path=output_path
        )

    def generate_metrics_with_trends(
        self, services: Optional[List[str]] = None, days_back: int = 30
    ) -> List[SLOMetric]:
        """
        Generate metrics with historical trend data.

        Args:
            services: List of service names (uses defaults if None)
            days_back: Number of days of historical data

        Returns:
            List of SLO metrics with trend data
        """
        return self.metrics_generator.generate_metrics_with_trends(
            services=services, days_back=days_back
        )

    def create_trend_visualizations(
        self, metrics: List[SLOMetric], save_images: bool = False
    ) -> Dict[str, str]:
        """
        Create trend visualization charts.

        Args:
            metrics: List of SLO metrics
            save_images: If True, save as image files; if False, return base64

        Returns:
            Dict mapping metric names to image paths or base64 strings
        """
        return self.chart_generator.create_trend_visualizations(
            metrics=metrics, save_images=save_images
        )

    def generate_incident_report(
        self, application_name: str, incident_time: datetime, duration_hours: float = 1.0
    ) -> IncidentData:
        """
        Generate incident report with RCA analysis.

        Args:
            application_name: Name of the application
            incident_time: When the incident occurred
            duration_hours: Duration of incident in hours

        Returns:
            IncidentData object with analysis
        """
        return self.incident_generator.generate_incident_report(
            application_name=application_name,
            incident_time=incident_time,
            duration_hours=duration_hours,
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dictionary with system and component status
        """
        return {
            "system": {
                "app_name": self.app_name,
                "config_dir": str(self.config_dir),
                "version": "2.0.0-refactored",
                "architecture": "modular",
            },
            "orchestrator": self.orchestrator.get_component_status(),
        }

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"EnhancedSREReportSystem(app_name='{self.app_name}', config_dir='{self.config_dir}')"
        )


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🚀 Enhanced SRE Report System - Refactored Architecture")
    print("=" * 60)
    print()

    # Get user input
    app_name = input("Enter application name (default: E-Commerce Platform): ").strip()
    app_name = app_name or "E-Commerce Platform"

    # Initialize system
    print(f"\\nInitializing SRE Report System for: {app_name}")
    system = EnhancedSREReportSystem(app_name=app_name)

    # Ask if user wants incident analysis
    want_incident = input("\\nInclude incident analysis? (y/n): ").strip().lower() == "y"

    incident_time = None
    incident_duration = 1.0

    if want_incident:
        hours_ago_input = input("Hours ago incident started (default: 2): ").strip()
        try:
            hours_ago = float(hours_ago_input) if hours_ago_input else 2.0
            incident_time = datetime.now() - timedelta(hours=hours_ago)

            duration_input = input("Incident duration in hours (default: 1.0): ").strip()
            incident_duration = float(duration_input) if duration_input else 1.0
        except ValueError:
            print("Invalid input, using defaults")
            incident_time = datetime.now() - timedelta(hours=2)
            incident_duration = 1.0

    # Generate report suite
    print("\\n📊 Generating comprehensive report suite...")
    print("This includes:")
    print("  • Performance metrics with 30-day trends")
    if want_incident:
        print("  • Incident analysis with RCA")
    print("  • AI-powered recommendations")
    print("  • HTML, PDF, and JSON exports")
    print()

    try:
        from datetime import timedelta

        results = system.generate_full_report_suite(
            application_name=app_name,
            incident_time=incident_time,
            incident_duration=incident_duration,
        )

        # Display results
        print("\\n✅ Report generation completed!")
        print("=" * 60)
        print("\\n📁 Generated Files:")
        for report_type, file_path in results.items():
            print(f"  • {report_type.replace('_', ' ').title()}: {file_path}")

        print("\\n💡 Features included:")
        print("  • 📈 30-day performance trend analysis")
        print("  • 🎯 SLO/SLA compliance tracking")
        print("  • 🚨 Incident RCA with LLM analysis")
        print("  • 📊 Interactive visualizations")
        print("  • 📄 PDF export for stakeholders")
        print("  • 📋 JSON data for API integration")
        print("  • 🤖 AI-powered recommendations")
        print("\\n🎉 Perfect for SRE teams tracking system reliability!")

    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
