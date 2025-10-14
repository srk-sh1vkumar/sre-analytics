"""
Unit Tests for PDFReportGenerator Module

Tests the multi-tier PDF generation system including:
- Browser PDF generation (primary)
- WeasyPrint PDF generation (fallback)
- ReportLab PDF generation (last resort)
- Template optimization for PDF
- Error handling and fallback logic
"""

# Add src to path for imports
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.exceptions import PDFGenerationError
from src.reports.llm_analyzer import IncidentData, SLOMetric
from src.reports.pdf_report_generator import (
    BROWSER_PDF_AVAILABLE,
    REPORTLAB_AVAILABLE,
    WEASYPRINT_AVAILABLE,
    PDFReportGenerator,
)


class TestPDFReportGeneratorInitialization:
    """Test PDFReportGenerator initialization"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        generator = PDFReportGenerator()

        assert generator.app_name == "Application"
        assert generator.logger is not None

    def test_init_with_custom_app_name(self):
        """Test initialization with custom app name"""
        generator = PDFReportGenerator(app_name="TestApp")

        assert generator.app_name == "TestApp"

    def test_log_capabilities_called(self):
        """Test that PDF capabilities are logged on init"""
        with patch.object(PDFReportGenerator, "_log_capabilities") as mock_log:
            generator = PDFReportGenerator()
            mock_log.assert_called_once()


class TestPDFReportGeneratorEnhancedPDF:
    """Test enhanced PDF creation with fallback strategy"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    @pytest.fixture
    def mock_metrics(self):
        return [
            SLOMetric(
                service_name="api-service",
                metric_name="latency",
                current_value=100.0,
                slo_target=200.0,
                sla_target=220.0,  # Added for test compatibility
                unit="ms",
                status="compliant",
                timestamp=datetime.now(),
                error_budget_consumed=10.0,
                trend_data=[95, 98, 100, 102, 100],
            )
        ]

    @pytest.fixture
    def mock_incident(self):
        return IncidentData(
            incident_id="INC-001",
            title="Test Incident",
            description="Test incident description",
            application_name="TestApp",
            severity="High",
            start_time=datetime.now(),
            end_time=datetime.now(),
            affected_services=["api"],
            root_cause="Issue",
            resolution_steps=["Fixed"],
            lessons_learned="Lesson",
            llm_analysis="Analysis",
        )

    def test_create_enhanced_pdf_browser_success(self, generator, mock_metrics, tmp_path):
        """Test successful PDF creation using browser method"""
        output_path = str(tmp_path / "test.pdf")
        html_content = "<html><body>TestApp Report</body></html>"

        with patch.object(generator, "_create_browser_pdf", return_value=output_path):
            result = generator.create_enhanced_pdf(
                html_content=html_content,
                metrics=mock_metrics,
                incident=None,
                output_path=output_path,
                use_browser=True,
            )

            assert result == output_path
            generator._create_browser_pdf.assert_called_once()

    @pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="WeasyPrint not available")
    def test_create_enhanced_pdf_weasyprint_fallback(self, generator, mock_metrics, tmp_path):
        """Test PDF creation falls back to WeasyPrint when browser fails"""
        output_path = str(tmp_path / "test.pdf")
        html_content = "<html><body>Test Report</body></html>"

        with patch.object(
            generator, "_create_browser_pdf", side_effect=Exception("Browser failed")
        ):
            with patch.object(generator, "_create_weasyprint_pdf", return_value=output_path):
                result = generator.create_enhanced_pdf(
                    html_content=html_content,
                    metrics=mock_metrics,
                    incident=None,
                    output_path=output_path,
                    use_browser=True,
                )

                assert result == output_path
                generator._create_browser_pdf.assert_called_once()
                generator._create_weasyprint_pdf.assert_called_once()

    def test_create_enhanced_pdf_reportlab_last_resort(self, generator, mock_metrics, tmp_path):
        """Test PDF creation falls back to ReportLab as last resort"""
        output_path = str(tmp_path / "test.pdf")
        html_content = "<html><body>Test Report</body></html>"

        with patch.object(
            generator, "_create_browser_pdf", side_effect=Exception("Browser failed")
        ):
            with patch.object(
                generator, "_create_weasyprint_pdf", side_effect=Exception("WeasyPrint failed")
            ):
                with patch.object(generator, "_create_reportlab_pdf", return_value=output_path):
                    result = generator.create_enhanced_pdf(
                        html_content=html_content,
                        metrics=mock_metrics,
                        incident=None,
                        output_path=output_path,
                        use_browser=True,
                    )

                    assert result == output_path
                    generator._create_reportlab_pdf.assert_called_once()

    def test_create_enhanced_pdf_auto_generates_path(self, generator, mock_metrics):
        """Test that output path is auto-generated if not provided"""
        html_content = "<html><body>Test Report</body></html>"

        with patch.object(
            generator,
            "_create_reportlab_pdf",
            return_value="reports/generated/enhanced_sre_report_20250101_120000.pdf",
        ):
            result = generator.create_enhanced_pdf(
                html_content=html_content,
                metrics=mock_metrics,
                incident=None,
                output_path=None,
                use_browser=False,
            )

            assert "reports/generated/" in result
            assert result.endswith(".pdf")


class TestPDFReportGeneratorBrowserPDF:
    """Test browser PDF generation method"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    def test_create_browser_pdf_success(self, generator, tmp_path):
        """Test successful browser PDF generation"""
        html_content = "<html><body>Test Content</body></html>"
        output_path = str(tmp_path / "test.pdf")

        with patch("src.reports.pdf_report_generator.BrowserPDFGenerator") as mock_browser:
            mock_instance = Mock()
            mock_instance.create_pdf_from_html_sync.return_value = True
            mock_browser.return_value = mock_instance

            result = generator._create_browser_pdf(html_content, output_path)

            assert result == output_path
            mock_instance.create_pdf_from_html_sync.assert_called_once_with(
                html_content, output_path
            )

    def test_create_browser_pdf_failure(self, generator, tmp_path):
        """Test browser PDF generation failure"""
        html_content = "<html><body>Test</body></html>"
        output_path = str(tmp_path / "test.pdf")

        with patch("src.reports.pdf_report_generator.BrowserPDFGenerator") as mock_browser:
            mock_instance = Mock()
            mock_instance.create_pdf_from_html_sync.return_value = False
            mock_browser.return_value = mock_instance

            with pytest.raises(PDFGenerationError):
                generator._create_browser_pdf(html_content, output_path)


class TestPDFReportGeneratorWeasyPrintPDF:
    """Test WeasyPrint PDF generation method"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    @pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="WeasyPrint not available")
    def test_create_weasyprint_pdf_success(self, generator, tmp_path):
        """Test successful WeasyPrint PDF generation"""
        html_content = "<html><body>Test</body></html>"
        output_path = str(tmp_path / "test.pdf")

        with patch("src.reports.pdf_report_generator.WeasyPrintPDFGenerator") as mock_weasy:
            mock_instance = Mock()
            mock_instance.create_pdf_from_html.return_value = True
            mock_weasy.return_value = mock_instance

            with patch.object(generator, "_setup_weasyprint_environment"):
                result = generator._create_weasyprint_pdf(html_content, output_path)

                assert result == output_path
                mock_instance.create_pdf_from_html.assert_called_once()

    @pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="WeasyPrint not available")
    def test_create_weasyprint_pdf_failure(self, generator, tmp_path):
        """Test WeasyPrint PDF generation failure"""
        html_content = "<html><body>Test</body></html>"
        output_path = str(tmp_path / "test.pdf")

        with patch("src.reports.pdf_report_generator.WeasyPrintPDFGenerator") as mock_weasy:
            mock_instance = Mock()
            mock_instance.create_pdf_from_html.return_value = False
            mock_weasy.return_value = mock_instance

            with patch.object(generator, "_setup_weasyprint_environment"):
                with pytest.raises(PDFGenerationError):
                    generator._create_weasyprint_pdf(html_content, output_path)


class TestPDFReportGeneratorReportLabPDF:
    """Test ReportLab PDF generation method"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    @pytest.fixture
    def mock_metrics(self):
        return [
            SLOMetric(
                service_name="api",
                metric_name="latency",
                current_value=100.0,
                slo_target=200.0,
                sla_target=220.0,  # Added for test compatibility
                unit="ms",
                status="compliant",
                timestamp=datetime.now(),
                error_budget_consumed=10.0,
                trend_data=[],
            ),
            SLOMetric(
                service_name="db",
                metric_name="latency",
                current_value=250.0,
                slo_target=200.0,
                sla_target=220.0,  # Added for test compatibility
                unit="ms",
                status="breached",
                timestamp=datetime.now(),
                error_budget_consumed=90.0,
                trend_data=[],
            ),
        ]

    @pytest.mark.skipif(not REPORTLAB_AVAILABLE, reason="ReportLab not available")
    def test_create_reportlab_pdf_without_incident(self, generator, mock_metrics, tmp_path):
        """Test ReportLab PDF generation without incident"""
        output_path = str(tmp_path / "test.pdf")

        with patch("src.reports.pdf_report_generator.REPORTLAB_AVAILABLE", True):
            with patch("src.reports.pdf_report_generator.SimpleDocTemplate") as mock_doc:
                with patch.object(
                    generator,
                    "_create_summary_stats",
                    return_value={
                        "total_services": 2,
                        "total_metrics": 2,
                        "compliant_count": 1,
                        "at_risk_count": 0,
                        "breached_count": 1,
                        "compliance_percentage": 50.0,
                        "health_status": "Critical",
                    },
                ):
                    # Mock chart generator (imported locally in the method)
                    with patch("src.reports.chart_generator.ChartGenerator") as mock_chart:
                        mock_chart_instance = Mock()
                        mock_chart_instance.create_trend_visualizations.return_value = {}
                        mock_chart.return_value = mock_chart_instance

                        # Mock LLM analyzer (imported locally in the method)
                        with patch("src.reports.llm_analyzer.LLMAnalyzer") as mock_llm:
                            mock_llm_instance = Mock()
                            mock_llm_instance.analyze_performance_metrics.return_value = "Analysis"
                            mock_llm.return_value = mock_llm_instance

                            result = generator._create_reportlab_pdf(
                                metrics=mock_metrics, incident=None, output_path=output_path
                            )

                            assert result == output_path

    def test_create_reportlab_pdf_not_available(self, generator, mock_metrics, tmp_path):
        """Test ReportLab PDF when library not available"""
        output_path = str(tmp_path / "test.pdf")

        with patch("src.reports.pdf_report_generator.REPORTLAB_AVAILABLE", False):
            with pytest.raises(PDFGenerationError):
                generator._create_reportlab_pdf(mock_metrics, None, output_path)


class TestPDFReportGeneratorTemplateOptimization:
    """Test HTML template optimization for PDF"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    def test_optimize_template_removes_scripts(self, generator):
        """Test that JavaScript and interactive elements are removed"""
        html_content = """
        <html>
        <head>
            <script src="chart.js"></script>
            <script src="tailwindcss"></script>
            <link rel="stylesheet" href="font-awesome.css">
        </head>
        <body onclick="alert('test')">
            <div class="floating-menu">Menu</div>
            Content
        </body>
        </html>
        """

        result = generator._optimize_template_for_pdf(html_content)

        # Verify scripts removed
        assert "chart.js" not in result
        assert "tailwindcss" not in result
        assert "font-awesome" not in result
        assert "onclick=" not in result
        assert "floating-menu" not in result

    def test_optimize_template_converts_tailwind(self, generator):
        """Test that Tailwind classes are converted to inline CSS"""
        html_content = '<div class="text-lg font-semibold text-white">Test</div>'

        result = generator._optimize_template_for_pdf(html_content)

        # Should convert to inline styles
        assert "style=" in result
        assert "font-size" in result or "font-weight" in result

    def test_optimize_template_converts_icons(self, generator):
        """Test that Font Awesome icons are converted to symbols"""
        html_content = '<i class="fas fa-tachometer-alt"></i><i class="fas fa-brain"></i>'

        result = generator._optimize_template_for_pdf(html_content)

        # Should convert to emoji/symbols
        assert "⚡" in result or "🧠" in result or "•" in result

    def test_optimize_template_updates_body(self, generator):
        """Test that body tag is updated with print-friendly styles"""
        html_content = '<body class="dark-mode interactive">'

        result = generator._optimize_template_for_pdf(html_content)

        # Should have print-friendly body styles
        assert 'style="font-family: Inter' in result


class TestPDFReportGeneratorSummaryStats:
    """Test summary statistics generation"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    def test_create_summary_stats_all_compliant(self, generator):
        """Test summary with all compliant metrics"""
        metrics = [
            SLOMetric(
                service_name="api",
                metric_name="latency",
                current_value=100,
                slo_target=200,
                sla_target=220.0,
                status="compliant",
                error_budget_consumed=10.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
            SLOMetric(
                service_name="db",
                metric_name="latency",
                current_value=150,
                slo_target=200,
                sla_target=220.0,
                status="compliant",
                error_budget_consumed=20.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
        ]

        summary = generator._create_summary_stats(metrics)

        assert summary["total_services"] == 2
        assert summary["total_metrics"] == 2
        assert summary["compliant_count"] == 2
        assert summary["at_risk_count"] == 0
        assert summary["breached_count"] == 0
        assert summary["compliance_percentage"] == 100.0
        assert summary["health_status"] == "Healthy"

    def test_create_summary_stats_with_breaches(self, generator):
        """Test summary with breached SLOs"""
        metrics = [
            SLOMetric(
                service_name="api",
                metric_name="latency",
                current_value=100,
                slo_target=200,
                sla_target=220.0,
                status="compliant",
                error_budget_consumed=10.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
            SLOMetric(
                service_name="db",
                metric_name="latency",
                current_value=300,
                slo_target=200,
                sla_target=220.0,
                status="breached",
                error_budget_consumed=95.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
        ]

        summary = generator._create_summary_stats(metrics)

        assert summary["breached_count"] == 1
        assert summary["health_status"] == "Unhealthy"

    def test_create_summary_stats_at_risk(self, generator):
        """Test summary with at-risk services"""
        metrics = [
            SLOMetric(
                service_name="api",
                metric_name="latency",
                current_value=100,
                slo_target=200,
                sla_target=220.0,
                status="compliant",
                error_budget_consumed=10.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
            SLOMetric(
                service_name="db",
                metric_name="latency",
                current_value=190,
                slo_target=200,
                sla_target=220.0,
                status="at_risk",
                error_budget_consumed=70.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
            SLOMetric(
                service_name="cache",
                metric_name="latency",
                current_value=195,
                slo_target=200,
                sla_target=220.0,
                status="at_risk",
                error_budget_consumed=75.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[],
            ),
        ]

        summary = generator._create_summary_stats(metrics)

        assert summary["at_risk_count"] == 2
        # 2 at-risk out of 3 = 66%, which is > 30% threshold
        assert summary["health_status"] == "Degraded"


class TestPDFReportGeneratorRecommendations:
    """Test recommendation generation"""

    @pytest.fixture
    def generator(self):
        return PDFReportGenerator(app_name="TestApp")

    def test_generate_recommendations_with_breaches(self, generator):
        """Test recommendations include breach alerts"""
        summary = {"breached_count": 2, "at_risk_count": 1}

        recommendations = generator._generate_recommendations(summary)

        # Should have urgent recommendation for breaches
        assert any("URGENT" in rec and "2" in rec for rec in recommendations)
        assert any("Monitor" in rec and "1" in rec for rec in recommendations)

    def test_generate_recommendations_healthy(self, generator):
        """Test recommendations for healthy system"""
        summary = {"breached_count": 0, "at_risk_count": 0}

        recommendations = generator._generate_recommendations(summary)

        # Should still have general recommendations
        assert len(recommendations) > 0
        assert any("error budget" in rec.lower() for rec in recommendations)


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
