"""
Unit Tests for ReportOrchestrator Module

Tests the main coordination logic for report generation including:
- Full report suite generation
- HTML report creation
- PDF report creation
- JSON data export
- Component coordination
- Error handling
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reports.report_orchestrator import ReportOrchestrator
from src.reports.llm_analyzer import SLOMetric, IncidentData
from src.exceptions import ReportGenerationError, FileWriteError


class TestReportOrchestratorInitialization:
    """Test ReportOrchestrator initialization and component setup"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        orchestrator = ReportOrchestrator()

        assert orchestrator.app_name == "Application"
        assert orchestrator.config_dir == Path("config")
        assert orchestrator.config_loader is not None
        assert orchestrator.llm_analyzer is not None
        assert orchestrator.metrics_generator is not None
        assert orchestrator.chart_generator is not None
        assert orchestrator.pdf_generator is not None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters"""
        orchestrator = ReportOrchestrator(
            app_name="TestApp",
            config_dir="custom_config"
        )

        assert orchestrator.app_name == "TestApp"
        assert orchestrator.config_dir == Path("custom_config")

    def test_components_initialized(self):
        """Test all sub-components are properly initialized"""
        orchestrator = ReportOrchestrator()

        # Verify all components exist
        components = [
            'config_loader',
            'llm_analyzer',
            'incident_generator',
            'metrics_generator',
            'chart_generator',
            'html_template_builder',
            'pdf_generator'
        ]

        for component in components:
            assert hasattr(orchestrator, component)
            assert getattr(orchestrator, component) is not None


class TestReportOrchestratorFullReportSuite:
    """Test full report suite generation"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for testing"""
        return ReportOrchestrator(app_name="TestApp")

    @pytest.fixture
    def mock_metrics(self):
        """Create mock SLO metrics"""
        return [
            SLOMetric(
                service_name="api-service",
                metric_name="response_time",
                current_value=150.0,
                slo_target=200.0,
                sla_target=220.0,  # Added for test compatibility
                unit="ms",
                status="compliant",
                timestamp=datetime.now(),
                error_budget_consumed=25.0,
                trend_data=[140, 145, 150, 155, 150]
            ),
            SLOMetric(
                service_name="db-service",
                metric_name="availability",
                current_value=99.9,
                slo_target=99.95,
                sla_target=220.0,  # Added for test compatibility
                unit="%",
                status="at_risk",
                timestamp=datetime.now(),
                error_budget_consumed=80.0,
                trend_data=[99.95, 99.92, 99.90, 99.88, 99.90]
            )
        ]

    @pytest.fixture
    def mock_incident(self):
        """Create mock incident data"""
        return IncidentData(
            incident_id="INC-001",
            title="Test Incident",
            description="Test incident description",
            application_name="TestApp",
            severity="High",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
            affected_services=["api-service", "db-service"],
            root_cause="Database connection pool exhaustion",
            resolution_steps=["Increased pool size", "Restarted service"],
            lessons_learned="Monitor connection pool metrics",
            llm_analysis="AI analysis: Database saturation detected"
        )

    def test_generate_full_report_suite_success(self, orchestrator, mock_metrics, tmp_path):
        """Test successful full report suite generation"""
        with patch.object(orchestrator.metrics_generator, 'generate_metrics_with_trends', return_value=mock_metrics):
            with patch.object(orchestrator, 'create_html_report', return_value=str(tmp_path / "test.html")):
                with patch.object(orchestrator, 'create_pdf_report', return_value=str(tmp_path / "test.pdf")):
                    with patch.object(orchestrator, 'export_json_data', return_value=str(tmp_path / "test.json")):

                        results = orchestrator.generate_full_report_suite(
                            application_name="TestApp",
                            services=["api-service"],
                            output_dir=str(tmp_path)
                        )

                        # Verify all report types generated
                        assert 'html_report' in results
                        assert 'pdf_report' in results
                        assert 'json_data' in results

                        # Verify paths
                        assert results['html_report'].endswith('.html')
                        assert results['pdf_report'].endswith('.pdf')
                        assert results['json_data'].endswith('.json')

    def test_generate_full_report_suite_with_incident(self, orchestrator, mock_metrics, mock_incident, tmp_path):
        """Test report suite generation with incident analysis"""
        with patch.object(orchestrator.metrics_generator, 'generate_metrics_with_trends', return_value=mock_metrics):
            with patch.object(orchestrator.incident_generator, 'generate_incident_report', return_value=mock_incident):
                with patch.object(orchestrator, 'create_html_report', return_value=str(tmp_path / "test.html")):
                    with patch.object(orchestrator, 'create_pdf_report', return_value=str(tmp_path / "test.pdf")):
                        with patch.object(orchestrator, 'export_json_data', return_value=str(tmp_path / "test.json")):

                            results = orchestrator.generate_full_report_suite(
                                application_name="TestApp",
                                incident_time=datetime.now() - timedelta(hours=2),
                                incident_duration=1.0,
                                output_dir=str(tmp_path)
                            )

                            # Verify incident generator was called
                            orchestrator.incident_generator.generate_incident_report.assert_called_once()

                            # Verify reports generated
                            assert len(results) >= 3

    def test_generate_full_report_suite_no_metrics(self, orchestrator, tmp_path):
        """Test report generation with no metrics"""
        with patch.object(orchestrator.metrics_generator, 'generate_metrics_with_trends', return_value=[]):
            with patch.object(orchestrator, 'create_html_report', return_value=str(tmp_path / "test.html")):
                with patch.object(orchestrator, 'create_pdf_report', return_value=str(tmp_path / "test.pdf")):
                    with patch.object(orchestrator, 'export_json_data', return_value=str(tmp_path / "test.json")):

                        results = orchestrator.generate_full_report_suite(
                            application_name="TestApp",
                            output_dir=str(tmp_path)
                        )

                        # Should still generate reports
                        assert 'html_report' in results
                        assert 'json_data' in results


class TestReportOrchestratorHTMLReport:
    """Test HTML report creation"""

    @pytest.fixture
    def orchestrator(self):
        return ReportOrchestrator(app_name="TestApp")

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
                trend_data=[95, 98, 100, 102, 100]
            )
        ]

    def test_create_html_report_success(self, orchestrator, mock_metrics, tmp_path):
        """Test successful HTML report creation"""
        output_path = str(tmp_path / "test_report.html")

        with patch.object(orchestrator.chart_generator, 'create_trend_visualizations', return_value={}):
            with patch.object(orchestrator.llm_analyzer, 'analyze_performance_metrics', return_value="Analysis"):
                with patch.object(orchestrator, '_load_html_template', return_value="<html>{{app_name}}</html>"):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file

                        result = orchestrator.create_html_report(
                            metrics=mock_metrics,
                            output_path=output_path
                        )

                        assert result == output_path
                        mock_file.write.assert_called_once()

    def test_create_html_report_with_incident(self, orchestrator, mock_metrics, tmp_path):
        """Test HTML report with incident data"""
        incident = IncidentData(
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
            llm_analysis="Analysis"
        )

        output_path = str(tmp_path / "test_report_incident.html")

        with patch.object(orchestrator.chart_generator, 'create_trend_visualizations', return_value={}):
            with patch.object(orchestrator.llm_analyzer, 'analyze_performance_metrics', return_value="Analysis"):
                with patch.object(orchestrator, '_load_html_template', return_value="<html>{{app_name}}</html>"):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file

                        result = orchestrator.create_html_report(
                            metrics=mock_metrics,
                            incident=incident,
                            output_path=output_path
                        )

                        assert result == output_path


class TestReportOrchestratorPDFReport:
    """Test PDF report creation"""

    @pytest.fixture
    def orchestrator(self):
        return ReportOrchestrator(app_name="TestApp")

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
                trend_data=[95, 98, 100, 102, 100]
            )
        ]

    def test_create_pdf_report_success(self, orchestrator, mock_metrics, tmp_path):
        """Test successful PDF report creation"""
        output_path = str(tmp_path / "test_report.pdf")

        with patch.object(orchestrator.chart_generator, 'create_trend_visualizations', return_value={}):
            with patch.object(orchestrator.llm_analyzer, 'analyze_performance_metrics', return_value="Analysis"):
                with patch.object(orchestrator, '_load_html_template', return_value="<html>template</html>"):
                    with patch.object(orchestrator.pdf_generator, 'create_enhanced_pdf', return_value=output_path):

                        result = orchestrator.create_pdf_report(
                            metrics=mock_metrics,
                            output_path=output_path
                        )

                        assert result == output_path
                        orchestrator.pdf_generator.create_enhanced_pdf.assert_called_once()

    def test_create_pdf_report_fallback_on_failure(self, orchestrator, mock_metrics, tmp_path):
        """Test PDF report returns empty string on failure"""
        output_path = str(tmp_path / "test_report.pdf")

        with patch.object(orchestrator.chart_generator, 'create_trend_visualizations', return_value={}):
            with patch.object(orchestrator.llm_analyzer, 'analyze_performance_metrics', return_value="Analysis"):
                with patch.object(orchestrator, '_load_html_template', return_value="<html>template</html>"):
                    with patch.object(orchestrator.pdf_generator, 'create_enhanced_pdf', side_effect=Exception("PDF failed")):

                        result = orchestrator.create_pdf_report(
                            metrics=mock_metrics,
                            output_path=output_path
                        )

                        # Should return empty string on failure
                        assert result == ""


class TestReportOrchestratorJSONExport:
    """Test JSON data export"""

    @pytest.fixture
    def orchestrator(self):
        return ReportOrchestrator(app_name="TestApp")

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
                trend_data=[95, 98, 100, 102, 100]
            )
        ]

    def test_export_json_data_success(self, orchestrator, mock_metrics, tmp_path):
        """Test successful JSON export"""
        output_path = str(tmp_path / "test_data.json")

        result = orchestrator.export_json_data(
            metrics=mock_metrics,
            output_path=output_path
        )

        # Verify file created
        assert Path(result).exists()

        # Verify JSON structure
        with open(result, 'r') as f:
            data = json.load(f)

        assert 'report_metadata' in data
        assert 'slo_metrics' in data
        assert 'summary' in data
        assert data['report_metadata']['application_name'] == "TestApp"
        assert len(data['slo_metrics']) == 1

    def test_export_json_with_incident(self, orchestrator, mock_metrics, tmp_path):
        """Test JSON export with incident data"""
        incident = IncidentData(
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
            llm_analysis="Analysis"
        )

        output_path = str(tmp_path / "test_data_incident.json")

        result = orchestrator.export_json_data(
            metrics=mock_metrics,
            incident=incident,
            output_path=output_path
        )

        # Verify file created
        assert Path(result).exists()

        # Verify incident included
        with open(result, 'r') as f:
            data = json.load(f)

        assert data['incident'] is not None
        assert data['incident']['incident_id'] == "INC-001"


class TestReportOrchestratorSummaryStats:
    """Test summary statistics generation"""

    @pytest.fixture
    def orchestrator(self):
        return ReportOrchestrator(app_name="TestApp")

    def test_create_summary_stats_with_metrics(self, orchestrator):
        """Test summary statistics with various metrics"""
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
                trend_data=[]
            ),
            SLOMetric(
                service_name="api",
                metric_name="errors",
                current_value=0.5,
                slo_target=1.0,
                sla_target=1.1,
                status="compliant",
                error_budget_consumed=20.0,
                timestamp=datetime.now(),
                unit="%",
                trend_data=[]
            ),
            SLOMetric(
                service_name="db",
                metric_name="latency",
                current_value=250,
                slo_target=200,
                sla_target=220.0,
                status="at_risk",
                error_budget_consumed=60.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[]
            ),
            SLOMetric(
                service_name="cache",
                metric_name="latency",
                current_value=500,
                slo_target=200,
                sla_target=220.0,
                status="breached",
                error_budget_consumed=90.0,
                timestamp=datetime.now(),
                unit="ms",
                trend_data=[]
            ),
        ]

        summary = orchestrator._create_summary_stats(metrics)

        assert summary['total_services'] == 3  # api, db, cache
        assert summary['total_metrics'] == 4
        assert summary['compliant_count'] == 2
        assert summary['at_risk_count'] == 1
        assert summary['breached_count'] == 1
        assert summary['compliance_percentage'] == 50.0
        assert summary['health_status'] == 'Critical'

    def test_create_summary_stats_empty_metrics(self, orchestrator):
        """Test summary statistics with no metrics"""
        summary = orchestrator._create_summary_stats([])

        assert summary['total_services'] == 0
        assert summary['total_metrics'] == 0
        assert summary['compliant_count'] == 0
        assert summary['health_status'] == 'No Data'


class TestReportOrchestratorComponentStatus:
    """Test component status reporting"""

    def test_get_component_status(self):
        """Test retrieving component status"""
        orchestrator = ReportOrchestrator(app_name="TestApp", config_dir="config")

        status = orchestrator.get_component_status()

        # Verify structure
        assert 'orchestrator' in status
        assert 'components' in status
        assert 'pdf_capabilities' in status

        # Verify orchestrator info
        assert status['orchestrator']['app_name'] == "TestApp"
        assert status['orchestrator']['initialized'] is True

        # Verify all components reported
        components = status['components']
        assert components['llm_analyzer'] is True
        assert components['metrics_generator'] is True
        assert components['chart_generator'] is True
        assert components['pdf_generator'] is True


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
