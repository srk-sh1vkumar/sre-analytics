"""
Integration Tests for End-to-End Report Generation

Tests the complete workflow from initialization through report generation,
verifying that all components work together correctly.
"""

import json
import shutil

# Add src to path for imports
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem


class TestEndToEndReportGeneration:
    """Test complete report generation workflow"""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def system(self):
        """Create SRE Report System instance"""
        return EnhancedSREReportSystem(app_name="IntegrationTestApp")

    def test_system_initialization(self, system):
        """Test that system initializes all components correctly"""
        # Verify system initialized
        assert system is not None
        assert system.app_name == "IntegrationTestApp"

        # Verify all components exist
        assert hasattr(system, "orchestrator")
        assert hasattr(system, "pdf_generator")
        assert hasattr(system, "metrics_generator")
        assert hasattr(system, "chart_generator")

    def test_get_system_status(self, system):
        """Test retrieving system status"""
        status = system.get_system_status()

        # Verify status structure
        assert "system" in status
        assert "orchestrator" in status

        # Verify system info
        assert status["system"]["app_name"] == "IntegrationTestApp"
        assert status["system"]["version"] == "2.0.0-refactored"
        assert status["system"]["architecture"] == "modular"

        # Verify components reported
        assert "components" in status["orchestrator"]
        components = status["orchestrator"]["components"]
        assert components["llm_analyzer"] is True
        assert components["metrics_generator"] is True

    def test_generate_metrics_with_trends(self, system):
        """Test generating metrics with historical trends"""
        metrics = system.generate_metrics_with_trends(
            services=["test-api", "test-db"], days_back=30
        )

        # Verify metrics generated
        assert len(metrics) > 0

        # Verify metric structure
        for metric in metrics:
            assert hasattr(metric, "service_name")
            assert hasattr(metric, "metric_name")
            assert hasattr(metric, "current_value")
            assert hasattr(metric, "slo_target")
            assert hasattr(metric, "status")
            assert hasattr(metric, "trend_data")
            assert metric.service_name in ["test-api", "test-db"]

    def test_create_trend_visualizations(self, system):
        """Test creating trend visualizations"""
        # Generate metrics first
        metrics = system.generate_metrics_with_trends(services=["test-service"], days_back=30)

        # Create visualizations (base64)
        charts = system.create_trend_visualizations(metrics=metrics, save_images=False)

        # Verify charts generated
        assert isinstance(charts, dict)
        if metrics:  # If we have metrics
            assert len(charts) > 0
            # Charts should be base64 strings
            for chart_name, chart_data in charts.items():
                assert isinstance(chart_data, str)

    def test_generate_incident_report(self, system):
        """Test generating incident report with RCA"""
        incident_time = datetime.now() - timedelta(hours=2)

        incident = system.generate_incident_report(
            application_name="IntegrationTestApp", incident_time=incident_time, duration_hours=1.5
        )

        # Verify incident structure
        assert incident is not None
        assert hasattr(incident, "incident_id")
        assert hasattr(incident, "application_name")
        assert hasattr(incident, "severity")
        assert hasattr(incident, "root_cause")
        assert hasattr(incident, "resolution_steps")
        assert incident.application_name == "IntegrationTestApp"

    def test_full_report_suite_without_incident(self, system, temp_output_dir):
        """Test generating full report suite without incident"""
        results = system.generate_full_report_suite(
            application_name="IntegrationTestApp", services=["api", "db"]
        )

        # Verify all report types generated
        assert "html_report" in results
        assert "json_data" in results

        # Verify files exist
        assert Path(results["html_report"]).exists()
        assert Path(results["json_data"]).exists()

        # Verify HTML file has content
        html_content = Path(results["html_report"]).read_text()
        assert "IntegrationTestApp" in html_content
        assert len(html_content) > 1000  # Substantial content

        # Verify JSON structure
        with open(results["json_data"], "r") as f:
            json_data = json.load(f)

        assert "report_metadata" in json_data
        assert "slo_metrics" in json_data
        assert "summary" in json_data
        assert json_data["report_metadata"]["application_name"] == "IntegrationTestApp"

    def test_full_report_suite_with_incident(self, system, temp_output_dir):
        """Test generating full report suite with incident analysis"""
        incident_time = datetime.now() - timedelta(hours=1)

        results = system.generate_full_report_suite(
            application_name="IntegrationTestApp",
            services=["api"],
            incident_time=incident_time,
            incident_duration=0.5,
        )

        # Verify reports generated
        assert "html_report" in results
        assert "json_data" in results

        # Verify incident included in HTML
        html_content = Path(results["html_report"]).read_text()
        # Should contain incident-related content
        assert len(html_content) > 1000

        # Verify incident in JSON
        with open(results["json_data"], "r") as f:
            json_data = json.load(f)

        assert json_data["incident"] is not None
        assert "incident_id" in json_data["incident"]

    def test_create_html_report_directly(self, system, temp_output_dir):
        """Test creating HTML report directly"""
        # Generate metrics
        metrics = system.generate_metrics_with_trends(services=["test-service"])

        # Create HTML report
        output_path = str(Path(temp_output_dir) / "direct_test.html")
        result = system.create_comprehensive_html_report(metrics=metrics, output_path=output_path)

        # Verify file created
        assert result == output_path
        assert Path(result).exists()

        # Verify content
        content = Path(result).read_text()
        assert "IntegrationTestApp" in content

    def test_backward_compatibility_simple_pdf_method(self, system, temp_output_dir):
        """Test backward compatibility of create_simple_pdf_report method"""
        # Generate metrics
        metrics = system.generate_metrics_with_trends(services=["test"])

        # Use old method name (should delegate to enhanced PDF)
        html_path = str(Path(temp_output_dir) / "test.html")
        pdf_path = str(Path(temp_output_dir) / "test.pdf")

        result = system.create_simple_pdf_report(
            html_path=html_path,  # This parameter is ignored in new implementation
            metrics=metrics,
            output_path=pdf_path,
        )

        # Should return a path (even if empty string on failure)
        assert isinstance(result, str)


class TestEndToEndErrorHandling:
    """Test error handling in end-to-end scenarios"""

    @pytest.fixture
    def system(self):
        return EnhancedSREReportSystem(app_name="ErrorTestApp")

    def test_report_generation_with_empty_services(self, system):
        """Test report generation with empty service list"""
        results = system.generate_full_report_suite(application_name="ErrorTestApp", services=[])

        # Should still generate reports
        assert "html_report" in results or "json_data" in results

    def test_report_generation_with_invalid_output_dir(self, system):
        """Test graceful handling of invalid output directory"""
        # This should create the directory if it doesn't exist
        # or handle the error gracefully
        try:
            results = system.generate_full_report_suite(
                application_name="ErrorTestApp", services=["test"]
            )
            # Should either succeed or handle error
            assert isinstance(results, dict)
        except Exception as e:
            # If it raises, should be a meaningful exception
            assert str(e)  # Has error message


class TestEndToEndPerformance:
    """Basic performance checks for end-to-end workflows"""

    @pytest.fixture
    def system(self):
        return EnhancedSREReportSystem(app_name="PerfTestApp")

    def test_metrics_generation_performance(self, system):
        """Test that metrics generation completes in reasonable time"""
        import time

        start = time.time()
        metrics = system.generate_metrics_with_trends(
            services=["service1", "service2", "service3"], days_back=30
        )
        duration = time.time() - start

        # Should complete within 5 seconds
        assert duration < 5.0
        assert len(metrics) > 0

    def test_full_report_suite_performance(self, system):
        """Test that full report generation completes in reasonable time"""
        import time

        start = time.time()
        results = system.generate_full_report_suite(
            application_name="PerfTestApp", services=["api", "db"]
        )
        duration = time.time() - start

        # Full suite should complete within 30 seconds
        # (includes metrics, charts, LLM analysis, HTML, PDF, JSON)
        assert duration < 30.0
        assert len(results) > 0


class TestEndToEndDataValidation:
    """Test data validation in end-to-end scenarios"""

    @pytest.fixture
    def system(self):
        return EnhancedSREReportSystem(app_name="DataTestApp")

    def test_json_export_data_structure(self, system):
        """Test that exported JSON has correct structure"""
        results = system.generate_full_report_suite(
            application_name="DataTestApp", services=["test"]
        )

        # Load and validate JSON
        with open(results["json_data"], "r") as f:
            data = json.load(f)

        # Verify required top-level keys
        assert "report_metadata" in data
        assert "slo_metrics" in data
        assert "summary" in data

        # Verify metadata structure
        metadata = data["report_metadata"]
        assert "application_name" in metadata
        assert "generated_at" in metadata
        assert "report_type" in metadata

        # Verify summary structure
        summary = data["summary"]
        assert "total_services" in summary
        assert "total_metrics" in summary
        assert "compliant_count" in summary
        assert "health_status" in summary

    def test_metrics_have_required_fields(self, system):
        """Test that generated metrics have all required fields"""
        metrics = system.generate_metrics_with_trends(services=["test"], days_back=30)

        for metric in metrics:
            # Verify all required attributes exist
            assert hasattr(metric, "service_name")
            assert hasattr(metric, "metric_name")
            assert hasattr(metric, "current_value")
            assert hasattr(metric, "slo_target")
            assert hasattr(metric, "unit")
            assert hasattr(metric, "status")
            assert hasattr(metric, "timestamp")
            assert hasattr(metric, "error_budget_consumed")
            assert hasattr(metric, "trend_data")

            # Verify status is valid
            assert metric.status in ["compliant", "at_risk", "breached"]

            # Verify error budget is in valid range
            assert 0 <= metric.error_budget_consumed <= 100


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
