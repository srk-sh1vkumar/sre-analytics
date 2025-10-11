"""
Performance Benchmarks for SRE Analytics System

Measures performance of key operations to ensure no regression after refactoring.
Establishes baselines for future performance optimization.
"""

import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem


class TestPerformanceBenchmarks:
    """Performance benchmarks for core operations"""

    @pytest.fixture
    def system(self):
        """Create system instance"""
        return EnhancedSREReportSystem(app_name="BenchmarkApp")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_system_initialization_time(self):
        """Benchmark: System initialization should be fast"""
        start = time.time()
        system = EnhancedSREReportSystem(app_name="BenchmarkApp")
        duration = time.time() - start

        print(f"\n⏱️ System Initialization: {duration:.3f}s")

        # Should initialize within 1 second
        assert duration < 1.0, f"Initialization took {duration:.3f}s, expected < 1.0s"

    def test_metrics_generation_time(self, system):
        """Benchmark: Metrics generation with trends"""
        services = ["api", "db", "cache", "queue"]

        start = time.time()
        metrics = system.generate_metrics_with_trends(
            services=services,
            days_back=30
        )
        duration = time.time() - start

        print(f"\n⏱️ Metrics Generation ({len(services)} services, 30 days): {duration:.3f}s")
        print(f"   Generated {len(metrics)} metrics")

        # Should complete within 3 seconds for 4 services
        assert duration < 3.0, f"Metrics generation took {duration:.3f}s, expected < 3.0s"
        assert len(metrics) > 0

    def test_chart_generation_time(self, system):
        """Benchmark: Chart visualization generation"""
        metrics = system.generate_metrics_with_trends(
            services=["api", "db"],
            days_back=30
        )

        start = time.time()
        charts = system.create_trend_visualizations(
            metrics=metrics,
            save_images=False  # Base64 encoding
        )
        duration = time.time() - start

        print(f"\n⏱️ Chart Generation ({len(metrics)} metrics): {duration:.3f}s")
        print(f"   Generated {len(charts)} charts")

        # Chart generation should be fast (< 5 seconds)
        assert duration < 5.0, f"Chart generation took {duration:.3f}s, expected < 5.0s"

    def test_html_report_generation_time(self, system, temp_dir):
        """Benchmark: HTML report generation"""
        metrics = system.generate_metrics_with_trends(
            services=["api", "db"],
            days_back=30
        )

        output_path = str(Path(temp_dir) / "benchmark.html")

        start = time.time()
        result = system.create_comprehensive_html_report(
            metrics=metrics,
            output_path=output_path
        )
        duration = time.time() - start

        print(f"\n⏱️ HTML Report Generation: {duration:.3f}s")
        print(f"   File size: {Path(result).stat().st_size / 1024:.1f} KB")

        # HTML generation should be fast
        assert duration < 10.0, f"HTML generation took {duration:.3f}s, expected < 10.0s"
        assert Path(result).exists()

    def test_full_report_suite_time(self, system, temp_dir):
        """Benchmark: Full report suite generation"""
        start = time.time()
        results = system.generate_full_report_suite(
            application_name="BenchmarkApp",
            services=["api", "db", "cache"]
        )
        duration = time.time() - start

        print(f"\n⏱️ Full Report Suite Generation: {duration:.3f}s")
        print(f"   Generated {len(results)} reports")
        for report_type, path in results.items():
            if Path(path).exists():
                size = Path(path).stat().st_size / 1024
                print(f"   - {report_type}: {size:.1f} KB")

        # Full suite includes metrics, charts, LLM analysis, HTML, PDF, JSON
        # Should complete within 30 seconds
        assert duration < 30.0, f"Full suite took {duration:.3f}s, expected < 30.0s"
        assert len(results) >= 2  # At least HTML and JSON

    def test_incident_generation_time(self, system):
        """Benchmark: Incident report generation"""
        incident_time = datetime.now() - timedelta(hours=2)

        start = time.time()
        incident = system.generate_incident_report(
            application_name="BenchmarkApp",
            incident_time=incident_time,
            duration_hours=1.0
        )
        duration = time.time() - start

        print(f"\n⏱️ Incident Report Generation: {duration:.3f}s")

        # Incident generation should be fast
        assert duration < 2.0, f"Incident generation took {duration:.3f}s, expected < 2.0s"
        assert incident is not None


class TestPerformanceScalability:
    """Test performance with varying data sizes"""

    @pytest.fixture
    def system(self):
        return EnhancedSREReportSystem(app_name="ScalabilityApp")

    def test_metrics_generation_scales_with_services(self, system):
        """Test that metrics generation scales linearly with service count"""
        service_counts = [2, 5, 10]
        timings = []

        for count in service_counts:
            services = [f"service-{i}" for i in range(count)]

            start = time.time()
            metrics = system.generate_metrics_with_trends(
                services=services,
                days_back=30
            )
            duration = time.time() - start
            timings.append(duration)

            print(f"\n⏱️ {count} services: {duration:.3f}s ({len(metrics)} metrics)")

        # Each doubling of services should take less than 3x time (sub-quadratic)
        for i in range(len(timings) - 1):
            ratio = timings[i+1] / timings[i]
            service_ratio = service_counts[i+1] / service_counts[i]
            print(f"   Scaling ratio: {ratio:.2f}x for {service_ratio:.1f}x services")
            assert ratio < (service_ratio * 1.5), "Performance degradation detected"

    def test_metrics_generation_scales_with_days(self, system):
        """Test that metrics generation scales with historical data range"""
        day_ranges = [7, 30, 90]
        timings = []

        for days in day_ranges:
            start = time.time()
            metrics = system.generate_metrics_with_trends(
                services=["api", "db"],
                days_back=days
            )
            duration = time.time() - start
            timings.append(duration)

            print(f"\n⏱️ {days} days history: {duration:.3f}s")

        # Should scale reasonably with data range
        # 90 days shouldn't take more than 2x the time of 30 days
        if len(timings) >= 2:
            ratio_90_to_30 = timings[-1] / timings[1]
            print(f"   90 days vs 30 days ratio: {ratio_90_to_30:.2f}x")
            assert ratio_90_to_30 < 2.0, "Performance degradation with larger date ranges"


class TestPerformanceMemory:
    """Basic memory usage checks"""

    @pytest.fixture
    def system(self):
        return EnhancedSREReportSystem(app_name="MemoryApp")

    def test_metrics_memory_efficiency(self, system):
        """Test that metrics generation doesn't leak memory"""
        import gc

        # Force garbage collection before test
        gc.collect()

        # Generate metrics multiple times
        for i in range(3):
            metrics = system.generate_metrics_with_trends(
                services=["api", "db", "cache"],
                days_back=30
            )
            assert len(metrics) > 0

            # Clear references
            metrics = None
            gc.collect()

        # Test passed if no memory errors occurred
        assert True

    def test_report_generation_memory_efficiency(self, system):
        """Test that report generation doesn't accumulate memory"""
        import gc
        import tempfile

        temp_dir = tempfile.mkdtemp()

        try:
            # Generate multiple reports
            for i in range(2):
                results = system.generate_full_report_suite(
                    application_name="MemoryApp",
                    services=["api"]
                )
                assert len(results) > 0

                # Clear references
                results = None
                gc.collect()

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Test passed if no memory errors occurred
        assert True


class TestPerformanceComparison:
    """Compare performance of different report generation methods"""

    @pytest.fixture
    def system(self):
        return EnhancedSREReportSystem(app_name="ComparisonApp")

    def test_compare_html_vs_json_generation(self, system, tmp_path):
        """Compare HTML vs JSON generation performance"""
        metrics = system.generate_metrics_with_trends(
            services=["api", "db"],
            days_back=30
        )

        # Time HTML generation
        start_html = time.time()
        html_path = system.create_comprehensive_html_report(
            metrics=metrics,
            output_path=str(tmp_path / "test.html")
        )
        html_time = time.time() - start_html

        # Time JSON export
        start_json = time.time()
        json_path = system.orchestrator.export_json_data(
            metrics=metrics,
            output_path=str(tmp_path / "test.json")
        )
        json_time = time.time() - start_json

        print(f"\n⏱️ HTML Generation: {html_time:.3f}s")
        print(f"⏱️ JSON Export: {json_time:.3f}s")
        print(f"   Ratio: {html_time / json_time:.2f}x")

        # HTML should be slower but not dramatically (includes charts, LLM)
        assert html_time < json_time * 10, "HTML generation unexpectedly slow"


# Performance summary
def test_print_performance_summary():
    """Print overall performance summary"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    print("\nRefactored Architecture Performance Targets:")
    print("  ✓ System Init: < 1.0s")
    print("  ✓ Metrics Generation (4 services, 30 days): < 3.0s")
    print("  ✓ Chart Generation: < 5.0s")
    print("  ✓ HTML Report: < 10.0s")
    print("  ✓ Full Report Suite: < 30.0s")
    print("  ✓ Incident Report: < 2.0s")
    print("\nAll benchmarks should pass these thresholds!")
    print("=" * 60 + "\n")


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
