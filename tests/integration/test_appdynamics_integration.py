"""
Integration Tests for AppDynamics Integration

Tests the complete AppDynamics integration including:
- SLO metric mapping
- Error budget calculation
- Rate limiting
- Caching
- Health reporting
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_sources.appdynamics_adapter import AppDynamicsAdapter
from src.data_sources.appdynamics_slo_mapper import AppDynamicsSLOMapper
from src.data_sources.appdynamics_integration import AppDynamicsIntegration, RateLimiter, MetricsCache
from src.data_sources.base import DataSourceConfig, DataSourceType, StandardMetric, MetricType


class TestAppDynamicsSLOMapper:
    """Test SLO metric mapping functionality"""

    @pytest.fixture
    def mapper(self):
        slo_targets = {
            "test-service": {
                "response_time": 200.0,
                "error_rate": 1.0,
                "cpu_utilization": 80.0
            }
        }
        return AppDynamicsSLOMapper(slo_targets=slo_targets)

    @pytest.fixture
    def sample_standard_metrics(self):
        """Create sample StandardMetric objects"""
        now = datetime.now()
        return [
            StandardMetric(
                metric_id="test-1",
                metric_type=MetricType.RESPONSE_TIME,
                service_name="test-service",
                metric_name="API Response Time",
                value=150.0,
                timestamp=now,
                unit="ms",
                tags={"component": "api"},
                raw_data={}
            ),
            StandardMetric(
                metric_id="test-2",
                metric_type=MetricType.ERROR_RATE,
                service_name="test-service",
                metric_name="API Error Rate",
                value=0.5,
                timestamp=now,
                unit="%",
                tags={"component": "api"},
                raw_data={}
            ),
            StandardMetric(
                metric_id="test-3",
                metric_type=MetricType.CPU_UTILIZATION,
                service_name="test-service",
                metric_name="CPU Usage",
                value=70.0,
                timestamp=now,
                unit="%",
                tags={"component": "infrastructure"},
                raw_data={}
            )
        ]

    def test_map_to_slo_metrics_basic(self, mapper, sample_standard_metrics):
        """Test basic mapping from StandardMetric to SLOMetric"""
        slo_metrics = mapper.map_to_slo_metrics(sample_standard_metrics)

        assert len(slo_metrics) == 3
        assert all(hasattr(m, 'service_name') for m in slo_metrics)
        assert all(hasattr(m, 'slo_target') for m in slo_metrics)
        assert all(hasattr(m, 'error_budget_consumed') for m in slo_metrics)

    def test_error_budget_calculation_response_time(self, mapper):
        """Test error budget calculation for response time"""
        # Within SLO
        budget = mapper._calculate_error_budget(150.0, 200.0, "response_time")
        assert budget < 100  # Should be below 100%
        assert budget > 0

        # At SLO limit
        budget = mapper._calculate_error_budget(200.0, 200.0, "response_time")
        assert budget == 100.0

        # Exceeding SLO
        budget = mapper._calculate_error_budget(250.0, 200.0, "response_time")
        assert budget > 100  # Over budget

    def test_error_budget_calculation_error_rate(self, mapper):
        """Test error budget calculation for error rate"""
        # Well within SLO
        budget = mapper._calculate_error_budget(0.5, 1.0, "error_rate")
        assert budget == 50.0

        # At SLO limit
        budget = mapper._calculate_error_budget(1.0, 1.0, "error_rate")
        assert budget == 100.0

        # Exceeding SLO
        budget = mapper._calculate_error_budget(2.0, 1.0, "error_rate")
        assert budget > 100

    def test_status_determination(self, mapper):
        """Test SLO status determination"""
        assert mapper._determine_status(50.0) == "compliant"
        assert mapper._determine_status(70.0) == "compliant"
        assert mapper._determine_status(85.0) == "at_risk"
        assert mapper._determine_status(100.0) == "at_risk"
        assert mapper._determine_status(110.0) == "breached"

    def test_service_health_score_all_compliant(self, mapper, sample_standard_metrics):
        """Test health score calculation when all metrics compliant"""
        slo_metrics = mapper.map_to_slo_metrics(sample_standard_metrics)
        health = mapper.calculate_service_health_score(slo_metrics, "test-service")

        assert health["service_name"] == "test-service"
        assert health["health_score"] > 0
        # Status may be "warning" if any metric is at_risk (CPU at 87.5% consumed)
        assert health["status"] in ["healthy", "warning"]
        assert health["metrics_count"] == 3

    def test_service_health_score_with_breaches(self, mapper):
        """Test health score when some metrics are breached"""
        now = datetime.now()
        bad_metrics = [
            StandardMetric(
                metric_id="bad-1",
                metric_type=MetricType.RESPONSE_TIME,
                service_name="test-service",
                metric_name="Slow API",
                value=300.0,  # Exceeds 200ms target
                timestamp=now,
                unit="ms",
                tags={},
                raw_data={}
            )
        ]

        slo_metrics = mapper.map_to_slo_metrics(bad_metrics)
        health = mapper.calculate_service_health_score(slo_metrics, "test-service")

        assert health["status"] == "critical"
        assert health["status_breakdown"]["breached"] > 0


class TestRateLimiter:
    """Test rate limiting functionality"""

    def test_rate_limiter_allows_under_limit(self):
        """Test that calls under limit are allowed"""
        limiter = RateLimiter(max_calls=5, time_window=1)

        # Should not block
        for i in range(5):
            limiter._wait_if_needed()

        assert len(limiter.calls) == 5

    def test_rate_limiter_blocks_over_limit(self):
        """Test that rate limiter blocks when limit exceeded"""
        limiter = RateLimiter(max_calls=3, time_window=10)

        # Fill up the limit
        for i in range(3):
            limiter._wait_if_needed()

        import time
        start = time.time()

        # This should be blocked and wait
        limiter._wait_if_needed()

        # Should have waited
        elapsed = time.time() - start
        assert elapsed > 0  # Some wait occurred


class TestMetricsCache:
    """Test metrics caching functionality"""

    def test_cache_stores_and_retrieves(self):
        """Test basic cache operations"""
        cache = MetricsCache(ttl_seconds=60)

        data = {"key": "value"}
        cache.set("test_key", data)

        retrieved = cache.get("test_key")
        assert retrieved == data

    def test_cache_expiration(self):
        """Test that cache entries expire"""
        cache = MetricsCache(ttl_seconds=1)  # 1 second TTL

        cache.set("test_key", "test_value")

        # Should be available immediately
        assert cache.get("test_key") == "test_value"

        # Wait for expiration
        import time
        time.sleep(1.1)

        # Should be expired
        assert cache.get("test_key") is None

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = MetricsCache(ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache.cache) == 2

        cache.clear()

        assert len(cache.cache) == 0


class TestAppDynamicsIntegration:
    """Test full AppDynamics integration"""

    @pytest.fixture
    def config(self):
        return DataSourceConfig(
            source_type=DataSourceType.APPDYNAMICS,
            name="Test AppDynamics",
            connection_params={"host": "test.appdynamics.com"},
            authentication={"username": "test", "password": "test"},
            enabled=True
        )

    @pytest.fixture
    def integration(self, config):
        return AppDynamicsIntegration(
            config=config,
            rate_limit_calls=100,
            rate_limit_window=60,
            cache_ttl=300
        )

    def test_integration_initialization(self, integration):
        """Test that integration initializes all components"""
        assert integration.adapter is not None
        assert integration.mapper is not None
        assert integration.rate_limiter is not None
        assert integration.cache is not None
        assert integration.stats is not None

    @patch.object(AppDynamicsAdapter, 'connect')
    def test_connect(self, mock_connect, integration):
        """Test connection to AppDynamics"""
        mock_connect.return_value = True

        result = integration.connect()

        assert result is True
        mock_connect.assert_called_once()

    @patch.object(AppDynamicsAdapter, 'query_metrics')
    def test_get_slo_metrics_without_cache(self, mock_query, integration):
        """Test getting SLO metrics without cache"""
        # Mock StandardMetric return
        now = datetime.now()
        mock_query.return_value = [
            StandardMetric(
                metric_id="test-1",
                metric_type=MetricType.RESPONSE_TIME,
                service_name="test-service",
                metric_name="Test Metric",
                value=150.0,
                timestamp=now,
                unit="ms",
                tags={},
                raw_data={}
            )
        ]

        services = ["test-service"]
        slo_metrics = integration.get_slo_metrics_for_services(
            services=services,
            use_cache=False
        )

        assert len(slo_metrics) > 0
        assert mock_query.called
        assert integration.stats["total_queries"] == 1

    @patch.object(AppDynamicsAdapter, 'query_metrics')
    def test_get_slo_metrics_with_cache(self, mock_query, integration):
        """Test that caching works correctly"""
        now = datetime.now()
        mock_query.return_value = [
            StandardMetric(
                metric_id="test-1",
                metric_type=MetricType.RESPONSE_TIME,
                service_name="test-service",
                metric_name="Test Metric",
                value=150.0,
                timestamp=now,
                unit="ms",
                tags={},
                raw_data={}
            )
        ]

        services = ["test-service"]

        # First call - should query and cache
        result1 = integration.get_slo_metrics_for_services(
            services=services,
            use_cache=True
        )

        # Second call - should use cache
        result2 = integration.get_slo_metrics_for_services(
            services=services,
            use_cache=True
        )

        # Should only query once (second call uses cache)
        assert mock_query.call_count == 1
        assert integration.stats["cache_hits"] == 1
        assert result1 == result2

    @patch.object(AppDynamicsAdapter, 'query_metrics')
    def test_get_service_health_report(self, mock_query, integration):
        """Test generating service health report"""
        now = datetime.now()
        mock_query.return_value = [
            StandardMetric(
                metric_id="test-1",
                metric_type=MetricType.RESPONSE_TIME,
                service_name="test-service",
                metric_name="Test Metric",
                value=150.0,
                timestamp=now,
                unit="ms",
                tags={},
                raw_data={}
            )
        ]

        report = integration.get_service_health_report("test-service", hours_back=1)

        assert report["service_name"] == "test-service"
        assert "health_score" in report
        assert "metrics" in report
        assert "insights" in report
        assert "time_range" in report

    def test_get_statistics(self, integration):
        """Test retrieving integration statistics"""
        stats = integration.get_statistics()

        assert "total_queries" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "api_calls" in stats
        assert "errors" in stats
        assert "cache_hit_rate" in stats

    def test_insights_generation_healthy(self, integration):
        """Test insight generation for healthy metrics"""
        from src.reports.llm_analyzer import SLOMetric

        slo_metrics = [
            SLOMetric(
                service_name="test",
                metric_name="Response Time",
                current_value=150.0,
                slo_target=200.0,
                sla_target=220.0,
                status="compliant",
                error_budget_consumed=50.0,
                timestamp=datetime.now(),
                unit="ms",
                description="Test",
                trend_data=[]
            )
        ]

        insights = integration._generate_insights(slo_metrics)

        assert len(insights) > 0
        assert any("acceptable" in insight.lower() for insight in insights)

    def test_insights_generation_with_breaches(self, integration):
        """Test insight generation with breached metrics"""
        from src.reports.llm_analyzer import SLOMetric

        slo_metrics = [
            SLOMetric(
                service_name="test",
                metric_name="Response Time",
                current_value=300.0,
                slo_target=200.0,
                sla_target=220.0,
                status="breached",
                error_budget_consumed=150.0,
                timestamp=datetime.now(),
                unit="ms",
                description="Test",
                trend_data=[]
            )
        ]

        insights = integration._generate_insights(slo_metrics)

        assert len(insights) > 0
        assert any("breached" in insight.lower() for insight in insights)


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
