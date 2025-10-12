"""
Prometheus Full Integration Module

Provides a complete integration between Prometheus and the SRE Analytics system.
Includes rate limiting, caching, and error budget tracking.

Reuses infrastructure from AppDynamics integration but optimized for Prometheus patterns.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

# Reuse rate limiter and cache from AppDynamics integration
from .appdynamics_integration import RateLimiter, MetricsCache
from .prometheus_adapter import PrometheusAdapter
from .prometheus_slo_mapper import PrometheusSLOMapper
from .base import DataSourceConfig, QueryParams, StandardMetric, MetricType
from ..reports.llm_analyzer import SLOMetric


class PrometheusIntegration:
    """
    Complete Prometheus integration with SLO tracking

    Similar to AppDynamicsIntegration but optimized for Prometheus query patterns.
    """

    def __init__(
        self,
        config: DataSourceConfig,
        slo_targets: Optional[Dict[str, Dict[str, float]]] = None,
        rate_limit_calls: int = 100,
        rate_limit_window: int = 60,
        cache_ttl: int = 300
    ):
        """
        Initialize Prometheus integration

        Args:
            config: Prometheus data source configuration
            slo_targets: SLO targets for services
            rate_limit_calls: Max API calls per window
            rate_limit_window: Rate limit window in seconds
            cache_ttl: Cache TTL in seconds
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.adapter = PrometheusAdapter(config)
        self.mapper = PrometheusSLOMapper(slo_targets)
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        self.cache = MetricsCache(ttl_seconds=cache_ttl)

        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "errors": 0,
            "last_query_time": None
        }

        self.logger.info(
            f"Prometheus integration initialized for {config.name}"
        )

    def connect(self) -> bool:
        """Establish connection to Prometheus"""
        try:
            connected = self.adapter.connect()
            if connected:
                self.logger.info("Successfully connected to Prometheus")
            else:
                self.logger.error("Failed to connect to Prometheus")
            return connected
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    def get_slo_metrics_for_services(
        self,
        services: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metric_types: Optional[List[MetricType]] = None,
        use_cache: bool = True
    ) -> List[SLOMetric]:
        """
        Get SLO metrics for specified services from Prometheus

        Args:
            services: List of service names (Prometheus job/instance labels)
            start_time: Start of time range (default: 1 hour ago)
            end_time: End of time range (default: now)
            metric_types: Specific metric types to query
            use_cache: Whether to use cached results

        Returns:
            List of SLO metrics ready for reporting
        """
        self.stats["total_queries"] += 1

        # Set default time range
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)

        # Check cache
        cache_key = self._generate_cache_key(services, start_time, end_time, metric_types)
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                self.logger.info(f"Returning cached SLO metrics for {len(services)} services")
                return cached_result

        self.stats["cache_misses"] += 1

        try:
            # Query Prometheus
            standard_metrics = self._query_prometheus_with_rate_limit(
                services, start_time, end_time, metric_types
            )

            # Map to SLO metrics
            slo_metrics = self.mapper.map_to_slo_metrics(
                standard_metrics, calculate_trends=True
            )

            # Cache results
            if use_cache:
                self.cache.set(cache_key, slo_metrics)

            self.stats["last_query_time"] = datetime.now()

            self.logger.info(
                f"Retrieved {len(slo_metrics)} SLO metrics for {len(services)} services"
            )

            return slo_metrics

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Error getting SLO metrics: {e}")
            return []

    def _query_prometheus_with_rate_limit(
        self,
        services: List[str],
        start_time: datetime,
        end_time: datetime,
        metric_types: Optional[List[MetricType]]
    ) -> List[StandardMetric]:
        """Query Prometheus with rate limiting"""

        # Apply rate limiting
        self._wait_if_needed = self.rate_limiter._wait_if_needed
        self._wait_if_needed()

        # Create query params
        params = QueryParams(
            start_time=start_time,
            end_time=end_time,
            services=services,
            metric_types=metric_types
        )

        # Execute query
        self.stats["api_calls"] += 1
        standard_metrics = self.adapter.query_metrics(params)

        return standard_metrics

    def get_service_health_report(
        self, service_name: str, hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get comprehensive health report for a service from Prometheus

        Args:
            service_name: Name of the service (Prometheus job/instance)
            hours_back: How many hours of data to analyze

        Returns:
            Dictionary with health metrics and insights
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        # Get SLO metrics
        slo_metrics = self.get_slo_metrics_for_services(
            services=[service_name],
            start_time=start_time,
            end_time=end_time
        )

        # Calculate health score
        health_score = self.mapper.calculate_service_health_score(
            slo_metrics, service_name
        )

        # Organize metrics by type
        metrics_by_type = {}
        for metric in slo_metrics:
            metrics_by_type[metric.metric_name] = {
                "current_value": metric.current_value,
                "slo_target": metric.slo_target,
                "status": metric.status,
                "error_budget_consumed": metric.error_budget_consumed,
                "unit": metric.unit,
                "trend": metric.trend_data[-5:] if metric.trend_data else []
            }

        # Generate insights
        insights = self._generate_insights(slo_metrics)

        return {
            "service_name": service_name,
            "report_time": end_time.isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours_back
            },
            "health_score": health_score,
            "metrics": metrics_by_type,
            "insights": insights,
            "total_metrics": len(slo_metrics),
            "data_source": "Prometheus"
        }

    def query_promql(
        self, query: str, time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Execute a raw PromQL query

        Useful for custom queries not covered by standard metric types.

        Args:
            query: PromQL query string
            time: Time to evaluate query (default: now)

        Returns:
            Prometheus query result
        """
        try:
            from urllib.parse import urljoin

            time_param = time.timestamp() if time else datetime.now().timestamp()

            url = urljoin(self.adapter.base_url, '/api/v1/query')
            params = {
                'query': query,
                'time': time_param
            }

            # Apply rate limiting
            self._wait_if_needed = self.rate_limiter._wait_if_needed
            self._wait_if_needed()

            response = self.adapter.session.get(url, params=params, timeout=30)
            self.stats["api_calls"] += 1

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"PromQL query failed: {response.status_code}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            self.logger.error(f"Error executing PromQL query: {e}")
            return {"status": "error", "error": str(e)}

    def get_available_services(self) -> List[str]:
        """
        Get list of available services from Prometheus

        Returns:
            List of service names discovered from Prometheus labels
        """
        try:
            return self.adapter.get_available_services()
        except Exception as e:
            self.logger.error(f"Error getting available services: {e}")
            return []

    def _generate_insights(self, slo_metrics: List[SLOMetric]) -> List[str]:
        """Generate human-readable insights from metrics"""
        insights = []

        # Check for breaches
        breached = [m for m in slo_metrics if m.status == "breached"]
        if breached:
            insight = f"⚠️ {len(breached)} metric(s) breached SLO: "
            insight += ", ".join([m.metric_name for m in breached])
            insights.append(insight)

        # Check for at-risk metrics
        at_risk = [m for m in slo_metrics if m.status == "at_risk"]
        if at_risk:
            insight = f"⚡ {len(at_risk)} metric(s) at risk: "
            insight += ", ".join([m.metric_name for m in at_risk])
            insights.append(insight)

        # Check trends
        for metric in slo_metrics:
            if metric.trend_data and len(metric.trend_data) >= 3:
                if self._is_trending_up(metric.trend_data):
                    insights.append(
                        f"📈 {metric.metric_name} trending upward"
                    )
                elif self._is_trending_down(metric.trend_data):
                    insights.append(
                        f"📉 {metric.metric_name} trending downward"
                    )

        if not insights:
            insights.append("✅ All metrics within acceptable ranges")

        return insights

    def _is_trending_up(self, trend_data: List[float]) -> bool:
        """Check if trend is going up"""
        if len(trend_data) < 3:
            return False

        recent = trend_data[-3:]
        return all(recent[i] < recent[i+1] for i in range(len(recent)-1))

    def _is_trending_down(self, trend_data: List[float]) -> bool:
        """Check if trend is going down"""
        if len(trend_data) < 3:
            return False

        recent = trend_data[-3:]
        return all(recent[i] > recent[i+1] for i in range(len(recent)-1))

    def _generate_cache_key(
        self,
        services: List[str],
        start_time: datetime,
        end_time: datetime,
        metric_types: Optional[List[MetricType]]
    ) -> str:
        """Generate cache key for query"""
        services_str = ",".join(sorted(services))
        start_str = start_time.strftime("%Y%m%d%H%M")
        end_str = end_time.strftime("%Y%m%d%H%M")

        types_str = ""
        if metric_types:
            types_str = ",".join(sorted([mt.value for mt in metric_types]))

        return f"prometheus:{services_str}:{start_str}:{end_str}:{types_str}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache.cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] /
                max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            ) * 100
        }

    def clear_cache(self):
        """Clear metrics cache"""
        self.cache.clear()
        self.logger.info("Cache cleared manually")

    def get_prometheus_info(self) -> Dict[str, Any]:
        """Get Prometheus server information"""
        try:
            health_status = self.adapter.get_health_status()

            # Get additional Prometheus-specific info
            build_info = self.query_promql("prometheus_build_info")

            return {
                "connection_status": health_status,
                "build_info": build_info.get("data", {}),
                "base_url": self.adapter.base_url,
                "integration_stats": self.get_statistics()
            }
        except Exception as e:
            self.logger.error(f"Error getting Prometheus info: {e}")
            return {"error": str(e)}
