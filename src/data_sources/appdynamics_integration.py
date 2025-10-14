"""
AppDynamics Full Integration Module

Provides a complete integration between AppDynamics and the SRE Analytics system.
Includes rate limiting, caching, and error budget tracking.
"""

import logging
import time
from collections import deque
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

from ..reports.llm_analyzer import SLOMetric
from .appdynamics_adapter import AppDynamicsAdapter
from .appdynamics_slo_mapper import AppDynamicsSLOMapper
from .base import DataSourceConfig, MetricType, QueryParams, StandardMetric


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.logger = logging.getLogger(__name__)

    def __call__(self, func):
        """Decorator for rate limiting"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            self._wait_if_needed()
            return func(*args, **kwargs)

        return wrapper

    def _wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()

        # Remove old calls outside time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()

        # Check if we've hit the limit
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                self._wait_if_needed()  # Recurse to recheck

        # Record this call
        self.calls.append(now)


class MetricsCache:
    """Simple time-based cache for metrics"""

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache

        Args:
            ttl_seconds: Time to live for cached items in seconds
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.logger.debug(f"Cache hit for key: {key}")
                return data
            else:
                # Expired
                del self.cache[key]
                self.logger.debug(f"Cache expired for key: {key}")

        return None

    def set(self, key: str, data: Any):
        """Store item in cache"""
        self.cache[key] = (data, time.time())
        self.logger.debug(f"Cached data for key: {key}")

    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")


class AppDynamicsIntegration:
    """
    Complete AppDynamics integration with SLO tracking
    """

    def __init__(
        self,
        config: DataSourceConfig,
        slo_targets: Optional[Dict[str, Dict[str, float]]] = None,
        rate_limit_calls: int = 100,
        rate_limit_window: int = 60,
        cache_ttl: int = 300,
    ):
        """
        Initialize AppDynamics integration

        Args:
            config: AppDynamics data source configuration
            slo_targets: SLO targets for services
            rate_limit_calls: Max API calls per window
            rate_limit_window: Rate limit window in seconds
            cache_ttl: Cache TTL in seconds
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.adapter = AppDynamicsAdapter(config)
        self.mapper = AppDynamicsSLOMapper(slo_targets)
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_window)
        self.cache = MetricsCache(ttl_seconds=cache_ttl)

        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "errors": 0,
            "last_query_time": None,
        }

        self.logger.info(f"AppDynamics integration initialized for {config.name}")

    def connect(self) -> bool:
        """Establish connection to AppDynamics"""
        try:
            connected = self.adapter.connect()
            if connected:
                self.logger.info("Successfully connected to AppDynamics")
            else:
                self.logger.error("Failed to connect to AppDynamics")
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
        use_cache: bool = True,
    ) -> List[SLOMetric]:
        """
        Get SLO metrics for specified services

        Args:
            services: List of service/application names
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
            # Query AppDynamics
            standard_metrics = self._query_appdynamics_with_rate_limit(
                services, start_time, end_time, metric_types
            )

            # Map to SLO metrics
            slo_metrics = self.mapper.map_to_slo_metrics(standard_metrics, calculate_trends=True)

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

    def _query_appdynamics_with_rate_limit(
        self,
        services: List[str],
        start_time: datetime,
        end_time: datetime,
        metric_types: Optional[List[MetricType]],
    ) -> List[StandardMetric]:
        """Query AppDynamics with rate limiting"""

        # Apply rate limiting
        self._wait_if_needed = self.rate_limiter._wait_if_needed
        self._wait_if_needed()

        # Create query params
        params = QueryParams(
            start_time=start_time, end_time=end_time, services=services, metric_types=metric_types
        )

        # Execute query
        self.stats["api_calls"] += 1
        standard_metrics = self.adapter.query_metrics(params)

        return standard_metrics

    def get_service_health_report(self, service_name: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive health report for a service

        Args:
            service_name: Name of the service
            hours_back: How many hours of data to analyze

        Returns:
            Dictionary with health metrics and insights
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        # Get SLO metrics
        slo_metrics = self.get_slo_metrics_for_services(
            services=[service_name], start_time=start_time, end_time=end_time
        )

        # Calculate health score
        health_score = self.mapper.calculate_service_health_score(slo_metrics, service_name)

        # Organize metrics by type
        metrics_by_type = {}
        for metric in slo_metrics:
            metrics_by_type[metric.metric_name] = {
                "current_value": metric.current_value,
                "slo_target": metric.slo_target,
                "status": metric.status,
                "error_budget_consumed": metric.error_budget_consumed,
                "unit": metric.unit,
                "trend": metric.trend_data[-5:] if metric.trend_data else [],
            }

        # Generate insights
        insights = self._generate_insights(slo_metrics)

        return {
            "service_name": service_name,
            "report_time": end_time.isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours_back,
            },
            "health_score": health_score,
            "metrics": metrics_by_type,
            "insights": insights,
            "total_metrics": len(slo_metrics),
        }

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
                    insights.append(f"📈 {metric.metric_name} trending upward")
                elif self._is_trending_down(metric.trend_data):
                    insights.append(f"📉 {metric.metric_name} trending downward")

        if not insights:
            insights.append("✅ All metrics within acceptable ranges")

        return insights

    def _is_trending_up(self, trend_data: List[float]) -> bool:
        """Check if trend is going up"""
        if len(trend_data) < 3:
            return False

        recent = trend_data[-3:]
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))

    def _is_trending_down(self, trend_data: List[float]) -> bool:
        """Check if trend is going down"""
        if len(trend_data) < 3:
            return False

        recent = trend_data[-3:]
        return all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

    def _generate_cache_key(
        self,
        services: List[str],
        start_time: datetime,
        end_time: datetime,
        metric_types: Optional[List[MetricType]],
    ) -> str:
        """Generate cache key for query"""
        services_str = ",".join(sorted(services))
        start_str = start_time.strftime("%Y%m%d%H%M")
        end_str = end_time.strftime("%Y%m%d%H%M")

        types_str = ""
        if metric_types:
            types_str = ",".join(sorted([mt.value for mt in metric_types]))

        return f"appdynamics:{services_str}:{start_str}:{end_str}:{types_str}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache.cache),
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
            * 100,
        }

    def clear_cache(self):
        """Clear metrics cache"""
        self.cache.clear()
        self.logger.info("Cache cleared manually")
