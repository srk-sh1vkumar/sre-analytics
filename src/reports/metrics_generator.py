"""
Metrics Generator Module

Handles generation of SLO metrics with trend data and compliance status.
"""

import logging
import random
from datetime import datetime
from typing import List

import numpy as np

from src.config.constants import (
    AVAILABILITY_MAX,
    COMPLIANCE_THRESHOLD_COMPLIANT,
    DEFAULT_TREND_DAYS,
    ERROR_RATE_CRITICAL,
    ERROR_RATE_TARGET,
    LATENCY_P95_CRITICAL_MS,
    LATENCY_P95_TARGET_MS,
    METRIC_AVAILABILITY,
    METRIC_ERROR_RATE,
    METRIC_LATENCY_P95,
    STATUS_AT_RISK,
    STATUS_BREACHED,
    STATUS_COMPLIANT,
    UNIT_MILLISECONDS,
    UNIT_PERCENTAGE,
)
from src.reports.llm_analyzer import SLOMetric


class MetricsGenerator:
    """Generates realistic SLO metrics with trends and compliance status"""

    def __init__(self):
        """Initialize metrics generator"""
        self.logger = logging.getLogger(__name__)

    def generate_metrics_with_trends(
        self, services: List[str], days_back: int = DEFAULT_TREND_DAYS
    ) -> List[SLOMetric]:
        """
        Generate comprehensive SLO metrics with trend data

        Args:
            services: List of service names
            days_back: Number of days for trend data

        Returns:
            List[SLOMetric]: Generated metrics with trends
        """
        metrics = []
        current_time = datetime.now()

        for service_name in services:
            trend_days = days_back

            # Availability metric
            availability_trend = self._generate_trend_data(AVAILABILITY_MAX, 0.05, trend_days)
            current_availability = availability_trend[-1]

            availability_metric = SLOMetric(
                service_name=service_name,
                metric_name=METRIC_AVAILABILITY,
                current_value=current_availability,
                slo_target=AVAILABILITY_MAX,
                sla_target=AVAILABILITY_MAX,
                status=self._get_compliance_status(current_availability, AVAILABILITY_MAX),
                error_budget_consumed=max(0, (AVAILABILITY_MAX - current_availability) / 0.1 * 100),
                timestamp=current_time,
                unit=UNIT_PERCENTAGE,
                description=f"Service availability for {service_name}",
                trend_data=availability_trend,
            )
            metrics.append(availability_metric)

            # Latency metric
            latency_trend = self._generate_trend_data(
                LATENCY_P95_TARGET_MS, 30, trend_days, min_val=50
            )
            current_latency = latency_trend[-1]

            latency_metric = SLOMetric(
                service_name=service_name,
                metric_name=METRIC_LATENCY_P95,
                current_value=current_latency,
                slo_target=LATENCY_P95_TARGET_MS,
                sla_target=LATENCY_P95_CRITICAL_MS,
                status=self._get_compliance_status(
                    current_latency, LATENCY_P95_TARGET_MS, inverse=True
                ),
                error_budget_consumed=max(
                    0, (current_latency - LATENCY_P95_TARGET_MS) / LATENCY_P95_TARGET_MS * 100
                ),
                timestamp=current_time,
                unit=UNIT_MILLISECONDS,
                description=f"95th percentile response time for {service_name}",
                trend_data=latency_trend,
            )
            metrics.append(latency_metric)

            # Error rate metric
            error_trend = self._generate_trend_data(0.1, 0.03, trend_days, min_val=0)
            current_error_rate = error_trend[-1]

            error_rate_metric = SLOMetric(
                service_name=service_name,
                metric_name=METRIC_ERROR_RATE,
                current_value=current_error_rate,
                slo_target=ERROR_RATE_TARGET,
                sla_target=ERROR_RATE_CRITICAL,
                status=self._get_compliance_status(
                    current_error_rate, ERROR_RATE_TARGET, inverse=True
                ),
                error_budget_consumed=(
                    max(0, (current_error_rate - ERROR_RATE_TARGET) / ERROR_RATE_TARGET * 100)
                    if current_error_rate > ERROR_RATE_TARGET
                    else 0
                ),
                timestamp=current_time,
                unit=UNIT_PERCENTAGE,
                description=f"Error rate for {service_name}",
                trend_data=error_trend,
            )
            metrics.append(error_rate_metric)

        return metrics

    def _generate_trend_data(
        self, mean: float, std: float, days: int, min_val: float = 0
    ) -> List[float]:
        """
        Generate realistic trend data with weekly patterns and random walk

        Args:
            mean: Mean value for trend
            std: Standard deviation for noise
            days: Number of days of trend data
            min_val: Minimum allowed value

        Returns:
            List[float]: Trend data points
        """
        trend = []
        current = mean

        for day in range(days):
            # Add weekly patterns and random walk
            weekly_effect = 0.1 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
            daily_change = np.random.normal(0, std * 0.1)  # Random walk

            current += weekly_effect + daily_change
            current = max(min_val, current)  # Ensure minimum value
            trend.append(current)

        return trend

    def _get_compliance_status(self, current: float, target: float, inverse: bool = False) -> str:
        """
        Determine compliance status based on current vs target value

        Args:
            current: Current metric value
            target: Target/SLO value
            inverse: True if lower is better (e.g., latency, error rate)

        Returns:
            str: Compliance status (compliant/at_risk/breached)
        """
        if inverse:
            # Lower is better (latency, error rate)
            if current <= target:
                return STATUS_COMPLIANT
            elif current <= target * 1.2:
                return STATUS_AT_RISK
            else:
                return STATUS_BREACHED
        else:
            # Higher is better (availability)
            if current >= target:
                return STATUS_COMPLIANT
            elif current >= target * COMPLIANCE_THRESHOLD_COMPLIANT:
                return STATUS_AT_RISK
            else:
                return STATUS_BREACHED

    def calculate_metrics_summary(self, metrics: List[SLOMetric]) -> dict:
        """
        Calculate summary statistics for a set of metrics

        Args:
            metrics: List of SLO metrics

        Returns:
            dict: Summary statistics
        """
        total_metrics = len(metrics)
        compliant_count = sum(1 for m in metrics if m.status == STATUS_COMPLIANT)
        at_risk_count = sum(1 for m in metrics if m.status == STATUS_AT_RISK)
        breached_count = sum(1 for m in metrics if m.status == STATUS_BREACHED)

        # Count unique services
        services = set(m.service_name for m in metrics)

        # Calculate health status
        if breached_count > 0:
            health_status = "Unhealthy"
        elif at_risk_count > total_metrics * 0.3:
            health_status = "Degraded"
        else:
            health_status = "Healthy"

        return {
            "total_metrics": total_metrics,
            "total_services": len(services),
            "compliant_count": compliant_count,
            "at_risk_count": at_risk_count,
            "breached_count": breached_count,
            "compliance_percentage": (
                (compliant_count / total_metrics * 100) if total_metrics > 0 else 0
            ),
            "health_status": health_status,
            "avg_error_budget_consumed": (
                sum(m.error_budget_consumed for m in metrics) / total_metrics
                if total_metrics > 0
                else 0
            ),
        }
