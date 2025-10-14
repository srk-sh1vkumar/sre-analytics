"""
AppDynamics to SLO Metric Mapper

Maps AppDynamics StandardMetric data to SLOMetric format used by the reporting system.
Includes error budget calculation and SLO target mapping.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..reports.llm_analyzer import SLOMetric
from .base import MetricType, StandardMetric


class AppDynamicsSLOMapper:
    """
    Maps AppDynamics metrics to SLO metrics with error budget calculations
    """

    def __init__(self, slo_targets: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the SLO mapper

        Args:
            slo_targets: Dictionary mapping service names to SLO targets
                        Example: {
                            "service-name": {
                                "response_time": 200.0,  # ms
                                "error_rate": 1.0,       # %
                                "availability": 99.9,    # %
                                "cpu_utilization": 80.0  # %
                            }
                        }
        """
        self.logger = logging.getLogger(__name__)
        self.slo_targets = slo_targets or self._get_default_slo_targets()

    def _get_default_slo_targets(self) -> Dict[str, Dict[str, float]]:
        """Get default SLO targets for common metrics"""
        return {
            "_default": {
                "response_time": 200.0,  # 200ms
                "error_rate": 1.0,  # 1% error rate
                "throughput": 100.0,  # 100 requests/min minimum
                "availability": 99.9,  # 99.9% uptime
                "cpu_utilization": 80.0,  # 80% max CPU
                "memory_utilization": 85.0,  # 85% max memory
            }
        }

    def map_to_slo_metrics(
        self, standard_metrics: List[StandardMetric], calculate_trends: bool = True
    ) -> List[SLOMetric]:
        """
        Convert StandardMetrics from AppDynamics to SLOMetrics

        Args:
            standard_metrics: List of StandardMetric objects from AppDynamics
            calculate_trends: Whether to calculate trend data

        Returns:
            List of SLOMetric objects ready for reporting
        """
        # Group metrics by service and metric type
        grouped_metrics = self._group_metrics(standard_metrics)

        slo_metrics = []
        for (service_name, metric_type), metrics in grouped_metrics.items():
            try:
                slo_metric = self._create_slo_metric(
                    service_name, metric_type, metrics, calculate_trends
                )
                if slo_metric:
                    slo_metrics.append(slo_metric)
            except Exception as e:
                self.logger.error(
                    f"Error creating SLO metric for {service_name}:{metric_type}: {e}"
                )

        self.logger.info(
            f"Mapped {len(standard_metrics)} StandardMetrics to " f"{len(slo_metrics)} SLOMetrics"
        )

        return slo_metrics

    def _group_metrics(self, metrics: List[StandardMetric]) -> Dict[tuple, List[StandardMetric]]:
        """Group metrics by service and metric type"""
        grouped = defaultdict(list)

        for metric in metrics:
            key = (metric.service_name, metric.metric_type.value)
            grouped[key].append(metric)

        return grouped

    def _create_slo_metric(
        self,
        service_name: str,
        metric_type: str,
        metrics: List[StandardMetric],
        calculate_trends: bool,
    ) -> Optional[SLOMetric]:
        """Create a single SLO metric from grouped StandardMetrics"""

        if not metrics:
            return None

        # Sort by timestamp
        metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Get most recent value
        current_metric = metrics[-1]
        current_value = current_metric.value

        # Get SLO and SLA targets
        service_targets = self.slo_targets.get(service_name, self.slo_targets.get("_default", {}))
        slo_target = service_targets.get(metric_type, 0.0)

        # SLA target is typically 10% higher than SLO
        sla_target = slo_target * 1.1

        # Calculate error budget consumed
        error_budget_consumed = self._calculate_error_budget(current_value, slo_target, metric_type)

        # Determine compliance status
        status = self._determine_status(error_budget_consumed)

        # Calculate trend data
        trend_data = []
        if calculate_trends and len(metrics) > 1:
            # Get last 10 data points or all if less than 10
            recent_metrics = metrics[-10:]
            trend_data = [m.value for m in recent_metrics]

        # Map unit from StandardMetric
        unit = current_metric.unit or self._get_unit_for_metric_type(metric_type)

        # Generate description
        description = self._generate_description(
            service_name, metric_type, current_value, slo_target, unit
        )

        # Create SLOMetric
        slo_metric = SLOMetric(
            service_name=service_name,
            metric_name=self._get_metric_display_name(metric_type),
            current_value=current_value,
            slo_target=slo_target,
            sla_target=sla_target,
            status=status,
            error_budget_consumed=error_budget_consumed,
            timestamp=current_metric.timestamp,
            unit=unit,
            description=description,
            trend_data=trend_data,
        )

        return slo_metric

    def _calculate_error_budget(
        self, current_value: float, slo_target: float, metric_type: str
    ) -> float:
        """
        Calculate error budget consumed as a percentage

        Error budget represents how much "room" we have before breaching SLO.
        100% means we've fully consumed the error budget (at or beyond target).
        """
        if slo_target == 0:
            return 0.0

        # Different calculation based on metric type
        if metric_type in [
            "response_time",
            "cpu_utilization",
            "memory_utilization",
            "disk_utilization",
        ]:
            # Lower is better - error budget increases as value approaches target
            if current_value <= slo_target:
                # Within SLO - calculate how close we are to the limit
                consumed = (current_value / slo_target) * 100
            else:
                # Exceeded SLO - budget is over 100%
                consumed = 100 + ((current_value - slo_target) / slo_target) * 50
                consumed = min(consumed, 150)  # Cap at 150%

        elif metric_type in ["availability", "throughput"]:
            # Higher is better - error budget increases as value falls below target
            if current_value >= slo_target:
                # Within SLO - minimal budget consumed
                consumed = max(0, (1 - (current_value / slo_target)) * 100)
            else:
                # Below SLO - significant budget consumed
                consumed = ((slo_target - current_value) / slo_target) * 100
                consumed = min(consumed, 100)

        elif metric_type == "error_rate":
            # Special case: error rate (lower is better, target is max acceptable)
            consumed = (current_value / slo_target) * 100
            consumed = min(consumed, 150)  # Cap at 150%

        else:
            # Default: assume lower is better
            consumed = (current_value / slo_target) * 100

        return round(min(max(consumed, 0.0), 150.0), 2)

    def _determine_status(self, error_budget_consumed: float) -> str:
        """
        Determine SLO compliance status

        Returns: "compliant", "at_risk", or "breached"
        """
        if error_budget_consumed <= 70:
            return "compliant"
        elif error_budget_consumed <= 100:
            return "at_risk"
        else:
            return "breached"

    def _get_unit_for_metric_type(self, metric_type: str) -> str:
        """Get display unit for metric type"""
        unit_map = {
            "response_time": "ms",
            "error_rate": "%",
            "throughput": "rpm",
            "availability": "%",
            "cpu_utilization": "%",
            "memory_utilization": "%",
            "disk_utilization": "%",
            "network_io": "MB/s",
            "database_connections": "connections",
        }
        return unit_map.get(metric_type, "")

    def _get_metric_display_name(self, metric_type: str) -> str:
        """Get human-readable metric name"""
        name_map = {
            "response_time": "Response Time",
            "error_rate": "Error Rate",
            "throughput": "Throughput",
            "availability": "Availability",
            "cpu_utilization": "CPU Utilization",
            "memory_utilization": "Memory Utilization",
            "disk_utilization": "Disk Utilization",
            "network_io": "Network I/O",
            "database_connections": "Database Connections",
        }
        return name_map.get(metric_type, metric_type.replace("_", " ").title())

    def _generate_description(
        self,
        service_name: str,
        metric_type: str,
        current_value: float,
        slo_target: float,
        unit: str,
    ) -> str:
        """Generate a descriptive text for the metric"""
        metric_display = self._get_metric_display_name(metric_type)

        return (
            f"{metric_display} for {service_name}: "
            f"Current {current_value:.2f}{unit} "
            f"(Target: {slo_target:.2f}{unit})"
        )

    def calculate_service_health_score(
        self, slo_metrics: List[SLOMetric], service_name: str
    ) -> Dict[str, Any]:
        """
        Calculate overall health score for a service

        Args:
            slo_metrics: List of SLO metrics for the service
            service_name: Name of the service

        Returns:
            Dictionary with health score and details
        """
        service_metrics = [m for m in slo_metrics if m.service_name == service_name]

        if not service_metrics:
            return {
                "service_name": service_name,
                "health_score": 0.0,
                "status": "unknown",
                "metrics_count": 0,
            }

        # Calculate health score (0-100)
        # Lower error budget consumption = higher health
        total_budget = sum(m.error_budget_consumed for m in service_metrics)
        avg_budget = total_budget / len(service_metrics)
        health_score = max(0, 100 - avg_budget)

        # Count statuses
        status_counts = {
            "compliant": sum(1 for m in service_metrics if m.status == "compliant"),
            "at_risk": sum(1 for m in service_metrics if m.status == "at_risk"),
            "breached": sum(1 for m in service_metrics if m.status == "breached"),
        }

        # Overall status
        if status_counts["breached"] > 0:
            overall_status = "critical"
        elif status_counts["at_risk"] > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "service_name": service_name,
            "health_score": round(health_score, 2),
            "status": overall_status,
            "metrics_count": len(service_metrics),
            "status_breakdown": status_counts,
            "avg_error_budget_consumed": round(avg_budget, 2),
        }
