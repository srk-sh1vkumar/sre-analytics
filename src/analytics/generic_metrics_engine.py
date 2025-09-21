"""
Generic Metrics Processing Engine
Processes metrics from multiple data sources and provides unified analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from ..data_sources.base import (
    StandardMetric,
    MetricType,
    DataSourceRegistry,
    MetricAggregator,
    QueryParams
)


@dataclass
class SLOTarget:
    """SLO target configuration"""
    metric_type: MetricType
    target_value: float
    comparison: str  # "less_than", "greater_than", "equals"
    unit: str = ""
    description: str = ""


@dataclass
class SLOResult:
    """SLO evaluation result"""
    service_name: str
    metric_type: MetricType
    current_value: float
    target_value: float
    compliance: bool
    compliance_percentage: float
    error_budget_consumed: float
    trend: str  # "improving", "degrading", "stable"
    status: str  # "compliant", "at_risk", "breached"


@dataclass
class AnalysisResult:
    """Analysis result for a service or system"""
    service_name: str
    overall_health: str  # "healthy", "degraded", "critical"
    slo_results: List[SLOResult]
    recommendations: List[str]
    key_insights: List[str]
    timestamp: datetime


class GenericMetricsEngine:
    """Generic metrics processing engine for multi-source analytics"""

    def __init__(self, registry: DataSourceRegistry):
        self.registry = registry
        self.aggregator = MetricAggregator(registry)
        self.logger = logging.getLogger(__name__)
        self.slo_targets = {}
        self.analysis_cache = {}

    def set_slo_targets(self, service_name: str, targets: List[SLOTarget]):
        """Set SLO targets for a service"""
        self.slo_targets[service_name] = targets

    def collect_and_analyze(self, params: QueryParams) -> Dict[str, AnalysisResult]:
        """Collect metrics from all sources and perform analysis"""
        try:
            # Collect metrics from all enabled data sources
            source_metrics = self.aggregator.collect_metrics(params)

            # Merge metrics from multiple sources
            merged_metrics = self.aggregator.merge_metrics(source_metrics)

            # Group metrics by service
            service_metrics = self._group_metrics_by_service(merged_metrics)

            # Perform analysis for each service
            analysis_results = {}
            for service_name, metrics in service_metrics.items():
                try:
                    analysis = self._analyze_service_metrics(
                        service_name, metrics, params
                    )
                    analysis_results[service_name] = analysis
                except Exception as e:
                    self.logger.error(f"Error analyzing {service_name}: {e}")

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in collect_and_analyze: {e}")
            return {}

    def _group_metrics_by_service(self, metrics: List[StandardMetric]) -> Dict[str, List[StandardMetric]]:
        """Group metrics by service name"""
        service_metrics = {}
        for metric in metrics:
            service_name = metric.service_name
            if service_name not in service_metrics:
                service_metrics[service_name] = []
            service_metrics[service_name].append(metric)
        return service_metrics

    def _analyze_service_metrics(self, service_name: str, metrics: List[StandardMetric],
                                params: QueryParams) -> AnalysisResult:
        """Analyze metrics for a specific service"""
        # Convert metrics to DataFrame for easier analysis
        df = self._metrics_to_dataframe(metrics)

        # Calculate SLO compliance
        slo_results = self._evaluate_slos(service_name, df)

        # Generate insights and recommendations
        insights = self._generate_insights(service_name, df, slo_results)
        recommendations = self._generate_recommendations(service_name, df, slo_results)

        # Determine overall health
        overall_health = self._calculate_overall_health(slo_results)

        return AnalysisResult(
            service_name=service_name,
            overall_health=overall_health,
            slo_results=slo_results,
            recommendations=recommendations,
            key_insights=insights,
            timestamp=datetime.now()
        )

    def _metrics_to_dataframe(self, metrics: List[StandardMetric]) -> pd.DataFrame:
        """Convert StandardMetric list to pandas DataFrame"""
        data = []
        for metric in metrics:
            data.append({
                'service_name': metric.service_name,
                'metric_type': metric.metric_type.value,
                'metric_name': metric.metric_name,
                'value': metric.value,
                'timestamp': metric.timestamp,
                'unit': metric.unit,
                'source': metric.tags.get('source', 'unknown') if metric.tags else 'unknown'
            })

        return pd.DataFrame(data)

    def _evaluate_slos(self, service_name: str, df: pd.DataFrame) -> List[SLOResult]:
        """Evaluate SLO compliance for a service"""
        slo_results = []

        # Get SLO targets for this service
        targets = self.slo_targets.get(service_name, self._get_default_slo_targets())

        for target in targets:
            try:
                # Filter metrics for this metric type
                metric_data = df[df['metric_type'] == target.metric_type.value]

                if metric_data.empty:
                    continue

                # Calculate current value (latest or average)
                if len(metric_data) == 1:
                    current_value = metric_data['value'].iloc[0]
                else:
                    # Use average for aggregation
                    current_value = metric_data['value'].mean()

                # Evaluate compliance
                compliance = self._check_compliance(
                    current_value, target.target_value, target.comparison
                )

                # Calculate compliance percentage
                compliance_percentage = self._calculate_compliance_percentage(
                    current_value, target.target_value, target.comparison
                )

                # Calculate error budget consumption
                error_budget_consumed = max(0, 100 - compliance_percentage)

                # Determine trend
                trend = self._calculate_trend(metric_data)

                # Determine status
                status = self._determine_status(compliance_percentage, trend)

                slo_result = SLOResult(
                    service_name=service_name,
                    metric_type=target.metric_type,
                    current_value=current_value,
                    target_value=target.target_value,
                    compliance=compliance,
                    compliance_percentage=compliance_percentage,
                    error_budget_consumed=error_budget_consumed,
                    trend=trend,
                    status=status
                )

                slo_results.append(slo_result)

            except Exception as e:
                self.logger.error(f"Error evaluating SLO for {target.metric_type}: {e}")

        return slo_results

    def _get_default_slo_targets(self) -> List[SLOTarget]:
        """Get default SLO targets"""
        return [
            SLOTarget(MetricType.RESPONSE_TIME, 200, "less_than", "ms", "Response time should be under 200ms"),
            SLOTarget(MetricType.ERROR_RATE, 1, "less_than", "%", "Error rate should be under 1%"),
            SLOTarget(MetricType.AVAILABILITY, 99.9, "greater_than", "%", "Availability should be above 99.9%"),
            SLOTarget(MetricType.CPU_UTILIZATION, 80, "less_than", "%", "CPU utilization should be under 80%"),
            SLOTarget(MetricType.MEMORY_UTILIZATION, 85, "less_than", "%", "Memory utilization should be under 85%")
        ]

    def _check_compliance(self, current_value: float, target_value: float, comparison: str) -> bool:
        """Check if current value meets SLO target"""
        if comparison == "less_than":
            return current_value < target_value
        elif comparison == "greater_than":
            return current_value > target_value
        elif comparison == "equals":
            return abs(current_value - target_value) < 0.01
        else:
            return False

    def _calculate_compliance_percentage(self, current_value: float, target_value: float, comparison: str) -> float:
        """Calculate compliance percentage"""
        if comparison == "less_than":
            if current_value <= target_value:
                return 100.0
            else:
                # Calculate percentage over target
                excess = (current_value - target_value) / target_value * 100
                return max(0, 100 - excess)
        elif comparison == "greater_than":
            if current_value >= target_value:
                return 100.0
            else:
                # Calculate percentage under target
                shortfall = (target_value - current_value) / target_value * 100
                return max(0, 100 - shortfall)
        else:
            return 100.0 if self._check_compliance(current_value, target_value, comparison) else 0.0

    def _calculate_trend(self, metric_data: pd.DataFrame) -> str:
        """Calculate trend direction for metrics"""
        if len(metric_data) < 2:
            return "stable"

        # Sort by timestamp
        sorted_data = metric_data.sort_values('timestamp')
        values = sorted_data['value'].values

        # Calculate trend using linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Determine trend based on slope
        if abs(slope) < 0.01:  # Threshold for stability
            return "stable"
        elif slope > 0:
            return "improving" if sorted_data['metric_type'].iloc[0] in ['availability', 'throughput'] else "degrading"
        else:
            return "degrading" if sorted_data['metric_type'].iloc[0] in ['availability', 'throughput'] else "improving"

    def _determine_status(self, compliance_percentage: float, trend: str) -> str:
        """Determine SLO status based on compliance and trend"""
        if compliance_percentage >= 95:
            return "compliant"
        elif compliance_percentage >= 85:
            return "at_risk" if trend == "degrading" else "compliant"
        else:
            return "breached"

    def _calculate_overall_health(self, slo_results: List[SLOResult]) -> str:
        """Calculate overall service health"""
        if not slo_results:
            return "unknown"

        # Count status types
        compliant_count = sum(1 for r in slo_results if r.status == "compliant")
        at_risk_count = sum(1 for r in slo_results if r.status == "at_risk")
        breached_count = sum(1 for r in slo_results if r.status == "breached")

        total_count = len(slo_results)
        compliant_ratio = compliant_count / total_count

        if breached_count > 0:
            return "critical"
        elif at_risk_count > 0 or compliant_ratio < 0.8:
            return "degraded"
        else:
            return "healthy"

    def _generate_insights(self, service_name: str, df: pd.DataFrame,
                          slo_results: List[SLOResult]) -> List[str]:
        """Generate key insights from metrics analysis"""
        insights = []

        try:
            # Insight 1: Overall performance summary
            compliant_count = sum(1 for r in slo_results if r.status == "compliant")
            total_count = len(slo_results)
            if total_count > 0:
                compliance_rate = (compliant_count / total_count) * 100
                insights.append(f"Service achieves {compliance_rate:.1f}% SLO compliance across {total_count} metrics")

            # Insight 2: Worst performing metrics
            breached_metrics = [r for r in slo_results if r.status == "breached"]
            if breached_metrics:
                worst_metric = max(breached_metrics, key=lambda x: x.error_budget_consumed)
                insights.append(f"Highest concern: {worst_metric.metric_type.value} "
                               f"({worst_metric.error_budget_consumed:.1f}% error budget consumed)")

            # Insight 3: Trending analysis
            degrading_metrics = [r for r in slo_results if r.trend == "degrading"]
            if degrading_metrics:
                insights.append(f"Warning: {len(degrading_metrics)} metrics showing degrading trends")

            # Insight 4: Data source analysis
            if not df.empty:
                sources = df['source'].unique()
                if len(sources) > 1:
                    insights.append(f"Data collected from {len(sources)} sources: {', '.join(sources)}")

            # Insight 5: Time range analysis
            if not df.empty and 'timestamp' in df.columns:
                time_span = df['timestamp'].max() - df['timestamp'].min()
                insights.append(f"Analysis covers {time_span.total_seconds() / 3600:.1f} hours of data")

        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            insights.append("Error generating detailed insights")

        return insights

    def _generate_recommendations(self, service_name: str, df: pd.DataFrame,
                                 slo_results: List[SLOResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        try:
            # Recommendation 1: Address breached SLOs
            breached_metrics = [r for r in slo_results if r.status == "breached"]
            for metric in breached_metrics:
                if metric.metric_type == MetricType.RESPONSE_TIME:
                    recommendations.append("Optimize response time through caching, database optimization, or load balancing")
                elif metric.metric_type == MetricType.ERROR_RATE:
                    recommendations.append("Investigate error patterns and implement better error handling or circuit breakers")
                elif metric.metric_type == MetricType.CPU_UTILIZATION:
                    recommendations.append("Consider horizontal scaling or CPU optimization for better resource utilization")
                elif metric.metric_type == MetricType.MEMORY_UTILIZATION:
                    recommendations.append("Review memory usage patterns and consider memory optimization or scaling")

            # Recommendation 2: Address at-risk metrics
            at_risk_metrics = [r for r in slo_results if r.status == "at_risk"]
            if at_risk_metrics:
                recommendations.append(f"Monitor {len(at_risk_metrics)} at-risk metrics closely and implement preventive measures")

            # Recommendation 3: Trending issues
            degrading_metrics = [r for r in slo_results if r.trend == "degrading"]
            if degrading_metrics:
                recommendations.append("Set up alerting for degrading metrics to catch issues early")

            # Recommendation 4: Data source improvements
            if not df.empty:
                sources = df['source'].unique()
                if len(sources) == 1:
                    recommendations.append("Consider adding additional monitoring sources for better observability")

            # Recommendation 5: General improvements
            if len(recommendations) == 0:
                recommendations.append("Maintain current performance levels and consider optimizing further")

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Review metrics and investigate any anomalies")

        return recommendations

    def export_analysis_results(self, results: Dict[str, AnalysisResult],
                               format: str = "json") -> str:
        """Export analysis results in specified format"""
        try:
            if format.lower() == "json":
                return self._export_as_json(results)
            elif format.lower() == "csv":
                return self._export_as_csv(results)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return ""

    def _export_as_json(self, results: Dict[str, AnalysisResult]) -> str:
        """Export results as JSON"""
        import json

        export_data = {}
        for service_name, result in results.items():
            export_data[service_name] = {
                "overall_health": result.overall_health,
                "timestamp": result.timestamp.isoformat(),
                "slo_results": [asdict(slo) for slo in result.slo_results],
                "recommendations": result.recommendations,
                "key_insights": result.key_insights
            }

        return json.dumps(export_data, indent=2, default=str)

    def _export_as_csv(self, results: Dict[str, AnalysisResult]) -> str:
        """Export results as CSV"""
        rows = []
        for service_name, result in results.items():
            for slo in result.slo_results:
                rows.append({
                    "service_name": service_name,
                    "overall_health": result.overall_health,
                    "metric_type": slo.metric_type.value,
                    "current_value": slo.current_value,
                    "target_value": slo.target_value,
                    "compliance": slo.compliance,
                    "compliance_percentage": slo.compliance_percentage,
                    "error_budget_consumed": slo.error_budget_consumed,
                    "trend": slo.trend,
                    "status": slo.status,
                    "timestamp": result.timestamp.isoformat()
                })

        if rows:
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        else:
            return "No data to export"