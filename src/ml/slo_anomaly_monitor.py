"""
SLO Anomaly Monitor

Integrates anomaly detection with SLO monitoring.
Provides proactive alerts before SLO breaches occur.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..reports.llm_analyzer import SLOMetric
from .anomaly_detector import Anomaly, AnomalyDetector, AnomalyMethod, AnomalySeverity


@dataclass
class SLOAnomalyReport:
    """Report of anomalies detected in SLO metrics"""

    service_name: str
    metric_name: str
    anomalies: List[Anomaly]
    baseline_health: str  # healthy, warning, critical
    prediction: Dict[str, Any]  # SLO breach prediction
    recommendations: List[str]
    generated_at: datetime


class SLOAnomalyMonitor:
    """
    Monitor SLO metrics for anomalies and predict breaches

    Combines anomaly detection with SLO tracking for proactive monitoring.
    """

    def __init__(
        self,
        detector: Optional[AnomalyDetector] = None,
        enable_predictions: bool = True,
        alert_lead_time_minutes: int = 30,
    ):
        """
        Initialize SLO anomaly monitor

        Args:
            detector: AnomalyDetector instance (creates default if None)
            enable_predictions: Enable SLO breach prediction
            alert_lead_time_minutes: How far ahead to predict breaches
        """
        self.detector = detector or AnomalyDetector(
            sensitivity=2.5, baseline_window_hours=168, min_samples=30  # 1 week
        )
        self.enable_predictions = enable_predictions
        self.alert_lead_time_minutes = alert_lead_time_minutes
        self.logger = logging.getLogger(__name__)

    def analyze_slo_metrics(
        self,
        slo_metrics: List[SLOMetric],
        detection_method: AnomalyMethod = AnomalyMethod.MODIFIED_Z_SCORE,
    ) -> List[SLOAnomalyReport]:
        """
        Analyze SLO metrics for anomalies

        Args:
            slo_metrics: List of SLO metrics with trend data
            detection_method: Anomaly detection method to use

        Returns:
            List of anomaly reports per metric
        """
        reports = []

        for metric in slo_metrics:
            if not metric.trend_data or len(metric.trend_data) < 10:
                self.logger.debug(
                    f"Skipping {metric.service_name}.{metric.metric_name}: "
                    f"insufficient trend data"
                )
                continue

            try:
                report = self._analyze_single_metric(metric, detection_method)
                if report:
                    reports.append(report)
            except Exception as e:
                self.logger.error(
                    f"Error analyzing {metric.service_name}.{metric.metric_name}: {e}"
                )

        self.logger.info(f"Generated {len(reports)} anomaly reports")
        return reports

    def _analyze_single_metric(
        self, metric: SLOMetric, detection_method: AnomalyMethod
    ) -> Optional[SLOAnomalyReport]:
        """Analyze a single SLO metric for anomalies"""

        # Generate timestamps for trend data
        # Assuming trend data is evenly spaced, ending at metric.timestamp
        timestamps = self._generate_timestamps(
            end_time=metric.timestamp,
            count=len(metric.trend_data),
            interval_minutes=5,  # Assume 5-minute intervals
        )

        # Detect anomalies
        anomalies = self.detector.detect_anomalies(
            values=metric.trend_data,
            timestamps=timestamps,
            metric_name=metric.metric_name,
            service_name=metric.service_name,
            method=detection_method,
        )

        # Determine baseline health
        baseline_health = self._determine_baseline_health(metric, anomalies)

        # Predict SLO breach
        prediction = {}
        if self.enable_predictions:
            prediction = self.detector.predict_slo_breach(
                values=metric.trend_data,
                timestamps=timestamps,
                slo_target=metric.slo_target,
                forecast_minutes=self.alert_lead_time_minutes,
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(metric, anomalies, prediction)

        # Only create report if there are anomalies or predictions
        if anomalies or (prediction and prediction.get("will_breach")):
            return SLOAnomalyReport(
                service_name=metric.service_name,
                metric_name=metric.metric_name,
                anomalies=anomalies,
                baseline_health=baseline_health,
                prediction=prediction,
                recommendations=recommendations,
                generated_at=datetime.now(),
            )

        return None

    def _generate_timestamps(
        self, end_time: datetime, count: int, interval_minutes: int
    ) -> List[datetime]:
        """Generate evenly-spaced timestamps for trend data"""
        timestamps = []
        for i in range(count):
            offset_minutes = (count - 1 - i) * interval_minutes
            ts = end_time - timedelta(minutes=offset_minutes)
            timestamps.append(ts)
        return timestamps

    def _determine_baseline_health(self, metric: SLOMetric, anomalies: List[Anomaly]) -> str:
        """Determine overall health status"""

        # Check SLO compliance first
        if metric.status == "breached":
            return "critical"
        elif metric.status == "at_risk":
            return "warning"

        # Check anomalies
        if not anomalies:
            return "healthy"

        # Count anomalies by severity
        critical_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        warning_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.WARNING)

        if critical_count > 0:
            return "critical"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"

    def _generate_recommendations(
        self, metric: SLOMetric, anomalies: List[Anomaly], prediction: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Anomaly-based recommendations
        if anomalies:
            critical_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
            if critical_anomalies:
                recommendations.append(
                    f"⚠️ {len(critical_anomalies)} critical anomalies detected. "
                    f"Immediate investigation recommended for {metric.service_name}.{metric.metric_name}"
                )

            # Check for recent spike
            recent_anomalies = [
                a
                for a in anomalies
                if (datetime.now() - a.timestamp).total_seconds() < 600  # Last 10 minutes
            ]
            if recent_anomalies:
                recommendations.append(
                    f"📈 Recent anomaly spike detected ({len(recent_anomalies)} in last 10 min). "
                    f"Monitor {metric.service_name} closely."
                )

        # Prediction-based recommendations
        if prediction and prediction.get("will_breach"):
            confidence = prediction.get("confidence", 0)
            forecast_time = prediction.get("forecast_time")
            if forecast_time:
                time_until = (forecast_time - datetime.now()).total_seconds() / 60
                recommendations.append(
                    f"🔮 Predicted SLO breach in ~{time_until:.0f} minutes "
                    f"(confidence: {confidence*100:.0f}%). "
                    f"Consider scaling up {metric.service_name} or investigating root cause."
                )

        # SLO status recommendations
        if metric.status == "at_risk":
            recommendations.append(
                f"⚡ Error budget {metric.error_budget_consumed:.0f}% consumed. "
                f"Approaching SLO limit for {metric.metric_name}."
            )
        elif metric.status == "breached":
            recommendations.append(
                f"🚨 SLO breached! {metric.metric_name} exceeded target. "
                f"Immediate action required for {metric.service_name}."
            )

        # Trend-based recommendations
        if metric.trend_data and len(metric.trend_data) >= 3:
            if self._is_trending_up(metric.trend_data):
                recommendations.append(
                    f"📊 {metric.metric_name} trending upward. "
                    f"Monitor for continued degradation."
                )

        if not recommendations:
            recommendations.append(
                f"✅ {metric.metric_name} operating normally. No action required."
            )

        return recommendations

    def _is_trending_up(self, trend_data: List[float]) -> bool:
        """Check if metric is trending upward"""
        if len(trend_data) < 3:
            return False

        recent = trend_data[-3:]
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))

    def get_summary_report(self, reports: List[SLOAnomalyReport]) -> Dict[str, Any]:
        """
        Generate summary report across all services

        Args:
            reports: List of individual anomaly reports

        Returns:
            Summary statistics and insights
        """
        total_anomalies = sum(len(r.anomalies) for r in reports)

        # Count by severity
        critical_anomalies = sum(
            sum(1 for a in r.anomalies if a.severity == AnomalySeverity.CRITICAL) for r in reports
        )
        warning_anomalies = sum(
            sum(1 for a in r.anomalies if a.severity == AnomalySeverity.WARNING) for r in reports
        )

        # Count predicted breaches
        predicted_breaches = sum(
            1 for r in reports if r.prediction and r.prediction.get("will_breach")
        )

        # Services at risk
        at_risk_services = set(
            r.service_name for r in reports if r.baseline_health in ["warning", "critical"]
        )

        # Top recommendations
        all_recommendations = []
        for report in reports:
            all_recommendations.extend(report.recommendations)

        # Get unique critical recommendations
        critical_recommendations = [
            rec
            for rec in all_recommendations
            if any(keyword in rec for keyword in ["⚠️", "🚨", "🔮"])
        ]

        return {
            "total_metrics_analyzed": len(reports),
            "total_anomalies": total_anomalies,
            "critical_anomalies": critical_anomalies,
            "warning_anomalies": warning_anomalies,
            "predicted_breaches": predicted_breaches,
            "at_risk_services": list(at_risk_services),
            "critical_recommendations": critical_recommendations[:10],  # Top 10
            "overall_health": self._determine_overall_health(reports),
            "generated_at": datetime.now().isoformat(),
        }

    def _determine_overall_health(self, reports: List[SLOAnomalyReport]) -> str:
        """Determine overall platform health"""
        critical_count = sum(1 for r in reports if r.baseline_health == "critical")
        warning_count = sum(1 for r in reports if r.baseline_health == "warning")

        if critical_count > 0:
            return "critical"
        elif warning_count > len(reports) * 0.3:  # >30% warnings
            return "warning"
        else:
            return "healthy"
