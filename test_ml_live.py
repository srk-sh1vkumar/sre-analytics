#!/usr/bin/env python3
"""
Live Test Script for ML Anomaly Detection

Tests ML anomaly detection against real Prometheus data.
Connects to localhost:9090 (ecommerce-microservices Prometheus).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_sources.prometheus_integration import PrometheusIntegration
from src.data_sources.base import DataSourceConfig, DataSourceType
from src.ml.slo_anomaly_monitor import SLOAnomalyMonitor
from src.ml.anomaly_detector import AnomalyMethod


def main():
    print("=" * 70)
    print("ML ANOMALY DETECTION - LIVE TEST WITH PROMETHEUS")
    print("=" * 70)
    print()

    # Create Prometheus configuration
    config = DataSourceConfig(
        source_type=DataSourceType.PROMETHEUS,
        name="Local Prometheus",
        connection_params={"url": "http://localhost:9090"},
        enabled=True
    )

    # SLO targets for ecommerce services
    slo_targets = {
        "_default": {
            "response_time": 200.0,
            "error_rate": 0.01,
            "availability": 0.999,
            "cpu_utilization": 80.0,
            "memory_utilization": 85.0
        },
        "product-service": {
            "response_time": 150.0,
            "error_rate": 0.005,
            "availability": 0.995
        },
        "order-service": {
            "response_time": 300.0,
            "error_rate": 0.005,
            "availability": 0.9995
        }
    }

    print("📊 Initializing Prometheus Integration...")
    prometheus = PrometheusIntegration(
        config=config,
        slo_targets=slo_targets
    )

    # Test 1: Connect to Prometheus
    print("\n🔌 Test 1: Connecting to Prometheus...")
    connected = prometheus.connect()
    if not connected:
        print("   ❌ Failed to connect to Prometheus at http://localhost:9090")
        print("   💡 Make sure Prometheus is running: docker ps | grep prometheus")
        return 1

    print("   ✅ Connected to Prometheus successfully")

    # Test 2: Get available services
    print("\n📋 Test 2: Discovering Services...")
    services = prometheus.get_available_services()
    if not services:
        print("   ⚠️  No services found in Prometheus")
        print("   💡 Make sure ecommerce-microservices are running and scraped by Prometheus")
        return 1

    print(f"   ✅ Found {len(services)} services: {', '.join(services[:5])}")
    if len(services) > 5:
        print(f"      ... and {len(services) - 5} more")

    # Test 3: Fetch SLO metrics with trend data
    print("\n📈 Test 3: Fetching SLO Metrics with Trend Data...")
    test_services = services[:3]  # Test with first 3 services
    print(f"   Fetching metrics for: {', '.join(test_services)}")

    try:
        # Fetch last 4 hours of data for trend analysis
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        slo_metrics = prometheus.get_slo_metrics_for_services(
            services=test_services,
            start_time=start_time,
            end_time=end_time,
            use_cache=False
        )

        if not slo_metrics:
            print("   ⚠️  No SLO metrics retrieved")
            print("   💡 Services may not have matching metrics in Prometheus")
            return 1

        print(f"   ✅ Retrieved {len(slo_metrics)} SLO metrics with trend data")

        # Show sample metric
        sample_metric = slo_metrics[0]
        print(f"\n   📊 Sample Metric:")
        print(f"      Service: {sample_metric.service_name}")
        print(f"      Metric: {sample_metric.metric_name}")
        print(f"      Current: {sample_metric.current_value:.2f}{sample_metric.unit}")
        print(f"      Target: {sample_metric.slo_target:.2f}{sample_metric.unit}")
        print(f"      Status: {sample_metric.status}")
        print(f"      Trend Data Points: {len(sample_metric.trend_data) if sample_metric.trend_data else 0}")

    except Exception as e:
        print(f"   ❌ Error fetching metrics: {e}")
        return 1

    # Test 4: Initialize ML Anomaly Monitor
    print("\n🤖 Test 4: Initializing ML Anomaly Monitor...")
    monitor = SLOAnomalyMonitor(
        enable_predictions=True,
        alert_lead_time_minutes=30
    )
    print("   ✅ ML Anomaly Monitor initialized")
    print(f"      Detection Method: Modified Z-Score (MAD-based)")
    print(f"      Prediction: Enabled (30-minute lead time)")
    print(f"      Sensitivity: 2.5σ threshold")

    # Test 5: Run Anomaly Detection
    print("\n🔍 Test 5: Running Anomaly Detection on SLO Metrics...")
    print(f"   Analyzing {len(slo_metrics)} metrics for anomalies...")

    anomaly_reports = monitor.analyze_slo_metrics(
        slo_metrics=slo_metrics,
        detection_method=AnomalyMethod.MODIFIED_Z_SCORE
    )

    print(f"   ✅ Analysis complete: {len(anomaly_reports)} anomaly reports generated")

    # Test 6: Display Anomaly Reports
    print("\n📋 Test 6: Anomaly Detection Results...")

    if anomaly_reports:
        print(f"   Found {len(anomaly_reports)} services with anomalies or predictions:\n")

        for report in anomaly_reports[:5]:  # Show first 5 reports
            print(f"   🔹 {report.service_name} - {report.metric_name}")
            print(f"      Health: {report.baseline_health.upper()}")
            print(f"      Anomalies Detected: {len(report.anomalies)}")

            # Show critical anomalies
            critical_anomalies = [a for a in report.anomalies if a.severity.value == "critical"]
            if critical_anomalies:
                print(f"      ⚠️  Critical Anomalies: {len(critical_anomalies)}")
                for anomaly in critical_anomalies[:2]:
                    print(f"         • {anomaly.description}")
                    print(f"           Value: {anomaly.value:.2f}, Expected: {anomaly.expected_value:.2f}")
                    print(f"           Confidence: {anomaly.confidence*100:.0f}%")

            # Show predictions
            if report.prediction and report.prediction.get("will_breach"):
                print(f"      🔮 SLO Breach Prediction:")
                print(f"         Will breach: Yes (confidence: {report.prediction.get('confidence', 0)*100:.0f}%)")
                forecast_time = report.prediction.get("forecast_time")
                if forecast_time:
                    minutes_until = (forecast_time - datetime.now()).total_seconds() / 60
                    print(f"         Estimated time: ~{minutes_until:.0f} minutes")

            # Show top recommendations
            if report.recommendations:
                print(f"      💡 Top Recommendations:")
                for rec in report.recommendations[:2]:
                    print(f"         • {rec}")

            print()

        if len(anomaly_reports) > 5:
            print(f"   ... and {len(anomaly_reports) - 5} more reports")

    else:
        print("   ✅ No anomalies or predictions detected")
        print("   💡 All services are operating within normal parameters")

    # Test 7: Summary Report
    print("\n📊 Test 7: Generating Summary Report...")
    summary = monitor.get_summary_report(anomaly_reports)

    print("   ✅ Summary Report:")
    print(f"      Metrics Analyzed: {summary['total_metrics_analyzed']}")
    print(f"      Total Anomalies: {summary['total_anomalies']}")
    print(f"      Critical Anomalies: {summary['critical_anomalies']}")
    print(f"      Warning Anomalies: {summary['warning_anomalies']}")
    print(f"      Predicted Breaches: {summary['predicted_breaches']}")
    print(f"      At-Risk Services: {len(summary['at_risk_services'])}")
    if summary['at_risk_services']:
        print(f"         {', '.join(summary['at_risk_services'])}")
    print(f"      Overall Health: {summary['overall_health'].upper()}")

    # Show critical recommendations
    if summary['critical_recommendations']:
        print(f"\n      🚨 Critical Recommendations:")
        for rec in summary['critical_recommendations'][:5]:
            print(f"         • {rec}")

    # Test 8: Test Different Detection Methods
    print("\n🔬 Test 8: Testing Multiple Detection Methods...")

    if slo_metrics:
        # Test with a single metric using different methods
        test_metric = slo_metrics[0]
        methods = [
            ("Z-Score", AnomalyMethod.Z_SCORE),
            ("Modified Z-Score", AnomalyMethod.MODIFIED_Z_SCORE),
            ("IQR", AnomalyMethod.IQR),
            ("Moving Average", AnomalyMethod.MOVING_AVERAGE)
        ]

        print(f"   Testing metric: {test_metric.service_name}.{test_metric.metric_name}")
        print(f"   Methods:")

        for method_name, method in methods:
            reports = monitor.analyze_slo_metrics(
                [test_metric],
                detection_method=method
            )
            anomaly_count = sum(len(r.anomalies) for r in reports)
            print(f"      • {method_name}: {anomaly_count} anomalies detected")

    # Summary
    print("\n" + "=" * 70)
    print("✅ ML ANOMALY DETECTION TEST COMPLETE")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Review anomaly reports and validate against known issues")
    print("   2. Adjust sensitivity thresholds if needed")
    print("   3. Integrate with alerting system for proactive monitoring")
    print("   4. Use predictions to prevent SLO breaches")
    print("   5. Enable Prophet/LSTM for seasonal pattern detection")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
