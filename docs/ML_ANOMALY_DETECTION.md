# ML-Based Anomaly Detection for SRE Analytics

**Status:** ✅ Complete
**Priority:** 3 (Phase 4: Intelligence & Automation)
**Date:** 2025-10-12

---

## Overview

ML-based anomaly detection provides proactive SLO breach detection by identifying unusual patterns in metrics before they impact service availability. The system uses statistical methods by default (fast, no dependencies) with optional ML models for seasonal patterns.

---

## Features

### Core Capabilities

1. **Multiple Detection Methods**
   - **Z-Score**: Standard deviation-based detection (fast)
   - **Modified Z-Score**: MAD-based, robust to outliers (default)
   - **IQR**: Quartile-based outlier detection
   - **Moving Average**: Deviation from moving average
   - **Prophet**: Optional, Facebook Prophet for seasonal patterns
   - **LSTM**: Optional, deep learning for complex patterns

2. **SLO Breach Prediction**
   - Linear regression-based trend extrapolation
   - Forecast SLO breaches 30-60 minutes ahead
   - Confidence scores for predictions
   - Configurable lead time

3. **Automatic Baseline Learning**
   - Historical data analysis (default: 1 week window)
   - Dynamic threshold calculation
   - Statistical baselines: mean, median, std, IQR
   - Adaptive to metric patterns

4. **Severity Classification**
   - **INFO**: Minor deviation, informational only
   - **WARNING**: Moderate deviation, monitor closely
   - **CRITICAL**: Major deviation, immediate action needed

5. **Actionable Recommendations**
   - Proactive alerts before SLO breaches
   - Trend-based recommendations
   - Error budget warnings
   - Service-specific guidance

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SLO Metrics                              │
│                   (with trend_data from                         │
│                    Prometheus/AppD)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SLOAnomalyMonitor                             │
│  • Integrates with existing SLO framework                       │
│  • Generates timestamps for trend data                          │
│  • Orchestrates detection and prediction                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AnomalyDetector                              │
│  • detect_anomalies(): Statistical/ML detection                 │
│  • predict_slo_breach(): Linear regression prediction           │
│  • _calculate_baseline(): Historical analysis                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SLOAnomalyReport                               │
│  • Anomalies: List[Anomaly]                                     │
│  • Predictions: Dict (will_breach, confidence, time)            │
│  • Recommendations: List[str]                                   │
│  • Baseline Health: healthy/warning/critical                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Basic Usage

```python
from src.ml.slo_anomaly_monitor import SLOAnomalyMonitor
from src.ml.anomaly_detector import AnomalyMethod

# Initialize monitor
monitor = SLOAnomalyMonitor(
    enable_predictions=True,
    alert_lead_time_minutes=30
)

# Analyze SLO metrics (fetched from Prometheus/AppD)
reports = monitor.analyze_slo_metrics(
    slo_metrics=slo_metrics,  # List[SLOMetric]
    detection_method=AnomalyMethod.MODIFIED_Z_SCORE
)

# Review anomaly reports
for report in reports:
    print(f"Service: {report.service_name}")
    print(f"Health: {report.baseline_health}")
    print(f"Anomalies: {len(report.anomalies)}")

    if report.prediction and report.prediction.get("will_breach"):
        print(f"⚠️  Predicted breach in {report.prediction['forecast_time']}")

    for recommendation in report.recommendations:
        print(f"💡 {recommendation}")
```

### Advanced: Custom Detector

```python
from src.ml.anomaly_detector import AnomalyDetector

# Create custom detector with different sensitivity
detector = AnomalyDetector(
    sensitivity=3.0,  # Higher = less sensitive (fewer false positives)
    baseline_window_hours=336,  # 2 weeks of baseline data
    min_samples=50,  # Minimum data points required
    enable_prophet=True  # Enable seasonal decomposition
)

monitor = SLOAnomalyMonitor(detector=detector)
```

### Generate Summary Report

```python
# Analyze multiple services
reports = monitor.analyze_slo_metrics(all_slo_metrics)

# Get aggregated summary
summary = monitor.get_summary_report(reports)

print(f"Total Anomalies: {summary['total_anomalies']}")
print(f"Critical Anomalies: {summary['critical_anomalies']}")
print(f"Predicted Breaches: {summary['predicted_breaches']}")
print(f"Overall Health: {summary['overall_health']}")
print(f"At-Risk Services: {', '.join(summary['at_risk_services'])}")
```

---

## Detection Methods

### Modified Z-Score (Default)

**Best for:** General-purpose anomaly detection, robust to outliers

```python
detection_method=AnomalyMethod.MODIFIED_Z_SCORE
```

Uses Median Absolute Deviation (MAD) instead of standard deviation:
```
Modified Z-Score = 0.6745 * (x - median) / MAD
```

**Advantages:**
- Robust to outliers
- Works well with skewed distributions
- Fast computation

### Z-Score

**Best for:** Normal distributions without outliers

```python
detection_method=AnomalyMethod.Z_SCORE
```

Standard statistical method:
```
Z-Score = (x - mean) / std
```

**Advantages:**
- Well-understood
- Fast
- Good for normal distributions

### IQR (Interquartile Range)

**Best for:** Non-parametric outlier detection

```python
detection_method=AnomalyMethod.IQR
```

Detects values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`

**Advantages:**
- No distribution assumptions
- Resistant to outliers
- Works with small samples

### Moving Average

**Best for:** Time-series with local patterns

```python
detection_method=AnomalyMethod.MOVING_AVERAGE
```

Compares values to moving average of surrounding points.

**Advantages:**
- Captures local trends
- Good for slowly changing baselines
- Adaptive to changes

### Prophet (Optional)

**Best for:** Metrics with seasonal patterns (daily, weekly)

```python
detector = AnomalyDetector(enable_prophet=True)
detection_method=AnomalyMethod.PROPHET
```

**Requires:** `pip install prophet`

**Advantages:**
- Handles seasonality automatically
- Decomposes trend + seasonal + residual
- Confidence intervals
- Best for metrics with predictable patterns (e.g., traffic)

---

## Configuration

### Sensitivity Tuning

```python
# Less sensitive - fewer false positives, might miss subtle anomalies
detector = AnomalyDetector(sensitivity=3.5)

# Default - balanced
detector = AnomalyDetector(sensitivity=2.5)

# More sensitive - catches more anomalies, more false positives
detector = AnomalyDetector(sensitivity=1.5)
```

### Baseline Window

```python
# Short baseline - responsive to recent changes (1 day)
detector = AnomalyDetector(baseline_window_hours=24)

# Default - 1 week baseline
detector = AnomalyDetector(baseline_window_hours=168)

# Long baseline - stable, less responsive (1 month)
detector = AnomalyDetector(baseline_window_hours=720)
```

### Prediction Lead Time

```python
# Short lead time - 15 minutes ahead
monitor = SLOAnomalyMonitor(alert_lead_time_minutes=15)

# Default - 30 minutes ahead
monitor = SLOAnomalyMonitor(alert_lead_time_minutes=30)

# Long lead time - 1 hour ahead
monitor = SLOAnomalyMonitor(alert_lead_time_minutes=60)
```

---

## Integration with Data Sources

### With Prometheus

```python
from src.data_sources.prometheus_integration import PrometheusIntegration
from src.ml.slo_anomaly_monitor import SLOAnomalyMonitor

# Fetch SLO metrics with trend data
prometheus = PrometheusIntegration(config, slo_targets)
prometheus.connect()

slo_metrics = prometheus.get_slo_metrics_for_services(
    services=["product-service", "order-service"],
    start_time=datetime.now() - timedelta(hours=4),  # 4 hours for trend
    end_time=datetime.now()
)

# Run anomaly detection
monitor = SLOAnomalyMonitor()
reports = monitor.analyze_slo_metrics(slo_metrics)
```

### With AppDynamics

```python
from src.data_sources.appdynamics_integration import AppDynamicsIntegration

# Same pattern as Prometheus
appdynamics = AppDynamicsIntegration(config, slo_targets)
slo_metrics = appdynamics.get_slo_metrics_for_services(...)
reports = monitor.analyze_slo_metrics(slo_metrics)
```

---

## Testing

### Unit Tests

```bash
# Run all ML tests (60 tests)
pytest tests/ml/ -v

# Test specific module
pytest tests/ml/test_anomaly_detector.py -v
pytest tests/ml/test_slo_anomaly_monitor.py -v
```

### Live Testing with Prometheus

```bash
# Test with real Prometheus data
python3 test_ml_live.py
```

**Prerequisites:**
- Prometheus running on localhost:9090
- Services scraped and available in Prometheus
- At least 1 hour of historical data for trends

---

## Output Examples

### Anomaly Report Example

```
🔹 product-service - response_time
   Health: WARNING
   Anomalies Detected: 3
   ⚠️  Critical Anomalies: 1
      • Modified Z-score 4.56 exceeds threshold 3.75
        Value: 285.32ms, Expected: 145.20ms
        Confidence: 95%
   🔮 SLO Breach Prediction:
      Will breach: Yes (confidence: 78%)
      Estimated time: ~23 minutes
   💡 Top Recommendations:
      • ⚠️ 1 critical anomalies detected. Immediate investigation recommended
      • 🔮 Predicted SLO breach in ~23 minutes. Consider scaling up
```

### Summary Report Example

```
📊 Summary Report:
   Metrics Analyzed: 12
   Total Anomalies: 8
   Critical Anomalies: 2
   Warning Anomalies: 6
   Predicted Breaches: 1
   At-Risk Services: 2 (product-service, cart-service)
   Overall Health: WARNING

   🚨 Critical Recommendations:
      • ⚠️ 2 critical anomalies detected in product-service
      • 🔮 Predicted SLO breach in ~23 minutes for response_time
      • ⚡ Error budget 87% consumed for cart-service
```

---

## Performance Characteristics

### Computational Complexity

| Method | Time Complexity | Space Complexity | Speed |
|--------|----------------|------------------|-------|
| Z-Score | O(n) | O(1) | Very Fast |
| Modified Z-Score | O(n log n) | O(n) | Fast |
| IQR | O(n log n) | O(n) | Fast |
| Moving Average | O(n*w) | O(w) | Medium |
| Prophet | O(n²) | O(n) | Slow |

**Recommendation:** Use Modified Z-Score (default) for production. Enable Prophet only for metrics with known seasonal patterns.

### Typical Performance

- **Statistical methods**: 50-100 metrics/second
- **Prophet**: 5-10 metrics/second
- **Memory usage**: ~50MB baseline + 1MB per 10k data points

---

## Best Practices

### 1. Choose Appropriate Detection Method

```python
# For most metrics (default)
method = AnomalyMethod.MODIFIED_Z_SCORE

# For traffic metrics with daily/weekly patterns
method = AnomalyMethod.PROPHET

# For slowly changing resource metrics
method = AnomalyMethod.MOVING_AVERAGE
```

### 2. Ensure Sufficient Historical Data

- Minimum: 30 data points
- Recommended: 100+ data points (4+ hours at 5-min intervals)
- Baseline window: 1 week minimum

### 3. Tune Sensitivity Based on Metric Type

```python
# Low-variance metrics (e.g., error rate) - more sensitive
detector = AnomalyDetector(sensitivity=2.0)

# High-variance metrics (e.g., queue depth) - less sensitive
detector = AnomalyDetector(sensitivity=3.5)
```

### 4. Act on Recommendations

- **CRITICAL anomalies**: Investigate immediately, correlate with logs/traces
- **Predicted breaches**: Scale resources proactively
- **WARNING anomalies**: Monitor closely, prepare runbooks

### 5. Combine with Traditional Alerting

ML anomaly detection complements (not replaces) traditional threshold-based alerts:
- Traditional alerts: Known failure modes, hard SLO breaches
- ML anomaly detection: Unknown issues, early warning, trend analysis

---

## Limitations

1. **Requires Historical Data**: Minimum 30 data points needed
2. **Not Real-Time**: Batch processing (acceptable for 5-minute intervals)
3. **No Context**: Doesn't understand business events (deployments, sales)
4. **Cold Start**: New metrics have no baseline (use fixed thresholds initially)
5. **Seasonal Patterns**: Requires Prophet for automatic detection

---

## Future Enhancements

- [ ] LSTM neural networks for complex pattern learning
- [ ] Anomaly clustering (group related anomalies)
- [ ] Root cause analysis (correlate anomalies across services)
- [ ] Feedback loop (learn from user confirmations)
- [ ] Multi-variate anomaly detection (detect correlated anomalies)
- [ ] Anomaly visualization dashboard
- [ ] Model persistence (save/load trained models)
- [ ] Online learning (update baselines in real-time)

---

## References

### Academic Papers

- Hochenbaum, J., et al. "Automatic Anomaly Detection in the Cloud Via Statistical Learning" (2017)
- Taylor, S., & Letham, B. "Forecasting at Scale" - Facebook Prophet (2017)
- Laptev, N., et al. "Time-Series Extreme Event Forecasting with Neural Networks at Uber" (2017)

### Implementation Resources

- Prophet Documentation: https://facebook.github.io/prophet/
- Modified Z-Score: Iglewicz & Hoaglin, "How to Detect and Handle Outliers" (1993)
- SciPy Statistical Methods: https://scipy.org/

---

## Support

For questions or issues:
- Review unit tests: `tests/ml/`
- Run live test: `python3 test_ml_live.py`
- Check logs: `self.logger` in AnomalyDetector and SLOAnomalyMonitor

---

**Version:** 1.0.0
**Last Updated:** 2025-10-12
**Maintainers:** SRE Analytics Team
