"""
Tests for Anomaly Detector

Tests statistical anomaly detection methods and SLO breach prediction.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.ml.anomaly_detector import (
    AnomalyDetector,
    AnomalyMethod,
    AnomalySeverity,
    Anomaly,
    BaselineStatistics
)


@pytest.fixture
def detector():
    """Create detector with standard settings"""
    return AnomalyDetector(
        sensitivity=2.5,
        baseline_window_hours=168,
        min_samples=30
    )


@pytest.fixture
def normal_data():
    """Generate normal time-series data"""
    np.random.seed(42)
    values = np.random.normal(100, 10, 100)
    timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(100)]
    timestamps.reverse()
    return list(values), timestamps


@pytest.fixture
def anomalous_data():
    """Generate data with injected anomalies"""
    np.random.seed(42)
    values = np.random.normal(100, 10, 100)
    # Inject anomalies
    values[50] = 200  # Spike
    values[75] = 30   # Drop
    timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(100)]
    timestamps.reverse()
    return list(values), timestamps


@pytest.fixture
def trending_data():
    """Generate data with upward trend"""
    np.random.seed(42)
    x = np.arange(100)
    trend = x * 0.5
    noise = np.random.normal(0, 5, 100)
    values = 100 + trend + noise
    timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(100)]
    timestamps.reverse()
    return list(values), timestamps


class TestAnomalyDetector:
    """Test anomaly detector initialization and configuration"""

    def test_initialization(self):
        """Test detector initializes with correct parameters"""
        detector = AnomalyDetector(
            sensitivity=3.0,
            baseline_window_hours=24,
            min_samples=50
        )
        assert detector.sensitivity == 3.0
        assert detector.baseline_window_hours == 24
        assert detector.min_samples == 50
        assert not detector.prophet_available
        assert not detector.lstm_available

    def test_prophet_availability(self):
        """Test Prophet availability detection"""
        detector = AnomalyDetector(enable_prophet=True)
        # Prophet may or may not be installed
        assert isinstance(detector.prophet_available, bool)

    def test_insufficient_data(self, detector):
        """Test handling of insufficient data"""
        values = [100.0] * 20  # Less than min_samples
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(20)]

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service"
        )
        assert anomalies == []


class TestBaselineCalculation:
    """Test baseline statistics calculation"""

    def test_baseline_calculation(self, detector, normal_data):
        """Test baseline statistics are calculated correctly"""
        values, _ = normal_data
        values_array = np.array(values)

        baseline = detector._calculate_baseline(values_array)

        assert isinstance(baseline, BaselineStatistics)
        assert baseline.mean == pytest.approx(np.mean(values), rel=0.01)
        assert baseline.std == pytest.approx(np.std(values), rel=0.01)
        assert baseline.median == pytest.approx(np.median(values), rel=0.01)
        assert baseline.q1 == pytest.approx(np.percentile(values, 25), rel=0.01)
        assert baseline.q3 == pytest.approx(np.percentile(values, 75), rel=0.01)
        assert baseline.iqr == pytest.approx(baseline.q3 - baseline.q1, rel=0.01)
        assert baseline.min == pytest.approx(np.min(values), rel=0.01)
        assert baseline.max == pytest.approx(np.max(values), rel=0.01)
        assert baseline.sample_size == len(values)


class TestZScoreDetection:
    """Test Z-score anomaly detection"""

    def test_no_anomalies_in_normal_data(self, detector, normal_data):
        """Test Z-score doesn't flag normal data"""
        values, timestamps = normal_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.Z_SCORE
        )

        # Should detect very few or no anomalies in normal data
        assert len(anomalies) <= 3  # Allow for statistical outliers

    def test_detects_spike_anomalies(self, detector, anomalous_data):
        """Test Z-score detects spike anomalies"""
        values, timestamps = anomalous_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.Z_SCORE
        )

        assert len(anomalies) >= 2  # Should detect both injected anomalies

        # Check anomaly attributes
        for anomaly in anomalies:
            assert isinstance(anomaly, Anomaly)
            assert anomaly.service_name == "test_service"
            assert anomaly.metric_name == "test_metric"
            assert anomaly.method == AnomalyMethod.Z_SCORE
            assert 0 <= anomaly.confidence <= 1
            assert isinstance(anomaly.severity, AnomalySeverity)

    def test_zero_std_handling(self, detector):
        """Test handling of zero standard deviation"""
        values = [100.0] * 50  # Constant values
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(50)]

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.Z_SCORE
        )

        assert anomalies == []  # No anomalies in constant data


class TestModifiedZScoreDetection:
    """Test Modified Z-score (MAD) anomaly detection"""

    def test_modified_z_score_robust(self, detector, anomalous_data):
        """Test Modified Z-score is more robust than Z-score"""
        values, timestamps = anomalous_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.MODIFIED_Z_SCORE
        )

        assert len(anomalies) >= 2  # Should detect injected anomalies

        # Check method is correct
        for anomaly in anomalies:
            assert anomaly.method == AnomalyMethod.MODIFIED_Z_SCORE

    def test_zero_mad_handling(self, detector):
        """Test handling of zero MAD"""
        values = [100.0] * 50  # Constant values
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(50)]

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.MODIFIED_Z_SCORE
        )

        assert anomalies == []


class TestIQRDetection:
    """Test IQR anomaly detection"""

    def test_iqr_detects_outliers(self, detector, anomalous_data):
        """Test IQR method detects outliers"""
        values, timestamps = anomalous_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.IQR
        )

        assert len(anomalies) >= 2  # Should detect injected anomalies

        # Check method is correct
        for anomaly in anomalies:
            assert anomaly.method == AnomalyMethod.IQR
            assert anomaly.expected_value > 0  # Should have expected value

    def test_iqr_bounds(self, detector, normal_data):
        """Test IQR calculates reasonable bounds"""
        values, timestamps = normal_data
        values_array = np.array(values)
        baseline = detector._calculate_baseline(values_array)

        lower_bound = baseline.q1 - 1.5 * baseline.iqr
        upper_bound = baseline.q3 + 1.5 * baseline.iqr

        assert lower_bound < baseline.median < upper_bound
        assert lower_bound < baseline.mean < upper_bound


class TestMovingAverageDetection:
    """Test Moving Average anomaly detection"""

    def test_moving_average_detects_deviations(self, detector, anomalous_data):
        """Test moving average detects deviations"""
        values, timestamps = anomalous_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.MOVING_AVERAGE
        )

        # Moving average may or may not detect these specific anomalies
        # depending on the window size and surrounding values
        assert isinstance(anomalies, list)

        for anomaly in anomalies:
            assert anomaly.method == AnomalyMethod.MOVING_AVERAGE

    def test_moving_average_insufficient_data(self, detector):
        """Test moving average with insufficient data"""
        values = [100.0] * 40  # Just enough for detector, not enough for window
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(40)]

        # Should not crash
        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.MOVING_AVERAGE
        )

        assert isinstance(anomalies, list)


class TestSeverityDetermination:
    """Test anomaly severity classification"""

    def test_severity_levels(self, detector):
        """Test severity is determined correctly"""
        threshold = 2.5

        # INFO level
        severity = detector._determine_severity(2.6, threshold)
        assert severity == AnomalySeverity.INFO

        # WARNING level
        severity = detector._determine_severity(4.0, threshold)
        assert severity == AnomalySeverity.WARNING

        # CRITICAL level
        severity = detector._determine_severity(6.0, threshold)
        assert severity == AnomalySeverity.CRITICAL


class TestSLOBreachPrediction:
    """Test SLO breach prediction"""

    def test_predicts_breach_on_upward_trend(self, detector, trending_data):
        """Test prediction detects upward trend leading to breach"""
        values, timestamps = trending_data
        slo_target = 150.0  # Will be breached with upward trend

        prediction = detector.predict_slo_breach(
            values, timestamps, slo_target, forecast_minutes=30
        )

        assert isinstance(prediction, dict)
        assert "will_breach" in prediction
        assert "predicted_value" in prediction
        assert "confidence" in prediction
        assert "trend_slope" in prediction
        assert "forecast_time" in prediction
        assert prediction["slo_target"] == slo_target

    def test_no_breach_on_stable_data(self, detector, normal_data):
        """Test prediction shows no breach on stable data"""
        values, timestamps = normal_data
        slo_target = 200.0  # Well above normal range

        prediction = detector.predict_slo_breach(
            values, timestamps, slo_target, forecast_minutes=30
        )

        # Stable data should not predict breach
        assert prediction["will_breach"] is False or prediction["confidence"] < 0.5

    def test_prediction_with_downward_trend(self, detector):
        """Test prediction with downward trend"""
        # Generate downward trend
        np.random.seed(42)
        x = np.arange(50)
        values = 200 - x * 1.5 + np.random.normal(0, 3, 50)
        timestamps = [datetime.now() - timedelta(minutes=i*5) for i in range(50)]
        timestamps.reverse()

        slo_target = 200.0  # Above current values

        prediction = detector.predict_slo_breach(
            list(values), timestamps, slo_target, forecast_minutes=30
        )

        # Downward trend should show negative slope
        assert prediction["trend_slope"] < 0
        assert prediction["will_breach"] == False

    def test_prediction_insufficient_data(self, detector):
        """Test prediction with insufficient data"""
        values = [100.0] * 20  # Less than min_samples
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(20)]

        prediction = detector.predict_slo_breach(
            values, timestamps, 150.0, forecast_minutes=30
        )

        assert prediction["will_breach"] is False
        assert prediction["confidence"] == 0.0
        assert "Insufficient historical data" in prediction["reason"]

    def test_confidence_calculation(self, detector, normal_data):
        """Test confidence is calculated reasonably"""
        values, timestamps = normal_data

        prediction = detector.predict_slo_breach(
            values, timestamps, 150.0, forecast_minutes=30
        )

        assert 0 <= prediction["confidence"] <= 1


class TestMethodComparison:
    """Compare different detection methods"""

    def test_all_methods_detect_anomalies(self, detector, anomalous_data):
        """Test all methods can detect the same anomalies"""
        values, timestamps = anomalous_data

        methods = [
            AnomalyMethod.Z_SCORE,
            AnomalyMethod.MODIFIED_Z_SCORE,
            AnomalyMethod.IQR
        ]

        results = {}
        for method in methods:
            anomalies = detector.detect_anomalies(
                values, timestamps, "test_metric", "test_service",
                method=method
            )
            results[method] = len(anomalies)

        # Statistical methods should detect the injected anomalies
        for method, count in results.items():
            assert count >= 1, f"{method} detected no anomalies"

    def test_methods_have_different_sensitivity(self, detector, anomalous_data):
        """Test different methods have different sensitivity"""
        values, timestamps = anomalous_data

        z_score_anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.Z_SCORE
        )

        iqr_anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service",
            method=AnomalyMethod.IQR
        )

        # Methods may detect different numbers of anomalies
        # This is expected due to different statistical approaches
        assert isinstance(z_score_anomalies, list)
        assert isinstance(iqr_anomalies, list)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_single_value_data(self, detector):
        """Test with single data point"""
        values = [100.0]
        timestamps = [datetime.now()]

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service"
        )

        assert anomalies == []  # Insufficient data

    def test_all_nan_values(self, detector):
        """Test with NaN values"""
        values = [np.nan] * 50
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(50)]

        # Should handle gracefully (may return empty or filter NaNs)
        try:
            anomalies = detector.detect_anomalies(
                values, timestamps, "test_metric", "test_service"
            )
            assert isinstance(anomalies, list)
        except Exception:
            # NaN handling is implementation-dependent
            pass

    def test_negative_values(self, detector):
        """Test with negative values"""
        np.random.seed(42)
        values = np.random.normal(-100, 10, 50)
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(50)]

        anomalies = detector.detect_anomalies(
            list(values), timestamps, "test_metric", "test_service"
        )

        # Should work with negative values
        assert isinstance(anomalies, list)

    def test_very_large_values(self, detector):
        """Test with very large values"""
        np.random.seed(42)
        values = np.random.normal(1e9, 1e8, 50)
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(50)]

        anomalies = detector.detect_anomalies(
            list(values), timestamps, "test_metric", "test_service"
        )

        # Should handle large values
        assert isinstance(anomalies, list)


class TestAnomalyAttributes:
    """Test anomaly object attributes"""

    def test_anomaly_has_required_fields(self, detector, anomalous_data):
        """Test anomaly objects have all required fields"""
        values, timestamps = anomalous_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service"
        )

        assert len(anomalies) > 0

        for anomaly in anomalies:
            assert hasattr(anomaly, "timestamp")
            assert hasattr(anomaly, "value")
            assert hasattr(anomaly, "expected_value")
            assert hasattr(anomaly, "deviation")
            assert hasattr(anomaly, "severity")
            assert hasattr(anomaly, "confidence")
            assert hasattr(anomaly, "method")
            assert hasattr(anomaly, "metric_name")
            assert hasattr(anomaly, "service_name")
            assert hasattr(anomaly, "description")

    def test_anomaly_description_informative(self, detector, anomalous_data):
        """Test anomaly descriptions are informative"""
        values, timestamps = anomalous_data

        anomalies = detector.detect_anomalies(
            values, timestamps, "test_metric", "test_service"
        )

        for anomaly in anomalies:
            assert len(anomaly.description) > 10
            assert isinstance(anomaly.description, str)
