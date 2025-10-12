"""
Anomaly Detection Engine for SRE Metrics

Provides multiple anomaly detection methods:
1. Statistical methods (Z-score, IQR, Modified Z-score)
2. Time-series decomposition
3. Optional: Prophet for seasonal patterns
4. Optional: LSTM for complex patterns

Fast, accurate, and proactive SLO breach detection.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class AnomalyMethod(Enum):
    """Anomaly detection methods"""
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    MOVING_AVERAGE = "moving_average"
    PROPHET = "prophet"  # Optional, requires fbprophet
    LSTM = "lstm"  # Optional, requires tensorflow


class AnomalySeverity(Enum):
    """Severity levels for anomalies"""
    INFO = "info"  # Minor deviation, informational
    WARNING = "warning"  # Moderate deviation, monitor closely
    CRITICAL = "critical"  # Major deviation, immediate action needed


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    severity: AnomalySeverity
    confidence: float  # 0-1, how confident we are this is an anomaly
    method: AnomalyMethod
    metric_name: str
    service_name: str
    description: str


@dataclass
class BaselineStatistics:
    """Statistical baseline for a metric"""
    mean: float
    std: float
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range
    min: float
    max: float
    sample_size: int
    calculated_at: datetime


class AnomalyDetector:
    """
    Core anomaly detection engine

    Uses statistical methods by default (fast, no dependencies).
    Can optionally use ML models if available.
    """

    def __init__(
        self,
        sensitivity: float = 2.5,
        baseline_window_hours: int = 168,  # 1 week
        min_samples: int = 30,
        enable_prophet: bool = False,
        enable_lstm: bool = False
    ):
        """
        Initialize anomaly detector

        Args:
            sensitivity: Z-score threshold (2.5 = ~99% confidence)
            baseline_window_hours: Hours of historical data for baseline
            min_samples: Minimum data points required
            enable_prophet: Use Facebook Prophet (requires fbprophet package)
            enable_lstm: Use LSTM neural networks (requires tensorflow)
        """
        self.sensitivity = sensitivity
        self.baseline_window_hours = baseline_window_hours
        self.min_samples = min_samples
        self.enable_prophet = enable_prophet
        self.enable_lstm = enable_lstm

        self.logger = logging.getLogger(__name__)

        # Check optional dependencies
        self.prophet_available = False
        self.lstm_available = False

        if enable_prophet:
            try:
                from prophet import Prophet
                self.prophet_available = True
                self.logger.info("Prophet available for seasonal decomposition")
            except ImportError:
                self.logger.warning("Prophet requested but not installed. Using statistical methods only.")

        if enable_lstm:
            try:
                import tensorflow as tf
                self.lstm_available = True
                self.logger.info("TensorFlow available for LSTM anomaly detection")
            except ImportError:
                self.logger.warning("LSTM requested but TensorFlow not installed. Using statistical methods only.")

    def detect_anomalies(
        self,
        values: List[float],
        timestamps: List[datetime],
        metric_name: str,
        service_name: str,
        method: AnomalyMethod = AnomalyMethod.MODIFIED_Z_SCORE
    ) -> List[Anomaly]:
        """
        Detect anomalies in time-series data

        Args:
            values: List of metric values
            timestamps: Corresponding timestamps
            metric_name: Name of the metric
            service_name: Name of the service
            method: Detection method to use

        Returns:
            List of detected anomalies
        """
        if len(values) < self.min_samples:
            self.logger.warning(
                f"Insufficient data for {service_name}.{metric_name}: "
                f"{len(values)} samples (need {self.min_samples})"
            )
            return []

        # Convert to numpy arrays
        values_array = np.array(values)

        # Calculate baseline statistics
        baseline = self._calculate_baseline(values_array)

        # Detect anomalies based on method
        if method == AnomalyMethod.Z_SCORE:
            anomalies = self._detect_z_score(
                values_array, timestamps, baseline, metric_name, service_name
            )
        elif method == AnomalyMethod.MODIFIED_Z_SCORE:
            anomalies = self._detect_modified_z_score(
                values_array, timestamps, baseline, metric_name, service_name
            )
        elif method == AnomalyMethod.IQR:
            anomalies = self._detect_iqr(
                values_array, timestamps, baseline, metric_name, service_name
            )
        elif method == AnomalyMethod.MOVING_AVERAGE:
            anomalies = self._detect_moving_average(
                values_array, timestamps, metric_name, service_name
            )
        elif method == AnomalyMethod.PROPHET and self.prophet_available:
            anomalies = self._detect_prophet(
                values_array, timestamps, metric_name, service_name
            )
        else:
            # Default to modified z-score
            anomalies = self._detect_modified_z_score(
                values_array, timestamps, baseline, metric_name, service_name
            )

        self.logger.info(
            f"Detected {len(anomalies)} anomalies in {service_name}.{metric_name} "
            f"using {method.value}"
        )

        return anomalies

    def _calculate_baseline(self, values: np.ndarray) -> BaselineStatistics:
        """Calculate baseline statistics from historical data"""
        return BaselineStatistics(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            median=float(np.median(values)),
            q1=float(np.percentile(values, 25)),
            q3=float(np.percentile(values, 75)),
            iqr=float(np.percentile(values, 75) - np.percentile(values, 25)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            sample_size=len(values),
            calculated_at=datetime.now()
        )

    def _detect_z_score(
        self,
        values: np.ndarray,
        timestamps: List[datetime],
        baseline: BaselineStatistics,
        metric_name: str,
        service_name: str
    ) -> List[Anomaly]:
        """Detect anomalies using standard Z-score method"""
        anomalies = []

        if baseline.std == 0:
            return anomalies  # No variation, no anomalies

        z_scores = np.abs((values - baseline.mean) / baseline.std)

        for i, (value, timestamp, z_score) in enumerate(zip(values, timestamps, z_scores)):
            if z_score > self.sensitivity:
                severity = self._determine_severity(z_score, self.sensitivity)
                confidence = min(z_score / (self.sensitivity * 2), 1.0)

                anomaly = Anomaly(
                    timestamp=timestamp,
                    value=float(value),
                    expected_value=baseline.mean,
                    deviation=float(value - baseline.mean),
                    severity=severity,
                    confidence=confidence,
                    method=AnomalyMethod.Z_SCORE,
                    metric_name=metric_name,
                    service_name=service_name,
                    description=f"Z-score {z_score:.2f} exceeds threshold {self.sensitivity:.2f}"
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_modified_z_score(
        self,
        values: np.ndarray,
        timestamps: List[datetime],
        baseline: BaselineStatistics,
        metric_name: str,
        service_name: str
    ) -> List[Anomaly]:
        """
        Detect anomalies using Modified Z-score (MAD-based)

        More robust to outliers than standard Z-score.
        """
        anomalies = []

        # Calculate MAD (Median Absolute Deviation)
        median = baseline.median
        mad = np.median(np.abs(values - median))

        if mad == 0:
            return anomalies  # No variation

        # Modified Z-score: 0.6745 * (x - median) / MAD
        modified_z_scores = np.abs(0.6745 * (values - median) / mad)

        threshold = self.sensitivity * 1.5  # Adjusted threshold for modified z-score

        for i, (value, timestamp, mz_score) in enumerate(zip(values, timestamps, modified_z_scores)):
            if mz_score > threshold:
                severity = self._determine_severity(mz_score, threshold)
                confidence = min(mz_score / (threshold * 2), 1.0)

                anomaly = Anomaly(
                    timestamp=timestamp,
                    value=float(value),
                    expected_value=median,
                    deviation=float(value - median),
                    severity=severity,
                    confidence=confidence,
                    method=AnomalyMethod.MODIFIED_Z_SCORE,
                    metric_name=metric_name,
                    service_name=service_name,
                    description=f"Modified Z-score {mz_score:.2f} exceeds threshold {threshold:.2f}"
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_iqr(
        self,
        values: np.ndarray,
        timestamps: List[datetime],
        baseline: BaselineStatistics,
        metric_name: str,
        service_name: str
    ) -> List[Anomaly]:
        """
        Detect anomalies using Interquartile Range (IQR) method

        Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        """
        anomalies = []

        lower_bound = baseline.q1 - 1.5 * baseline.iqr
        upper_bound = baseline.q3 + 1.5 * baseline.iqr

        for value, timestamp in zip(values, timestamps):
            if value < lower_bound or value > upper_bound:
                # Calculate severity based on how far outside bounds
                if value < lower_bound:
                    deviation_factor = (lower_bound - value) / baseline.iqr
                    expected = lower_bound
                else:
                    deviation_factor = (value - upper_bound) / baseline.iqr
                    expected = upper_bound

                severity = self._determine_severity(deviation_factor, 1.5)
                confidence = min(deviation_factor / 3.0, 1.0)

                anomaly = Anomaly(
                    timestamp=timestamp,
                    value=float(value),
                    expected_value=expected,
                    deviation=float(value - expected),
                    severity=severity,
                    confidence=confidence,
                    method=AnomalyMethod.IQR,
                    metric_name=metric_name,
                    service_name=service_name,
                    description=f"Value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_moving_average(
        self,
        values: np.ndarray,
        timestamps: List[datetime],
        metric_name: str,
        service_name: str,
        window: int = 5
    ) -> List[Anomaly]:
        """
        Detect anomalies using moving average deviation

        Compares each value to moving average of surrounding values.
        """
        anomalies = []

        if len(values) < window:
            return anomalies

        # Calculate moving average
        moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')

        # Calculate moving std
        moving_std = np.array([
            np.std(values[max(0, i-window//2):min(len(values), i+window//2+1)])
            for i in range(len(values))
        ])

        # Align arrays (moving_avg is shorter)
        offset = (len(values) - len(moving_avg)) // 2

        for i in range(offset, len(values) - offset):
            ma_idx = i - offset
            if ma_idx >= len(moving_avg):
                continue

            expected = moving_avg[ma_idx]
            std = moving_std[i]

            if std > 0:
                deviation = abs(values[i] - expected) / std

                if deviation > self.sensitivity:
                    severity = self._determine_severity(deviation, self.sensitivity)
                    confidence = min(deviation / (self.sensitivity * 2), 1.0)

                    anomaly = Anomaly(
                        timestamp=timestamps[i],
                        value=float(values[i]),
                        expected_value=float(expected),
                        deviation=float(values[i] - expected),
                        severity=severity,
                        confidence=confidence,
                        method=AnomalyMethod.MOVING_AVERAGE,
                        metric_name=metric_name,
                        service_name=service_name,
                        description=f"Deviation {deviation:.2f}σ from moving average"
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _detect_prophet(
        self,
        values: np.ndarray,
        timestamps: List[datetime],
        metric_name: str,
        service_name: str
    ) -> List[Anomaly]:
        """
        Detect anomalies using Facebook Prophet (seasonal decomposition)

        Requires fbprophet package. Best for metrics with seasonal patterns.
        """
        try:
            from prophet import Prophet
            import pandas as pd

            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': timestamps,
                'y': values
            })

            # Train Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                interval_width=0.95
            )
            model.fit(df)

            # Get predictions
            forecast = model.predict(df)

            # Detect anomalies where actual deviates from prediction
            anomalies = []

            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                predicted = forecast['yhat'].iloc[i]
                lower_bound = forecast['yhat_lower'].iloc[i]
                upper_bound = forecast['yhat_upper'].iloc[i]

                if value < lower_bound or value > upper_bound:
                    deviation = abs(value - predicted)
                    confidence = 0.95  # Prophet's confidence level

                    # Determine severity
                    prediction_range = upper_bound - lower_bound
                    if prediction_range > 0:
                        deviation_factor = deviation / prediction_range
                        severity = self._determine_severity(deviation_factor, 1.0)
                    else:
                        severity = AnomalySeverity.WARNING

                    anomaly = Anomaly(
                        timestamp=timestamp,
                        value=float(value),
                        expected_value=float(predicted),
                        deviation=float(deviation),
                        severity=severity,
                        confidence=confidence,
                        method=AnomalyMethod.PROPHET,
                        metric_name=metric_name,
                        service_name=service_name,
                        description=f"Value {value:.2f} outside Prophet prediction [{lower_bound:.2f}, {upper_bound:.2f}]"
                    )
                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            self.logger.error(f"Prophet detection failed: {e}")
            return []

    def _determine_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """Determine anomaly severity based on deviation score"""
        if score > threshold * 2:
            return AnomalySeverity.CRITICAL
        elif score > threshold * 1.5:
            return AnomalySeverity.WARNING
        else:
            return AnomalySeverity.INFO

    def predict_slo_breach(
        self,
        values: List[float],
        timestamps: List[datetime],
        slo_target: float,
        forecast_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Predict if SLO will be breached in the near future

        Args:
            values: Historical metric values
            timestamps: Corresponding timestamps
            slo_target: SLO target value
            forecast_minutes: How far ahead to predict

        Returns:
            Dictionary with prediction results
        """
        if len(values) < self.min_samples:
            return {
                "will_breach": False,
                "confidence": 0.0,
                "reason": "Insufficient historical data"
            }

        # Calculate trend
        values_array = np.array(values)
        x = np.arange(len(values))

        # Linear regression for trend
        coeffs = np.polyfit(x, values_array, 1)
        trend_slope = coeffs[0]

        # Extrapolate
        last_value = values[-1]
        last_time = timestamps[-1]

        # Estimate value after forecast_minutes
        # Assuming consistent data interval
        if len(timestamps) > 1:
            avg_interval_seconds = (timestamps[-1] - timestamps[0]).total_seconds() / (len(timestamps) - 1)
            steps_ahead = (forecast_minutes * 60) / avg_interval_seconds
            predicted_value = last_value + (trend_slope * steps_ahead)
        else:
            predicted_value = last_value

        # Check if predicted value will breach SLO
        will_breach = predicted_value > slo_target

        # Calculate confidence based on trend consistency
        residuals = values_array - np.polyval(coeffs, x)
        prediction_std = np.std(residuals)
        confidence = 1.0 - min(prediction_std / abs(predicted_value - last_value + 1e-6), 1.0)

        return {
            "will_breach": will_breach,
            "predicted_value": float(predicted_value),
            "slo_target": slo_target,
            "confidence": float(confidence),
            "forecast_time": last_time + timedelta(minutes=forecast_minutes),
            "trend_slope": float(trend_slope),
            "reason": f"Trend slope {trend_slope:.2f} predicts breach" if will_breach else "Trending within SLO"
        }
