"""
Pydantic Models for API Request/Response Schemas

Defines data structures for API endpoints with validation.
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class ServiceStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    """Anomaly severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """Anomaly detection methods"""
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    MOVING_AVERAGE = "moving_average"
    PROPHET = "prophet"


# ============================================================================
# Base Models
# ============================================================================

class BaseResponse(BaseModel):
    """Base response model with timestamp"""
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Health & Status Models
# ============================================================================

class HealthResponse(BaseResponse):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component health status"
    )


class ErrorResponse(BaseResponse):
    """Error response"""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )


# ============================================================================
# Metrics Models
# ============================================================================

class MetricValue(BaseModel):
    """Single metric value"""
    service_name: str = Field(..., description="Service name")
    metric_name: str = Field(..., description="Metric name (e.g., response_time)")
    current_value: float = Field(..., description="Current metric value")
    slo_target: float = Field(..., description="SLO target value")
    sla_target: Optional[float] = Field(None, description="SLA target value")
    unit: str = Field(..., description="Metric unit (e.g., ms, %, count)")
    status: str = Field(..., description="SLO status (healthy, at_risk, breached)")
    error_budget_consumed: float = Field(
        ...,
        description="Error budget consumed (%)",
        ge=0,
        le=100
    )

    class Config:
        schema_extra = {
            "example": {
                "service_name": "product-service",
                "metric_name": "response_time",
                "current_value": 145.5,
                "slo_target": 150.0,
                "sla_target": 200.0,
                "unit": "ms",
                "status": "healthy",
                "error_budget_consumed": 15.3
            }
        }


class MetricsResponse(BaseResponse):
    """Response for /metrics endpoint"""
    services: List[str] = Field(..., description="List of service names")
    metrics: List[MetricValue] = Field(..., description="List of metric values")
    count: int = Field(..., description="Total number of metrics returned")

    class Config:
        schema_extra = {
            "example": {
                "services": ["product-service", "order-service"],
                "metrics": [],
                "count": 0,
                "timestamp": "2025-10-12T10:30:00"
            }
        }


class ServiceHealth(BaseResponse):
    """Service health status"""
    service_name: str = Field(..., description="Service name")
    health_score: float = Field(
        ...,
        description="Health score (0-100)",
        ge=0,
        le=100
    )
    status: ServiceStatus = Field(..., description="Overall status")
    metrics_count: int = Field(..., description="Number of metrics analyzed")
    slo_compliance: float = Field(
        ...,
        description="SLO compliance percentage",
        ge=0,
        le=100
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Health insights and observations"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )

    class Config:
        schema_extra = {
            "example": {
                "service_name": "product-service",
                "health_score": 95.0,
                "status": "healthy",
                "metrics_count": 5,
                "slo_compliance": 98.5,
                "insights": ["Response time trending upward", "Error rate stable"],
                "recommendations": ["Monitor response time closely"],
                "timestamp": "2025-10-12T10:30:00"
            }
        }


# ============================================================================
# Anomaly Detection Models
# ============================================================================

class AnomalyDetail(BaseModel):
    """Single anomaly detail"""
    timestamp: datetime = Field(..., description="When anomaly occurred")
    value: float = Field(..., description="Actual value")
    expected_value: float = Field(..., description="Expected value")
    deviation: float = Field(..., description="Deviation from expected")
    severity: SeverityLevel = Field(..., description="Anomaly severity")
    confidence: float = Field(
        ...,
        description="Detection confidence (0-1)",
        ge=0,
        le=1
    )
    description: str = Field(..., description="Anomaly description")


class BreachPrediction(BaseModel):
    """SLO breach prediction"""
    will_breach: bool = Field(..., description="Whether breach is predicted")
    predicted_value: Optional[float] = Field(None, description="Predicted value")
    slo_target: float = Field(..., description="SLO target value")
    confidence: float = Field(
        ...,
        description="Prediction confidence (0-1)",
        ge=0,
        le=1
    )
    forecast_time: Optional[datetime] = Field(
        None,
        description="When breach is predicted to occur"
    )
    trend_slope: float = Field(..., description="Trend slope")
    reason: str = Field(..., description="Prediction reasoning")


class AnomalyReport(BaseResponse):
    """Anomaly detection report"""
    service_name: str = Field(..., description="Service name")
    metric_name: str = Field(..., description="Metric name")
    anomalies: List[AnomalyDetail] = Field(
        default_factory=list,
        description="Detected anomalies"
    )
    baseline_health: str = Field(..., description="Baseline health status")
    prediction: Optional[BreachPrediction] = Field(
        None,
        description="SLO breach prediction"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    detection_method: DetectionMethod = Field(
        DetectionMethod.MODIFIED_Z_SCORE,
        description="Detection method used"
    )

    class Config:
        schema_extra = {
            "example": {
                "service_name": "product-service",
                "metric_name": "response_time",
                "anomalies": [],
                "baseline_health": "healthy",
                "prediction": None,
                "recommendations": ["No action required"],
                "detection_method": "modified_z_score",
                "timestamp": "2025-10-12T10:30:00"
            }
        }


# ============================================================================
# Report Models
# ============================================================================

class CreateReportRequest(BaseModel):
    """Request to generate a new report"""
    services: List[str] = Field(
        ...,
        description="Services to include in report",
        min_items=1
    )
    start_time: datetime = Field(..., description="Report start time")
    end_time: datetime = Field(..., description="Report end time")
    include_anomalies: bool = Field(
        True,
        description="Include anomaly detection"
    )
    include_recommendations: bool = Field(
        True,
        description="Include recommendations"
    )
    format: str = Field(
        "json",
        description="Report format (json, html, pdf)"
    )

    @validator("end_time")
    def end_after_start(cls, v, values):
        """Validate end_time is after start_time"""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v

    @validator("format")
    def valid_format(cls, v):
        """Validate report format"""
        valid_formats = ["json", "html", "pdf"]
        if v.lower() not in valid_formats:
            raise ValueError(f"format must be one of: {', '.join(valid_formats)}")
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "services": ["product-service", "order-service"],
                "start_time": "2025-10-12T00:00:00",
                "end_time": "2025-10-12T23:59:59",
                "include_anomalies": True,
                "include_recommendations": True,
                "format": "json"
            }
        }


class ReportResponse(BaseResponse):
    """Report generation response"""
    report_id: str = Field(..., description="Unique report ID")
    status: str = Field(..., description="Report status (generating, ready, failed)")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )
    download_url: Optional[str] = Field(
        None,
        description="URL to download completed report"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if failed"
    )

    class Config:
        schema_extra = {
            "example": {
                "report_id": "report_20251012_103000",
                "status": "generating",
                "created_at": "2025-10-12T10:30:00",
                "completed_at": None,
                "download_url": None,
                "error": None,
                "timestamp": "2025-10-12T10:30:00"
            }
        }


# ============================================================================
# Pagination Models
# ============================================================================

class PaginatedResponse(BaseResponse):
    """Base paginated response"""
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="Whether more items are available")


class PaginatedMetricsResponse(PaginatedResponse):
    """Paginated metrics response"""
    items: List[MetricValue] = Field(..., description="Metric items")


class PaginatedAnomaliesResponse(PaginatedResponse):
    """Paginated anomalies response"""
    items: List[AnomalyReport] = Field(..., description="Anomaly reports")
