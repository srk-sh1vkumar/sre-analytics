"""
Type definitions and TypedDict classes for SRE Analytics

This module contains TypedDict definitions for complex data structures
used throughout the application for better type safety.
"""

from typing import TypedDict, Optional, List, Dict, Any, Union
from datetime import datetime


# ============================================================================
# Configuration Types
# ============================================================================

class ControllerConfig(TypedDict):
    """AppDynamics controller configuration"""
    host: str
    port: Optional[int]


class ApplicationsConfig(TypedDict, total=False):
    """Applications configuration"""
    primary_app: str


class AppDynamicsConfigDict(TypedDict):
    """Complete AppDynamics configuration dictionary"""
    controller: ControllerConfig
    applications: ApplicationsConfig


# ============================================================================
# Metric Types
# ============================================================================

class MetricDict(TypedDict):
    """Basic metric data dictionary"""
    metric_name: str
    metric_path: str
    value: float
    timestamp: datetime
    unit: str
    tier: str
    application: str


class SLOMetricDict(TypedDict):
    """SLO metric data dictionary"""
    service_name: str
    metric_name: str
    current_value: float
    slo_target: float
    sla_target: float
    status: str
    error_budget_consumed: float
    timestamp: datetime
    unit: str
    description: str
    trend_data: Optional[List[float]]


class MetricsSummary(TypedDict):
    """Metrics summary statistics"""
    total_services: int
    total_metrics: int
    compliant_count: int
    at_risk_count: int
    breached_count: int
    compliance_percentage: float
    health_status: str
    avg_error_budget_consumed: float


# ============================================================================
# Incident Types
# ============================================================================

class IncidentDict(TypedDict):
    """Incident data dictionary"""
    incident_id: str
    title: str
    description: str
    severity: str
    application_name: str
    start_time: datetime
    end_time: Optional[datetime]
    affected_services: List[str]
    root_cause: str
    resolution_steps: List[str]
    llm_analysis: str
    lessons_learned: str


class PerformanceSnapshotDict(TypedDict):
    """Performance snapshot dictionary"""
    service_name: str
    timestamp: datetime
    metrics: Dict[str, float]
    logs: List[str]
    errors: List[str]


# ============================================================================
# Business Transaction Types
# ============================================================================

class BusinessTransactionDict(TypedDict):
    """Business transaction metrics dictionary"""
    name: str
    tier: str
    calls_per_minute: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    timestamp: datetime


# ============================================================================
# API Response Types
# ============================================================================

class OAuthTokenDict(TypedDict):
    """OAuth token response dictionary"""
    access_token: str
    token_type: str
    expires_in: int
    expires_at: datetime


class APIErrorResponse(TypedDict):
    """API error response dictionary"""
    error: str
    message: Optional[str]
    status_code: int


class APISuccessResponse(TypedDict):
    """Generic API success response"""
    success: bool
    data: Dict[str, Any]
    message: Optional[str]


# ============================================================================
# Application Health Types
# ============================================================================

class ApplicationHealthDict(TypedDict):
    """Application health metrics dictionary"""
    availability_percentage: float
    total_calls: int
    errors_per_minute: float
    average_response_time: float
    health_status: str


# ============================================================================
# Report Types
# ============================================================================

class ReportTemplateData(TypedDict):
    """Report template data dictionary"""
    app_name: str
    report_date: str
    report_time: str
    metrics: List[SLOMetricDict]
    trend_charts: Dict[str, str]
    incident: Optional[IncidentDict]
    summary: MetricsSummary
    has_incident: bool
    llm_analysis: str


class ChartDataDict(TypedDict):
    """Chart data dictionary"""
    labels: List[str]
    values: List[float]
    colors: List[str]
    title: str
    chart_type: str


# ============================================================================
# Collector Types
# ============================================================================

class CollectorMetricsDict(TypedDict):
    """Complete metrics collection result"""
    application_name: str
    collection_time: datetime
    business_transactions: List[BusinessTransactionDict]
    infrastructure_metrics: List[MetricDict]
    application_health: ApplicationHealthDict
    slo_metrics: List[SLOMetricDict]


class ConnectionTestResults(TypedDict):
    """Connection test results dictionary"""
    controller_reachable: bool
    oauth_authentication: bool
    applications_access: bool
    primary_app_found: bool
    error_message: Optional[str]


class DiagnosisResults(TypedDict):
    """Diagnosis results dictionary"""
    controller_reachable: bool
    oauth_endpoint_available: bool
    credentials_valid: bool
    ssl_issues: bool
    network_issues: bool
    recommendations: List[str]


# ============================================================================
# PDF Generation Types
# ============================================================================

class PDFGenerationOptions(TypedDict, total=False):
    """PDF generation options"""
    output_path: str
    use_browser: bool
    page_size: str
    margins: Dict[str, str]
    include_charts: bool
    include_llm_analysis: bool


class PDFMetadata(TypedDict):
    """PDF document metadata"""
    title: str
    author: str
    subject: str
    creator: str
    creation_date: datetime


# ============================================================================
# LLM Types
# ============================================================================

class LLMRequestDict(TypedDict):
    """LLM request parameters"""
    prompt: str
    model: str
    max_tokens: int
    temperature: float
    provider: str


class LLMResponseDict(TypedDict):
    """LLM response dictionary"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    provider: str


# ============================================================================
# Error Context Types
# ============================================================================

class ErrorContextDict(TypedDict, total=False):
    """Error context dictionary"""
    operation: str
    service: str
    endpoint: str
    user_id: str
    request_id: str
    timestamp: datetime
    additional_info: Dict[str, Any]


# ============================================================================
# Monitoring Types
# ============================================================================

class AlertCondition(TypedDict):
    """Alert condition definition"""
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '=='
    duration_minutes: int
    severity: str


class MonitoringRule(TypedDict):
    """Monitoring rule definition"""
    rule_id: str
    name: str
    description: str
    conditions: List[AlertCondition]
    enabled: bool
    notification_channels: List[str]


# ============================================================================
# Utility Type Aliases
# ============================================================================

# JSON-serializable types
JSONValue = Optional[Union[str, int, float, bool, Dict[str, Any], List[Any]]]
JSONDict = Dict[str, JSONValue]

# Metric value types
MetricValue = Union[float, int]
MetricName = str
ServiceName = str

# Time range types
TimeRangeMinutes = int
TimeRangeDays = int

# Status types
ComplianceStatus = str  # 'compliant' | 'at_risk' | 'breached'
HealthStatus = str  # 'Healthy' | 'Degraded' | 'Unhealthy'
SeverityLevel = str  # 'Critical' | 'High' | 'Medium' | 'Low'
