"""
FastAPI Application for SRE Analytics

Provides RESTful API access to metrics, reports, incidents, and anomaly detection.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Header, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging

from .auth import (
    key_manager, rate_limiter, has_permission,
    Role, APIKey
)
from .models import (
    MetricsResponse, ServiceHealth, AnomalyReport,
    CreateReportRequest, ReportResponse,
    ErrorResponse, HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SRE Analytics API",
    description="RESTful API for SLO monitoring, anomaly detection, and SRE reporting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Dependency: Authentication & Rate Limiting
# ============================================================================

async def get_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> APIKey:
    """
    Validate API key from header

    Raises:
        HTTPException: 401 if invalid or missing
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header."
        )

    validated_key = key_manager.validate_key(api_key)
    if not validated_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key"
        )

    return validated_key


async def check_rate_limit(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    """
    Check rate limit for API key

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    allowed, info = rate_limiter.check_rate_limit(api_key)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Reset in {info['reset_in']} seconds.",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info["reset_in"])
            }
        )

    return api_key


async def require_write_access(api_key: APIKey = Depends(check_rate_limit)) -> APIKey:
    """Require WRITE or ADMIN role"""
    if not has_permission(api_key, Role.WRITE):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. WRITE role required."
        )
    return api_key


async def require_admin_access(api_key: APIKey = Depends(check_rate_limit)) -> APIKey:
    """Require ADMIN role"""
    if not has_permission(api_key, Role.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. ADMIN role required."
        )
    return api_key


# ============================================================================
# Middleware: Add rate limit headers to all responses
# ============================================================================

@app.middleware("http")
async def add_rate_limit_headers(request, call_next):
    """Add rate limit info to response headers"""
    response = await call_next(request)

    # Try to get API key from request
    api_key_value = request.headers.get("X-API-Key")
    if api_key_value:
        validated_key = key_manager.validate_key(api_key_value)
        if validated_key:
            rate_info = rate_limiter.get_rate_limit_info(validated_key)
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset_in"])

    return response


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check API health status

    Returns system health, version, and uptime information.
    No authentication required.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        components={
            "api": "healthy",
            "authentication": "healthy",
            "rate_limiter": "healthy"
        }
    )


@app.get(
    "/api/v1/status",
    response_model=Dict[str, Any],
    tags=["Health"],
    summary="API status and statistics"
)
async def api_status(api_key: APIKey = Depends(check_rate_limit)):
    """
    Get API status and usage statistics

    Requires authentication. Returns rate limit info and API key details.
    """
    rate_info = rate_limiter.get_rate_limit_info(api_key)

    return {
        "api_version": "1.0.0",
        "authenticated_as": api_key.name,
        "role": api_key.role.value,
        "rate_limit": rate_info,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["Metrics"],
    summary="Get current metrics for services"
)
async def get_metrics(
    services: Optional[str] = None,
    metric_types: Optional[str] = None,
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Get current SLO metrics for services

    Query Parameters:
    - services: Comma-separated list of service names (optional, default: all)
    - metric_types: Comma-separated list of metric types (optional, default: all)
      Types: response_time, error_rate, availability, cpu, memory

    Returns:
    - List of current metric values with SLO targets and status
    """
    # Parse query parameters
    service_list = services.split(",") if services else None
    metric_list = metric_types.split(",") if metric_types else None

    # TODO: Integrate with actual data sources
    # For now, return mock data structure
    return MetricsResponse(
        services=service_list or ["product-service", "order-service"],
        metrics=[],
        timestamp=datetime.now(),
        count=0
    )


@app.get(
    "/api/v1/services/{service_name}/health",
    response_model=ServiceHealth,
    tags=["Metrics"],
    summary="Get health status for a specific service"
)
async def get_service_health(
    service_name: str,
    hours_back: int = 1,
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Get comprehensive health status for a service

    Path Parameters:
    - service_name: Name of the service

    Query Parameters:
    - hours_back: Hours of historical data to analyze (default: 1)

    Returns:
    - Health score, SLO compliance, trend analysis, and recommendations
    """
    # TODO: Integrate with Prometheus/AppDynamics
    return ServiceHealth(
        service_name=service_name,
        health_score=95.0,
        status="healthy",
        metrics_count=0,
        slo_compliance=98.5,
        timestamp=datetime.now()
    )


# ============================================================================
# Anomaly Detection Endpoints
# ============================================================================

@app.get(
    "/api/v1/anomalies",
    response_model=List[AnomalyReport],
    tags=["Anomaly Detection"],
    summary="Get recent anomalies across all services"
)
async def get_anomalies(
    services: Optional[str] = None,
    severity: Optional[str] = None,
    hours_back: int = 4,
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Get recent anomalies detected across services

    Query Parameters:
    - services: Comma-separated list of service names (optional)
    - severity: Filter by severity (info, warning, critical)
    - hours_back: Hours of historical data to analyze (default: 4)

    Returns:
    - List of anomaly reports with predictions and recommendations
    """
    service_list = services.split(",") if services else None

    # TODO: Integrate with ML anomaly detection
    return []


@app.post(
    "/api/v1/anomalies/detect",
    response_model=List[AnomalyReport],
    tags=["Anomaly Detection"],
    summary="Run anomaly detection on specified services"
)
async def detect_anomalies(
    services: List[str],
    detection_method: str = "modified_z_score",
    hours_back: int = 4,
    api_key: APIKey = Depends(require_write_access)
):
    """
    Trigger anomaly detection for specified services

    Requires WRITE or ADMIN role.

    Request Body:
    - services: List of service names to analyze
    - detection_method: Method to use (z_score, modified_z_score, iqr, moving_average, prophet)
    - hours_back: Hours of historical data to analyze

    Returns:
    - List of anomaly reports
    """
    # TODO: Integrate with ML anomaly detection
    return []


# ============================================================================
# Report Endpoints
# ============================================================================

@app.get(
    "/api/v1/reports",
    response_model=List[Dict[str, Any]],
    tags=["Reports"],
    summary="List available reports"
)
async def list_reports(
    limit: int = 10,
    offset: int = 0,
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    List generated SRE reports

    Query Parameters:
    - limit: Maximum number of reports to return (default: 10)
    - offset: Number of reports to skip (default: 0)

    Returns:
    - List of report metadata
    """
    # TODO: Implement report storage and retrieval
    return []


@app.post(
    "/api/v1/reports/generate",
    response_model=ReportResponse,
    tags=["Reports"],
    summary="Generate a new SRE report"
)
async def generate_report(
    request: CreateReportRequest,
    api_key: APIKey = Depends(require_write_access)
):
    """
    Generate a new SRE report

    Requires WRITE or ADMIN role.

    Request Body:
    - services: List of services to include
    - start_time: Report start time (ISO format)
    - end_time: Report end time (ISO format)
    - include_anomalies: Include anomaly detection (default: true)
    - include_recommendations: Include recommendations (default: true)

    Returns:
    - Report ID and generation status
    """
    # TODO: Integrate with report generation system
    return ReportResponse(
        report_id="report_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        status="generating",
        created_at=datetime.now()
    )


@app.get(
    "/api/v1/reports/{report_id}",
    response_model=Dict[str, Any],
    tags=["Reports"],
    summary="Get a specific report"
)
async def get_report(
    report_id: str,
    api_key: APIKey = Depends(check_rate_limit)
):
    """
    Retrieve a generated report

    Path Parameters:
    - report_id: Report ID

    Returns:
    - Complete report data
    """
    # TODO: Implement report retrieval
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Report {report_id} not found"
    )


# ============================================================================
# Admin Endpoints (API Key Management)
# ============================================================================

@app.post(
    "/api/v1/admin/keys",
    response_model=Dict[str, Any],
    tags=["Admin"],
    summary="Create a new API key"
)
async def create_api_key(
    name: str,
    role: str = "read",
    rate_limit: int = 100,
    api_key: APIKey = Depends(require_admin_access)
):
    """
    Create a new API key

    Requires ADMIN role.

    Query Parameters:
    - name: Descriptive name for the key
    - role: Access role (read, write, admin)
    - rate_limit: Requests per minute limit (default: 100)

    Returns:
    - New API key (only shown once!) and key metadata

    ⚠️  Save the API key immediately - it cannot be retrieved later!
    """
    try:
        role_enum = Role(role.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: read, write, admin"
        )

    raw_key, new_key = key_manager.generate_key(
        name=name,
        role=role_enum,
        rate_limit=rate_limit
    )

    return {
        "api_key": raw_key,  # Only time this is shown!
        "key_id": new_key.key_id,
        "name": new_key.name,
        "role": new_key.role.value,
        "rate_limit": new_key.rate_limit,
        "created_at": new_key.created_at.isoformat(),
        "warning": "Save this API key now! It cannot be retrieved later."
    }


@app.get(
    "/api/v1/admin/keys",
    response_model=List[Dict[str, Any]],
    tags=["Admin"],
    summary="List all API keys"
)
async def list_api_keys(api_key: APIKey = Depends(require_admin_access)):
    """
    List all API keys (without secrets)

    Requires ADMIN role.

    Returns:
    - List of API key metadata
    """
    return key_manager.list_keys()


@app.delete(
    "/api/v1/admin/keys/{key_id}",
    response_model=Dict[str, str],
    tags=["Admin"],
    summary="Revoke an API key"
)
async def revoke_api_key(
    key_id: str,
    api_key: APIKey = Depends(require_admin_access)
):
    """
    Revoke an API key

    Requires ADMIN role.

    Path Parameters:
    - key_id: API key ID to revoke

    Returns:
    - Confirmation message
    """
    success = key_manager.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found"
        )

    return {"message": f"API key {key_id} revoked successfully"}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp=datetime.now()
        ).dict()
    )
