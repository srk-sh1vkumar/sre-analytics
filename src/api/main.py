"""
FastAPI Application for SRE Analytics

Provides RESTful API access to metrics, reports, incidents, and anomaly detection.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import logging

from pydantic import BaseModel, Field

from src.api.auth import key_manager, rate_limiter, has_permission, Role, APIKey
from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem
from src.ml.slo_anomaly_monitor import SLOAnomalyMonitor
from src.data_sources.prometheus_integration import PrometheusIntegration
from src.data_sources.appdynamics_integration import AppDynamicsIntegration
from src.config.app_config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="SRE Analytics API",
    description="RESTful API for Site Reliability Engineering metrics, reports, and anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global storage for background tasks
task_storage: Dict[str, Dict] = {}


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class HealthResponse(BaseModel):
    """System health check response"""
    status: str
    version: str
    timestamp: datetime
    data_sources: Dict[str, str]


class MetricQuery(BaseModel):
    """Query parameters for fetching metrics"""
    service: str = Field(..., description="Service name")
    metric_type: str = Field(..., description="Metric type (latency, error_rate, availability)")
    start_time: Optional[datetime] = Field(None, description="Start time (default: 1 hour ago)")
    end_time: Optional[datetime] = Field(None, description="End time (default: now)")
    data_source: str = Field("prometheus", description="Data source (prometheus, appdynamics)")


class MetricResponse(BaseModel):
    """Response containing metric data"""
    service: str
    metric_type: str
    timestamp: datetime
    values: List[Dict[str, Any]]
    summary: Dict[str, float]


class ReportGenerationRequest(BaseModel):
    """Request to generate a new report"""
    application_name: str = Field(..., description="Application name")
    services: List[str] = Field(..., description="List of services to analyze")
    report_type: str = Field("performance", description="Report type: performance or incident")
    incident_time: Optional[datetime] = Field(None, description="Incident time (for incident reports)")
    incident_duration: Optional[float] = Field(1.0, description="Incident duration in hours")
    output_format: str = Field("html", description="Output format: html, pdf, or json")


class ReportResponse(BaseModel):
    """Response after report generation"""
    task_id: str
    status: str
    message: str
    estimated_time: int = Field(..., description="Estimated completion time in seconds")


class ReportStatusResponse(BaseModel):
    """Report generation status"""
    task_id: str
    status: str  # pending, in_progress, completed, failed
    progress: int  # 0-100
    message: str
    report_url: Optional[str] = None
    created_at: datetime


class IncidentCreate(BaseModel):
    """Create a new incident"""
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    affected_services: List[str] = Field(..., description="List of affected services")
    start_time: datetime = Field(..., description="Incident start time")
    end_time: Optional[datetime] = Field(None, description="Incident end time (if resolved)")
    tags: Optional[List[str]] = Field([], description="Tags for categorization")


class IncidentResponse(BaseModel):
    """Incident details response"""
    incident_id: str
    title: str
    description: str
    severity: str
    affected_services: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]  # Hours
    status: str  # active, resolved
    tags: List[str]
    created_at: datetime


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    service: str = Field(..., description="Service name")
    metric_type: str = Field(..., description="Metric type")
    detection_method: str = Field("z_score", description="Detection method: z_score, modified_z_score, iqr, moving_average")
    sensitivity: float = Field(2.5, description="Detection sensitivity (sigma threshold)")
    lookback_hours: int = Field(24, description="Historical data lookback period")


class AnomalyResponse(BaseModel):
    """Anomaly detection results"""
    service: str
    metric_type: str
    anomalies_detected: int
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]
    recommendations: List[str]


class APIKeyCreateRequest(BaseModel):
    """Request to create new API key"""
    name: str = Field(..., description="Descriptive name for the key")
    role: str = Field("read", description="Role: admin, write, or read")
    rate_limit: int = Field(100, description="Requests per minute limit")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata")


class APIKeyResponse(BaseModel):
    """API key creation response"""
    key_id: str
    api_key: str  # Only returned on creation
    name: str
    role: str
    rate_limit: int
    created_at: datetime


# ============================================================================
# Authentication & Authorization
# ============================================================================

async def get_api_key(api_key_value: str = Security(api_key_header)) -> APIKey:
    """
    Validate API key from header

    Args:
        api_key_value: API key from X-API-Key header

    Returns:
        Validated APIKey object

    Raises:
        HTTPException: If key is invalid or disabled
    """
    if not api_key_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header."
        )

    api_key = key_manager.validate_key(api_key_value)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or disabled API key"
        )

    # Check rate limit
    allowed, rate_info = rate_limiter.check_rate_limit(api_key)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {rate_info['reset_in']} seconds.",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info["reset_in"])
            }
        )

    return api_key


def require_role(required_role: Role):
    """
    Dependency to check if API key has required role

    Args:
        required_role: Minimum required role

    Returns:
        Dependency function
    """
    async def _check_role(api_key: APIKey = Depends(get_api_key)) -> APIKey:
        if not has_permission(api_key, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role.value}"
            )
        return api_key

    return _check_role


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SRE Analytics API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    System health check endpoint

    Returns system status and data source connectivity.
    """
    # Check data source connectivity
    data_sources = {}

    try:
        prometheus = PrometheusIntegration()
        prometheus_health = "healthy" if prometheus.test_connection() else "unavailable"
    except Exception as e:
        prometheus_health = f"error: {str(e)}"

    data_sources["prometheus"] = prometheus_health

    try:
        appdynamics = AppDynamicsIntegration()
        appdynamics_health = "healthy" if appdynamics.test_connection() else "unavailable"
    except Exception as e:
        appdynamics_health = f"error: {str(e)}"

    data_sources["appdynamics"] = appdynamics_health

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        data_sources=data_sources
    )


@app.get("/metrics", response_model=MetricResponse, tags=["Metrics"])
async def get_metrics(
    query: MetricQuery,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Fetch current metrics for a service

    Requires: READ permission
    """
    try:
        # Set default time range if not provided
        end_time = query.end_time or datetime.now()
        start_time = query.start_time or (end_time - timedelta(hours=1))

        # Select data source
        if query.data_source == "prometheus":
            data_source = PrometheusIntegration()
        elif query.data_source == "appdynamics":
            data_source = AppDynamicsIntegration()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported data source: {query.data_source}"
            )

        # Fetch metrics
        metrics = data_source.get_service_metrics(
            service_name=query.service,
            start_time=start_time,
            end_time=end_time
        )

        # Filter by metric type
        filtered_metrics = [m for m in metrics if m.metric_type == query.metric_type]

        if not filtered_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for service '{query.service}' with type '{query.metric_type}'"
            )

        # Calculate summary statistics
        values_list = [m.value for m in filtered_metrics if m.value is not None]
        summary = {
            "count": len(values_list),
            "min": min(values_list) if values_list else None,
            "max": max(values_list) if values_list else None,
            "avg": sum(values_list) / len(values_list) if values_list else None
        }

        # Format response
        values = [
            {
                "timestamp": m.timestamp.isoformat(),
                "value": m.value,
                "unit": m.unit
            }
            for m in filtered_metrics
        ]

        return MetricResponse(
            service=query.service,
            metric_type=query.metric_type,
            timestamp=datetime.now(),
            values=values,
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metrics: {str(e)}"
        )


@app.get("/reports", tags=["Reports"])
async def list_reports(
    limit: int = 10,
    offset: int = 0,
    api_key: APIKey = Depends(get_api_key)
):
    """
    List available reports

    Requires: READ permission
    """
    try:
        reports_dir = Path("reports/generated")

        if not reports_dir.exists():
            return {"reports": [], "total": 0, "limit": limit, "offset": offset}

        # Get all report files
        all_reports = []
        for file_path in reports_dir.glob("*.html"):
            stats = file_path.stat()
            all_reports.append({
                "filename": file_path.name,
                "size_kb": round(stats.st_size / 1024, 2),
                "created_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "path": str(file_path)
            })

        # Sort by creation time (newest first)
        all_reports.sort(key=lambda x: x["created_at"], reverse=True)

        # Paginate
        total = len(all_reports)
        paginated_reports = all_reports[offset:offset + limit]

        return {
            "reports": paginated_reports,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reports: {str(e)}"
        )


async def generate_report_task(
    task_id: str,
    request: ReportGenerationRequest
):
    """Background task to generate report"""
    try:
        task_storage[task_id]["status"] = "in_progress"
        task_storage[task_id]["progress"] = 10
        task_storage[task_id]["message"] = "Initializing report generation..."

        # Initialize report system
        sre_system = EnhancedSREReportSystem(app_name=request.application_name)

        task_storage[task_id]["progress"] = 30
        task_storage[task_id]["message"] = "Collecting metrics data..."

        # Generate report
        if request.report_type == "incident" and request.incident_time:
            report_paths = sre_system.generate_full_report_suite(
                application_name=request.application_name,
                services=request.services,
                incident_time=request.incident_time,
                incident_duration=request.incident_duration
            )
        else:
            report_paths = sre_system.generate_full_report_suite(
                application_name=request.application_name,
                services=request.services
            )

        task_storage[task_id]["progress"] = 100
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["message"] = "Report generated successfully"
        task_storage[task_id]["report_paths"] = report_paths

        # Determine report URL based on format
        if request.output_format == "html" and report_paths.get("html_report"):
            task_storage[task_id]["report_url"] = f"/reports/download/{task_id}/html"
        elif request.output_format == "pdf" and report_paths.get("pdf_report"):
            task_storage[task_id]["report_url"] = f"/reports/download/{task_id}/pdf"
        elif request.output_format == "json" and report_paths.get("json_data"):
            task_storage[task_id]["report_url"] = f"/reports/download/{task_id}/json"

    except Exception as e:
        logger.error(f"Report generation failed for task {task_id}: {e}")
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["message"] = f"Report generation failed: {str(e)}"


@app.post("/reports/generate", response_model=ReportResponse, tags=["Reports"])
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(require_role(Role.WRITE))
):
    """
    Generate a new SRE report

    Requires: WRITE permission
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())

        # Initialize task status
        task_storage[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "Report generation queued",
            "created_at": datetime.now(),
            "request": request.dict()
        }

        # Start background task
        background_tasks.add_task(generate_report_task, task_id, request)

        return ReportResponse(
            task_id=task_id,
            status="pending",
            message="Report generation started",
            estimated_time=60  # Estimate 60 seconds
        )

    except Exception as e:
        logger.error(f"Error starting report generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start report generation: {str(e)}"
        )


@app.get("/reports/status/{task_id}", response_model=ReportStatusResponse, tags=["Reports"])
async def get_report_status(
    task_id: str,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Get report generation status

    Requires: READ permission
    """
    task = task_storage.get(task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    return ReportStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        report_url=task.get("report_url"),
        created_at=task["created_at"]
    )


@app.get("/reports/download/{task_id}/{format}", tags=["Reports"])
async def download_report(
    task_id: str,
    format: str,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Download generated report

    Requires: READ permission

    Args:
        task_id: Report generation task ID
        format: Report format (html, pdf, json)
    """
    task = task_storage.get(task_id)

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )

    if task["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report not ready. Current status: {task['status']}"
        )

    report_paths = task.get("report_paths", {})

    if format == "html" and report_paths.get("html_report"):
        file_path = Path(report_paths["html_report"])
        if file_path.exists():
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="text/html"
            )
    elif format == "pdf" and report_paths.get("pdf_report"):
        file_path = Path(report_paths["pdf_report"])
        if file_path.exists():
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="application/pdf"
            )
    elif format == "json" and report_paths.get("json_data"):
        file_path = Path(report_paths["json_data"])
        if file_path.exists():
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="application/json"
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Report file not found for format: {format}"
    )


# In-memory incident storage (replace with database in production)
incidents_db: Dict[str, Dict] = {}


@app.post("/incidents", response_model=IncidentResponse, tags=["Incidents"])
async def create_incident(
    incident: IncidentCreate,
    api_key: APIKey = Depends(require_role(Role.WRITE))
):
    """
    Report a new incident

    Requires: WRITE permission
    """
    try:
        incident_id = str(uuid.uuid4())

        # Calculate duration if end time is provided
        duration = None
        if incident.end_time:
            duration = (incident.end_time - incident.start_time).total_seconds() / 3600

        incident_data = {
            "incident_id": incident_id,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity,
            "affected_services": incident.affected_services,
            "start_time": incident.start_time,
            "end_time": incident.end_time,
            "duration": duration,
            "status": "resolved" if incident.end_time else "active",
            "tags": incident.tags or [],
            "created_at": datetime.now()
        }

        incidents_db[incident_id] = incident_data

        return IncidentResponse(**incident_data)

    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create incident: {str(e)}"
        )


@app.get("/incidents", tags=["Incidents"])
async def list_incidents(
    status_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    service_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: APIKey = Depends(get_api_key)
):
    """
    List incidents with optional filtering

    Requires: READ permission
    """
    try:
        # Filter incidents
        filtered_incidents = list(incidents_db.values())

        if status_filter:
            filtered_incidents = [
                i for i in filtered_incidents
                if i["status"] == status_filter
            ]

        if severity_filter:
            filtered_incidents = [
                i for i in filtered_incidents
                if i["severity"] == severity_filter
            ]

        if service_filter:
            filtered_incidents = [
                i for i in filtered_incidents
                if service_filter in i["affected_services"]
            ]

        # Sort by start time (newest first)
        filtered_incidents.sort(key=lambda x: x["start_time"], reverse=True)

        # Paginate
        total = len(filtered_incidents)
        paginated_incidents = filtered_incidents[offset:offset + limit]

        return {
            "incidents": paginated_incidents,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing incidents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list incidents: {str(e)}"
        )


@app.get("/incidents/{incident_id}", response_model=IncidentResponse, tags=["Incidents"])
async def get_incident(
    incident_id: str,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Get incident details by ID

    Requires: READ permission
    """
    incident = incidents_db.get(incident_id)

    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found"
        )

    return IncidentResponse(**incident)


@app.post("/anomalies/detect", response_model=AnomalyResponse, tags=["Anomaly Detection"])
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Detect anomalies in service metrics

    Requires: READ permission
    """
    try:
        # Fetch historical metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=request.lookback_hours)

        prometheus = PrometheusIntegration()
        metrics = prometheus.get_service_metrics(
            service_name=request.service,
            start_time=start_time,
            end_time=end_time
        )

        # Filter by metric type
        filtered_metrics = [m for m in metrics if m.metric_type == request.metric_type]

        if not filtered_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for service '{request.service}' with type '{request.metric_type}'"
            )

        # Initialize anomaly monitor
        monitor = SLOAnomalyMonitor()

        # Analyze metrics
        anomaly_results = monitor.analyze_slo_metrics(
            slo_metrics=filtered_metrics,
            method=request.detection_method,
            sensitivity=request.sensitivity
        )

        # Format response
        anomalies = []
        for result in anomaly_results.get("anomalies", []):
            anomalies.append({
                "timestamp": result.get("timestamp"),
                "value": result.get("value"),
                "expected_range": result.get("expected_range"),
                "deviation": result.get("deviation"),
                "confidence": result.get("confidence")
            })

        return AnomalyResponse(
            service=request.service,
            metric_type=request.metric_type,
            anomalies_detected=len(anomalies),
            anomalies=anomalies,
            summary=anomaly_results.get("summary", {}),
            recommendations=anomaly_results.get("recommendations", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect anomalies: {str(e)}"
        )


# ============================================================================
# Admin Endpoints (API Key Management)
# ============================================================================

@app.post("/admin/api-keys", response_model=APIKeyResponse, tags=["Admin"])
async def create_api_key(
    request: APIKeyCreateRequest,
    api_key: APIKey = Depends(require_role(Role.ADMIN))
):
    """
    Create a new API key

    Requires: ADMIN permission
    """
    try:
        # Validate role
        role_map = {
            "admin": Role.ADMIN,
            "write": Role.WRITE,
            "read": Role.READ
        }

        role = role_map.get(request.role.lower())
        if not role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role. Must be: admin, write, or read"
            )

        # Generate key
        raw_key, new_key = key_manager.generate_key(
            name=request.name,
            role=role,
            rate_limit=request.rate_limit,
            metadata=request.metadata
        )

        return APIKeyResponse(
            key_id=new_key.key_id,
            api_key=raw_key,  # Only returned on creation
            name=new_key.name,
            role=new_key.role.value,
            rate_limit=new_key.rate_limit,
            created_at=new_key.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )


@app.get("/admin/api-keys", tags=["Admin"])
async def list_api_keys(
    api_key: APIKey = Depends(require_role(Role.ADMIN))
):
    """
    List all API keys

    Requires: ADMIN permission
    """
    return {"api_keys": key_manager.list_keys()}


@app.delete("/admin/api-keys/{key_id}", tags=["Admin"])
async def revoke_api_key(
    key_id: str,
    api_key: APIKey = Depends(require_role(Role.ADMIN))
):
    """
    Revoke an API key

    Requires: ADMIN permission
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
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.flask.debug else "An unexpected error occurred"
        }
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting SRE Analytics API...")
    logger.info(f"📚 API documentation available at: /docs")

    # Create default admin key if none exists
    if not key_manager.list_keys():
        raw_key, admin_key = key_manager.generate_key(
            name="Default Admin Key",
            role=Role.ADMIN,
            rate_limit=1000,
            metadata={"created_by": "system", "default": True}
        )
        logger.info(f"🔑 Default admin API key created: {raw_key}")
        logger.info("⚠️  Save this key securely - it won't be shown again!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down SRE Analytics API...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
