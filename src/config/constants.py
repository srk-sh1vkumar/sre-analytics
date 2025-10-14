"""
Centralized Constants Module

All magic numbers, hardcoded values, and configuration defaults are defined here
for better maintainability and consistency across the codebase.
"""

from typing import Final, Tuple

# ============================================================================
# TIME AND DATE CONSTANTS
# ============================================================================

DAYS_IN_MONTH: Final[int] = 30
DEFAULT_TREND_DAYS: Final[int] = 30
DEFAULT_INCIDENT_DURATION_HOURS: Final[float] = 1.0

# ============================================================================
# SLO/SLA THRESHOLDS AND PERCENTILES
# ============================================================================

# Availability thresholds
AVAILABILITY_MIN: Final[float] = 99.5
AVAILABILITY_TARGET: Final[float] = 99.9
AVAILABILITY_MAX: Final[float] = 99.99

# Latency thresholds (milliseconds)
LATENCY_P95_TARGET_MS: Final[int] = 200
LATENCY_P99_TARGET_MS: Final[int] = 300
LATENCY_P95_WARNING_MS: Final[int] = 400
LATENCY_P95_CRITICAL_MS: Final[int] = 500

# Error rate thresholds (percentage)
ERROR_RATE_TARGET: Final[float] = 1.0
ERROR_RATE_WARNING: Final[float] = 2.0
ERROR_RATE_CRITICAL: Final[float] = 5.0

# Compliance thresholds
COMPLIANCE_THRESHOLD_COMPLIANT: Final[float] = 0.999  # 99.9%
COMPLIANCE_THRESHOLD_AT_RISK: Final[float] = 0.95  # 95%

# ============================================================================
# API AND NETWORK CONSTANTS
# ============================================================================

# Timeouts (seconds)
API_TIMEOUT_DEFAULT: Final[int] = 30
API_TIMEOUT_SHORT: Final[int] = 10
API_TIMEOUT_LONG: Final[int] = 60
API_TIMEOUT_CONNECTION_TEST: Final[int] = 5

# Retry configuration
API_RETRY_ATTEMPTS: Final[int] = 3
API_RETRY_DELAY_SECONDS: Final[int] = 5

# HTTP Status Codes
HTTP_OK: Final[int] = 200
HTTP_CREATED: Final[int] = 201
HTTP_BAD_REQUEST: Final[int] = 400
HTTP_UNAUTHORIZED: Final[int] = 401
HTTP_FORBIDDEN: Final[int] = 403
HTTP_NOT_FOUND: Final[int] = 404
HTTP_SERVER_ERROR: Final[int] = 500
HTTP_SERVICE_UNAVAILABLE: Final[int] = 503

# ============================================================================
# PORT NUMBERS
# ============================================================================

PORT_FLASK_APP: Final[int] = 5001
PORT_REPORT_WEB: Final[int] = 8080
PORT_PROMETHEUS: Final[int] = 9090
PORT_GRAFANA: Final[int] = 3000
PORT_EUREKA: Final[int] = 8761

# ============================================================================
# PDF GENERATION CONSTANTS
# ============================================================================

# Page sizes
PDF_PAGE_WIDTH: Final[int] = 595  # A4 width in points
PDF_PAGE_HEIGHT: Final[int] = 842  # A4 height in points

# Margins (points)
PDF_MARGIN_LEFT: Final[int] = 50
PDF_MARGIN_RIGHT: Final[int] = 50
PDF_MARGIN_TOP: Final[int] = 50
PDF_MARGIN_BOTTOM: Final[int] = 50

# Font sizes
PDF_FONT_SIZE_TITLE: Final[int] = 24
PDF_FONT_SIZE_HEADING: Final[int] = 16
PDF_FONT_SIZE_SUBHEADING: Final[int] = 14
PDF_FONT_SIZE_BODY: Final[int] = 11
PDF_FONT_SIZE_SMALL: Final[int] = 9

# Colors (RGB tuples)
PDF_COLOR_PRIMARY: Final[Tuple[int, int, int]] = (0, 122, 204)  # #007acc
PDF_COLOR_SUCCESS: Final[Tuple[int, int, int]] = (40, 167, 69)  # #28a745
PDF_COLOR_WARNING: Final[Tuple[int, int, int]] = (255, 193, 7)  # #ffc107
PDF_COLOR_DANGER: Final[Tuple[int, int, int]] = (220, 53, 69)  # #dc3545
PDF_COLOR_GRAY_LIGHT: Final[Tuple[float, float, float]] = (0.95, 0.95, 0.95)
PDF_COLOR_GRAY_MEDIUM: Final[Tuple[float, float, float]] = (0.75, 0.75, 0.75)

# ============================================================================
# CHART AND VISUALIZATION CONSTANTS
# ============================================================================

# Chart dimensions (pixels)
CHART_WIDTH: Final[int] = 1200
CHART_HEIGHT: Final[int] = 600
CHART_DPI: Final[int] = 100

# Chart colors (hex)
CHART_COLOR_PRIMARY: Final[str] = "#007acc"
CHART_COLOR_SUCCESS: Final[str] = "#28a745"
CHART_COLOR_WARNING: Final[str] = "#ffc107"
CHART_COLOR_DANGER: Final[str] = "#dc3545"
CHART_COLOR_INFO: Final[str] = "#17a2b8"

# ============================================================================
# METRICS AND MONITORING CONSTANTS
# ============================================================================

# Metric names
METRIC_AVAILABILITY: Final[str] = "availability"
METRIC_LATENCY_P95: Final[str] = "latency_p95"
METRIC_LATENCY_P99: Final[str] = "latency_p99"
METRIC_ERROR_RATE: Final[str] = "error_rate"
METRIC_THROUGHPUT: Final[str] = "throughput"
METRIC_CPU_USAGE: Final[str] = "cpu_usage"
METRIC_MEMORY_USAGE: Final[str] = "memory_usage"

# Metric units
UNIT_PERCENTAGE: Final[str] = "%"
UNIT_MILLISECONDS: Final[str] = "ms"
UNIT_SECONDS: Final[str] = "s"
UNIT_REQUESTS_PER_MINUTE: Final[str] = "req/min"

# ============================================================================
# INCIDENT SEVERITY LEVELS
# ============================================================================

SEVERITY_CRITICAL: Final[str] = "Critical"
SEVERITY_HIGH: Final[str] = "High"
SEVERITY_MEDIUM: Final[str] = "Medium"
SEVERITY_LOW: Final[str] = "Low"

# ============================================================================
# SLO STATUS CATEGORIES
# ============================================================================

STATUS_COMPLIANT: Final[str] = "compliant"
STATUS_AT_RISK: Final[str] = "at_risk"
STATUS_BREACHED: Final[str] = "breached"

# ============================================================================
# HEALTH STATUS
# ============================================================================

HEALTH_HEALTHY: Final[str] = "Healthy"
HEALTH_DEGRADED: Final[str] = "Degraded"
HEALTH_UNHEALTHY: Final[str] = "Unhealthy"

# ============================================================================
# FILE AND DIRECTORY PATHS
# ============================================================================

DEFAULT_REPORT_OUTPUT_DIR: Final[str] = "reports/generated"
DEFAULT_CONFIG_DIR: Final[str] = "config"
DEFAULT_LOGS_DIR: Final[str] = "logs"

# ============================================================================
# DATA GENERATION AND SIMULATION
# ============================================================================

# Random variation ranges
NOISE_STANDARD_DEVIATION: Final[float] = 0.5
AVAILABILITY_NOISE_STD: Final[float] = 0.1
LATENCY_NOISE_STD: Final[float] = 15.0
ERROR_RATE_NOISE_STD: Final[float] = 0.3

# Degradation factors
DEGRADATION_FACTOR_MIN: Final[float] = 1.0
DEGRADATION_FACTOR_MAX: Final[float] = 3.0

# ============================================================================
# REPORT FORMATTING
# ============================================================================

# Maximum content lengths
MAX_DESCRIPTION_LENGTH: Final[int] = 500
MAX_RECOMMENDATION_LENGTH: Final[int] = 1000

# Grid and layout
SUMMARY_GRID_MIN_WIDTH: Final[int] = 200
TABLE_BORDER_WIDTH: Final[int] = 1

# ============================================================================
# LLM AND AI CONSTANTS
# ============================================================================

LLM_PROVIDER_OPENAI: Final[str] = "openai"
LLM_PROVIDER_ANTHROPIC: Final[str] = "anthropic"

# Model names
LLM_MODEL_GPT4: Final[str] = "gpt-4"
LLM_MODEL_GPT35: Final[str] = "gpt-3.5-turbo"
LLM_MODEL_CLAUDE_OPUS: Final[str] = "claude-3-opus-20240229"
LLM_MODEL_CLAUDE_SONNET: Final[str] = "claude-3-5-sonnet-20241022"

# ============================================================================
# BROWSER PDF CONSTANTS
# ============================================================================

BROWSER_PDF_TIMEOUT_MS: Final[int] = 30000
BROWSER_PDF_WAIT_FOR_SELECTOR: Final[str] = "body"
BROWSER_PDF_MARGIN_TOP: Final[str] = "20mm"
BROWSER_PDF_MARGIN_BOTTOM: Final[str] = "20mm"
BROWSER_PDF_MARGIN_LEFT: Final[str] = "15mm"
BROWSER_PDF_MARGIN_RIGHT: Final[str] = "15mm"

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

LOG_LEVEL_DEBUG: Final[str] = "DEBUG"
LOG_LEVEL_INFO: Final[str] = "INFO"
LOG_LEVEL_WARNING: Final[str] = "WARNING"
LOG_LEVEL_ERROR: Final[str] = "ERROR"
LOG_LEVEL_CRITICAL: Final[str] = "CRITICAL"

# ============================================================================
# CACHE AND PERFORMANCE
# ============================================================================

CACHE_TTL_SECONDS: Final[int] = 300  # 5 minutes
MAX_CONCURRENT_REQUESTS: Final[int] = 10

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

MIN_SERVICE_NAME_LENGTH: Final[int] = 1
MAX_SERVICE_NAME_LENGTH: Final[int] = 100
MIN_METRIC_VALUE: Final[float] = 0.0
MAX_PERCENTAGE: Final[float] = 100.0
