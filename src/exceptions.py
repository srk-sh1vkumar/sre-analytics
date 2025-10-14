"""
Custom Exception Classes for SRE Analytics

Provides a comprehensive hierarchy of exceptions for better error handling
and debugging across the application.
"""

from typing import Any, Dict, List, Optional, Type, cast

# ============================================================================
# Base Exception Classes
# ============================================================================


class SREAnalyticsError(Exception):
    """Base exception for all SRE Analytics errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional context

        Args:
            message: Human-readable error message
            context: Additional context information for debugging
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationError(SREAnalyticsError):
    """Base class for configuration-related errors"""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing"""

    pass


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid"""

    pass


class ConfigLoadError(ConfigurationError):
    """Failed to load configuration file"""

    pass


# ============================================================================
# API and Network Exceptions
# ============================================================================


class APIError(SREAnalyticsError):
    """Base class for API-related errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message, context)


class APIConnectionError(APIError):
    """Failed to connect to API"""

    pass


class APIAuthenticationError(APIError):
    """API authentication failed"""

    pass


class APIAuthorizationError(APIError):
    """API authorization failed (insufficient permissions)"""

    pass


class APITimeoutError(APIError):
    """API request timed out"""

    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded"""

    pass


class APIResponseError(APIError):
    """API returned an error response"""

    pass


class APIDataError(APIError):
    """API returned invalid or unexpected data"""

    pass


# ============================================================================
# Data Collection Exceptions
# ============================================================================


class DataCollectionError(SREAnalyticsError):
    """Base class for data collection errors"""

    pass


class MetricCollectionError(DataCollectionError):
    """Failed to collect metrics"""

    pass


class IncidentCollectionError(DataCollectionError):
    """Failed to collect incident data"""

    pass


class ServiceDiscoveryError(DataCollectionError):
    """Failed to discover services"""

    pass


# ============================================================================
# Data Processing Exceptions
# ============================================================================


class DataProcessingError(SREAnalyticsError):
    """Base class for data processing errors"""

    pass


class DataValidationError(DataProcessingError):
    """Data validation failed"""

    pass


class DataTransformationError(DataProcessingError):
    """Data transformation failed"""

    pass


class DataAggregationError(DataProcessingError):
    """Data aggregation failed"""

    pass


# ============================================================================
# Report Generation Exceptions
# ============================================================================


class ReportGenerationError(SREAnalyticsError):
    """Base class for report generation errors"""

    pass


class TemplateError(ReportGenerationError):
    """Template rendering failed"""

    pass


class PDFGenerationError(ReportGenerationError):
    """PDF generation failed"""

    pass


class ChartGenerationError(ReportGenerationError):
    """Chart/visualization generation failed"""

    pass


class HTMLGenerationError(ReportGenerationError):
    """HTML generation failed"""

    pass


# ============================================================================
# LLM and AI Exceptions
# ============================================================================


class LLMError(SREAnalyticsError):
    """Base class for LLM-related errors"""

    pass


class LLMConnectionError(LLMError):
    """Failed to connect to LLM service"""

    pass


class LLMAuthenticationError(LLMError):
    """LLM authentication failed"""

    pass


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded"""

    pass


class LLMResponseError(LLMError):
    """LLM returned invalid response"""

    pass


class LLMContextLengthError(LLMError):
    """Input exceeds LLM context length"""

    pass


# ============================================================================
# File and I/O Exceptions
# ============================================================================


class FileOperationError(SREAnalyticsError):
    """Base class for file operation errors"""

    pass


class FileNotFoundError(FileOperationError):
    """Required file not found"""

    pass


class FileReadError(FileOperationError):
    """Failed to read file"""

    pass


class FileWriteError(FileOperationError):
    """Failed to write file"""

    pass


class FilePermissionError(FileOperationError):
    """Insufficient permissions for file operation"""

    pass


# ============================================================================
# Database Exceptions
# ============================================================================


class DatabaseError(SREAnalyticsError):
    """Base class for database errors"""

    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database"""

    pass


class DatabaseQueryError(DatabaseError):
    """Database query failed"""

    pass


class DatabaseTransactionError(DatabaseError):
    """Database transaction failed"""

    pass


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationError(SREAnalyticsError):
    """Base class for validation errors"""

    pass


class SchemaValidationError(ValidationError):
    """Data does not match expected schema"""

    pass


class InputValidationError(ValidationError):
    """User input validation failed"""

    pass


class MetricValidationError(ValidationError):
    """Metric value validation failed"""

    pass


# ============================================================================
# Resource Exceptions
# ============================================================================


class ResourceError(SREAnalyticsError):
    """Base class for resource-related errors"""

    pass


class ResourceNotFoundError(ResourceError):
    """Required resource not found"""

    pass


class ResourceExhaustedError(ResourceError):
    """Resource limit exceeded"""

    pass


class ResourceLockError(ResourceError):
    """Failed to acquire resource lock"""

    pass


# ============================================================================
# Business Logic Exceptions
# ============================================================================


class BusinessLogicError(SREAnalyticsError):
    """Base class for business logic errors"""

    pass


class SLOViolationError(BusinessLogicError):
    """SLO threshold violated"""

    pass


class SLAViolationError(BusinessLogicError):
    """SLA threshold violated"""

    pass


class ThresholdExceededError(BusinessLogicError):
    """Threshold exceeded"""

    pass


# ============================================================================
# Dependency Exceptions
# ============================================================================


class DependencyError(SREAnalyticsError):
    """Base class for dependency errors"""

    pass


class MissingDependencyError(DependencyError):
    """Required dependency is missing"""

    pass


class IncompatibleDependencyError(DependencyError):
    """Dependency version is incompatible"""

    pass


# ============================================================================
# Helper Functions
# ============================================================================


def wrap_exception(
    exc: Exception, new_type: type, message: str, context: Optional[Dict[str, Any]] = None
) -> SREAnalyticsError:
    """
    Wrap a standard exception in a custom exception type

    Args:
        exc: Original exception
        new_type: Custom exception class to wrap in
        message: Custom error message
        context: Additional context

    Returns:
        Custom exception instance
    """
    ctx = context or {}
    ctx["original_error"] = str(exc)
    ctx["original_type"] = type(exc).__name__
    return cast(SREAnalyticsError, new_type(message, context=ctx))


def get_error_chain(exc: Exception) -> List[str]:
    """
    Get the chain of exceptions that led to this error

    Args:
        exc: Exception to trace

    Returns:
        List of exception messages in the chain
    """
    chain: List[str] = []
    current: Optional[BaseException] = exc
    while current is not None:
        chain.append(str(current))
        current = current.__cause__ if hasattr(current, "__cause__") else None
    return chain
