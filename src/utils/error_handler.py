"""
Error Handler Utilities

Provides decorators and utilities for consistent error handling with
retry logic, fallback strategies, and error context.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, cast

from src.exceptions import (
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APITimeoutError,
    FileOperationError,
    LLMError,
    LLMRateLimitError,
    SREAnalyticsError,
)

# Type variables for generic function signatures
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


logger = logging.getLogger(__name__)


# ============================================================================
# Retry Decorators
# ============================================================================


def retry_on_error(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    retry_on: Tuple[Type[Exception], ...] = (APIConnectionError, APITimeoutError),
    raise_on: Tuple[Type[Exception], ...] = (),
) -> Callable[[F], F]:
    """
    Decorator to retry function on specific exceptions

    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        retry_on: Tuple of exception types to retry on
        raise_on: Tuple of exception types to immediately raise

    Example:
        @retry_on_error(max_attempts=3, retry_on=(APIConnectionError,))
        def fetch_data():
            return api.get_data()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay: float = delay_seconds

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except raise_on as e:
                    # Don't retry these exceptions
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    raise
                except retry_on as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}",
                            extra={"attempts": max_attempts, "function": func.__name__},
                        )
                        break

                    logger.warning(
                        f"{func.__name__} failed on attempt {attempt}/{max_attempts}: {e}. "
                        f"Retrying in {current_delay}s...",
                        extra={"attempt": attempt, "delay": current_delay},
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier
                except Exception as e:
                    # Unexpected exception - log and re-raise
                    logger.error(
                        f"{func.__name__} failed with unexpected error: {e}", exc_info=True
                    )
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception

        return cast(F, wrapper)

    return decorator


def retry_with_fallback(
    fallback_func: Callable[..., T],
    max_attempts: int = 3,
    retry_on: Tuple[Type[Exception], ...] = (APIError, LLMError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry with fallback function on failure

    Args:
        fallback_func: Function to call if all retries fail
        max_attempts: Maximum retry attempts
        retry_on: Exception types to retry on

    Example:
        def fallback_analysis():
            return "Basic analysis"

        @retry_with_fallback(fallback_func=fallback_analysis, max_attempts=2)
        def llm_analysis():
            return llm.analyze()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    if attempt == max_attempts:
                        logger.warning(
                            f"{func.__name__} failed after {max_attempts} attempts. "
                            f"Using fallback function.",
                            extra={"error": str(e)},
                        )
                        try:
                            return fallback_func(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback function also failed: {fallback_error}", exc_info=True
                            )
                            raise

                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}")
                    time.sleep(1.0 * attempt)  # Linear backoff

            # This should never be reached but satisfies type checker
            raise RuntimeError("Unexpected state in retry_with_fallback")

        return wrapper

    return decorator


# ============================================================================
# Error Context Managers
# ============================================================================


class ErrorContext:
    """
    Context manager for adding error context and handling exceptions

    Example:
        with ErrorContext("fetching metrics", service="web-api"):
            metrics = fetch_metrics()
    """

    def __init__(self, operation: str, **context: Any) -> None:
        """
        Initialize error context

        Args:
            operation: Description of the operation
            **context: Additional context key-value pairs
        """
        self.operation = operation
        self.context = context
        self.logger = logging.getLogger(__name__)

    def __enter__(self) -> "ErrorContext":
        self.logger.debug(f"Starting: {self.operation}", extra=self.context)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if exc_type is None:
            self.logger.debug(f"Completed: {self.operation}", extra=self.context)
            return

        # Log error with full context
        self.logger.error(
            f"Error during {self.operation}: {exc_val}",
            extra={
                **self.context,
                "exception_type": exc_type.__name__,
                "operation": self.operation,
            },
            exc_info=True,
        )


# ============================================================================
# Error Handling Utilities
# ============================================================================


def safe_execute(
    func: Callable[[], T],
    default_value: Optional[T] = None,
    log_errors: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[T]:
    """
    Safely execute a function with error handling

    Args:
        func: Function to execute
        default_value: Value to return on error
        log_errors: Whether to log errors
        context: Additional context for error logging

    Returns:
        Function result or default_value on error

    Example:
        result = safe_execute(
            lambda: risky_operation(),
            default_value=[],
            context={'operation': 'data_fetch'}
        )
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error in safe_execute: {e}", extra=context or {}, exc_info=True)
        return default_value


def handle_api_error(
    response_status: int, response_body: str, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convert HTTP status codes to appropriate API exceptions

    Args:
        response_status: HTTP status code
        response_body: Response body
        context: Additional context

    Raises:
        Appropriate APIError subclass
    """
    from src.config.constants import (
        HTTP_FORBIDDEN,
        HTTP_NOT_FOUND,
        HTTP_SERVER_ERROR,
        HTTP_SERVICE_UNAVAILABLE,
        HTTP_UNAUTHORIZED,
    )

    ctx = context or {}
    ctx["status_code"] = response_status

    if response_status == HTTP_UNAUTHORIZED:
        from src.exceptions import APIAuthenticationError

        raise APIAuthenticationError(
            "API authentication failed",
            status_code=response_status,
            response_body=response_body,
            context=ctx,
        )
    elif response_status == HTTP_FORBIDDEN:
        from src.exceptions import APIAuthorizationError

        raise APIAuthorizationError(
            "API authorization failed - insufficient permissions",
            status_code=response_status,
            response_body=response_body,
            context=ctx,
        )
    elif response_status == HTTP_NOT_FOUND:
        from src.exceptions import ResourceNotFoundError

        raise ResourceNotFoundError("API resource not found", context=ctx)
    elif response_status == 429:  # Rate limit
        raise APIRateLimitError(
            "API rate limit exceeded",
            status_code=response_status,
            response_body=response_body,
            context=ctx,
        )
    elif response_status == 408:  # Request timeout
        raise APITimeoutError(
            "API request timed out",
            status_code=response_status,
            response_body=response_body,
            context=ctx,
        )
    elif response_status >= HTTP_SERVER_ERROR:
        from src.exceptions import APIResponseError

        raise APIResponseError(
            f"API server error: {response_status}",
            status_code=response_status,
            response_body=response_body,
            context=ctx,
        )
    else:
        from src.exceptions import APIResponseError

        raise APIResponseError(
            f"API request failed with status {response_status}",
            status_code=response_status,
            response_body=response_body,
            context=ctx,
        )


def log_exception_with_context(
    exc: Exception,
    logger_instance: logging.Logger,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log exception with full context

    Args:
        exc: Exception to log
        logger_instance: Logger instance to use
        operation: Description of operation that failed
        context: Additional context
        level: Logging level
    """
    ctx = context or {}

    if isinstance(exc, SREAnalyticsError):
        # Custom exception with built-in context
        ctx.update(exc.context)

    logger_instance.log(
        level,
        f"Operation '{operation}' failed: {exc}",
        extra={**ctx, "exception_type": type(exc).__name__, "operation": operation},
        exc_info=True,
    )


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_required_fields(
    data: Dict[str, Any], required_fields: List[str], context_name: str = "data"
) -> None:
    """
    Validate that required fields are present

    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        context_name: Name for error context

    Raises:
        ValidationError: If required fields are missing
    """
    from src.exceptions import InputValidationError

    missing_fields = [
        field for field in required_fields if field not in data or data[field] is None
    ]

    if missing_fields:
        raise InputValidationError(
            f"Missing required fields in {context_name}",
            context={"missing_fields": missing_fields, "provided_fields": list(data.keys())},
        )


def validate_config_value(
    config_dict: Dict[str, Any],
    key: str,
    expected_type: Optional[Type[Any]] = None,
    required: bool = True,
    default: Optional[Any] = None,
) -> Any:
    """
    Validate and retrieve configuration value

    Args:
        config_dict: Configuration dictionary
        key: Configuration key
        expected_type: Expected value type
        required: Whether the key is required
        default: Default value if not required

    Returns:
        Configuration value

    Raises:
        MissingConfigError: If required config is missing
        InvalidConfigError: If config value is invalid type
    """
    from src.exceptions import InvalidConfigError, MissingConfigError

    if key not in config_dict:
        if required:
            raise MissingConfigError(
                f"Required configuration '{key}' is missing",
                context={"config_dict_keys": list(config_dict.keys())},
            )
        return default

    value = config_dict[key]

    if expected_type and not isinstance(value, expected_type):
        raise InvalidConfigError(
            f"Configuration '{key}' has invalid type",
            context={
                "expected_type": expected_type.__name__,
                "actual_type": type(value).__name__,
                "value": str(value),
            },
        )

    return value
