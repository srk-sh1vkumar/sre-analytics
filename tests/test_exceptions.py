"""
Tests for custom exception classes
"""

import pytest

from src.exceptions import (
    APIAuthenticationError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APITimeoutError,
    ConfigurationError,
    DataCollectionError,
    FileOperationError,
    FileWriteError,
    InputValidationError,
    InvalidConfigError,
    LLMError,
    LLMRateLimitError,
    MetricCollectionError,
    MissingConfigError,
    PDFGenerationError,
    ReportGenerationError,
    SREAnalyticsError,
    ValidationError,
    get_error_chain,
    wrap_exception,
)


class TestBaseException:
    """Tests for SREAnalyticsError base class"""

    def test_basic_exception(self):
        """Test creating basic exception"""
        exc = SREAnalyticsError("Test error")
        assert str(exc) == "Test error"
        assert exc.message == "Test error"
        assert exc.context == {}

    def test_exception_with_context(self):
        """Test exception with context"""
        context = {"user_id": "123", "action": "test"}
        exc = SREAnalyticsError("Test error", context=context)

        assert exc.message == "Test error"
        assert exc.context == context
        assert "user_id=123" in str(exc)
        assert "action=test" in str(exc)

    def test_exception_inheritance(self):
        """Test exception inheritance chain"""
        exc = APIConnectionError("Connection failed")
        assert isinstance(exc, APIError)
        assert isinstance(exc, SREAnalyticsError)
        assert isinstance(exc, Exception)


class TestConfigurationExceptions:
    """Tests for configuration-related exceptions"""

    def test_missing_config_error(self):
        """Test MissingConfigError"""
        exc = MissingConfigError("Required config missing", context={"config_key": "database_url"})
        assert isinstance(exc, ConfigurationError)
        assert "config_key=database_url" in str(exc)

    def test_invalid_config_error(self):
        """Test InvalidConfigError"""
        exc = InvalidConfigError("Invalid value", context={"expected": "int", "got": "str"})
        assert isinstance(exc, ConfigurationError)
        assert exc.context["expected"] == "int"


class TestAPIExceptions:
    """Tests for API-related exceptions"""

    def test_api_error_with_status_code(self):
        """Test APIError with status code"""
        exc = APIError(
            "API request failed",
            status_code=500,
            response_body='{"error": "Internal Server Error"}',
        )
        assert exc.status_code == 500
        assert exc.response_body == '{"error": "Internal Server Error"}'

    def test_api_connection_error(self):
        """Test APIConnectionError"""
        exc = APIConnectionError(
            "Failed to connect", context={"host": "api.example.com", "port": 443}
        )
        assert isinstance(exc, APIError)
        assert exc.context["host"] == "api.example.com"

    def test_api_authentication_error(self):
        """Test APIAuthenticationError"""
        exc = APIAuthenticationError("Authentication failed", status_code=401)
        assert exc.status_code == 401
        assert isinstance(exc, APIError)

    def test_api_timeout_error(self):
        """Test APITimeoutError"""
        exc = APITimeoutError("Request timed out", status_code=408, context={"timeout_seconds": 30})
        assert exc.status_code == 408
        assert exc.context["timeout_seconds"] == 30

    def test_api_rate_limit_error(self):
        """Test APIRateLimitError"""
        exc = APIRateLimitError("Rate limit exceeded", status_code=429, context={"retry_after": 60})
        assert exc.status_code == 429
        assert exc.context["retry_after"] == 60


class TestDataCollectionExceptions:
    """Tests for data collection exceptions"""

    def test_metric_collection_error(self):
        """Test MetricCollectionError"""
        exc = MetricCollectionError(
            "Failed to collect metrics", context={"service": "web-api", "metric": "latency"}
        )
        assert isinstance(exc, DataCollectionError)
        assert exc.context["service"] == "web-api"


class TestReportGenerationExceptions:
    """Tests for report generation exceptions"""

    def test_pdf_generation_error(self):
        """Test PDFGenerationError"""
        exc = PDFGenerationError(
            "PDF generation failed", context={"output_path": "/tmp/report.pdf"}
        )
        assert isinstance(exc, ReportGenerationError)
        assert exc.context["output_path"] == "/tmp/report.pdf"


class TestLLMExceptions:
    """Tests for LLM-related exceptions"""

    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError"""
        exc = LLMRateLimitError(
            "LLM rate limit exceeded", context={"provider": "openai", "retry_after": 120}
        )
        assert isinstance(exc, LLMError)
        assert exc.context["provider"] == "openai"


class TestFileOperationExceptions:
    """Tests for file operation exceptions"""

    def test_file_write_error(self):
        """Test FileWriteError"""
        exc = FileWriteError(
            "Failed to write file", context={"path": "/tmp/test.json", "error": "Permission denied"}
        )
        assert isinstance(exc, FileOperationError)
        assert exc.context["path"] == "/tmp/test.json"


class TestValidationExceptions:
    """Tests for validation exceptions"""

    def test_input_validation_error(self):
        """Test InputValidationError"""
        exc = InputValidationError(
            "Invalid input", context={"field": "email", "value": "invalid-email"}
        )
        assert isinstance(exc, ValidationError)
        assert exc.context["field"] == "email"


class TestExceptionHelpers:
    """Tests for exception helper functions"""

    def test_wrap_exception(self):
        """Test wrap_exception helper"""
        original_exc = ValueError("Invalid value")
        wrapped_exc = wrap_exception(
            original_exc, APIError, "API validation failed", context={"field": "age"}
        )

        assert isinstance(wrapped_exc, APIError)
        assert wrapped_exc.message == "API validation failed"
        assert wrapped_exc.context["original_error"] == "Invalid value"
        assert wrapped_exc.context["original_type"] == "ValueError"
        assert wrapped_exc.context["field"] == "age"

    def test_get_error_chain_single(self):
        """Test get_error_chain with single exception"""
        exc = SREAnalyticsError("Test error")
        chain = get_error_chain(exc)

        assert len(chain) == 1
        assert chain[0] == "Test error"

    def test_get_error_chain_multiple(self):
        """Test get_error_chain with exception chain"""
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise APIError("Outer error") from e
        except APIError as final_exc:
            chain = get_error_chain(final_exc)
            assert len(chain) == 2
            assert "Outer error" in chain[0]
            assert "Inner error" in chain[1]


class TestExceptionContext:
    """Tests for exception context functionality"""

    def test_context_preservation(self):
        """Test that context is preserved through exception handling"""
        original_context = {"service": "api", "endpoint": "/metrics"}

        try:
            raise APIError("Request failed", context=original_context)
        except APIError as e:
            assert e.context == original_context
            assert e.context["service"] == "api"

    def test_context_immutability(self):
        """Test that modifying context doesn't affect original"""
        original_context = {"key": "value"}
        exc = SREAnalyticsError("Test", context=original_context)

        exc.context["new_key"] = "new_value"
        assert "new_key" not in original_context

    def test_empty_context(self):
        """Test exception with no context"""
        exc = SREAnalyticsError("Test error")
        assert exc.context == {}
        assert "Context:" not in str(exc)
