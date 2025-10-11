"""
Tests for error handler utilities
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.utils.error_handler import (
    retry_on_error, retry_with_fallback, ErrorContext,
    safe_execute, handle_api_error, log_exception_with_context,
    validate_required_fields, validate_config_value
)
from src.exceptions import (
    APIConnectionError, APITimeoutError, APIAuthenticationError,
    APIAuthorizationError, ResourceNotFoundError, APIRateLimitError,
    APIResponseError, InputValidationError, MissingConfigError,
    InvalidConfigError
)


class TestRetryDecorator:
    """Tests for retry_on_error decorator"""

    def test_retry_success_on_first_attempt(self):
        """Test function succeeds on first attempt"""
        mock_func = Mock(return_value="success")

        @retry_on_error(max_attempts=3)
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_success_on_second_attempt(self):
        """Test function succeeds on second attempt after failure"""
        mock_func = Mock(side_effect=[APIConnectionError("Failed"), "success"])

        @retry_on_error(max_attempts=3, delay_seconds=0.01, retry_on=(APIConnectionError,))
        def test_func():
            return mock_func()

        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 2

    def test_retry_exhausted(self):
        """Test all retry attempts exhausted"""
        mock_func = Mock(side_effect=APIConnectionError("Failed"))

        @retry_on_error(max_attempts=3, delay_seconds=0.01, retry_on=(APIConnectionError,))
        def test_func():
            return mock_func()

        with pytest.raises(APIConnectionError):
            test_func()

        assert mock_func.call_count == 3

    def test_retry_with_backoff(self):
        """Test exponential backoff"""
        call_times = []

        @retry_on_error(max_attempts=3, delay_seconds=0.1, backoff_multiplier=2.0,
                       retry_on=(APIConnectionError,))
        def test_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise APIConnectionError("Failed")
            return "success"

        result = test_func()
        assert result == "success"

        # Check that delays increase (with some tolerance)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1 * 1.5  # Should be roughly 2x with backoff

    def test_retry_non_retryable_exception(self):
        """Test non-retryable exception is immediately raised"""
        mock_func = Mock(side_effect=APIAuthenticationError("Auth failed"))

        @retry_on_error(max_attempts=3, delay_seconds=0.01,
                       retry_on=(APIConnectionError,),
                       raise_on=(APIAuthenticationError,))
        def test_func():
            return mock_func()

        with pytest.raises(APIAuthenticationError):
            test_func()

        assert mock_func.call_count == 1  # No retries


class TestRetryWithFallback:
    """Tests for retry_with_fallback decorator"""

    def test_fallback_on_failure(self):
        """Test fallback function is called on failure"""
        main_func = Mock(side_effect=APIConnectionError("Failed"))
        fallback_func = Mock(return_value="fallback_result")

        @retry_with_fallback(fallback_func=fallback_func, max_attempts=2)
        def test_func():
            return main_func()

        result = test_func()
        assert result == "fallback_result"
        assert main_func.call_count == 2
        assert fallback_func.call_count == 1

    def test_no_fallback_on_success(self):
        """Test fallback is not called when main function succeeds"""
        main_func = Mock(return_value="success")
        fallback_func = Mock(return_value="fallback")

        @retry_with_fallback(fallback_func=fallback_func, max_attempts=2)
        def test_func():
            return main_func()

        result = test_func()
        assert result == "success"
        assert fallback_func.call_count == 0


class TestErrorContext:
    """Tests for ErrorContext context manager"""

    def test_error_context_success(self, caplog):
        """Test ErrorContext with successful operation"""
        with ErrorContext("test operation", user="test_user"):
            result = "success"

        assert "Starting: test operation" in caplog.text
        assert "Completed: test operation" in caplog.text

    def test_error_context_with_exception(self, caplog):
        """Test ErrorContext with exception"""
        with pytest.raises(ValueError):
            with ErrorContext("test operation", service="api"):
                raise ValueError("Test error")

        assert "Error during test operation" in caplog.text
        assert "ValueError" in caplog.text

    def test_error_context_captures_context(self, caplog):
        """Test ErrorContext captures additional context"""
        with pytest.raises(RuntimeError):
            with ErrorContext("processing", user_id="123", action="update"):
                raise RuntimeError("Processing failed")

        assert "user_id" in caplog.text or "123" in caplog.text


class TestSafeExecute:
    """Tests for safe_execute utility"""

    def test_safe_execute_success(self):
        """Test safe_execute with successful function"""
        result = safe_execute(lambda: "success", default_value="default")
        assert result == "success"

    def test_safe_execute_with_error(self):
        """Test safe_execute returns default on error"""
        def failing_func():
            raise ValueError("Failed")

        result = safe_execute(failing_func, default_value="default")
        assert result == "default"

    def test_safe_execute_logs_errors(self, caplog):
        """Test safe_execute logs errors"""
        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func, default_value=None, log_errors=True)
        assert "Error in safe_execute" in caplog.text

    def test_safe_execute_no_logging(self, caplog):
        """Test safe_execute without logging"""
        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func, default_value=None, log_errors=False)
        assert "Error in safe_execute" not in caplog.text


class TestHandleApiError:
    """Tests for handle_api_error utility"""

    def test_handle_401_error(self):
        """Test handling 401 Unauthorized"""
        with pytest.raises(APIAuthenticationError) as exc_info:
            handle_api_error(401, "Unauthorized", {'endpoint': '/api/data'})

        assert exc_info.value.status_code == 401
        assert exc_info.value.context['endpoint'] == '/api/data'

    def test_handle_403_error(self):
        """Test handling 403 Forbidden"""
        with pytest.raises(APIAuthorizationError) as exc_info:
            handle_api_error(403, "Forbidden")

        assert exc_info.value.status_code == 403

    def test_handle_404_error(self):
        """Test handling 404 Not Found"""
        with pytest.raises(ResourceNotFoundError):
            handle_api_error(404, "Not Found")

    def test_handle_429_error(self):
        """Test handling 429 Rate Limit"""
        with pytest.raises(APIRateLimitError) as exc_info:
            handle_api_error(429, "Too Many Requests")

        assert exc_info.value.status_code == 429

    def test_handle_408_error(self):
        """Test handling 408 Timeout"""
        with pytest.raises(APITimeoutError) as exc_info:
            handle_api_error(408, "Request Timeout")

        assert exc_info.value.status_code == 408

    def test_handle_500_error(self):
        """Test handling 500 Server Error"""
        with pytest.raises(APIResponseError) as exc_info:
            handle_api_error(500, "Internal Server Error")

        assert exc_info.value.status_code == 500

    def test_handle_generic_error(self):
        """Test handling other error codes"""
        with pytest.raises(APIResponseError) as exc_info:
            handle_api_error(418, "I'm a teapot")

        assert exc_info.value.status_code == 418


class TestValidateRequiredFields:
    """Tests for validate_required_fields utility"""

    def test_validate_all_fields_present(self):
        """Test validation passes when all fields present"""
        data = {'name': 'John', 'age': 30, 'email': 'john@example.com'}
        required = ['name', 'age', 'email']

        # Should not raise
        validate_required_fields(data, required)

    def test_validate_missing_fields(self):
        """Test validation fails when fields missing"""
        data = {'name': 'John'}
        required = ['name', 'age', 'email']

        with pytest.raises(InputValidationError) as exc_info:
            validate_required_fields(data, required)

        assert 'age' in exc_info.value.context['missing_fields']
        assert 'email' in exc_info.value.context['missing_fields']

    def test_validate_none_values(self):
        """Test validation fails when fields are None"""
        data = {'name': 'John', 'age': None, 'email': None}
        required = ['name', 'age', 'email']

        with pytest.raises(InputValidationError) as exc_info:
            validate_required_fields(data, required)

        assert 'age' in exc_info.value.context['missing_fields']


class TestValidateConfigValue:
    """Tests for validate_config_value utility"""

    def test_validate_required_present(self):
        """Test validation of required present value"""
        config = {'database_url': 'localhost:5432'}
        value = validate_config_value(config, 'database_url', str, required=True)

        assert value == 'localhost:5432'

    def test_validate_required_missing(self):
        """Test validation fails for missing required value"""
        config = {}

        with pytest.raises(MissingConfigError) as exc_info:
            validate_config_value(config, 'database_url', str, required=True)

        assert 'database_url' in str(exc_info.value)

    def test_validate_optional_missing(self):
        """Test validation passes for missing optional value"""
        config = {}
        value = validate_config_value(config, 'optional_key', str,
                                      required=False, default='default_value')

        assert value == 'default_value'

    def test_validate_wrong_type(self):
        """Test validation fails for wrong type"""
        config = {'port': '8080'}  # String instead of int

        with pytest.raises(InvalidConfigError) as exc_info:
            validate_config_value(config, 'port', int, required=True)

        assert 'int' in exc_info.value.context['expected_type']
        assert 'str' in exc_info.value.context['actual_type']

    def test_validate_correct_type(self):
        """Test validation passes for correct type"""
        config = {'port': 8080, 'debug': True}

        port = validate_config_value(config, 'port', int)
        debug = validate_config_value(config, 'debug', bool)

        assert port == 8080
        assert debug is True


class TestLogExceptionWithContext:
    """Tests for log_exception_with_context"""

    def test_log_standard_exception(self, caplog):
        """Test logging standard exception"""
        import logging
        logger = logging.getLogger('test')

        exc = ValueError("Test error")
        log_exception_with_context(
            exc, logger, "data processing",
            context={'data_id': '123'}
        )

        assert "data processing" in caplog.text
        assert "ValueError" in caplog.text

    def test_log_custom_exception_with_context(self, caplog):
        """Test logging custom exception with built-in context"""
        import logging
        logger = logging.getLogger('test')

        exc = APIConnectionError(
            "Connection failed",
            context={'host': 'api.example.com', 'port': 443}
        )
        log_exception_with_context(exc, logger, "API call")

        assert "API call" in caplog.text
        assert "api.example.com" in caplog.text or "host" in caplog.text
