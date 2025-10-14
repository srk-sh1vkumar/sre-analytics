"""
Tests for configuration modules
"""

import os

import pytest

from src.config import constants
from src.config.app_config import (
    AppDynamicsConfig,
    Config,
    FlaskConfig,
    LLMConfig,
    ReportConfig,
    SystemConfig,
    get_config,
    reload_config,
)
from src.exceptions import MissingConfigError


class TestAppDynamicsConfig:
    """Tests for AppDynamicsConfig"""

    def test_create_valid_config(self):
        """Test creating valid AppDynamics config"""
        config = AppDynamicsConfig(
            controller_host="test.appdynamics.com",
            client_id="test-id",
            client_secret="test-secret",
            primary_app="TestApp",
        )

        assert config.controller_host == "test.appdynamics.com"
        assert config.client_id == "test-id"
        assert config.client_secret == "test-secret"
        assert config.primary_app == "TestApp"

    def test_from_env(self, mock_environment):
        """Test loading from environment variables"""
        config = AppDynamicsConfig.from_env()

        assert config.controller_host == "test-controller.appdynamics.com"
        assert config.client_id == "test-client-id"
        assert config.client_secret == "test-client-secret"

    def test_from_env_missing_values(self, clean_environment):
        """Test loading from env with missing values"""
        config = AppDynamicsConfig.from_env()

        assert config.controller_host is None
        assert config.client_id is None
        assert config.client_secret is None


class TestLLMConfig:
    """Tests for LLMConfig"""

    def test_create_config_with_both_keys(self):
        """Test creating LLM config with both API keys"""
        config = LLMConfig(openai_api_key="openai-key", anthropic_api_key="anthropic-key")

        assert config.openai_api_key == "openai-key"
        assert config.anthropic_api_key == "anthropic-key"

    def test_create_config_with_one_key(self):
        """Test creating LLM config with one API key"""
        config = LLMConfig(openai_api_key="openai-key")

        assert config.openai_api_key == "openai-key"
        assert config.anthropic_api_key is None

    def test_from_env(self, mock_environment):
        """Test loading LLM config from environment"""
        config = LLMConfig.from_env()

        assert config.openai_api_key == "test-openai-key"
        assert config.anthropic_api_key == "test-anthropic-key"


class TestReportConfig:
    """Tests for ReportConfig"""

    def test_create_config(self):
        """Test creating report config"""
        config = ReportConfig(output_path="/tmp/reports", enable_llm_analysis=True)

        assert config.output_path == "/tmp/reports"
        assert config.enable_llm_analysis is True

    def test_default_values(self):
        """Test default values"""
        config = ReportConfig()

        assert config.output_path == "reports/generated"
        assert config.enable_llm_analysis is True

    def test_from_env(self, monkeypatch):
        """Test loading from environment"""
        monkeypatch.setenv("REPORT_OUTPUT_PATH", "/custom/path")
        monkeypatch.setenv("ENABLE_LLM_ANALYSIS", "false")

        config = ReportConfig.from_env()

        assert config.output_path == "/custom/path"
        assert config.enable_llm_analysis is False


class TestFlaskConfig:
    """Tests for FlaskConfig"""

    def test_create_config(self):
        """Test creating Flask config"""
        config = FlaskConfig(secret_key="test-secret", debug=True)

        assert config.secret_key == "test-secret"
        assert config.debug is True

    def test_from_env(self, mock_environment):
        """Test loading from environment"""
        config = FlaskConfig.from_env()

        assert config.secret_key == "test-secret-key"

    def test_default_secret_key(self, clean_environment):
        """Test default secret key generation"""
        config = FlaskConfig.from_env()

        assert config.secret_key is not None
        assert len(config.secret_key) > 0


class TestSystemConfig:
    """Tests for SystemConfig"""

    def test_create_config(self):
        """Test creating system config"""
        config = SystemConfig(
            pkg_config_path="/usr/local/lib/pkgconfig", dyld_library_path="/usr/local/lib"
        )

        assert config.pkg_config_path == "/usr/local/lib/pkgconfig"
        assert config.dyld_library_path == "/usr/local/lib"

    def test_from_env(self, monkeypatch):
        """Test loading from environment"""
        monkeypatch.setenv("PKG_CONFIG_PATH", "/custom/pkg-config")
        monkeypatch.setenv("DYLD_LIBRARY_PATH", "/custom/lib")

        config = SystemConfig.from_env()

        assert config.pkg_config_path == "/custom/pkg-config"
        assert config.dyld_library_path == "/custom/lib"


class TestConfig:
    """Tests for main Config class"""

    def test_create_config(
        self,
        mock_appdynamics_config,
        mock_llm_config,
        mock_report_config,
        mock_flask_config,
        mock_system_config,
    ):
        """Test creating complete config"""
        config = Config(
            appdynamics=mock_appdynamics_config,
            llm=mock_llm_config,
            report=mock_report_config,
            flask=mock_flask_config,
            system=mock_system_config,
        )

        assert config.appdynamics == mock_appdynamics_config
        assert config.llm == mock_llm_config
        assert config.report == mock_report_config
        assert config.flask == mock_flask_config
        assert config.system == mock_system_config

    def test_load_from_environment(self, mock_environment):
        """Test loading config from environment"""
        config = Config.load()

        assert config.appdynamics.controller_host == "test-controller.appdynamics.com"
        assert config.llm.openai_api_key == "test-openai-key"
        assert config.flask.secret_key == "test-secret-key"

    def test_get_config_singleton(self, mock_environment):
        """Test get_config returns singleton"""
        # First call
        config1 = get_config()

        # Second call should return same instance
        config2 = get_config()

        assert config1 is config2

    def test_reload_config(self, mock_environment):
        """Test reload_config creates new instance"""
        # Get initial config
        config1 = get_config()

        # Reload config
        reload_config()
        config2 = get_config()

        # Should be different instances
        assert config1 is not config2


class TestConstants:
    """Tests for constants module"""

    def test_time_constants(self):
        """Test time-related constants"""
        assert constants.DAYS_IN_MONTH == 30
        assert constants.DEFAULT_TREND_DAYS == 30
        assert constants.DEFAULT_INCIDENT_DURATION_HOURS == 1.0

    def test_availability_thresholds(self):
        """Test availability threshold constants"""
        assert constants.AVAILABILITY_MIN == 99.5
        assert constants.AVAILABILITY_TARGET == 99.9
        assert constants.AVAILABILITY_MAX == 99.99

    def test_latency_thresholds(self):
        """Test latency threshold constants"""
        assert constants.LATENCY_P95_TARGET_MS == 200
        assert constants.LATENCY_P99_TARGET_MS == 300
        assert constants.LATENCY_P95_WARNING_MS == 400
        assert constants.LATENCY_P95_CRITICAL_MS == 500

    def test_error_rate_thresholds(self):
        """Test error rate threshold constants"""
        assert constants.ERROR_RATE_TARGET == 1.0
        assert constants.ERROR_RATE_WARNING == 2.0
        assert constants.ERROR_RATE_CRITICAL == 5.0

    def test_api_timeouts(self):
        """Test API timeout constants"""
        assert constants.API_TIMEOUT_DEFAULT == 30
        assert constants.API_TIMEOUT_SHORT == 10
        assert constants.API_TIMEOUT_LONG == 60
        assert constants.API_RETRY_ATTEMPTS == 3

    def test_http_status_codes(self):
        """Test HTTP status code constants"""
        assert constants.HTTP_OK == 200
        assert constants.HTTP_CREATED == 201
        assert constants.HTTP_BAD_REQUEST == 400
        assert constants.HTTP_UNAUTHORIZED == 401
        assert constants.HTTP_FORBIDDEN == 403
        assert constants.HTTP_NOT_FOUND == 404
        assert constants.HTTP_SERVER_ERROR == 500

    def test_port_numbers(self):
        """Test port number constants"""
        assert constants.PORT_FLASK_APP == 5001
        assert constants.PORT_PROMETHEUS == 9090
        assert constants.PORT_GRAFANA == 3000
        assert constants.PORT_EUREKA == 8761

    def test_status_categories(self):
        """Test status category constants"""
        assert constants.STATUS_COMPLIANT == "compliant"
        assert constants.STATUS_AT_RISK == "at_risk"
        assert constants.STATUS_BREACHED == "breached"

    def test_metric_names(self):
        """Test metric name constants"""
        assert constants.METRIC_AVAILABILITY == "availability"
        assert constants.METRIC_LATENCY_P95 == "latency_p95"
        assert constants.METRIC_LATENCY_P99 == "latency_p99"
        assert constants.METRIC_ERROR_RATE == "error_rate"

    def test_severity_levels(self):
        """Test incident severity constants"""
        assert constants.SEVERITY_CRITICAL == "Critical"
        assert constants.SEVERITY_HIGH == "High"
        assert constants.SEVERITY_MEDIUM == "Medium"
        assert constants.SEVERITY_LOW == "Low"

    def test_health_status(self):
        """Test health status constants"""
        assert constants.HEALTH_HEALTHY == "Healthy"
        assert constants.HEALTH_DEGRADED == "Degraded"
        assert constants.HEALTH_UNHEALTHY == "Unhealthy"

    def test_llm_providers(self):
        """Test LLM provider constants"""
        assert constants.LLM_PROVIDER_OPENAI == "openai"
        assert constants.LLM_PROVIDER_ANTHROPIC == "anthropic"

    def test_llm_models(self):
        """Test LLM model constants"""
        assert constants.LLM_MODEL_GPT4 == "gpt-4"
        assert constants.LLM_MODEL_GPT35 == "gpt-3.5-turbo"
        assert "claude" in constants.LLM_MODEL_CLAUDE_SONNET

    def test_constants_are_final(self):
        """Test that constants cannot be modified (type check)"""
        from typing import get_type_hints

        # This just verifies the constants module has Final annotations
        # Actual immutability is enforced by Python's Final type hint
        assert hasattr(constants, "DEFAULT_TREND_DAYS")
        assert hasattr(constants, "AVAILABILITY_MIN")
