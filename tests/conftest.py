"""
Pytest configuration and shared fixtures

This file contains test fixtures that are shared across all test modules.
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.app_config import Config, AppDynamicsConfig, LLMConfig, ReportConfig, FlaskConfig, SystemConfig
from src.exceptions import SREAnalyticsError


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_appdynamics_config():
    """Mock AppDynamics configuration"""
    return AppDynamicsConfig(
        controller_host="test-controller.appdynamics.com",
        client_id="test-client-id",
        client_secret="test-client-secret",
        primary_app="TestApp"
    )


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration"""
    return LLMConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key"
    )


@pytest.fixture
def mock_report_config():
    """Mock report configuration"""
    return ReportConfig(
        output_path="test_reports",
        enable_llm_analysis=True
    )


@pytest.fixture
def mock_flask_config():
    """Mock Flask configuration"""
    return FlaskConfig(
        secret_key="test-secret-key",
        debug=False
    )


@pytest.fixture
def mock_system_config():
    """Mock system configuration"""
    return SystemConfig(
        pkg_config_path="/test/pkg-config",
        dyld_library_path="/test/lib"
    )


@pytest.fixture
def mock_config(mock_appdynamics_config, mock_llm_config, mock_report_config,
                mock_flask_config, mock_system_config):
    """Complete mock configuration"""
    return Config(
        appdynamics=mock_appdynamics_config,
        llm=mock_llm_config,
        report=mock_report_config,
        flask=mock_flask_config,
        system=mock_system_config
    )


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_metric_data():
    """Sample metric data for testing"""
    return {
        'service_name': 'test-service',
        'metric_name': 'availability',
        'current_value': 99.5,
        'slo_target': 99.9,
        'sla_target': 99.99,
        'status': 'at_risk',
        'error_budget_consumed': 40.0,
        'timestamp': datetime.now(),
        'unit': '%',
        'description': 'Test metric',
        'trend_data': [99.8, 99.7, 99.6, 99.5]
    }


@pytest.fixture
def sample_incident_data():
    """Sample incident data for testing"""
    return {
        'incident_id': 'INC-20250101-001',
        'title': 'Test Incident',
        'description': 'Test incident description',
        'severity': 'High',
        'application_name': 'TestApp',
        'start_time': datetime.now() - timedelta(hours=2),
        'end_time': datetime.now(),
        'affected_services': ['service-1', 'service-2'],
        'root_cause': 'Database connection pool exhaustion',
        'resolution_steps': ['Step 1', 'Step 2'],
        'llm_analysis': 'Test LLM analysis',
        'lessons_learned': 'Test lessons learned'
    }


@pytest.fixture
def sample_performance_snapshot():
    """Sample performance snapshot for testing"""
    return {
        'service_name': 'test-service',
        'timestamp': datetime.now(),
        'metrics': {
            'availability': 99.5,
            'latency_p95': 250.0,
            'error_rate': 1.5,
            'cpu_usage': 75.0,
            'memory_usage': 60.0
        },
        'logs': [
            '[2025-01-01 10:00:00] High latency detected',
            '[2025-01-01 10:05:00] Database connection timeout'
        ],
        'errors': [
            'TimeoutException: Request timed out',
            'DatabaseConnectionError: Pool exhausted'
        ]
    }


# ============================================================================
# Mock API Responses
# ============================================================================

@pytest.fixture
def mock_api_success_response():
    """Mock successful API response"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'success': True, 'data': {'test': 'value'}}
    mock_response.text = '{"success": true, "data": {"test": "value"}}'
    return mock_response


@pytest.fixture
def mock_api_error_response():
    """Mock error API response"""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {'error': 'Internal Server Error'}
    mock_response.text = '{"error": "Internal Server Error"}'
    return mock_response


@pytest.fixture
def mock_api_auth_error_response():
    """Mock authentication error response"""
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.json.return_value = {'error': 'Unauthorized'}
    mock_response.text = '{"error": "Unauthorized"}'
    return mock_response


# ============================================================================
# Mock External Services
# ============================================================================

@pytest.fixture
def mock_requests():
    """Mock requests library"""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        yield {
            'get': mock_get,
            'post': mock_post
        }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test OpenAI response"))]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test Anthropic response")]
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory"""
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_yaml_config(temp_config_dir):
    """Create sample YAML config file"""
    import yaml
    config_file = temp_config_dir / "test_config.yaml"
    config_data = {
        'controller': {
            'host': 'test-controller.appdynamics.com',
            'port': 8090
        },
        'applications': {
            'primary_app': 'TestApp'
        }
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    return config_file


# ============================================================================
# Environment Variable Fixtures
# ============================================================================

@pytest.fixture
def clean_environment(monkeypatch):
    """Clean environment variables for testing"""
    # Remove all test-related env vars
    env_vars = [
        'APPDYNAMICS_CONTROLLER_HOST',
        'APPDYNAMICS_CLIENT_ID',
        'APPDYNAMICS_CLIENT_SECRET',
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'FLASK_SECRET_KEY'
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def mock_environment(monkeypatch):
    """Set up mock environment variables"""
    monkeypatch.setenv('APPDYNAMICS_CONTROLLER_HOST', 'test-controller.appdynamics.com')
    monkeypatch.setenv('APPDYNAMICS_CLIENT_ID', 'test-client-id')
    monkeypatch.setenv('APPDYNAMICS_CLIENT_SECRET', 'test-client-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-key')
    monkeypatch.setenv('FLASK_SECRET_KEY', 'test-secret-key')
    return monkeypatch


# ============================================================================
# Test Helpers
# ============================================================================

@pytest.fixture
def assert_exception_context():
    """Helper to assert exception context"""
    def _assert_context(exception, expected_keys):
        assert isinstance(exception, SREAnalyticsError)
        assert hasattr(exception, 'context')
        for key in expected_keys:
            assert key in exception.context
    return _assert_context


@pytest.fixture
def create_mock_slo_metric():
    """Factory fixture for creating mock SLO metrics"""
    def _create_metric(service_name='test-service', metric_name='availability',
                      current_value=99.5, status='compliant'):
        from src.reports.llm_analyzer import SLOMetric
        return SLOMetric(
            service_name=service_name,
            metric_name=metric_name,
            current_value=current_value,
            slo_target=99.9,
            sla_target=99.99,
            status=status,
            error_budget_consumed=40.0,
            timestamp=datetime.now(),
            unit='%',
            description=f'Test {metric_name}',
            trend_data=[99.8, 99.7, 99.6, 99.5]
        )
    return _create_metric


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest"""
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports/coverage")
    reports_dir.mkdir(parents=True, exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers to tests based on their location
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
