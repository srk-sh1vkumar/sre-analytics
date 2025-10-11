# SRE Analytics Test Suite

Comprehensive test suite for the SRE Analytics platform.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_exceptions.py       # Exception hierarchy tests
├── test_error_handler.py    # Error handler utility tests
├── test_config.py          # Configuration module tests
├── test_collectors.py      # Data collector tests (TODO)
├── test_report_generators.py # Report generation tests (TODO)
├── test_metrics_generator.py # Metrics generation tests (TODO)
├── test_chart_generator.py  # Chart generation tests (TODO)
└── integration/            # Integration tests (TODO)
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_exceptions.py
```

### Run tests with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run tests in parallel
```bash
pytest -n auto
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Run tests excluding slow ones
```bash
pytest -m "not slow"
```

## Test Markers

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for workflows
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.requires_api` - Tests requiring external API access
- `@pytest.mark.requires_llm` - Tests requiring LLM API access
- `@pytest.mark.requires_docker` - Tests requiring Docker

## Writing Tests

### Basic Test Structure

```python
import pytest
from src.module import function_to_test

class TestFeature:
    """Tests for specific feature"""

    def test_basic_functionality(self):
        """Test basic use case"""
        result = function_to_test()
        assert result == expected_value

    def test_error_handling(self):
        """Test error cases"""
        with pytest.raises(ExpectedException):
            function_to_test(invalid_input)
```

### Using Fixtures

```python
def test_with_fixture(mock_config):
    """Use fixture from conftest.py"""
    assert mock_config.appdynamics.controller_host
```

### Mocking External Services

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked dependency"""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        result = function_that_uses_requests()
        assert result
```

## Coverage Requirements

- Minimum coverage: 70%
- Target coverage: 85%
- Critical modules should have >90% coverage

## Test Data

Test fixtures and sample data are defined in `conftest.py`:
- `mock_config` - Complete mock configuration
- `sample_metric_data` - Sample SLO metrics
- `sample_incident_data` - Sample incident data
- `mock_api_responses` - Mock API responses

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Scheduled daily runs

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Import Errors
Make sure src is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Missing Dependencies
Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Coverage Reports
View HTML coverage report:
```bash
open reports/coverage/index.html
```

## Best Practices

1. **One assertion per test** - Keep tests focused
2. **Descriptive names** - Test names should describe what they test
3. **Arrange-Act-Assert** - Follow AAA pattern
4. **Isolate tests** - Each test should be independent
5. **Mock external services** - Don't rely on real APIs
6. **Test edge cases** - Include boundary conditions
7. **Test error paths** - Verify exception handling
