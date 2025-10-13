"""
Tests for FastAPI main application

Tests all API endpoints including authentication, metrics, reports, and anomalies.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import uuid

from src.api.main import app
from src.api.auth import key_manager, Role, rate_limiter


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def admin_api_key():
    """Generate admin API key for testing"""
    raw_key, api_key = key_manager.generate_key(
        name="Test Admin Key",
        role=Role.ADMIN,
        rate_limit=1000
    )
    yield raw_key
    # Cleanup
    key_manager.revoke_key(api_key.key_id)


@pytest.fixture
def read_api_key():
    """Generate read-only API key for testing"""
    raw_key, api_key = key_manager.generate_key(
        name="Test Read Key",
        role=Role.READ,
        rate_limit=100
    )
    yield raw_key
    # Cleanup
    key_manager.revoke_key(api_key.key_id)


@pytest.fixture
def write_api_key():
    """Generate write API key for testing"""
    raw_key, api_key = key_manager.generate_key(
        name="Test Write Key",
        role=Role.WRITE,
        rate_limit=100
    )
    yield raw_key
    # Cleanup
    key_manager.revoke_key(api_key.key_id)


class TestGeneralEndpoints:
    """Test general API endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "SRE Analytics API"
        assert data["version"] == "1.0.0"
        assert "documentation" in data

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "data_sources" in data


class TestAuthentication:
    """Test API authentication and authorization"""

    def test_missing_api_key(self, client):
        """Test request without API key is rejected"""
        response = client.get("/metrics")
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]

    def test_invalid_api_key(self, client):
        """Test request with invalid API key is rejected"""
        response = client.get(
            "/metrics",
            headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code == 401
        assert "Invalid or disabled" in response.json()["detail"]

    def test_valid_api_key(self, client, read_api_key):
        """Test request with valid API key is accepted"""
        # Mock the data source to avoid actual connection
        with patch('src.api.main.PrometheusIntegration') as mock_prom:
            mock_prom.return_value.get_service_metrics.return_value = []

            response = client.get(
                "/metrics?service=test&metric_type=latency",
                headers={"X-API-Key": read_api_key}
            )

            # Should get 404 (no metrics) instead of 401 (unauthorized)
            assert response.status_code == 404

    def test_insufficient_permissions(self, client, read_api_key):
        """Test read-only key cannot access write endpoints"""
        response = client.post(
            "/reports/generate",
            headers={"X-API-Key": read_api_key},
            json={
                "application_name": "Test App",
                "services": ["service1"],
                "report_type": "performance"
            }
        )
        assert response.status_code == 403
        assert "Insufficient permissions" in response.json()["detail"]

    def test_admin_permissions(self, client, admin_api_key):
        """Test admin key can access admin endpoints"""
        response = client.get(
            "/admin/api-keys",
            headers={"X-API-Key": admin_api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert "api_keys" in data


class TestMetricsEndpoints:
    """Test metrics API endpoints"""

    @patch('src.api.main.PrometheusIntegration')
    def test_get_metrics_success(self, mock_prom_class, client, read_api_key):
        """Test successful metrics retrieval"""
        # Mock metrics data
        mock_metric = Mock()
        mock_metric.metric_type = "latency"
        mock_metric.value = 150.0
        mock_metric.unit = "ms"
        mock_metric.timestamp = datetime.now()

        mock_prom = mock_prom_class.return_value
        mock_prom.get_service_metrics.return_value = [mock_metric]

        response = client.get(
            "/metrics?service=product-service&metric_type=latency",
            headers={"X-API-Key": read_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "product-service"
        assert data["metric_type"] == "latency"
        assert "values" in data
        assert "summary" in data
        assert data["summary"]["count"] == 1

    @patch('src.api.main.PrometheusIntegration')
    def test_get_metrics_not_found(self, mock_prom_class, client, read_api_key):
        """Test metrics not found returns 404"""
        mock_prom = mock_prom_class.return_value
        mock_prom.get_service_metrics.return_value = []

        response = client.get(
            "/metrics?service=nonexistent&metric_type=latency",
            headers={"X-API-Key": read_api_key}
        )

        assert response.status_code == 404

    def test_get_metrics_invalid_data_source(self, client, read_api_key):
        """Test invalid data source returns 400"""
        response = client.get(
            "/metrics?service=test&metric_type=latency&data_source=invalid",
            headers={"X-API-Key": read_api_key}
        )

        assert response.status_code == 400
        assert "Unsupported data source" in response.json()["detail"]


class TestReportsEndpoints:
    """Test reports API endpoints"""

    def test_list_reports_empty(self, client, read_api_key):
        """Test listing reports when none exist"""
        response = client.get(
            "/reports",
            headers={"X-API-Key": read_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "reports" in data
        assert "total" in data
        assert data["total"] == 0

    @patch('src.api.main.EnhancedSREReportSystem')
    def test_generate_report(self, mock_report_class, client, write_api_key):
        """Test report generation"""
        response = client.post(
            "/reports/generate",
            headers={"X-API-Key": write_api_key},
            json={
                "application_name": "Test App",
                "services": ["service1", "service2"],
                "report_type": "performance",
                "output_format": "html"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "task_id" in data
        assert data["status"] == "pending"
        assert "estimated_time" in data

        # Verify UUID format
        try:
            uuid.UUID(data["task_id"])
        except ValueError:
            pytest.fail("Invalid task_id format")

    def test_generate_report_requires_write_permission(self, client, read_api_key):
        """Test report generation requires write permission"""
        response = client.post(
            "/reports/generate",
            headers={"X-API-Key": read_api_key},
            json={
                "application_name": "Test App",
                "services": ["service1"],
                "report_type": "performance"
            }
        )

        assert response.status_code == 403

    @patch('src.api.main.EnhancedSREReportSystem')
    def test_get_report_status(self, mock_report_class, client, write_api_key):
        """Test getting report generation status"""
        # First generate a report
        response = client.post(
            "/reports/generate",
            headers={"X-API-Key": write_api_key},
            json={
                "application_name": "Test App",
                "services": ["service1"],
                "report_type": "performance"
            }
        )

        task_id = response.json()["task_id"]

        # Check status
        status_response = client.get(
            f"/reports/status/{task_id}",
            headers={"X-API-Key": write_api_key}
        )

        assert status_response.status_code == 200
        status_data = status_response.json()

        assert status_data["task_id"] == task_id
        assert "status" in status_data
        assert "progress" in status_data
        assert "message" in status_data

    def test_get_report_status_not_found(self, client, read_api_key):
        """Test getting status for nonexistent task"""
        fake_task_id = str(uuid.uuid4())

        response = client.get(
            f"/reports/status/{fake_task_id}",
            headers={"X-API-Key": read_api_key}
        )

        assert response.status_code == 404


class TestIncidentsEndpoints:
    """Test incidents API endpoints"""

    def test_create_incident(self, client, write_api_key):
        """Test creating a new incident"""
        incident_data = {
            "title": "High Latency Detected",
            "description": "Product service experiencing high latency",
            "severity": "high",
            "affected_services": ["product-service"],
            "start_time": datetime.now().isoformat(),
            "tags": ["latency", "performance"]
        }

        response = client.post(
            "/incidents",
            headers={"X-API-Key": write_api_key},
            json=incident_data
        )

        assert response.status_code == 200
        data = response.json()

        assert "incident_id" in data
        assert data["title"] == incident_data["title"]
        assert data["severity"] == incident_data["severity"]
        assert data["status"] == "active"
        assert data["affected_services"] == incident_data["affected_services"]

    def test_create_incident_requires_write_permission(self, client, read_api_key):
        """Test creating incident requires write permission"""
        incident_data = {
            "title": "Test Incident",
            "description": "Test",
            "severity": "low",
            "affected_services": ["test"],
            "start_time": datetime.now().isoformat()
        }

        response = client.post(
            "/incidents",
            headers={"X-API-Key": read_api_key},
            json=incident_data
        )

        assert response.status_code == 403

    def test_list_incidents(self, client, write_api_key, read_api_key):
        """Test listing incidents"""
        # Create a test incident first
        incident_data = {
            "title": "Test Incident",
            "description": "Test description",
            "severity": "medium",
            "affected_services": ["test-service"],
            "start_time": datetime.now().isoformat()
        }

        create_response = client.post(
            "/incidents",
            headers={"X-API-Key": write_api_key},
            json=incident_data
        )

        assert create_response.status_code == 200

        # List incidents
        list_response = client.get(
            "/incidents",
            headers={"X-API-Key": read_api_key}
        )

        assert list_response.status_code == 200
        data = list_response.json()

        assert "incidents" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_get_incident_by_id(self, client, write_api_key, read_api_key):
        """Test getting incident details by ID"""
        # Create incident
        incident_data = {
            "title": "Specific Incident",
            "description": "Test",
            "severity": "critical",
            "affected_services": ["service1"],
            "start_time": datetime.now().isoformat()
        }

        create_response = client.post(
            "/incidents",
            headers={"X-API-Key": write_api_key},
            json=incident_data
        )

        incident_id = create_response.json()["incident_id"]

        # Get incident by ID
        get_response = client.get(
            f"/incidents/{incident_id}",
            headers={"X-API-Key": read_api_key}
        )

        assert get_response.status_code == 200
        data = get_response.json()

        assert data["incident_id"] == incident_id
        assert data["title"] == incident_data["title"]

    def test_get_incident_not_found(self, client, read_api_key):
        """Test getting nonexistent incident returns 404"""
        fake_id = str(uuid.uuid4())

        response = client.get(
            f"/incidents/{fake_id}",
            headers={"X-API-Key": read_api_key}
        )

        assert response.status_code == 404


class TestAnomalyDetectionEndpoints:
    """Test anomaly detection API endpoints"""

    @patch('src.api.main.PrometheusIntegration')
    @patch('src.api.main.SLOAnomalyMonitor')
    def test_detect_anomalies_success(
        self, mock_monitor_class, mock_prom_class, client, read_api_key
    ):
        """Test successful anomaly detection"""
        # Mock metrics
        mock_metric = Mock()
        mock_metric.metric_type = "latency"
        mock_metric.value = 250.0
        mock_metric.timestamp = datetime.now()

        mock_prom = mock_prom_class.return_value
        mock_prom.get_service_metrics.return_value = [mock_metric]

        # Mock anomaly results
        mock_monitor = mock_monitor_class.return_value
        mock_monitor.analyze_slo_metrics.return_value = {
            "anomalies": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "value": 250.0,
                    "expected_range": (100.0, 200.0),
                    "deviation": 2.5,
                    "confidence": 0.95
                }
            ],
            "summary": {
                "total_anomalies": 1,
                "severity": "high"
            },
            "recommendations": [
                "Investigate service performance"
            ]
        }

        response = client.post(
            "/anomalies/detect",
            headers={"X-API-Key": read_api_key},
            json={
                "service": "product-service",
                "metric_type": "latency",
                "detection_method": "z_score",
                "sensitivity": 2.5,
                "lookback_hours": 24
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "product-service"
        assert data["metric_type"] == "latency"
        assert data["anomalies_detected"] == 1
        assert len(data["anomalies"]) == 1
        assert "summary" in data
        assert "recommendations" in data

    @patch('src.api.main.PrometheusIntegration')
    def test_detect_anomalies_no_metrics(self, mock_prom_class, client, read_api_key):
        """Test anomaly detection with no metrics returns 404"""
        mock_prom = mock_prom_class.return_value
        mock_prom.get_service_metrics.return_value = []

        response = client.post(
            "/anomalies/detect",
            headers={"X-API-Key": read_api_key},
            json={
                "service": "nonexistent",
                "metric_type": "latency",
                "detection_method": "z_score"
            }
        )

        assert response.status_code == 404


class TestAdminEndpoints:
    """Test admin API key management endpoints"""

    def test_create_api_key(self, client, admin_api_key):
        """Test creating a new API key"""
        response = client.post(
            "/admin/api-keys",
            headers={"X-API-Key": admin_api_key},
            json={
                "name": "New Test Key",
                "role": "read",
                "rate_limit": 50,
                "metadata": {"purpose": "testing"}
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "key_id" in data
        assert "api_key" in data  # Raw key only returned on creation
        assert data["name"] == "New Test Key"
        assert data["role"] == "read"
        assert data["rate_limit"] == 50

    def test_create_api_key_invalid_role(self, client, admin_api_key):
        """Test creating API key with invalid role"""
        response = client.post(
            "/admin/api-keys",
            headers={"X-API-Key": admin_api_key},
            json={
                "name": "Invalid Key",
                "role": "superuser",  # Invalid role
                "rate_limit": 100
            }
        )

        assert response.status_code == 400
        assert "Invalid role" in response.json()["detail"]

    def test_list_api_keys(self, client, admin_api_key):
        """Test listing all API keys"""
        response = client.get(
            "/admin/api-keys",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 200
        data = response.json()

        assert "api_keys" in data
        assert isinstance(data["api_keys"], list)

    def test_revoke_api_key(self, client, admin_api_key):
        """Test revoking an API key"""
        # First create a key to revoke
        create_response = client.post(
            "/admin/api-keys",
            headers={"X-API-Key": admin_api_key},
            json={
                "name": "Key to Revoke",
                "role": "read",
                "rate_limit": 100
            }
        )

        key_id = create_response.json()["key_id"]

        # Revoke the key
        revoke_response = client.delete(
            f"/admin/api-keys/{key_id}",
            headers={"X-API-Key": admin_api_key}
        )

        assert revoke_response.status_code == 200
        assert "revoked successfully" in revoke_response.json()["message"]

    def test_revoke_api_key_not_found(self, client, admin_api_key):
        """Test revoking nonexistent API key"""
        fake_key_id = "nonexistent"

        response = client.delete(
            f"/admin/api-keys/{fake_key_id}",
            headers={"X-API-Key": admin_api_key}
        )

        assert response.status_code == 404

    def test_admin_endpoints_require_admin_role(self, client, write_api_key):
        """Test admin endpoints require admin role"""
        # Write role should not have access to admin endpoints
        response = client.get(
            "/admin/api-keys",
            headers={"X-API-Key": write_api_key}
        )

        assert response.status_code == 403


class TestRateLimiting:
    """Test API rate limiting"""

    def test_rate_limit_enforcement(self, client):
        """Test rate limit is enforced"""
        # Create a key with very low rate limit
        raw_key, api_key = key_manager.generate_key(
            name="Rate Limited Key",
            role=Role.READ,
            rate_limit=2  # Only 2 requests per minute
        )

        try:
            # First request should succeed
            response1 = client.get("/health", headers={"X-API-Key": raw_key})
            assert response1.status_code == 200

            # Second request should succeed
            response2 = client.get("/health", headers={"X-API-Key": raw_key})
            assert response2.status_code == 200

            # Third request should be rate limited
            response3 = client.get("/health", headers={"X-API-Key": raw_key})
            assert response3.status_code == 429
            assert "Rate limit exceeded" in response3.json()["detail"]

        finally:
            # Cleanup
            key_manager.revoke_key(api_key.key_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
