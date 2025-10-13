#!/usr/bin/env python3
"""
SRE Analytics API Client Example

Demonstrates how to use the SRE Analytics API with Python.
"""

import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class SREAnalyticsClient:
    """
    Python client for SRE Analytics API

    Example:
        client = SREAnalyticsClient(
            base_url="http://localhost:8000",
            api_key="sre_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

        # Get metrics
        metrics = client.get_metrics(services=["product-service"])

        # Get service health
        health = client.get_service_health("product-service")

        # Detect anomalies
        anomalies = client.detect_anomalies(
            services=["product-service"],
            detection_method="modified_z_score"
        )
    """

    def __init__(self, base_url: str, api_key: str):
        """
        Initialize API client

        Args:
            base_url: API base URL (e.g., http://localhost:8000)
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """
        Make HTTP request with error handling

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON data

        Raises:
            requests.HTTPError: On HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)

        # Print rate limit info
        if "X-RateLimit-Remaining" in response.headers:
            remaining = response.headers["X-RateLimit-Remaining"]
            limit = response.headers.get("X-RateLimit-Limit", "?")
            print(f"Rate limit: {remaining}/{limit} remaining")

        response.raise_for_status()
        return response.json()

    # ========================================================================
    # Health & Status
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health (no auth required)

        Returns:
            Health status dict
        """
        # Don't use session for health check (no auth)
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_status(self) -> Dict[str, Any]:
        """
        Get API status and authenticated user info

        Returns:
            Status dict with rate limit info
        """
        return self._request("GET", "/api/v1/status")

    # ========================================================================
    # Metrics
    # ========================================================================

    def get_metrics(
        self,
        services: Optional[List[str]] = None,
        metric_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get current SLO metrics

        Args:
            services: List of service names to filter (optional)
            metric_types: List of metric types to filter (optional)

        Returns:
            Metrics response with list of metric values
        """
        params = {}
        if services:
            params["services"] = ",".join(services)
        if metric_types:
            params["metric_types"] = ",".join(metric_types)

        return self._request("GET", "/api/v1/metrics", params=params)

    def get_service_health(
        self,
        service_name: str,
        hours_back: int = 1
    ) -> Dict[str, Any]:
        """
        Get comprehensive health status for a service

        Args:
            service_name: Service name
            hours_back: Hours of historical data to analyze

        Returns:
            Service health dict with score, status, and recommendations
        """
        return self._request(
            "GET",
            f"/api/v1/services/{service_name}/health",
            params={"hours_back": hours_back}
        )

    # ========================================================================
    # Anomaly Detection
    # ========================================================================

    def get_anomalies(
        self,
        services: Optional[List[str]] = None,
        severity: Optional[str] = None,
        hours_back: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Get recent anomalies

        Args:
            services: List of service names to filter (optional)
            severity: Filter by severity (info, warning, critical)
            hours_back: Hours of historical data

        Returns:
            List of anomaly reports
        """
        params = {"hours_back": hours_back}
        if services:
            params["services"] = ",".join(services)
        if severity:
            params["severity"] = severity

        return self._request("GET", "/api/v1/anomalies", params=params)

    def detect_anomalies(
        self,
        services: List[str],
        detection_method: str = "modified_z_score",
        hours_back: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Trigger anomaly detection for services (requires WRITE role)

        Args:
            services: List of service names to analyze
            detection_method: Detection method (z_score, modified_z_score, iqr, moving_average, prophet)
            hours_back: Hours of historical data

        Returns:
            List of anomaly reports
        """
        data = {
            "services": services,
            "detection_method": detection_method,
            "hours_back": hours_back
        }
        return self._request("POST", "/api/v1/anomalies/detect", json=data)

    # ========================================================================
    # Reports
    # ========================================================================

    def list_reports(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List generated reports

        Args:
            limit: Maximum reports to return
            offset: Number of reports to skip

        Returns:
            List of report metadata
        """
        params = {"limit": limit, "offset": offset}
        return self._request("GET", "/api/v1/reports", params=params)

    def generate_report(
        self,
        services: List[str],
        start_time: datetime,
        end_time: datetime,
        include_anomalies: bool = True,
        include_recommendations: bool = True,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate a new SRE report (requires WRITE role)

        Args:
            services: Services to include
            start_time: Report start time
            end_time: Report end time
            include_anomalies: Include anomaly detection
            include_recommendations: Include recommendations
            format: Report format (json, html, pdf)

        Returns:
            Report response with report_id and status
        """
        data = {
            "services": services,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "include_anomalies": include_anomalies,
            "include_recommendations": include_recommendations,
            "format": format
        }
        return self._request("POST", "/api/v1/reports/generate", json=data)

    def get_report(self, report_id: str) -> Dict[str, Any]:
        """
        Get a specific report

        Args:
            report_id: Report ID

        Returns:
            Complete report data
        """
        return self._request("GET", f"/api/v1/reports/{report_id}")

    # ========================================================================
    # Admin (API Key Management) - Requires ADMIN role
    # ========================================================================

    def create_api_key(
        self,
        name: str,
        role: str = "read",
        rate_limit: int = 100
    ) -> Dict[str, Any]:
        """
        Create new API key (requires ADMIN role)

        Args:
            name: Descriptive name
            role: Access role (read, write, admin)
            rate_limit: Requests per minute

        Returns:
            New API key and metadata (save the key immediately!)
        """
        params = {
            "name": name,
            "role": role,
            "rate_limit": rate_limit
        }
        return self._request("POST", "/api/v1/admin/keys", params=params)

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys (requires ADMIN role)

        Returns:
            List of API key metadata
        """
        return self._request("GET", "/api/v1/admin/keys")

    def revoke_api_key(self, key_id: str) -> Dict[str, str]:
        """
        Revoke an API key (requires ADMIN role)

        Args:
            key_id: API key ID to revoke

        Returns:
            Confirmation message
        """
        return self._request("DELETE", f"/api/v1/admin/keys/{key_id}")


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of SRE Analytics API client"""
    print("=" * 70)
    print("SRE ANALYTICS API CLIENT EXAMPLE")
    print("=" * 70)
    print()

    # Initialize client
    API_KEY = "sre_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your key
    client = SREAnalyticsClient(
        base_url="http://localhost:8000",
        api_key=API_KEY
    )

    # Example 1: Health check
    print("1️⃣  Health Check (no auth)")
    try:
        health = client.health_check()
        print(f"   ✅ API Status: {health['status']}")
        print(f"   Version: {health['version']}")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e}")
    print()

    # Example 2: Get API status
    print("2️⃣  API Status")
    try:
        status = client.get_status()
        print(f"   ✅ Authenticated as: {status['authenticated_as']}")
        print(f"   Role: {status['role']}")
        rate_limit = status['rate_limit']
        print(f"   Rate Limit: {rate_limit['remaining']}/{rate_limit['limit']} remaining")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e}")
    print()

    # Example 3: Get metrics
    print("3️⃣  Get Metrics")
    try:
        metrics = client.get_metrics(services=["product-service", "order-service"])
        print(f"   ✅ Found {metrics['count']} metrics for {len(metrics['services'])} services")
        for metric in metrics['metrics'][:3]:  # Show first 3
            print(f"      • {metric['service_name']}.{metric['metric_name']}: "
                  f"{metric['current_value']:.2f}{metric['unit']} "
                  f"(target: {metric['slo_target']:.2f}{metric['unit']}, status: {metric['status']})")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e}")
    print()

    # Example 4: Get service health
    print("4️⃣  Get Service Health")
    try:
        health = client.get_service_health("product-service", hours_back=4)
        print(f"   ✅ Health Score: {health['health_score']}/100")
        print(f"   Status: {health['status']}")
        print(f"   SLO Compliance: {health['slo_compliance']:.1f}%")
        if health['recommendations']:
            print(f"   Recommendations:")
            for rec in health['recommendations'][:2]:
                print(f"      • {rec}")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e}")
    print()

    # Example 5: Get anomalies
    print("5️⃣  Get Recent Anomalies")
    try:
        anomalies = client.get_anomalies(severity="critical", hours_back=4)
        print(f"   ✅ Found {len(anomalies)} critical anomalies")
        for anomaly in anomalies[:2]:  # Show first 2
            print(f"      • {anomaly['service_name']}.{anomaly['metric_name']}")
            print(f"        Baseline Health: {anomaly['baseline_health']}")
            print(f"        Anomalies: {len(anomaly['anomalies'])}")
            if anomaly.get('prediction', {}).get('will_breach'):
                print(f"        ⚠️  Predicted breach!")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e}")
    print()

    # Example 6: Generate report (requires WRITE role)
    print("6️⃣  Generate Report (requires WRITE role)")
    try:
        report = client.generate_report(
            services=["product-service"],
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            include_anomalies=True,
            include_recommendations=True,
            format="json"
        )
        print(f"   ✅ Report generated: {report['report_id']}")
        print(f"   Status: {report['status']}")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e.response.status_code} - {e.response.json()['error']}")
    print()

    # Example 7: List API keys (requires ADMIN role)
    print("7️⃣  List API Keys (requires ADMIN role)")
    try:
        keys = client.list_api_keys()
        print(f"   ✅ Found {len(keys)} API keys")
        for key in keys[:3]:  # Show first 3
            print(f"      • {key['name']} ({key['role']}) - {key['key_id']}")
    except requests.HTTPError as e:
        print(f"   ❌ Error: {e.response.status_code} - {e.response.json()['error']}")
    print()

    print("=" * 70)
    print("✅ API CLIENT EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
