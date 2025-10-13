"""
Splunk Data Source Adapter

Integrates with Splunk for log aggregation, error pattern detection,
and incident correlation.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import json
from urllib.parse import urljoin

from src.data_sources.base import BaseDataSourceAdapter, StandardMetric
from src.exceptions import (
    DataSourceConnectionError,
    DataSourceAuthenticationError,
    DataSourceQueryError
)

logger = logging.getLogger(__name__)


class SplunkAdapter(BaseDataSourceAdapter):
    """
    Adapter for Splunk log aggregation and search

    Features:
    - SPL (Search Processing Language) query execution
    - Error pattern detection
    - Log aggregation and parsing
    - Incident correlation with logs
    - Search job management
    """

    def __init__(
        self,
        host: str,
        port: int = 8089,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        scheme: str = "https",
        verify_ssl: bool = True,
        timeout: int = 30
    ):
        """
        Initialize Splunk adapter

        Args:
            host: Splunk server hostname
            port: Splunk REST API port (default: 8089)
            username: Splunk username (for basic auth)
            password: Splunk password (for basic auth)
            token: Splunk authentication token (alternative to username/password)
            scheme: http or https
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
        """
        super().__init__("Splunk")

        self.base_url = f"{scheme}://{host}:{port}"
        self.username = username
        self.password = password
        self.token = token
        self.verify_ssl = verify_ssl
        self.timeout = timeout

        # Session token for API calls
        self._session_key: Optional[str] = None

        # Validate configuration
        if not token and not (username and password):
            raise ValueError("Either token or username/password must be provided")

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}

        # If using username/password, get session key
        if not self._session_key:
            self._authenticate()

        return {"Authorization": f"Splunk {self._session_key}"}

    def _authenticate(self) -> None:
        """Authenticate and get session key"""
        if not self.username or not self.password:
            raise DataSourceAuthenticationError(
                "Splunk",
                "Username and password required for authentication"
            )

        url = urljoin(self.base_url, "/services/auth/login")

        try:
            response = requests.post(
                url,
                data={
                    "username": self.username,
                    "password": self.password,
                    "output_mode": "json"
                },
                verify=self.verify_ssl,
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            self._session_key = data.get("sessionKey")

            if not self._session_key:
                raise DataSourceAuthenticationError(
                    "Splunk",
                    "Failed to obtain session key"
                )

            logger.info("Successfully authenticated with Splunk")

        except requests.exceptions.RequestException as e:
            raise DataSourceAuthenticationError(
                "Splunk",
                f"Authentication failed: {str(e)}"
            )

    def test_connection(self) -> bool:
        """
        Test connection to Splunk

        Returns:
            True if connection successful
        """
        try:
            url = urljoin(self.base_url, "/services/server/info")

            response = requests.get(
                url,
                headers=self._get_auth_headers(),
                verify=self.verify_ssl,
                timeout=self.timeout,
                params={"output_mode": "json"}
            )

            response.raise_for_status()

            logger.info("Splunk connection test successful")
            return True

        except Exception as e:
            logger.error(f"Splunk connection test failed: {e}")
            return False

    def execute_search(
        self,
        search_query: str,
        earliest_time: Optional[datetime] = None,
        latest_time: Optional[datetime] = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute a Splunk search query

        Args:
            search_query: SPL query string
            earliest_time: Start time for search
            latest_time: End time for search
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        # Format time parameters
        earliest = self._format_time(earliest_time) if earliest_time else "-24h"
        latest = self._format_time(latest_time) if latest_time else "now"

        # Ensure search query starts with "search"
        if not search_query.strip().startswith("search "):
            search_query = f"search {search_query}"

        try:
            # Create search job
            job_sid = self._create_search_job(
                search_query,
                earliest,
                latest
            )

            # Wait for job completion
            self._wait_for_job(job_sid)

            # Get results
            results = self._get_job_results(job_sid, max_results)

            # Clean up job
            self._delete_job(job_sid)

            return results

        except Exception as e:
            raise DataSourceQueryError(
                "Splunk",
                f"Search execution failed: {str(e)}"
            )

    def _create_search_job(
        self,
        search_query: str,
        earliest_time: str,
        latest_time: str
    ) -> str:
        """Create a search job and return job ID"""
        url = urljoin(self.base_url, "/services/search/jobs")

        data = {
            "search": search_query,
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "output_mode": "json"
        }

        response = requests.post(
            url,
            headers=self._get_auth_headers(),
            data=data,
            verify=self.verify_ssl,
            timeout=self.timeout
        )

        response.raise_for_status()
        result = response.json()

        return result["sid"]

    def _wait_for_job(self, job_sid: str, max_wait: int = 60) -> None:
        """Wait for search job to complete"""
        url = urljoin(self.base_url, f"/services/search/jobs/{job_sid}")

        import time
        start_time = time.time()

        while time.time() - start_time < max_wait:
            response = requests.get(
                url,
                headers=self._get_auth_headers(),
                params={"output_mode": "json"},
                verify=self.verify_ssl,
                timeout=self.timeout
            )

            response.raise_for_status()
            job_info = response.json()

            # Check if job is done
            entry = job_info.get("entry", [{}])[0]
            content = entry.get("content", {})

            if content.get("isDone"):
                return

            time.sleep(1)

        raise TimeoutError(f"Search job {job_sid} did not complete within {max_wait} seconds")

    def _get_job_results(self, job_sid: str, max_results: int) -> List[Dict[str, Any]]:
        """Get results from completed search job"""
        url = urljoin(self.base_url, f"/services/search/jobs/{job_sid}/results")

        response = requests.get(
            url,
            headers=self._get_auth_headers(),
            params={
                "output_mode": "json",
                "count": max_results
            },
            verify=self.verify_ssl,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        # Extract results
        results = []
        for result in data.get("results", []):
            results.append(result)

        return results

    def _delete_job(self, job_sid: str) -> None:
        """Delete a search job"""
        url = urljoin(self.base_url, f"/services/search/jobs/{job_sid}")

        try:
            requests.delete(
                url,
                headers=self._get_auth_headers(),
                verify=self.verify_ssl,
                timeout=self.timeout
            )
        except Exception as e:
            logger.warning(f"Failed to delete search job {job_sid}: {e}")

    def _format_time(self, dt: datetime) -> str:
        """Format datetime for Splunk time format"""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def get_error_patterns(
        self,
        index: str,
        service: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Detect error patterns in logs

        Args:
            index: Splunk index to search
            service: Service name
            start_time: Start time for analysis
            end_time: End time for analysis
            min_count: Minimum error count to report

        Returns:
            List of error patterns with counts and examples
        """
        earliest = start_time or datetime.now() - timedelta(hours=24)
        latest = end_time or datetime.now()

        # SPL query to find error patterns
        query = f'''
        index={index} service={service} (level=ERROR OR level=FATAL OR status>=500)
        | rex field=_raw "(?<error_pattern>[A-Za-z]+Exception|Error:|FATAL:).*"
        | stats count by error_pattern
        | where count >= {min_count}
        | sort -count
        '''

        results = self.execute_search(query, earliest, latest)

        patterns = []
        for result in results:
            patterns.append({
                "pattern": result.get("error_pattern", "Unknown"),
                "count": int(result.get("count", 0)),
                "percentage": None  # Can be calculated if total is known
            })

        return patterns

    def correlate_logs_with_incident(
        self,
        index: str,
        service: str,
        incident_time: datetime,
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Correlate logs with an incident timeframe

        Args:
            index: Splunk index
            service: Service name
            incident_time: Incident start time
            duration_minutes: Duration to analyze

        Returns:
            Dict with log analysis results
        """
        start_time = incident_time - timedelta(minutes=5)  # 5 min before
        end_time = incident_time + timedelta(minutes=duration_minutes)

        # Query for errors during incident
        error_query = f'''
        index={index} service={service} (level=ERROR OR level=FATAL OR status>=500)
        | timechart span=1m count
        '''

        error_results = self.execute_search(error_query, start_time, end_time)

        # Query for specific exceptions
        exception_query = f'''
        index={index} service={service} (Exception OR Error)
        | rex field=_raw "(?<exception_type>[A-Za-z]+Exception)"
        | stats count by exception_type
        | sort -count
        | head 10
        '''

        exception_results = self.execute_search(exception_query, start_time, end_time)

        return {
            "incident_time": incident_time.isoformat(),
            "duration_minutes": duration_minutes,
            "error_timeline": error_results,
            "top_exceptions": exception_results,
            "total_errors": sum(int(r.get("count", 0)) for r in error_results)
        }

    def get_service_metrics_from_logs(
        self,
        index: str,
        service: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[StandardMetric]:
        """
        Extract metrics from log data

        Args:
            index: Splunk index
            service: Service name
            start_time: Start time
            end_time: End time

        Returns:
            List of StandardMetric objects
        """
        earliest = start_time or datetime.now() - timedelta(hours=1)
        latest = end_time or datetime.now()

        # Query for error rate
        error_rate_query = f'''
        index={index} service={service}
        | bin _time span=1m
        | stats count(eval(status>=500)) as errors, count as total by _time
        | eval error_rate = (errors / total) * 100
        '''

        results = self.execute_search(error_rate_query, earliest, latest)

        metrics = []
        for result in results:
            timestamp_str = result.get("_time")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            error_rate = float(result.get("error_rate", 0))

            metrics.append(StandardMetric(
                service_name=service,
                metric_type="error_rate",
                value=error_rate,
                unit="%",
                timestamp=timestamp,
                source="Splunk",
                labels={"index": index}
            ))

        return metrics

    def fetch_metrics(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        **kwargs
    ) -> List[StandardMetric]:
        """
        Fetch metrics from Splunk logs

        Args:
            service_name: Service to fetch metrics for
            start_time: Start time
            end_time: End time
            **kwargs: Additional parameters (index, etc.)

        Returns:
            List of StandardMetric objects
        """
        index = kwargs.get("index", "main")

        return self.get_service_metrics_from_logs(
            index=index,
            service=service_name,
            start_time=start_time,
            end_time=end_time
        )

    def get_available_services(self) -> List[str]:
        """
        Get list of services with log data

        Returns:
            List of service names
        """
        query = '''
        | metadata type=hosts index=*
        | table host
        '''

        try:
            results = self.execute_search(query)
            return [r.get("host", "") for r in results if r.get("host")]
        except Exception as e:
            logger.error(f"Failed to get available services: {e}")
            return []
