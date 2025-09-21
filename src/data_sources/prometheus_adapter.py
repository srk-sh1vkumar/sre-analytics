"""
Prometheus Data Source Adapter
Implements the generic data source interface for Prometheus
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

from .base import (
    DataSourceAdapter,
    StandardMetric,
    DataSourceConfig,
    QueryParams,
    MetricType,
    DataSourceType
)


class PrometheusAdapter(DataSourceAdapter):
    """Prometheus implementation of the generic data source adapter"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = config.connection_params.get('url', 'http://localhost:9090')
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self._setup_authentication()

    def _setup_authentication(self):
        """Setup authentication for Prometheus API"""
        auth_config = self.config.authentication or {}

        if auth_config.get('username') and auth_config.get('password'):
            self.session.auth = (
                auth_config['username'],
                auth_config['password']
            )
        elif auth_config.get('bearer_token'):
            self.session.headers.update({
                'Authorization': f"Bearer {auth_config['bearer_token']}"
            })

    def connect(self) -> bool:
        """Establish connection to Prometheus"""
        try:
            return self.test_connection()
        except Exception as e:
            self.logger.error(f"Failed to connect to Prometheus: {e}")
            return False

    def test_connection(self) -> bool:
        """Test Prometheus connection"""
        try:
            url = urljoin(self.base_url, '/api/v1/query')
            params = {'query': 'up'}
            response = self.session.get(url, params=params, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def get_available_services(self) -> List[str]:
        """Get list of services from Prometheus metrics"""
        try:
            # Query for unique service names from common service label
            queries = [
                'group by (service) ({__name__=~".*"})',
                'group by (job) ({__name__=~".*"})',
                'group by (instance) ({__name__=~".*"})'
            ]

            services = set()
            for query in queries:
                url = urljoin(self.base_url, '/api/v1/query')
                params = {'query': query}
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        result = data.get('data', {}).get('result', [])
                        for item in result:
                            metric = item.get('metric', {})
                            # Extract service names from various labels
                            for label in ['service', 'job', 'instance']:
                                if label in metric:
                                    services.add(metric[label])

            return sorted(list(services))
        except Exception as e:
            self.logger.error(f"Error getting available services: {e}")
            return []

    def get_available_metrics(self, service_name: str) -> List[str]:
        """Get available metrics for a Prometheus service"""
        try:
            # Query for all metrics for this service
            url = urljoin(self.base_url, '/api/v1/label/__name__/values')
            response = self.session.get(url, timeout=30)

            metrics = []
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    all_metrics = data.get('data', [])

                    # Filter metrics that might be relevant to the service
                    service_metrics = []
                    for metric in all_metrics:
                        # Check if metric might belong to this service
                        if self._metric_belongs_to_service(metric, service_name):
                            service_metrics.append(metric)

                    return service_metrics

            return metrics
        except Exception as e:
            self.logger.error(f"Error getting available metrics for {service_name}: {e}")
            return []

    def _metric_belongs_to_service(self, metric_name: str, service_name: str) -> bool:
        """Check if a metric belongs to a service"""
        try:
            # Query to see if this metric has data for the service
            queries = [
                f'{metric_name}{{service="{service_name}"}}',
                f'{metric_name}{{job="{service_name}"}}',
                f'{metric_name}{{instance=~".*{service_name}.*"}}'
            ]

            for query in queries:
                url = urljoin(self.base_url, '/api/v1/query')
                params = {'query': query}
                response = self.session.get(url, params=params, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        result = data.get('data', {}).get('result', [])
                        if result:
                            return True

            return False
        except Exception:
            return False

    def query_metrics(self, params: QueryParams) -> List[StandardMetric]:
        """Query metrics from Prometheus"""
        metrics = []

        # Define Prometheus queries for standard metric types
        metric_queries = self._build_metric_queries(params)

        for service in params.services or self.get_available_services():
            for metric_type, queries in metric_queries.items():
                if params.metric_types and metric_type not in params.metric_types:
                    continue

                for query_template in queries:
                    try:
                        # Substitute service name in query
                        query = query_template.format(service=service)
                        service_metrics = self._execute_range_query(
                            query, params.start_time, params.end_time,
                            service, metric_type
                        )
                        metrics.extend(service_metrics)
                    except Exception as e:
                        self.logger.error(f"Error querying {metric_type} for {service}: {e}")

        return metrics

    def _build_metric_queries(self, params: QueryParams) -> Dict[MetricType, List[str]]:
        """Build Prometheus queries for different metric types"""
        return {
            MetricType.RESPONSE_TIME: [
                'avg(http_request_duration_seconds{{service="{service}"}}) * 1000',
                'avg(response_time_ms{{service="{service}"}})',
                'avg(http_duration_ms{{service="{service}"}})'
            ],
            MetricType.ERROR_RATE: [
                'rate(http_requests_total{{service="{service}",status=~"4..|5.."}}[5m])',
                'rate(error_total{{service="{service}"}}[5m])',
                'rate(failed_requests{{service="{service}"}}[5m])'
            ],
            MetricType.THROUGHPUT: [
                'rate(http_requests_total{{service="{service}"}}[5m])',
                'rate(requests_total{{service="{service}"}}[5m])',
                'rate(throughput{{service="{service}"}}[5m])'
            ],
            MetricType.CPU_UTILIZATION: [
                'avg(cpu_usage_percent{{service="{service}"}}) * 100',
                'avg(process_cpu_seconds_total{{service="{service}"}}) * 100',
                'avg(container_cpu_usage_seconds_total{{service="{service}"}}) * 100'
            ],
            MetricType.MEMORY_UTILIZATION: [
                'avg(memory_usage_percent{{service="{service}"}}) * 100',
                'avg(process_resident_memory_bytes{{service="{service}"}}) / avg(memory_limit_bytes{{service="{service}"}}) * 100',
                'avg(container_memory_usage_bytes{{service="{service}"}}) / avg(container_spec_memory_limit_bytes{{service="{service}"}}) * 100'
            ],
            MetricType.AVAILABILITY: [
                'avg(up{{service="{service}"}}) * 100',
                'avg(service_up{{service="{service}"}}) * 100',
                'avg(health_check{{service="{service}"}}) * 100'
            ]
        }

    def _execute_range_query(self, query: str, start_time: datetime,
                           end_time: datetime, service_name: str,
                           metric_type: MetricType) -> List[StandardMetric]:
        """Execute a Prometheus range query"""
        metrics = []

        try:
            url = urljoin(self.base_url, '/api/v1/query_range')
            params = {
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '60s'  # 1 minute resolution
            }

            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    result = data.get('data', {}).get('result', [])

                    for series in result:
                        metric_labels = series.get('metric', {})
                        values = series.get('values', [])

                        for timestamp, value in values:
                            try:
                                dt = datetime.fromtimestamp(float(timestamp))
                                val = float(value)

                                # Determine unit based on metric type
                                unit = self._get_unit_for_metric_type(metric_type)

                                # Generate unique metric ID
                                metric_id = f"{service_name}:{metric_type.value}:{timestamp}"

                                standard_metric = StandardMetric(
                                    metric_id=metric_id,
                                    metric_type=metric_type,
                                    service_name=service_name,
                                    metric_name=f"{service_name} - {metric_type.value}",
                                    value=val,
                                    timestamp=dt,
                                    unit=unit,
                                    tags={
                                        "query": query,
                                        **metric_labels
                                    },
                                    raw_data={
                                        "timestamp": timestamp,
                                        "value": value,
                                        "labels": metric_labels
                                    }
                                )

                                metrics.append(standard_metric)
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Error parsing value {value}: {e}")

        except Exception as e:
            self.logger.error(f"Error executing range query: {e}")

        return metrics

    def _get_unit_for_metric_type(self, metric_type: MetricType) -> str:
        """Get appropriate unit for metric type"""
        unit_map = {
            MetricType.RESPONSE_TIME: "ms",
            MetricType.ERROR_RATE: "rate",
            MetricType.THROUGHPUT: "rps",
            MetricType.CPU_UTILIZATION: "%",
            MetricType.MEMORY_UTILIZATION: "%",
            MetricType.AVAILABILITY: "%",
            MetricType.DISK_UTILIZATION: "%",
            MetricType.NETWORK_IO: "bytes/s"
        }
        return unit_map.get(metric_type, "")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Prometheus connection"""
        try:
            connected = self.test_connection()
            services = self.get_available_services() if connected else []

            # Get Prometheus build info
            build_info = {}
            if connected:
                try:
                    url = urljoin(self.base_url, '/api/v1/query')
                    params = {'query': 'prometheus_build_info'}
                    response = self.session.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'success':
                            result = data.get('data', {}).get('result', [])
                            if result:
                                build_info = result[0].get('metric', {})
                except Exception:
                    pass

            return {
                "status": "healthy" if connected else "unhealthy",
                "connected": connected,
                "base_url": self.base_url,
                "available_services": len(services),
                "services": services[:5],  # Show first 5 services
                "build_info": build_info,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }