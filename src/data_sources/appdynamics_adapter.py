"""
AppDynamics Data Source Adapter
Implements the generic data source interface for AppDynamics
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base import (
    DataSourceAdapter,
    StandardMetric,
    DataSourceConfig,
    QueryParams,
    MetricType,
    DataSourceType
)


class AppDynamicsAdapter(DataSourceAdapter):
    """AppDynamics implementation of the generic data source adapter"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.controller_url = f"https://{config.connection_params.get('host')}"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self.applications = {}
        self._setup_authentication()

    def _setup_authentication(self):
        """Setup authentication for AppDynamics API"""
        auth_config = self.config.authentication or {}

        if auth_config.get('username') and auth_config.get('password'):
            # Basic authentication
            self.session.auth = (
                auth_config['username'],
                auth_config['password']
            )
        elif auth_config.get('api_key'):
            # API key authentication
            self.session.headers.update({
                'Authorization': f"Bearer {auth_config['api_key']}"
            })
        elif auth_config.get('oauth_token'):
            # OAuth token authentication
            self.session.headers.update({
                'Authorization': f"Bearer {auth_config['oauth_token']}"
            })

    def connect(self) -> bool:
        """Establish connection to AppDynamics"""
        try:
            return self.test_connection()
        except Exception as e:
            self.logger.error(f"Failed to connect to AppDynamics: {e}")
            return False

    def test_connection(self) -> bool:
        """Test AppDynamics connection"""
        try:
            url = f"{self.controller_url}/controller/rest/applications"
            response = self.session.get(url, timeout=30)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def get_available_services(self) -> List[str]:
        """Get list of applications from AppDynamics"""
        try:
            url = f"{self.controller_url}/controller/rest/applications"
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                apps = response.json()
                if isinstance(apps, list):
                    return [app.get('name', '') for app in apps if app.get('name')]
                else:
                    # Single application response
                    return [apps.get('name', '')] if apps.get('name') else []
            else:
                self.logger.error(f"Failed to get applications: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting available services: {e}")
            return []

    def get_available_metrics(self, service_name: str) -> List[str]:
        """Get available metrics for an AppDynamics application"""
        try:
            # Get business transactions for the application
            url = f"{self.controller_url}/controller/rest/applications/{service_name}/business-transactions"
            response = self.session.get(url, timeout=30)

            metrics = []
            if response.status_code == 200:
                transactions = response.json()
                if isinstance(transactions, list):
                    for tx in transactions:
                        tx_name = tx.get('name', '')
                        if tx_name:
                            metrics.extend([
                                f"{tx_name}:response_time",
                                f"{tx_name}:calls_per_minute",
                                f"{tx_name}:error_rate"
                            ])

            # Add standard application metrics
            standard_metrics = [
                "Overall Application Performance:response_time",
                "Overall Application Performance:calls_per_minute",
                "Overall Application Performance:error_rate",
                "Infrastructure:cpu_utilization",
                "Infrastructure:memory_utilization"
            ]
            metrics.extend(standard_metrics)

            return metrics
        except Exception as e:
            self.logger.error(f"Error getting available metrics for {service_name}: {e}")
            return []

    def query_metrics(self, params: QueryParams) -> List[StandardMetric]:
        """Query metrics from AppDynamics"""
        metrics = []

        for service in params.services or self.get_available_services():
            try:
                service_metrics = self._query_service_metrics(service, params)
                metrics.extend(service_metrics)
            except Exception as e:
                self.logger.error(f"Error querying metrics for {service}: {e}")

        return metrics

    def _query_service_metrics(self, service_name: str, params: QueryParams) -> List[StandardMetric]:
        """Query metrics for a specific service"""
        metrics = []

        # Calculate time range for AppDynamics API
        start_time_ms = int(params.start_time.timestamp() * 1000)
        end_time_ms = int(params.end_time.timestamp() * 1000)
        duration_mins = int((params.end_time - params.start_time).total_seconds() / 60)

        # Query business transaction metrics
        bt_metrics = self._query_business_transaction_metrics(
            service_name, start_time_ms, end_time_ms, duration_mins, params
        )
        metrics.extend(bt_metrics)

        # Query infrastructure metrics if requested
        if not params.metric_types or any(mt in [MetricType.CPU_UTILIZATION, MetricType.MEMORY_UTILIZATION] for mt in params.metric_types):
            infra_metrics = self._query_infrastructure_metrics(
                service_name, start_time_ms, end_time_ms, duration_mins, params
            )
            metrics.extend(infra_metrics)

        return metrics

    def _query_business_transaction_metrics(self, service_name: str, start_time_ms: int,
                                          end_time_ms: int, duration_mins: int,
                                          params: QueryParams) -> List[StandardMetric]:
        """Query business transaction metrics"""
        metrics = []

        try:
            # Get business transactions
            bt_url = f"{self.controller_url}/controller/rest/applications/{service_name}/business-transactions"
            bt_response = self.session.get(bt_url, timeout=30)

            if bt_response.status_code != 200:
                return metrics

            transactions = bt_response.json()
            if not isinstance(transactions, list):
                transactions = [transactions] if transactions else []

            for transaction in transactions[:5]:  # Limit to first 5 transactions
                bt_name = transaction.get('name', '')
                bt_id = transaction.get('id', '')

                if not bt_name or not bt_id:
                    continue

                # Query metrics for this business transaction
                metric_data_url = f"{self.controller_url}/controller/rest/applications/{service_name}/metric-data"

                # Response Time
                if not params.metric_types or MetricType.RESPONSE_TIME in params.metric_types:
                    rt_params = {
                        'metric-path': f"Business Transaction Performance|Business Transactions|{bt_name}|Average Response Time (ms)",
                        'time-range-type': 'BETWEEN_TIMES',
                        'start-time': start_time_ms,
                        'end-time': end_time_ms,
                        'rollup': 'false'
                    }

                    rt_response = self.session.get(metric_data_url, params=rt_params, timeout=30)
                    if rt_response.status_code == 200:
                        rt_data = rt_response.json()
                        rt_metrics = self._parse_metric_data(
                            rt_data, service_name, bt_name, MetricType.RESPONSE_TIME, "ms"
                        )
                        metrics.extend(rt_metrics)

                # Throughput (Calls per minute)
                if not params.metric_types or MetricType.THROUGHPUT in params.metric_types:
                    tp_params = {
                        'metric-path': f"Business Transaction Performance|Business Transactions|{bt_name}|Calls per Minute",
                        'time-range-type': 'BETWEEN_TIMES',
                        'start-time': start_time_ms,
                        'end-time': end_time_ms,
                        'rollup': 'false'
                    }

                    tp_response = self.session.get(metric_data_url, params=tp_params, timeout=30)
                    if tp_response.status_code == 200:
                        tp_data = tp_response.json()
                        tp_metrics = self._parse_metric_data(
                            tp_data, service_name, bt_name, MetricType.THROUGHPUT, "cpm"
                        )
                        metrics.extend(tp_metrics)

                # Error Rate
                if not params.metric_types or MetricType.ERROR_RATE in params.metric_types:
                    er_params = {
                        'metric-path': f"Business Transaction Performance|Business Transactions|{bt_name}|Errors per Minute",
                        'time-range-type': 'BETWEEN_TIMES',
                        'start-time': start_time_ms,
                        'end-time': end_time_ms,
                        'rollup': 'false'
                    }

                    er_response = self.session.get(metric_data_url, params=er_params, timeout=30)
                    if er_response.status_code == 200:
                        er_data = er_response.json()
                        er_metrics = self._parse_metric_data(
                            er_data, service_name, bt_name, MetricType.ERROR_RATE, "epm"
                        )
                        metrics.extend(er_metrics)

        except Exception as e:
            self.logger.error(f"Error querying business transaction metrics: {e}")

        return metrics

    def _query_infrastructure_metrics(self, service_name: str, start_time_ms: int,
                                    end_time_ms: int, duration_mins: int,
                                    params: QueryParams) -> List[StandardMetric]:
        """Query infrastructure metrics"""
        metrics = []

        try:
            metric_data_url = f"{self.controller_url}/controller/rest/applications/{service_name}/metric-data"

            # CPU Utilization
            if not params.metric_types or MetricType.CPU_UTILIZATION in params.metric_types:
                cpu_params = {
                    'metric-path': "Application Infrastructure Performance|*|Hardware Resources|CPU|%Busy",
                    'time-range-type': 'BETWEEN_TIMES',
                    'start-time': start_time_ms,
                    'end-time': end_time_ms,
                    'rollup': 'false'
                }

                cpu_response = self.session.get(metric_data_url, params=cpu_params, timeout=30)
                if cpu_response.status_code == 200:
                    cpu_data = cpu_response.json()
                    cpu_metrics = self._parse_metric_data(
                        cpu_data, service_name, "Infrastructure", MetricType.CPU_UTILIZATION, "%"
                    )
                    metrics.extend(cpu_metrics)

            # Memory Utilization
            if not params.metric_types or MetricType.MEMORY_UTILIZATION in params.metric_types:
                mem_params = {
                    'metric-path': "Application Infrastructure Performance|*|Hardware Resources|Memory|Used %",
                    'time-range-type': 'BETWEEN_TIMES',
                    'start-time': start_time_ms,
                    'end-time': end_time_ms,
                    'rollup': 'false'
                }

                mem_response = self.session.get(metric_data_url, params=mem_params, timeout=30)
                if mem_response.status_code == 200:
                    mem_data = mem_response.json()
                    mem_metrics = self._parse_metric_data(
                        mem_data, service_name, "Infrastructure", MetricType.MEMORY_UTILIZATION, "%"
                    )
                    metrics.extend(mem_metrics)

        except Exception as e:
            self.logger.error(f"Error querying infrastructure metrics: {e}")

        return metrics

    def _parse_metric_data(self, metric_data: List[Dict], service_name: str,
                          component_name: str, metric_type: MetricType,
                          unit: str) -> List[StandardMetric]:
        """Parse AppDynamics metric data into StandardMetric format"""
        metrics = []

        try:
            if not isinstance(metric_data, list):
                metric_data = [metric_data] if metric_data else []

            for metric_series in metric_data:
                metric_name = metric_series.get('metricName', component_name)
                metric_values = metric_series.get('metricValues', [])

                for value_point in metric_values:
                    timestamp_ms = value_point.get('startTimeInMillis', 0)
                    value = value_point.get('value', 0)

                    if timestamp_ms and value is not None:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

                        # Generate unique metric ID
                        metric_id = f"{service_name}:{component_name}:{metric_type.value}:{timestamp_ms}"

                        standard_metric = StandardMetric(
                            metric_id=metric_id,
                            metric_type=metric_type,
                            service_name=service_name,
                            metric_name=f"{component_name} - {metric_type.value}",
                            value=float(value),
                            timestamp=timestamp,
                            unit=unit,
                            tags={
                                "component": component_name,
                                "source_metric": metric_name
                            },
                            raw_data=value_point
                        )

                        metrics.append(standard_metric)

        except Exception as e:
            self.logger.error(f"Error parsing metric data: {e}")

        return metrics

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of AppDynamics connection"""
        try:
            connected = self.test_connection()
            services = self.get_available_services() if connected else []

            return {
                "status": "healthy" if connected else "unhealthy",
                "connected": connected,
                "controller_url": self.controller_url,
                "available_services": len(services),
                "services": services[:5],  # Show first 5 services
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }