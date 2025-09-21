"""
AppDynamics Data Collector
Collects performance metrics and business transaction data from AppDynamics API
"""

import requests
import json
import yaml
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MetricData:
    """Data class for metric information"""
    metric_name: str
    metric_path: str
    value: float
    timestamp: datetime
    unit: str = ""
    tier: str = ""
    application: str = ""

@dataclass
class BusinessTransaction:
    """Data class for business transaction metrics"""
    name: str
    tier: str
    calls_per_minute: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    timestamp: datetime

class AppDynamicsCollector:
    """Collects metrics from AppDynamics Controller API"""

    def __init__(self, config_path: str = "config/appdynamics_config.yaml"):
        self.config = self._load_config(config_path)
        self.controller_url = f"https://{self.config['controller']['host']}"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self._setup_authentication()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load AppDynamics configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Replace environment variables
            import os
            config['controller']['host'] = os.getenv(
                'APPDYNAMICS_CONTROLLER_HOST_NAME',
                config['controller']['host']
            )
            config['controller']['account'] = os.getenv(
                'APPDYNAMICS_AGENT_ACCOUNT_NAME',
                config['controller']['account']
            )
            config['controller']['access_key'] = os.getenv(
                'APPDYNAMICS_AGENT_ACCOUNT_ACCESS_KEY',
                config['controller']['access_key']
            )

            return config
        except Exception as e:
            self.logger.error(f"Failed to load AppDynamics config: {e}")
            raise

    def _setup_authentication(self):
        """Setup authentication for AppDynamics API"""
        auth_string = f"{self.config['controller']['account']}:{self.config['controller']['access_key']}"
        self.session.auth = (self.config['controller']['account'], self.config['controller']['access_key'])
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def get_applications(self) -> List[Dict[str, Any]]:
        """Get list of applications from AppDynamics"""
        url = f"{self.controller_url}/controller/rest/applications"

        try:
            response = self.session.get(url, timeout=self.config['api']['timeout'])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get applications: {e}")
            return []

    def get_business_transactions(self, app_id: int, time_range_type: str = "BEFORE_NOW",
                                duration_in_mins: int = 60) -> List[BusinessTransaction]:
        """Get business transaction metrics"""
        url = f"{self.controller_url}/controller/rest/applications/{app_id}/business-transactions"

        params = {
            'time-range-type': time_range_type,
            'duration-in-mins': duration_in_mins,
            'output': 'json'
        }

        try:
            response = self.session.get(url, params=params, timeout=self.config['api']['timeout'])
            response.raise_for_status()

            transactions = []
            current_time = datetime.now()

            for bt_data in response.json():
                # Get detailed metrics for each transaction
                bt_metrics = self._get_transaction_metrics(app_id, bt_data['id'], duration_in_mins)

                transaction = BusinessTransaction(
                    name=bt_data['name'],
                    tier=bt_data['tierName'],
                    calls_per_minute=bt_metrics.get('calls_per_minute', 0),
                    response_time_avg=bt_metrics.get('response_time_avg', 0),
                    response_time_p95=bt_metrics.get('response_time_p95', 0),
                    response_time_p99=bt_metrics.get('response_time_p99', 0),
                    error_rate=bt_metrics.get('error_rate', 0),
                    timestamp=current_time
                )
                transactions.append(transaction)

            return transactions

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get business transactions: {e}")
            return []

    def _get_transaction_metrics(self, app_id: int, bt_id: int, duration_in_mins: int) -> Dict[str, float]:
        """Get detailed metrics for a specific business transaction"""
        metrics = {}

        # Define metric paths to collect
        metric_paths = [
            "Business Transaction Performance|*|Calls per Minute",
            "Business Transaction Performance|*|Average Response Time (ms)",
            "Business Transaction Performance|*|95th Percentile Response Time (ms)",
            "Business Transaction Performance|*|99th Percentile Response Time (ms)",
            "Business Transaction Performance|*|Errors per Minute"
        ]

        for metric_path in metric_paths:
            url = f"{self.controller_url}/controller/rest/applications/{app_id}/metric-data"

            params = {
                'metric-path': metric_path,
                'time-range-type': 'BEFORE_NOW',
                'duration-in-mins': duration_in_mins,
                'output': 'json'
            }

            try:
                response = self.session.get(url, params=params, timeout=self.config['api']['timeout'])
                if response.status_code == 200:
                    metric_data = response.json()
                    if metric_data and len(metric_data) > 0:
                        # Get the latest value
                        latest_value = metric_data[0]['metricValues'][-1]['value'] if metric_data[0]['metricValues'] else 0

                        # Map metric path to friendly name
                        if "Calls per Minute" in metric_path:
                            metrics['calls_per_minute'] = latest_value
                        elif "Average Response Time" in metric_path:
                            metrics['response_time_avg'] = latest_value
                        elif "95th Percentile" in metric_path:
                            metrics['response_time_p95'] = latest_value
                        elif "99th Percentile" in metric_path:
                            metrics['response_time_p99'] = latest_value
                        elif "Errors per Minute" in metric_path:
                            metrics['error_rate'] = latest_value

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Failed to get metric {metric_path}: {e}")
                continue

        return metrics

    def get_infrastructure_metrics(self, app_id: int, duration_in_mins: int = 60) -> List[MetricData]:
        """Get infrastructure performance metrics"""
        infrastructure_metrics = []

        # Define infrastructure metric paths
        metric_paths = [
            "Application Infrastructure Performance|*|Individual Nodes|*|Memory|*|Used %",
            "Application Infrastructure Performance|*|Individual Nodes|*|CPU|*|%Busy",
            "Application Infrastructure Performance|*|Individual Nodes|*|Disks|*|%Used",
            "Overall Application Performance|*|Calls per Minute",
            "Overall Application Performance|*|Average Response Time (ms)",
            "Overall Application Performance|*|Errors per Minute"
        ]

        current_time = datetime.now()

        for metric_path in metric_paths:
            url = f"{self.controller_url}/controller/rest/applications/{app_id}/metric-data"

            params = {
                'metric-path': metric_path,
                'time-range-type': 'BEFORE_NOW',
                'duration-in-mins': duration_in_mins,
                'output': 'json'
            }

            try:
                response = self.session.get(url, params=params, timeout=self.config['api']['timeout'])
                if response.status_code == 200:
                    metric_data = response.json()

                    for metric in metric_data:
                        if metric['metricValues']:
                            latest_value = metric['metricValues'][-1]['value']

                            metric_obj = MetricData(
                                metric_name=metric['metricName'],
                                metric_path=metric['metricPath'],
                                value=latest_value,
                                timestamp=current_time,
                                unit=metric.get('unit', ''),
                                application=self.config['applications']['primary_app']
                            )
                            infrastructure_metrics.append(metric_obj)

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Failed to get infrastructure metric {metric_path}: {e}")
                continue

        return infrastructure_metrics

    def get_application_health(self, app_id: int) -> Dict[str, Any]:
        """Get overall application health status"""
        url = f"{self.controller_url}/controller/rest/applications/{app_id}/nodes"

        try:
            response = self.session.get(url, timeout=self.config['api']['timeout'])
            response.raise_for_status()

            nodes = response.json()
            total_nodes = len(nodes)
            healthy_nodes = len([node for node in nodes if node.get('available', False)])

            health_status = {
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'availability_percentage': (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0,
                'timestamp': datetime.now(),
                'nodes': nodes
            }

            return health_status

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get application health: {e}")
            return {
                'total_nodes': 0,
                'healthy_nodes': 0,
                'availability_percentage': 0,
                'timestamp': datetime.now(),
                'nodes': []
            }

    def collect_all_metrics(self, duration_in_mins: int = 60) -> Dict[str, Any]:
        """Collect comprehensive metrics from AppDynamics"""
        self.logger.info("Starting comprehensive metric collection from AppDynamics")

        # Get application ID
        applications = self.get_applications()
        app_id = None

        for app in applications:
            if app['name'] == self.config['applications']['primary_app']:
                app_id = app['id']
                break

        if not app_id:
            self.logger.error(f"Application {self.config['applications']['primary_app']} not found")
            return {}

        # Collect all metrics
        collection_timestamp = datetime.now()

        metrics_data = {
            'collection_timestamp': collection_timestamp,
            'application_id': app_id,
            'application_name': self.config['applications']['primary_app'],
            'business_transactions': self.get_business_transactions(app_id, duration_in_mins=duration_in_mins),
            'infrastructure_metrics': self.get_infrastructure_metrics(app_id, duration_in_mins),
            'application_health': self.get_application_health(app_id),
            'collection_duration_minutes': duration_in_mins
        }

        self.logger.info(f"Collected {len(metrics_data['business_transactions'])} business transactions and "
                        f"{len(metrics_data['infrastructure_metrics'])} infrastructure metrics")

        return metrics_data

    def save_metrics_to_file(self, metrics_data: Dict[str, Any], output_path: str = None):
        """Save collected metrics to JSON file"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/appdynamics_metrics_{timestamp}.json"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert datetime objects to strings for JSON serialization
        serializable_data = self._make_json_serializable(metrics_data)

        try:
            with open(output_path, 'w') as file:
                json.dump(serializable_data, file, indent=2)
            self.logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to file: {e}")

    def _make_json_serializable(self, obj):
        """Convert datetime objects to strings for JSON serialization"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    collector = AppDynamicsCollector()
    metrics = collector.collect_all_metrics(duration_in_mins=30)
    collector.save_metrics_to_file(metrics)

    print(f"Collected metrics for {len(metrics['business_transactions'])} business transactions")
    print(f"Application health: {metrics['application_health']['availability_percentage']:.2f}% availability")