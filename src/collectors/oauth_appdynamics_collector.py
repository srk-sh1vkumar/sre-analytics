"""
AppDynamics Data Collector with OAuth Token Authentication
Collects performance metrics and business transaction data from AppDynamics API using OAuth
"""

import requests
import json
import yaml
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

@dataclass
class OAuthToken:
    """OAuth token data structure"""
    access_token: str
    token_type: str
    expires_in: int
    expires_at: datetime

class OAuthAppDynamicsCollector:
    """Collects metrics from AppDynamics Controller API using OAuth authentication"""

    def __init__(self, config_path: str = "config/appdynamics_config.yaml"):
        self.config = self._load_config(config_path)
        self.controller_url = f"https://{self.config['controller']['host']}"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self.token: Optional[OAuthToken] = None

        # OAuth credentials from environment
        self.client_id = os.getenv('APPDYNAMICS_CLIENT_ID')
        self.client_secret = os.getenv('APPDYNAMICS_CLIENT_SECRET')

        if not self.client_id or not self.client_secret:
            raise ValueError("AppDynamics OAuth credentials not found in environment variables")

        # Get initial token
        self._get_oauth_token()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load AppDynamics configuration"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Update controller host from environment
            config['controller']['host'] = os.getenv(
                'APPDYNAMICS_CONTROLLER_HOST',
                config.get('controller', {}).get('host', 'localhost')
            )

            # Set primary application from environment if available
            if 'applications' not in config:
                config['applications'] = {}
            config['applications']['primary_app'] = os.getenv(
                'DEFAULT_APPLICATION_NAME',
                config.get('applications', {}).get('primary_app', 'Default-App')
            )

            return config
        except Exception as e:
            self.logger.warning(f"Failed to load AppDynamics config: {e}")
            # Return default config
            return {
                'controller': {
                    'host': os.getenv('APPDYNAMICS_CONTROLLER_HOST', 'localhost')
                },
                'applications': {
                    'primary_app': os.getenv('DEFAULT_APPLICATION_NAME', 'Default-App')
                },
                'api': {
                    'timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 5
                }
            }

    def _get_oauth_token(self) -> bool:
        """Get OAuth access token from AppDynamics"""
        token_url = f"{self.controller_url}/controller/api/oauth/access_token"

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }

        # Enhanced logging for troubleshooting
        self.logger.info("=== OAuth Authentication Debug ===")
        self.logger.info(f"Controller URL: {self.controller_url}")
        self.logger.info(f"Token URL: {token_url}")
        self.logger.info(f"Client ID: {self.client_id}")
        self.logger.info(f"Client Secret: {'*' * (len(self.client_secret) - 4) + self.client_secret[-4:] if self.client_secret else 'None'}")
        self.logger.info(f"Headers: {headers}")
        self.logger.info(f"Request Data: {dict(data)}")

        try:
            self.logger.info("Sending OAuth token request to AppDynamics...")
            response = requests.post(token_url, headers=headers, data=data, timeout=30)

            # Log response details
            self.logger.info(f"Response Status Code: {response.status_code}")
            self.logger.info(f"Response Headers: {dict(response.headers)}")
            self.logger.info(f"Response URL: {response.url}")

            if response.status_code != 200:
                self.logger.error(f"Response Content: {response.text[:1000]}...")  # First 1000 chars

            if response.status_code == 200:
                try:
                    token_data = response.json()
                    self.logger.info(f"Token response data: {token_data}")

                    self.token = OAuthToken(
                        access_token=token_data['access_token'],
                        token_type=token_data.get('token_type', 'Bearer'),
                        expires_in=token_data.get('expires_in', 3600),
                        expires_at=datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600))
                    )

                    # Update session headers with token
                    self.session.headers.update({
                        'Authorization': f'{self.token.token_type} {self.token.access_token}',
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    })

                    self.logger.info("✅ OAuth token obtained successfully")
                    self.logger.info(f"Token Type: {self.token.token_type}")
                    self.logger.info(f"Expires in: {self.token.expires_in} seconds")
                    self.logger.info(f"Expires at: {self.token.expires_at}")
                    return True
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.error(f"Failed to parse token response: {e}")
                    self.logger.error(f"Raw response: {response.text}")
                    return False
            else:
                self.logger.error(f"❌ Failed to get OAuth token: {response.status_code}")
                self.logger.error(f"Response text: {response.text}")

                # Additional troubleshooting info
                if response.status_code == 401:
                    self.logger.error("🔍 Troubleshooting 401 Unauthorized:")
                    self.logger.error("1. Verify client_id and client_secret are correct")
                    self.logger.error("2. Check if OAuth is enabled for your AppDynamics controller")
                    self.logger.error("3. Verify the client has appropriate permissions")
                elif response.status_code == 404:
                    self.logger.error("🔍 Troubleshooting 404 Not Found:")
                    self.logger.error("1. Check if controller URL is correct")
                    self.logger.error("2. Verify OAuth endpoint is available on this controller version")
                elif response.status_code == 500:
                    self.logger.error("🔍 Troubleshooting 500 Server Error:")
                    self.logger.error("1. Check AppDynamics controller status")
                    self.logger.error("2. Verify OAuth service is running")
                    self.logger.error("3. Check controller logs for more details")

                return False

        except requests.exceptions.ConnectTimeout as e:
            self.logger.error(f"❌ Connection timeout: {e}")
            self.logger.error("🔍 Check network connectivity to AppDynamics controller")
            return False
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"❌ Connection error: {e}")
            self.logger.error("🔍 Check if controller URL is correct and accessible")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ OAuth token request failed: {e}")
            return False

    def _refresh_token_if_needed(self) -> bool:
        """Refresh OAuth token if it's about to expire"""
        if not self.token:
            return self._get_oauth_token()

        # Refresh token if it expires in less than 5 minutes
        if datetime.now() >= (self.token.expires_at - timedelta(minutes=5)):
            self.logger.info("Token expiring soon, refreshing...")
            return self._get_oauth_token()

        return True

    def _make_authenticated_request(self, url: str, params: Dict = None,
                                  method: str = 'GET') -> Optional[requests.Response]:
        """Make authenticated request to AppDynamics API"""
        if not self._refresh_token_if_needed():
            self.logger.error("Failed to refresh OAuth token")
            return None

        max_retries = self.config.get('api', {}).get('retry_attempts', 3)
        retry_delay = self.config.get('api', {}).get('retry_delay', 5)
        timeout = self.config.get('api', {}).get('timeout', 30)

        for attempt in range(max_retries):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=timeout)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=params, timeout=timeout)
                else:
                    response = self.session.request(method, url, params=params, timeout=timeout)

                if response.status_code == 401:
                    # Token expired, try to refresh
                    self.logger.warning("Received 401, attempting to refresh token...")
                    if self._get_oauth_token():
                        continue  # Retry with new token
                    else:
                        self.logger.error("Failed to refresh token after 401")
                        return None

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All {max_retries} attempts failed for {url}")
                    return None

        return None

    def get_applications(self) -> List[Dict[str, Any]]:
        """Get list of applications from AppDynamics"""
        url = f"{self.controller_url}/controller/rest/applications"

        response = self._make_authenticated_request(url)
        if response:
            try:
                # AppDynamics returns XML by default, parse it
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.text)
                applications = []

                for app_elem in root.findall('.//application'):
                    app = {}
                    app['id'] = int(app_elem.find('id').text) if app_elem.find('id') is not None else None
                    app['name'] = app_elem.find('name').text if app_elem.find('name') is not None else None
                    app['description'] = app_elem.find('description').text if app_elem.find('description') is not None else ""
                    if app['name']:  # Only add if name exists
                        applications.append(app)

                self.logger.info(f"✅ Found {len(applications)} applications via XML parsing")
                return applications

            except Exception as xml_error:
                self.logger.error(f"Failed to parse XML response: {xml_error}")
                # Try JSON fallback
                try:
                    apps = response.json()
                    self.logger.info(f"✅ Found {len(apps)} applications via JSON fallback")
                    return apps
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse applications response: {e}")
                    self.logger.error(f"Response content: {response.text[:500]}")
                    return []
        else:
            self.logger.error("Failed to get applications")
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

        response = self._make_authenticated_request(url, params)
        if not response:
            return []

        try:
            transactions = []
            current_time = datetime.now()

            for bt_data in response.json():
                # Get detailed metrics for each transaction
                bt_metrics = self._get_transaction_metrics(app_id, bt_data['id'], duration_in_mins)

                transaction = BusinessTransaction(
                    name=bt_data['name'],
                    tier=bt_data.get('tierName', 'Unknown'),
                    calls_per_minute=bt_metrics.get('calls_per_minute', 0),
                    response_time_avg=bt_metrics.get('response_time_avg', 0),
                    response_time_p95=bt_metrics.get('response_time_p95', 0),
                    response_time_p99=bt_metrics.get('response_time_p99', 0),
                    error_rate=bt_metrics.get('error_rate', 0),
                    timestamp=current_time
                )
                transactions.append(transaction)

            return transactions

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse business transactions: {e}")
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

            response = self._make_authenticated_request(url, params)
            if response:
                try:
                    metric_data = response.json()
                    if metric_data and len(metric_data) > 0:
                        # Get the latest value
                        if metric_data[0].get('metricValues'):
                            latest_value = metric_data[0]['metricValues'][-1]['value']

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

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse metric data for {metric_path}: {e}")
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

            response = self._make_authenticated_request(url, params)
            if response:
                try:
                    metric_data = response.json()

                    for metric in metric_data:
                        if metric.get('metricValues'):
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

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse infrastructure metric {metric_path}: {e}")
                    continue

        return infrastructure_metrics

    def get_application_health(self, app_id: int) -> Dict[str, Any]:
        """Get overall application health status"""
        url = f"{self.controller_url}/controller/rest/applications/{app_id}/nodes"

        response = self._make_authenticated_request(url)
        if response:
            try:
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

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse application health: {e}")
                return self._get_default_health_status()
        else:
            self.logger.error("Failed to get application health")
            return self._get_default_health_status()

    def _get_default_health_status(self) -> Dict[str, Any]:
        """Return default health status when API call fails"""
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
        primary_app_name = self.config['applications']['primary_app']

        for app in applications:
            if app['name'] == primary_app_name:
                app_id = app['id']
                self.logger.info(f"Found application '{primary_app_name}' with ID: {app_id}")
                break

        if not app_id:
            self.logger.warning(f"Application '{primary_app_name}' not found")
            if applications:
                # Use first available application
                app_id = applications[0]['id']
                primary_app_name = applications[0]['name']
                self.logger.info(f"Using first available application: '{primary_app_name}' (ID: {app_id})")
            else:
                self.logger.error("No applications found")
                return self._get_empty_metrics_data()

        # Collect all metrics
        collection_timestamp = datetime.now()

        metrics_data = {
            'collection_timestamp': collection_timestamp,
            'application_id': app_id,
            'application_name': primary_app_name,
            'business_transactions': self.get_business_transactions(app_id, duration_in_mins=duration_in_mins),
            'infrastructure_metrics': self.get_infrastructure_metrics(app_id, duration_in_mins),
            'application_health': self.get_application_health(app_id),
            'collection_duration_minutes': duration_in_mins,
            'oauth_token_status': 'valid' if self.token else 'invalid'
        }

        self.logger.info(f"Collected {len(metrics_data['business_transactions'])} business transactions and "
                        f"{len(metrics_data['infrastructure_metrics'])} infrastructure metrics")

        return metrics_data

    def _get_empty_metrics_data(self) -> Dict[str, Any]:
        """Return empty metrics data structure"""
        return {
            'collection_timestamp': datetime.now(),
            'application_id': None,
            'application_name': 'No Application Found',
            'business_transactions': [],
            'infrastructure_metrics': [],
            'application_health': self._get_default_health_status(),
            'collection_duration_minutes': 0,
            'oauth_token_status': 'valid' if self.token else 'invalid'
        }

    def save_metrics_to_file(self, metrics_data: Dict[str, Any], output_path: str = None):
        """Save collected metrics to JSON file"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.getenv('REPORT_OUTPUT_PATH', 'reports/generated')
            output_path = f"{output_dir}/appdynamics_oauth_metrics_{timestamp}.json"

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

    def diagnose_connection_issues(self) -> Dict[str, Any]:
        """Diagnose connection and authentication issues"""
        diagnosis = {
            'controller_reachable': False,
            'oauth_endpoint_available': False,
            'credentials_valid': False,
            'ssl_issues': False,
            'network_issues': False,
            'recommendations': []
        }

        # Test 1: Basic controller connectivity
        try:
            basic_url = f"{self.controller_url}/controller"
            self.logger.info(f"Testing basic connectivity to: {basic_url}")
            response = requests.get(basic_url, timeout=10)
            diagnosis['controller_reachable'] = True
            self.logger.info("✅ Controller is reachable")
        except requests.exceptions.SSLError as e:
            diagnosis['ssl_issues'] = True
            diagnosis['recommendations'].append("SSL certificate issues detected - verify HTTPS configuration")
            self.logger.error(f"SSL Error: {e}")
        except requests.exceptions.ConnectionError as e:
            diagnosis['network_issues'] = True
            diagnosis['recommendations'].append("Network connectivity issues - check URL and firewall settings")
            self.logger.error(f"Connection Error: {e}")
        except Exception as e:
            self.logger.error(f"Basic connectivity test failed: {e}")

        # Test 2: OAuth endpoint availability
        try:
            oauth_url = f"{self.controller_url}/controller/api/oauth"
            self.logger.info(f"Testing OAuth endpoint availability: {oauth_url}")
            response = requests.get(oauth_url, timeout=10)
            if response.status_code != 404:
                diagnosis['oauth_endpoint_available'] = True
                self.logger.info("✅ OAuth endpoint is available")
            else:
                diagnosis['recommendations'].append("OAuth endpoint not found - check AppDynamics version and OAuth enablement")
        except Exception as e:
            self.logger.error(f"OAuth endpoint test failed: {e}")

        # Test 3: Try different authentication methods
        if not diagnosis['credentials_valid']:
            diagnosis['recommendations'].extend([
                "Verify client_id and client_secret are correct",
                "Check if client credentials are properly configured in AppDynamics",
                "Ensure OAuth client has necessary permissions",
                "Try accessing AppDynamics UI with same credentials"
            ])

        return diagnosis

    def test_connection(self) -> Dict[str, Any]:
        """Test OAuth connection and basic API access"""
        test_results = {
            'oauth_authentication': False,
            'applications_access': False,
            'applications_count': 0,
            'primary_app_found': False,
            'error_message': None,
            'diagnosis': None
        }

        try:
            # Run diagnosis first
            diagnosis = self.diagnose_connection_issues()
            test_results['diagnosis'] = diagnosis

            # Test OAuth token
            if self.token:
                test_results['oauth_authentication'] = True
                self.logger.info("✅ OAuth authentication successful")
            else:
                self.logger.warning("❌ OAuth token not available, running diagnosis...")
                test_results['error_message'] = "OAuth token not available"

                # Print diagnosis results
                if diagnosis['recommendations']:
                    self.logger.info("🔧 Recommendations:")
                    for i, rec in enumerate(diagnosis['recommendations'], 1):
                        self.logger.info(f"   {i}. {rec}")

                return test_results

            # Test applications access
            applications = self.get_applications()
            if applications:
                test_results['applications_access'] = True
                test_results['applications_count'] = len(applications)
                self.logger.info(f"✅ Applications access successful ({len(applications)} apps found)")

                # Check for primary application
                primary_app = self.config['applications']['primary_app']
                for app in applications:
                    if app['name'] == primary_app:
                        test_results['primary_app_found'] = True
                        self.logger.info(f"✅ Primary application '{primary_app}' found")
                        break
                else:
                    self.logger.warning(f"⚠️ Primary application '{primary_app}' not found")
            else:
                test_results['error_message'] = "No applications accessible"

        except Exception as e:
            test_results['error_message'] = str(e)
            self.logger.error(f"Connection test failed: {e}")

        return test_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        print("🔐 Testing AppDynamics OAuth Collector...")
        collector = OAuthAppDynamicsCollector()

        # Test connection
        test_results = collector.test_connection()
        print("\n📊 Connection Test Results:")
        print(f"• OAuth Authentication: {'✅' if test_results['oauth_authentication'] else '❌'}")
        print(f"• Applications Access: {'✅' if test_results['applications_access'] else '❌'}")
        print(f"• Applications Found: {test_results['applications_count']}")
        print(f"• Primary App Found: {'✅' if test_results['primary_app_found'] else '❌'}")

        if test_results['error_message']:
            print(f"• Error: {test_results['error_message']}")

        # If connection is successful, collect some basic metrics
        if test_results['oauth_authentication'] and test_results['applications_access']:
            print("\n📈 Collecting sample metrics...")
            metrics = collector.collect_all_metrics(duration_in_mins=10)
            collector.save_metrics_to_file(metrics)

            print(f"✅ Successfully collected metrics for {metrics['application_name']}")
            print(f"• Business Transactions: {len(metrics['business_transactions'])}")
            print(f"• Infrastructure Metrics: {len(metrics['infrastructure_metrics'])}")
            print(f"• Application Health: {metrics['application_health']['availability_percentage']:.2f}% availability")

    except Exception as e:
        print(f"❌ Error initializing collector: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check .env file has correct OAuth credentials")
        print("2. Verify controller URL is accessible")
        print("3. Ensure client ID and secret are valid")
        print("4. Check network connectivity to AppDynamics controller")