"""
Generic Data Source Abstraction Layer
Base interfaces for pluggable data source adapters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class MetricType(Enum):
    """Standard metric types across all data sources"""
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    DISK_UTILIZATION = "disk_utilization"
    NETWORK_IO = "network_io"
    DATABASE_CONNECTIONS = "database_connections"
    CUSTOM = "custom"


class DataSourceType(Enum):
    """Supported data source types"""
    APPDYNAMICS = "appdynamics"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SPLUNK = "splunk"
    ELASTICSEARCH = "elasticsearch"
    CUSTOM_API = "custom_api"
    CSV_FILE = "csv_file"
    JSON_FILE = "json_file"
    DATABASE = "database"


@dataclass
class StandardMetric:
    """Standardized metric structure across all data sources"""
    metric_id: str
    metric_type: MetricType
    service_name: str
    metric_name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = None
    raw_data: Dict[str, Any] = None  # Store original data for debugging


@dataclass
class DataSourceConfig:
    """Generic configuration for data sources"""
    source_type: DataSourceType
    name: str
    connection_params: Dict[str, Any]
    metric_mappings: Dict[str, str] = None  # Map source metrics to standard types
    authentication: Dict[str, Any] = None
    polling_interval: int = 300  # seconds
    enabled: bool = True


@dataclass
class QueryParams:
    """Parameters for querying data sources"""
    start_time: datetime
    end_time: datetime
    services: List[str] = None
    metric_types: List[MetricType] = None
    filters: Dict[str, Any] = None
    aggregation: str = "avg"  # avg, sum, max, min, p95, p99


class DataSourceAdapter(ABC):
    """Abstract base class for all data source adapters"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.source_type = config.source_type
        self.name = config.name
        self.enabled = config.enabled

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is working"""
        pass

    @abstractmethod
    def get_available_services(self) -> List[str]:
        """Get list of available services/applications"""
        pass

    @abstractmethod
    def get_available_metrics(self, service_name: str) -> List[str]:
        """Get available metrics for a service"""
        pass

    @abstractmethod
    def query_metrics(self, params: QueryParams) -> List[StandardMetric]:
        """Query metrics from the data source"""
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the data source"""
        pass

    def is_enabled(self) -> bool:
        """Check if this data source is enabled"""
        return self.enabled

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this data source"""
        return {
            "name": self.name,
            "type": self.source_type.value,
            "enabled": self.enabled,
            "polling_interval": self.config.polling_interval
        }


class DataSourceRegistry:
    """Registry for managing multiple data source adapters"""

    def __init__(self):
        self.adapters: Dict[str, DataSourceAdapter] = {}
        self.configs: Dict[str, DataSourceConfig] = {}

    def register_adapter(self, adapter: DataSourceAdapter):
        """Register a new data source adapter"""
        self.adapters[adapter.name] = adapter
        self.configs[adapter.name] = adapter.config

    def get_adapter(self, name: str) -> Optional[DataSourceAdapter]:
        """Get adapter by name"""
        return self.adapters.get(name)

    def get_enabled_adapters(self) -> List[DataSourceAdapter]:
        """Get all enabled adapters"""
        return [adapter for adapter in self.adapters.values() if adapter.is_enabled()]

    def list_adapters(self) -> List[str]:
        """List all registered adapter names"""
        return list(self.adapters.keys())

    def remove_adapter(self, name: str):
        """Remove an adapter from registry"""
        if name in self.adapters:
            del self.adapters[name]
        if name in self.configs:
            del self.configs[name]

    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of all registered adapters"""
        status = {}
        for name, adapter in self.adapters.items():
            try:
                status[name] = {
                    "enabled": adapter.is_enabled(),
                    "connected": adapter.test_connection(),
                    "metadata": adapter.get_metadata()
                }
            except Exception as e:
                status[name] = {
                    "enabled": adapter.is_enabled(),
                    "connected": False,
                    "error": str(e),
                    "metadata": adapter.get_metadata()
                }
        return status


class MetricAggregator:
    """Aggregates metrics from multiple data sources"""

    def __init__(self, registry: DataSourceRegistry):
        self.registry = registry

    def collect_metrics(self, params: QueryParams) -> Dict[str, List[StandardMetric]]:
        """Collect metrics from all enabled data sources"""
        results = {}

        for adapter in self.registry.get_enabled_adapters():
            try:
                metrics = adapter.query_metrics(params)
                results[adapter.name] = metrics
            except Exception as e:
                print(f"Error collecting from {adapter.name}: {e}")
                results[adapter.name] = []

        return results

    def merge_metrics(self, source_metrics: Dict[str, List[StandardMetric]]) -> List[StandardMetric]:
        """Merge metrics from multiple sources, handling duplicates"""
        merged = []
        seen_metrics = set()

        for source_name, metrics in source_metrics.items():
            for metric in metrics:
                # Create unique key for deduplication
                key = f"{metric.service_name}:{metric.metric_type.value}:{metric.timestamp}"

                if key not in seen_metrics:
                    # Add source information to tags
                    if metric.tags is None:
                        metric.tags = {}
                    metric.tags["source"] = source_name

                    merged.append(metric)
                    seen_metrics.add(key)

        return merged

    def get_consolidated_services(self) -> List[str]:
        """Get list of all services across all data sources"""
        services = set()

        for adapter in self.registry.get_enabled_adapters():
            try:
                services.update(adapter.get_available_services())
            except Exception as e:
                print(f"Error getting services from {adapter.name}: {e}")

        return sorted(list(services))