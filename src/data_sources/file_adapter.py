"""
File Data Source Adapter
Implements the generic data source interface for CSV and JSON files
"""

import pandas as pd
import json
import logging
from datetime import datetime
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


class FileAdapter(DataSourceAdapter):
    """File-based implementation of the generic data source adapter"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.file_path = config.connection_params.get('file_path')
        self.file_type = config.connection_params.get('file_type', 'auto')
        self.logger = logging.getLogger(__name__)
        self.data_cache = None
        self.last_modified = None

    def connect(self) -> bool:
        """Establish connection to file (load data)"""
        try:
            return self._load_data()
        except Exception as e:
            self.logger.error(f"Failed to connect to file {self.file_path}: {e}")
            return False

    def test_connection(self) -> bool:
        """Test file connection"""
        try:
            if not self.file_path:
                return False

            file_path = Path(self.file_path)
            return file_path.exists() and file_path.is_file()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def _load_data(self) -> bool:
        """Load data from file"""
        try:
            file_path = Path(self.file_path)
            if not file_path.exists():
                self.logger.error(f"File not found: {self.file_path}")
                return False

            # Check if file has been modified since last load
            current_modified = file_path.stat().st_mtime
            if self.data_cache is not None and self.last_modified == current_modified:
                return True

            # Determine file type
            if self.file_type == 'auto':
                if file_path.suffix.lower() == '.csv':
                    file_type = 'csv'
                elif file_path.suffix.lower() == '.json':
                    file_type = 'json'
                else:
                    self.logger.error(f"Unsupported file type: {file_path.suffix}")
                    return False
            else:
                file_type = self.file_type

            # Load data based on file type
            if file_type == 'csv':
                self.data_cache = pd.read_csv(file_path)
            elif file_type == 'json':
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        self.data_cache = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict) and 'data' in json_data:
                        self.data_cache = pd.DataFrame(json_data['data'])
                    else:
                        self.data_cache = pd.DataFrame([json_data])

            self.last_modified = current_modified

            # Standardize column names
            self._standardize_columns()

            return True
        except Exception as e:
            self.logger.error(f"Error loading data from {self.file_path}: {e}")
            return False

    def _standardize_columns(self):
        """Standardize column names for easier processing"""
        if self.data_cache is None:
            return

        # Create mapping of common column name variations
        column_mappings = {
            # Timestamp columns
            'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'ts'],
            'service_name': ['service_name', 'service', 'application', 'app', 'component'],
            'metric_name': ['metric_name', 'metric', 'name', 'metric_type'],
            'value': ['value', 'val', 'measurement', 'amount'],
            'metric_type': ['metric_type', 'type', 'category'],
            'unit': ['unit', 'units', 'measurement_unit']
        }

        # Apply mappings
        columns = self.data_cache.columns.tolist()
        for standard_col, variations in column_mappings.items():
            for col in columns:
                if col.lower() in [v.lower() for v in variations]:
                    if col != standard_col:
                        self.data_cache = self.data_cache.rename(columns={col: standard_col})
                    break

        # Ensure timestamp is datetime
        if 'timestamp' in self.data_cache.columns:
            self.data_cache['timestamp'] = pd.to_datetime(self.data_cache['timestamp'])

    def get_available_services(self) -> List[str]:
        """Get list of services from file data"""
        try:
            if not self._load_data():
                return []

            if 'service_name' in self.data_cache.columns:
                return sorted(self.data_cache['service_name'].unique().tolist())
            elif 'service' in self.data_cache.columns:
                return sorted(self.data_cache['service'].unique().tolist())
            else:
                # If no service column, return a default service name
                return ['default_service']
        except Exception as e:
            self.logger.error(f"Error getting available services: {e}")
            return []

    def get_available_metrics(self, service_name: str) -> List[str]:
        """Get available metrics for a service from file data"""
        try:
            if not self._load_data():
                return []

            # Filter data for the service
            service_data = self._filter_by_service(service_name)

            if 'metric_name' in service_data.columns:
                return sorted(service_data['metric_name'].unique().tolist())
            elif 'metric' in service_data.columns:
                return sorted(service_data['metric'].unique().tolist())
            else:
                # Return column names that might be metrics
                excluded_cols = ['timestamp', 'service_name', 'service', 'metric_type', 'unit']
                metric_cols = [col for col in service_data.columns if col not in excluded_cols]
                return metric_cols
        except Exception as e:
            self.logger.error(f"Error getting available metrics for {service_name}: {e}")
            return []

    def _filter_by_service(self, service_name: str) -> pd.DataFrame:
        """Filter data by service name"""
        if self.data_cache is None:
            return pd.DataFrame()

        if 'service_name' in self.data_cache.columns:
            return self.data_cache[self.data_cache['service_name'] == service_name]
        elif 'service' in self.data_cache.columns:
            return self.data_cache[self.data_cache['service'] == service_name]
        else:
            # If no service column, return all data
            return self.data_cache

    def query_metrics(self, params: QueryParams) -> List[StandardMetric]:
        """Query metrics from file data"""
        metrics = []

        try:
            if not self._load_data():
                return metrics

            for service in params.services or self.get_available_services():
                service_data = self._filter_by_service(service)

                # Filter by time range
                if 'timestamp' in service_data.columns:
                    mask = (
                        (service_data['timestamp'] >= params.start_time) &
                        (service_data['timestamp'] <= params.end_time)
                    )
                    service_data = service_data[mask]

                # Process metrics based on data structure
                service_metrics = self._extract_metrics_from_dataframe(
                    service_data, service, params
                )
                metrics.extend(service_metrics)

        except Exception as e:
            self.logger.error(f"Error querying metrics: {e}")

        return metrics

    def _extract_metrics_from_dataframe(self, df: pd.DataFrame, service_name: str,
                                      params: QueryParams) -> List[StandardMetric]:
        """Extract StandardMetric objects from DataFrame"""
        metrics = []

        if df.empty:
            return metrics

        try:
            # Check if data is in standard format (metric_name, value, timestamp columns)
            if all(col in df.columns for col in ['metric_name', 'value', 'timestamp']):
                # Standard format
                for _, row in df.iterrows():
                    metric_type = self._infer_metric_type(row.get('metric_name', ''))

                    # Filter by metric types if specified
                    if params.metric_types and metric_type not in params.metric_types:
                        continue

                    metric_id = f"{service_name}:{row['metric_name']}:{row['timestamp']}"

                    standard_metric = StandardMetric(
                        metric_id=metric_id,
                        metric_type=metric_type,
                        service_name=service_name,
                        metric_name=row['metric_name'],
                        value=float(row['value']),
                        timestamp=row['timestamp'],
                        unit=row.get('unit', ''),
                        tags={
                            "source_row": str(row.name)
                        },
                        raw_data=row.to_dict()
                    )

                    metrics.append(standard_metric)

            else:
                # Wide format - each column is a metric
                timestamp_col = None
                for col in ['timestamp', 'time', 'datetime']:
                    if col in df.columns:
                        timestamp_col = col
                        break

                excluded_cols = [timestamp_col, 'service_name', 'service']
                metric_columns = [col for col in df.columns if col not in excluded_cols]

                for _, row in df.iterrows():
                    timestamp = row[timestamp_col] if timestamp_col else datetime.now()

                    for metric_col in metric_columns:
                        if pd.isna(row[metric_col]):
                            continue

                        metric_type = self._infer_metric_type(metric_col)

                        # Filter by metric types if specified
                        if params.metric_types and metric_type not in params.metric_types:
                            continue

                        metric_id = f"{service_name}:{metric_col}:{timestamp}"

                        standard_metric = StandardMetric(
                            metric_id=metric_id,
                            metric_type=metric_type,
                            service_name=service_name,
                            metric_name=metric_col,
                            value=float(row[metric_col]),
                            timestamp=timestamp,
                            unit=self._infer_unit(metric_col),
                            tags={
                                "source_row": str(row.name),
                                "source_column": metric_col
                            },
                            raw_data=row.to_dict()
                        )

                        metrics.append(standard_metric)

        except Exception as e:
            self.logger.error(f"Error extracting metrics from DataFrame: {e}")

        return metrics

    def _infer_metric_type(self, metric_name: str) -> MetricType:
        """Infer metric type from metric name"""
        metric_name_lower = metric_name.lower()

        if any(term in metric_name_lower for term in ['response_time', 'latency', 'duration']):
            return MetricType.RESPONSE_TIME
        elif any(term in metric_name_lower for term in ['error_rate', 'error', 'failure']):
            return MetricType.ERROR_RATE
        elif any(term in metric_name_lower for term in ['throughput', 'requests', 'rps', 'qps']):
            return MetricType.THROUGHPUT
        elif any(term in metric_name_lower for term in ['cpu', 'processor']):
            return MetricType.CPU_UTILIZATION
        elif any(term in metric_name_lower for term in ['memory', 'ram']):
            return MetricType.MEMORY_UTILIZATION
        elif any(term in metric_name_lower for term in ['disk', 'storage']):
            return MetricType.DISK_UTILIZATION
        elif any(term in metric_name_lower for term in ['network', 'io', 'bandwidth']):
            return MetricType.NETWORK_IO
        elif any(term in metric_name_lower for term in ['availability', 'uptime', 'up']):
            return MetricType.AVAILABILITY
        else:
            return MetricType.CUSTOM

    def _infer_unit(self, metric_name: str) -> str:
        """Infer unit from metric name"""
        metric_name_lower = metric_name.lower()

        if any(term in metric_name_lower for term in ['percent', '%', 'rate']):
            return '%'
        elif any(term in metric_name_lower for term in ['time', 'latency', 'duration']):
            return 'ms'
        elif any(term in metric_name_lower for term in ['bytes', 'memory', 'disk']):
            return 'bytes'
        elif any(term in metric_name_lower for term in ['requests', 'calls']):
            return 'count'
        else:
            return ''

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of file data source"""
        try:
            connected = self.test_connection()
            services = self.get_available_services() if connected else []

            file_info = {}
            if connected:
                file_path = Path(self.file_path)
                file_info = {
                    "size_bytes": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "rows": len(self.data_cache) if self.data_cache is not None else 0,
                    "columns": list(self.data_cache.columns) if self.data_cache is not None else []
                }

            return {
                "status": "healthy" if connected else "unhealthy",
                "connected": connected,
                "file_path": self.file_path,
                "file_type": self.file_type,
                "available_services": len(services),
                "services": services[:5],  # Show first 5 services
                "file_info": file_info,
                "last_checked": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }