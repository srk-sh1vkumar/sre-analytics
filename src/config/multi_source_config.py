"""
Multi-Source Configuration System
Flexible configuration management for multiple data sources
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

from ..data_sources.base import DataSourceConfig, DataSourceType, MetricType
from ..analytics.generic_metrics_engine import SLOTarget


@dataclass
class AnalyticsConfig:
    """Configuration for analytics engine"""
    default_slo_targets: Dict[str, List[Dict[str, Any]]]
    analysis_window_hours: int = 24
    confidence_threshold: float = 0.6
    enable_ai_recommendations: bool = True
    export_formats: List[str] = None


@dataclass
class ReportingConfig:
    """Configuration for reporting system"""
    output_directory: str = "reports/generated"
    template_directory: str = "templates"
    default_format: str = "html"
    include_charts: bool = True
    chart_resolution: str = "high"
    enable_pdf_generation: bool = True


@dataclass
class MultiSourceConfig:
    """Main configuration class for multi-source analytics"""
    data_sources: List[DataSourceConfig]
    analytics: AnalyticsConfig
    reporting: ReportingConfig
    global_settings: Dict[str, Any]


class ConfigurationManager:
    """Manager for multi-source configuration"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/multi_source_config.yaml"
        self.logger = logging.getLogger(__name__)
        self.config: Optional[MultiSourceConfig] = None

    def load_config(self) -> MultiSourceConfig:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)

            if not config_file.exists():
                self.logger.warning(f"Config file not found: {self.config_path}. Creating default config.")
                self.config = self._create_default_config()
                self.save_config()
                return self.config

            # Load based on file extension
            if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")

            self.config = self._parse_config_data(config_data)
            return self.config

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config = self._create_default_config()
            return self.config

    def save_config(self, config: Optional[MultiSourceConfig] = None):
        """Save configuration to file"""
        try:
            config_to_save = config or self.config
            if not config_to_save:
                raise ValueError("No configuration to save")

            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dictionary
            config_dict = self._config_to_dict(config_to_save)

            # Save based on file extension
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

            self.logger.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def _create_default_config(self) -> MultiSourceConfig:
        """Create default configuration"""
        # Default data sources
        data_sources = [
            DataSourceConfig(
                source_type=DataSourceType.APPDYNAMICS,
                name="appdynamics_primary",
                connection_params={
                    "host": "your-controller.saas.appdynamics.com",
                    "port": 443,
                    "ssl": True
                },
                authentication={
                    "username": "${APPDYNAMICS_USERNAME}",
                    "password": "${APPDYNAMICS_PASSWORD}",
                    "account": "${APPDYNAMICS_ACCOUNT}"
                },
                polling_interval=300,
                enabled=False
            ),
            DataSourceConfig(
                source_type=DataSourceType.PROMETHEUS,
                name="prometheus_primary",
                connection_params={
                    "url": "http://localhost:9090"
                },
                authentication={
                    "bearer_token": "${PROMETHEUS_TOKEN}"
                },
                polling_interval=60,
                enabled=False
            ),
            DataSourceConfig(
                source_type=DataSourceType.CSV_FILE,
                name="sample_csv_data",
                connection_params={
                    "file_path": "data/sample_metrics.csv",
                    "file_type": "csv"
                },
                polling_interval=3600,
                enabled=True
            )
        ]

        # Default SLO targets
        default_slo_targets = {
            "default": [
                {
                    "metric_type": "response_time",
                    "target_value": 200,
                    "comparison": "less_than",
                    "unit": "ms",
                    "description": "Response time should be under 200ms"
                },
                {
                    "metric_type": "error_rate",
                    "target_value": 1.0,
                    "comparison": "less_than",
                    "unit": "%",
                    "description": "Error rate should be under 1%"
                },
                {
                    "metric_type": "availability",
                    "target_value": 99.9,
                    "comparison": "greater_than",
                    "unit": "%",
                    "description": "Availability should be above 99.9%"
                },
                {
                    "metric_type": "cpu_utilization",
                    "target_value": 80,
                    "comparison": "less_than",
                    "unit": "%",
                    "description": "CPU utilization should be under 80%"
                },
                {
                    "metric_type": "memory_utilization",
                    "target_value": 85,
                    "comparison": "less_than",
                    "unit": "%",
                    "description": "Memory utilization should be under 85%"
                }
            ]
        }

        analytics = AnalyticsConfig(
            default_slo_targets=default_slo_targets,
            analysis_window_hours=24,
            confidence_threshold=0.6,
            enable_ai_recommendations=True,
            export_formats=["json", "csv", "html"]
        )

        reporting = ReportingConfig(
            output_directory="reports/generated",
            template_directory="templates",
            default_format="html",
            include_charts=True,
            chart_resolution="high",
            enable_pdf_generation=True
        )

        global_settings = {
            "timezone": "UTC",
            "log_level": "INFO",
            "cache_duration_minutes": 15,
            "max_concurrent_sources": 5,
            "retry_attempts": 3,
            "retry_delay_seconds": 5
        }

        return MultiSourceConfig(
            data_sources=data_sources,
            analytics=analytics,
            reporting=reporting,
            global_settings=global_settings
        )

    def _parse_config_data(self, config_data: Dict[str, Any]) -> MultiSourceConfig:
        """Parse configuration data into MultiSourceConfig object"""
        # Parse data sources
        data_sources = []
        for ds_data in config_data.get('data_sources', []):
            source_type = DataSourceType(ds_data['source_type'])

            data_source = DataSourceConfig(
                source_type=source_type,
                name=ds_data['name'],
                connection_params=ds_data.get('connection_params', {}),
                metric_mappings=ds_data.get('metric_mappings'),
                authentication=ds_data.get('authentication'),
                polling_interval=ds_data.get('polling_interval', 300),
                enabled=ds_data.get('enabled', True)
            )
            data_sources.append(data_source)

        # Parse analytics config
        analytics_data = config_data.get('analytics', {})
        analytics = AnalyticsConfig(
            default_slo_targets=analytics_data.get('default_slo_targets', {}),
            analysis_window_hours=analytics_data.get('analysis_window_hours', 24),
            confidence_threshold=analytics_data.get('confidence_threshold', 0.6),
            enable_ai_recommendations=analytics_data.get('enable_ai_recommendations', True),
            export_formats=analytics_data.get('export_formats', ["json", "html"])
        )

        # Parse reporting config
        reporting_data = config_data.get('reporting', {})
        reporting = ReportingConfig(
            output_directory=reporting_data.get('output_directory', "reports/generated"),
            template_directory=reporting_data.get('template_directory', "templates"),
            default_format=reporting_data.get('default_format', "html"),
            include_charts=reporting_data.get('include_charts', True),
            chart_resolution=reporting_data.get('chart_resolution', "high"),
            enable_pdf_generation=reporting_data.get('enable_pdf_generation', True)
        )

        global_settings = config_data.get('global_settings', {})

        return MultiSourceConfig(
            data_sources=data_sources,
            analytics=analytics,
            reporting=reporting,
            global_settings=global_settings
        )

    def _config_to_dict(self, config: MultiSourceConfig) -> Dict[str, Any]:
        """Convert MultiSourceConfig to dictionary"""
        return {
            'data_sources': [
                {
                    'source_type': ds.source_type.value,
                    'name': ds.name,
                    'connection_params': ds.connection_params,
                    'metric_mappings': ds.metric_mappings,
                    'authentication': ds.authentication,
                    'polling_interval': ds.polling_interval,
                    'enabled': ds.enabled
                }
                for ds in config.data_sources
            ],
            'analytics': asdict(config.analytics),
            'reporting': asdict(config.reporting),
            'global_settings': config.global_settings
        }

    def add_data_source(self, data_source: DataSourceConfig):
        """Add a new data source to configuration"""
        if not self.config:
            self.config = self._create_default_config()

        # Check if data source with same name already exists
        existing_names = [ds.name for ds in self.config.data_sources]
        if data_source.name in existing_names:
            raise ValueError(f"Data source with name '{data_source.name}' already exists")

        self.config.data_sources.append(data_source)
        self.save_config()

    def remove_data_source(self, name: str):
        """Remove a data source from configuration"""
        if not self.config:
            return

        self.config.data_sources = [ds for ds in self.config.data_sources if ds.name != name]
        self.save_config()

    def update_data_source(self, name: str, updates: Dict[str, Any]):
        """Update a data source configuration"""
        if not self.config:
            return

        for ds in self.config.data_sources:
            if ds.name == name:
                for key, value in updates.items():
                    if hasattr(ds, key):
                        setattr(ds, key, value)
                break

        self.save_config()

    def get_enabled_sources(self) -> List[DataSourceConfig]:
        """Get list of enabled data sources"""
        if not self.config:
            self.load_config()

        return [ds for ds in self.config.data_sources if ds.enabled]

    def get_slo_targets(self, service_name: str) -> List[SLOTarget]:
        """Get SLO targets for a service"""
        if not self.config:
            self.load_config()

        # Check for service-specific targets first
        targets_data = self.config.analytics.default_slo_targets.get(
            service_name,
            self.config.analytics.default_slo_targets.get('default', [])
        )

        slo_targets = []
        for target_data in targets_data:
            try:
                metric_type = MetricType(target_data['metric_type'])
                slo_target = SLOTarget(
                    metric_type=metric_type,
                    target_value=target_data['target_value'],
                    comparison=target_data['comparison'],
                    unit=target_data.get('unit', ''),
                    description=target_data.get('description', '')
                )
                slo_targets.append(slo_target)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid SLO target configuration: {e}")

        return slo_targets

    def set_slo_targets(self, service_name: str, targets: List[SLOTarget]):
        """Set SLO targets for a service"""
        if not self.config:
            self.load_config()

        targets_data = []
        for target in targets:
            targets_data.append({
                'metric_type': target.metric_type.value,
                'target_value': target.target_value,
                'comparison': target.comparison,
                'unit': target.unit,
                'description': target.description
            })

        self.config.analytics.default_slo_targets[service_name] = targets_data
        self.save_config()

    def expand_environment_variables(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables in configuration"""
        def expand_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            else:
                return value

        return expand_value(config_dict)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        if not self.config:
            issues.append("No configuration loaded")
            return issues

        # Validate data sources
        enabled_sources = self.get_enabled_sources()
        if not enabled_sources:
            issues.append("No enabled data sources configured")

        source_names = [ds.name for ds in self.config.data_sources]
        if len(source_names) != len(set(source_names)):
            issues.append("Duplicate data source names found")

        # Validate each data source
        for ds in self.config.data_sources:
            if not ds.name:
                issues.append("Data source missing name")

            if not ds.connection_params:
                issues.append(f"Data source '{ds.name}' missing connection parameters")

            # Validate specific source types
            if ds.source_type == DataSourceType.APPDYNAMICS:
                required_params = ['host']
                for param in required_params:
                    if param not in ds.connection_params:
                        issues.append(f"AppDynamics source '{ds.name}' missing required parameter: {param}")

            elif ds.source_type == DataSourceType.PROMETHEUS:
                if 'url' not in ds.connection_params:
                    issues.append(f"Prometheus source '{ds.name}' missing required parameter: url")

            elif ds.source_type in [DataSourceType.CSV_FILE, DataSourceType.JSON_FILE]:
                if 'file_path' not in ds.connection_params:
                    issues.append(f"File source '{ds.name}' missing required parameter: file_path")

        # Validate analytics config
        if self.config.analytics.analysis_window_hours <= 0:
            issues.append("Analysis window hours must be positive")

        if not (0 <= self.config.analytics.confidence_threshold <= 1):
            issues.append("Confidence threshold must be between 0 and 1")

        # Validate reporting config
        if not self.config.reporting.output_directory:
            issues.append("Output directory not specified")

        return issues

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        if not self.config:
            self.load_config()

        enabled_sources = self.get_enabled_sources()

        return {
            "total_data_sources": len(self.config.data_sources),
            "enabled_data_sources": len(enabled_sources),
            "source_types": list(set([ds.source_type.value for ds in enabled_sources])),
            "analysis_window_hours": self.config.analytics.analysis_window_hours,
            "ai_recommendations_enabled": self.config.analytics.enable_ai_recommendations,
            "export_formats": self.config.analytics.export_formats,
            "pdf_generation_enabled": self.config.reporting.enable_pdf_generation,
            "validation_issues": len(self.validate_config())
        }