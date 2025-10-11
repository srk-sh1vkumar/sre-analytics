"""
Configuration Loader Module

Handles loading and managing configuration files for SRE reports.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


class ConfigurationLoader:
    """Handles loading YAML configuration files with defaults"""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML configuration file with defaults

        Args:
            filename: Name of the configuration file

        Returns:
            Dictionary containing configuration data
        """
        try:
            with open(self.config_dir / filename, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Could not load {filename}: {e}")
            return self._get_default_config(filename)

    def _get_default_config(self, filename: str) -> Dict[str, Any]:
        """
        Get default configuration when file cannot be loaded

        Args:
            filename: Name of the configuration file

        Returns:
            Dictionary containing default configuration
        """
        if "slo" in filename.lower():
            return self._get_default_slo_config()
        elif "sla" in filename.lower():
            return self._get_default_sla_config()
        return {}

    def _get_default_slo_config(self) -> Dict[str, Any]:
        """Get default SLO configuration"""
        return {
            'availability': {
                'target': 99.9,
                'sla': 99.5,
                'unit': '%'
            },
            'latency_p95': {
                'target': 200,
                'sla': 300,
                'unit': 'ms'
            },
            'error_rate': {
                'target': 1.0,
                'sla': 2.0,
                'unit': '%'
            }
        }

    def _get_default_sla_config(self) -> Dict[str, Any]:
        """Get default SLA configuration"""
        return {
            'thresholds': {
                'availability_min': 99.5,
                'latency_p95_max': 300,
                'error_rate_max': 2.0
            },
            'reporting': {
                'frequency': 'daily',
                'retention_days': 90
            }
        }
