"""
Centralized Configuration Management

This module provides a centralized Config class for managing all application
configuration including environment variables, with validation and defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppDynamicsConfig:
    """AppDynamics-specific configuration"""

    controller_host: str
    client_id: str
    client_secret: str
    account: Optional[str] = None
    access_key: Optional[str] = None
    primary_app: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AppDynamicsConfig":
        """Load AppDynamics config from environment variables"""
        return cls(
            controller_host=os.getenv("APPDYNAMICS_CONTROLLER_HOST", ""),
            client_id=os.getenv("APPDYNAMICS_CLIENT_ID", ""),
            client_secret=os.getenv("APPDYNAMICS_CLIENT_SECRET", ""),
            account=os.getenv("APPDYNAMICS_CONTROLLER_ACCOUNT"),
            access_key=os.getenv("APPDYNAMICS_CONTROLLER_ACCESS_KEY"),
            primary_app=os.getenv("DEFAULT_APPLICATION_NAME"),
        )

    def validate(self) -> bool:
        """Validate required AppDynamics configuration"""
        return bool(self.controller_host and self.client_id and self.client_secret)


@dataclass
class LLMConfig:
    """LLM provider configuration"""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM config from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def has_provider(self) -> bool:
        """Check if at least one LLM provider is configured"""
        return bool(self.openai_api_key or self.anthropic_api_key)


@dataclass
class ReportConfig:
    """Report generation configuration"""

    output_path: str = "reports/generated"

    @classmethod
    def from_env(cls) -> "ReportConfig":
        """Load report config from environment variables"""
        return cls(output_path=os.getenv("REPORT_OUTPUT_PATH", "reports/generated"))


@dataclass
class FlaskConfig:
    """Flask application configuration"""

    secret_key: str = "dev-key-change-in-production"
    env: str = "production"

    @classmethod
    def from_env(cls) -> "FlaskConfig":
        """Load Flask config from environment variables"""
        return cls(
            secret_key=os.getenv("FLASK_SECRET_KEY", "dev-key-change-in-production"),
            env=os.getenv("FLASK_ENV", "production"),
        )


@dataclass
class SystemConfig:
    """System-level configuration (paths, libraries, etc.)"""

    pkg_config_path: Optional[str] = None
    dyld_library_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load system config from environment variables"""
        return cls(
            pkg_config_path=os.getenv("PKG_CONFIG_PATH"),
            dyld_library_path=os.getenv("DYLD_LIBRARY_PATH"),
        )

    def apply_to_env(self) -> None:
        """Apply system configuration to environment variables"""
        if self.pkg_config_path:
            os.environ["PKG_CONFIG_PATH"] = self.pkg_config_path
        if self.dyld_library_path:
            os.environ["DYLD_LIBRARY_PATH"] = self.dyld_library_path


class Config:
    """
    Centralized configuration management for SRE Analytics application

    Loads all configuration from environment variables with proper validation
    and defaults. Provides type-safe access to configuration values.

    Usage:
        config = Config.load()

        # Access configuration
        print(config.appdynamics.controller_host)
        print(config.llm.openai_api_key)
        print(config.report.output_path)

        # Validate configuration
        if not config.appdynamics.validate():
            raise ValueError("AppDynamics configuration is incomplete")
    """

    def __init__(
        self,
        appdynamics: AppDynamicsConfig,
        llm: LLMConfig,
        report: ReportConfig,
        flask: FlaskConfig,
        system: SystemConfig,
    ):
        self.appdynamics = appdynamics
        self.llm = llm
        self.report = report
        self.flask = flask
        self.system = system

    @classmethod
    def load(cls) -> "Config":
        """
        Load configuration from environment variables

        Returns:
            Config: Fully initialized configuration object
        """
        return cls(
            appdynamics=AppDynamicsConfig.from_env(),
            llm=LLMConfig.from_env(),
            report=ReportConfig.from_env(),
            flask=FlaskConfig.from_env(),
            system=SystemConfig.from_env(),
        )

    def validate(self) -> Dict[str, bool]:
        """
        Validate all configuration sections

        Returns:
            Dict mapping section names to validation status
        """
        return {
            "appdynamics": self.appdynamics.validate(),
            "llm": self.llm.has_provider(),
        }

    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors

        Returns:
            List of error messages for invalid configuration
        """
        errors = []

        if not self.appdynamics.validate():
            errors.append(
                "AppDynamics configuration incomplete: missing controller_host, "
                "client_id, or client_secret"
            )

        if not self.llm.has_provider():
            errors.append("No LLM provider configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary (for debugging/logging)

        Returns:
            Dictionary representation of configuration (with secrets masked)
        """
        return {
            "appdynamics": {
                "controller_host": self.appdynamics.controller_host,
                "client_id": "***" if self.appdynamics.client_id else None,
                "client_secret": "***" if self.appdynamics.client_secret else None,
                "account": self.appdynamics.account,
                "primary_app": self.appdynamics.primary_app,
            },
            "llm": {
                "openai_configured": bool(self.llm.openai_api_key),
                "anthropic_configured": bool(self.llm.anthropic_api_key),
            },
            "report": {
                "output_path": self.report.output_path,
            },
            "flask": {
                "secret_key": "***" if self.flask.secret_key else None,
                "env": self.flask.env,
            },
            "system": {
                "pkg_config_path": self.system.pkg_config_path,
                "dyld_library_path": self.system.dyld_library_path,
            },
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance (singleton pattern)

    Loads configuration on first access and caches it.

    Returns:
        Config: Global configuration instance
    """
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config() -> Config:
    """
    Force reload configuration from environment

    Useful for testing or when environment variables change.

    Returns:
        Config: Newly loaded configuration instance
    """
    global _config
    _config = Config.load()
    return _config
