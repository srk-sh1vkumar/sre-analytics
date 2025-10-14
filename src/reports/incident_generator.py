"""
Incident Generator Module

Handles generation of realistic incident data with performance snapshots,
severity analysis, and affected service identification.
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.config.app_config import get_config
from src.config.constants import (
    AVAILABILITY_MIN,
    AVAILABILITY_NOISE_STD,
    DEFAULT_INCIDENT_DURATION_HOURS,
    DEGRADATION_FACTOR_MAX,
    DEGRADATION_FACTOR_MIN,
    ERROR_RATE_CRITICAL,
    ERROR_RATE_NOISE_STD,
    LATENCY_NOISE_STD,
    LATENCY_P95_CRITICAL_MS,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)
from src.reports.llm_analyzer import IncidentData, LLMAnalyzer, PerformanceSnapshot


class IncidentGenerator:
    """Generates realistic incident data with performance snapshots"""

    def __init__(self, llm_provider: str = "anthropic", llm_api_key: str = None):
        """
        Initialize incident generator

        Args:
            llm_provider: LLM provider for root cause analysis
            llm_api_key: Optional LLM API key
        """
        self.logger = logging.getLogger(__name__)
        self.llm_analyzer = LLMAnalyzer(provider=llm_provider, api_key=llm_api_key)

    def generate_incident_report(
        self,
        application_name: str,
        start_time: datetime,
        duration_hours: float = DEFAULT_INCIDENT_DURATION_HOURS,
    ) -> IncidentData:
        """
        Generate a comprehensive incident report with LLM analysis

        Args:
            application_name: Name of affected application
            start_time: Incident start time
            duration_hours: Duration of incident in hours

        Returns:
            IncidentData: Complete incident report with analysis
        """
        end_time = start_time + timedelta(hours=duration_hours)

        # Generate performance snapshots during incident
        snapshots = self._generate_incident_snapshots(application_name, start_time, end_time)

        # Analyze affected services
        affected_services = self._analyze_affected_services(snapshots)

        # Determine severity and initial root cause
        severity = self._determine_severity(snapshots)
        initial_root_cause = self._determine_initial_root_cause(snapshots)

        # Create incident data structure
        incident = IncidentData(
            incident_id=f"INC-{start_time.strftime('%Y%m%d-%H%M%S')}",
            title=f"{severity} Incident in {application_name}",
            description=f"Performance degradation detected in {application_name} starting at {start_time}",
            severity=severity,
            application_name=application_name,
            start_time=start_time,
            end_time=end_time,
            affected_services=affected_services,
            root_cause=initial_root_cause,
            resolution_steps=[
                "Identified performance degradation through monitoring alerts",
                "Scaled up affected service instances",
                "Applied database query optimization",
                "Implemented circuit breaker to prevent cascading failures",
                "Verified system recovery and performance normalization",
            ],
            llm_analysis="",
            lessons_learned="",
        )

        # Get LLM-powered root cause analysis
        try:
            llm_analysis = self.llm_analyzer.analyze_incident_root_cause(incident, snapshots)
            incident.llm_analysis = llm_analysis
            incident.lessons_learned = self._extract_lessons_learned(llm_analysis)
        except Exception as e:
            self.logger.warning(f"LLM analysis failed, using fallback: {e}")
            incident.llm_analysis = f"Automated analysis not available. Manual review required.\n\nInitial findings: {initial_root_cause}"
            incident.lessons_learned = "Conduct post-incident review to document lessons learned."

        return incident

    def _generate_incident_snapshots(
        self, app_name: str, start_time: datetime, end_time: datetime
    ) -> List[PerformanceSnapshot]:
        """
        Generate realistic performance snapshots during an incident

        Args:
            app_name: Application name
            start_time: Incident start time
            end_time: Incident end time

        Returns:
            List[PerformanceSnapshot]: Performance snapshots showing degradation
        """
        snapshots = []
        current_time = start_time
        interval = timedelta(minutes=5)

        # Simulate degradation pattern
        while current_time <= end_time:
            # Progressive degradation then recovery
            time_ratio = (current_time - start_time).total_seconds() / (
                end_time - start_time
            ).total_seconds()
            degradation_factor = DEGRADATION_FACTOR_MIN + (
                DEGRADATION_FACTOR_MAX - DEGRADATION_FACTOR_MIN
            ) * (
                1 - abs(2 * time_ratio - 1)  # Peak degradation at midpoint
            )

            snapshot = PerformanceSnapshot(
                service_name=app_name,
                timestamp=current_time,
                metrics={
                    "availability": max(
                        90.0,
                        AVAILABILITY_MIN
                        - degradation_factor * 2
                        + random.gauss(0, AVAILABILITY_NOISE_STD),
                    ),
                    "latency_p95": LATENCY_P95_CRITICAL_MS * degradation_factor
                    + random.gauss(0, LATENCY_NOISE_STD),
                    "error_rate": ERROR_RATE_CRITICAL * degradation_factor
                    + random.gauss(0, ERROR_RATE_NOISE_STD),
                    "cpu_usage": min(95.0, 60 + degradation_factor * 15),
                    "memory_usage": min(90.0, 50 + degradation_factor * 20),
                },
                logs=[
                    f"[{current_time}] High latency detected in API endpoints",
                    f"[{current_time}] Database connection pool exhausted",
                    f"[{current_time}] Increased error rate in downstream services",
                ],
                errors=(
                    [
                        "TimeoutException: Request timed out after 30s",
                        "DatabaseConnectionError: Connection pool exhausted",
                        "CircuitBreakerOpenException: Circuit breaker opened due to high failure rate",
                    ]
                    if degradation_factor > 2.0
                    else []
                ),
            )
            snapshots.append(snapshot)
            current_time += interval

        return snapshots

    def _analyze_affected_services(self, snapshots: List[PerformanceSnapshot]) -> List[str]:
        """
        Determine which services were affected based on performance snapshots

        Args:
            snapshots: Performance snapshots to analyze

        Returns:
            List[str]: Names of affected services
        """
        affected = set()
        for snapshot in snapshots:
            affected.add(snapshot.service_name)
            # Add related services based on error patterns
            if any("Database" in error for error in snapshot.errors):
                affected.add(f"{snapshot.service_name}-database")
            if snapshot.metrics.get("error_rate", 0) > ERROR_RATE_CRITICAL:
                affected.add(f"{snapshot.service_name}-api")

        return sorted(list(affected))

    def _determine_initial_root_cause(self, snapshots: List[PerformanceSnapshot]) -> str:
        """
        Determine initial root cause based on performance patterns

        Args:
            snapshots: Performance snapshots to analyze

        Returns:
            str: Initial root cause hypothesis
        """
        # Analyze error patterns
        db_errors = sum(1 for s in snapshots for e in s.errors if "Database" in e)
        timeout_errors = sum(1 for s in snapshots for e in s.errors if "Timeout" in e)
        circuit_breaker_errors = sum(
            1 for s in snapshots for e in s.errors if "CircuitBreaker" in e
        )

        # Determine root cause based on dominant error type
        if db_errors > max(timeout_errors, circuit_breaker_errors):
            return "Database connection pool exhaustion due to slow queries and increased load"
        elif timeout_errors > circuit_breaker_errors:
            return "Service timeout issues due to resource contention and high latency"
        elif circuit_breaker_errors > 0:
            return "Cascading failures triggering circuit breakers across service mesh"
        else:
            return "Performance degradation due to increased system load"

    def _determine_severity(self, snapshots: List[PerformanceSnapshot]) -> str:
        """
        Determine incident severity based on impact metrics

        Args:
            snapshots: Performance snapshots to analyze

        Returns:
            str: Severity level (Critical/High/Medium/Low)
        """
        max_error_rate = max(s.metrics.get("error_rate", 0) for s in snapshots)
        min_availability = min(s.metrics.get("availability", 100) for s in snapshots)
        max_latency = max(s.metrics.get("latency_p95", 0) for s in snapshots)

        # Critical: Major service disruption
        if min_availability < 95.0 or max_error_rate > 10.0:
            return SEVERITY_CRITICAL

        # High: Significant degradation
        if min_availability < 98.0 or max_error_rate > 5.0 or max_latency > 1000:
            return SEVERITY_HIGH

        # Medium: Noticeable impact
        if min_availability < AVAILABILITY_MIN or max_error_rate > ERROR_RATE_CRITICAL:
            return SEVERITY_MEDIUM

        # Low: Minor impact
        return SEVERITY_LOW

    def _extract_lessons_learned(self, llm_analysis: str) -> str:
        """
        Extract lessons learned section from LLM analysis

        Args:
            llm_analysis: Full LLM analysis text

        Returns:
            str: Lessons learned summary
        """
        # Try to extract lessons learned section
        if "lessons learned" in llm_analysis.lower():
            parts = llm_analysis.lower().split("lessons learned")
            if len(parts) > 1:
                # Get text after "lessons learned" header
                lessons_section = parts[1].split("\n\n")[0]  # First paragraph
                return lessons_section.strip()

        # Fallback: extract recommendations section
        if "recommendation" in llm_analysis.lower():
            parts = llm_analysis.lower().split("recommendation")
            if len(parts) > 1:
                return "Key recommendations: " + parts[1].split("\n\n")[0].strip()

        # Default fallback
        return "Post-incident review required to document lessons learned and preventive measures."
