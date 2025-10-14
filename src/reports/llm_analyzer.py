"""
LLM Analyzer Module

Handles LLM-powered analysis for incidents and performance metrics.
Supports OpenAI and Anthropic providers with fallback analysis.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from src.config.app_config import get_config

# LLM provider imports with availability flags
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Import dataclasses (these should eventually be moved to a shared models module)
@dataclass
class SLOMetric:
    """SLO metric data structure"""

    service_name: str
    metric_name: str
    current_value: float
    slo_target: float
    sla_target: float
    status: str
    error_budget_consumed: float
    timestamp: Any  # datetime
    unit: str = ""
    description: str = ""
    trend_data: List[float] = None


@dataclass
class IncidentData:
    """Incident data structure"""

    incident_id: str
    title: str
    description: str
    severity: str
    application_name: str
    start_time: Any  # datetime
    end_time: Any  # datetime
    affected_services: List[str]
    root_cause: str
    resolution_steps: List[str]
    llm_analysis: str
    lessons_learned: str


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""

    service_name: str
    timestamp: Any  # datetime
    metrics: Dict[str, float]
    logs: List[str]
    errors: List[str]


class LLMAnalyzer:
    """Enhanced LLM analyzer for incidents and performance"""

    def __init__(self, provider: str = "anthropic", api_key: str = None):
        """
        Initialize LLM analyzer

        Args:
            provider: LLM provider ('openai' or 'anthropic')
            api_key: Optional API key (uses config if not provided)
        """
        self.provider = provider.lower()
        self.logger = logging.getLogger(__name__)
        self.client = None
        config = get_config()

        # Initialize LLM client
        if api_key:
            self.api_key = api_key
        else:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                self.api_key = config.llm.openai_api_key
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                self.api_key = config.llm.anthropic_api_key
                if self.api_key:
                    self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_incident_root_cause(
        self, incident: IncidentData, snapshots: List[PerformanceSnapshot]
    ) -> str:
        """
        Analyze incident using LLM for root cause analysis

        Args:
            incident: Incident data to analyze
            snapshots: Performance snapshots during incident

        Returns:
            str: Detailed root cause analysis
        """
        if not self.client:
            return self._fallback_rca_analysis(incident, snapshots)

        context = self._prepare_incident_context(incident, snapshots)

        prompt = f"""
        As an expert Site Reliability Engineer, analyze this production incident and provide a comprehensive root cause analysis.

        {context}

        Please provide:
        1. Primary root cause identification
        2. Contributing factors analysis
        3. Impact assessment
        4. Prevention recommendations
        5. Monitoring improvements
        6. Process improvements

        Focus on actionable insights that will prevent similar incidents.
        """

        try:
            if self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=1500
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._fallback_rca_analysis(incident, snapshots)

    def _prepare_incident_context(
        self, incident: IncidentData, snapshots: List[PerformanceSnapshot]
    ) -> str:
        """Prepare context for incident analysis"""
        context = f"""
INCIDENT DETAILS:
- ID: {incident.incident_id}
- Application: {incident.application_name}
- Duration: {incident.start_time} to {incident.end_time or 'Ongoing'}
- Severity: {incident.severity}
- Title: {incident.title}
- Description: {incident.description}
- Affected Services: {', '.join(incident.affected_services)}
- Initial Root Cause: {incident.root_cause}

PERFORMANCE SNAPSHOTS:
"""
        for snapshot in snapshots[-5:]:  # Last 5 snapshots
            context += f"\n[{snapshot.timestamp}] {snapshot.service_name}:\n"
            context += f"  Metrics: {snapshot.metrics}\n"
            if snapshot.errors:
                context += f"  Errors: {snapshot.errors[:3]}\n"  # First 3 errors

        return context

    def _fallback_rca_analysis(
        self, incident: IncidentData, snapshots: List[PerformanceSnapshot]
    ) -> str:
        """Fallback analysis without LLM"""
        analysis = f"""
ROOT CAUSE ANALYSIS (Rule-based):

Primary Analysis:
- Incident Type: {incident.severity} severity incident in {incident.application_name}
- Duration: {(incident.end_time - incident.start_time).total_seconds() / 60:.1f} minutes
- Services Affected: {len(incident.affected_services)} services

Contributing Factors:
- Initial root cause identified: {incident.root_cause}
- Performance degradation observed across multiple snapshots
- Error patterns suggest {incident.affected_services[0] if incident.affected_services else 'unknown'} service issues

Recommendations:
1. Implement enhanced monitoring for {incident.application_name}
2. Add automated alerting for similar patterns
3. Review deployment processes and rollback procedures
4. Conduct post-incident review with team
5. Update runbooks based on lessons learned

Next Steps:
- Document incident in knowledge base
- Update monitoring thresholds
- Schedule follow-up review meeting
"""
        return analysis

    def analyze_performance_metrics(self, metrics: List[SLOMetric], summary: Dict[str, Any]) -> str:
        """
        Analyze performance metrics using LLM for insights

        Args:
            metrics: List of SLO metrics
            summary: Summary statistics dictionary

        Returns:
            str: Performance analysis and recommendations
        """
        if not self.client:
            return self._fallback_performance_analysis(metrics, summary)

        # Prepare metrics context
        context = self._prepare_performance_context(metrics, summary)

        prompt = f"""
        As an expert Site Reliability Engineer, analyze these SLO/SLA performance metrics and provide actionable insights.

        {context}

        Please provide:
        1. Overall system health assessment
        2. Key performance trends and patterns
        3. Risk areas and potential issues
        4. Specific recommendations for improvement
        5. Capacity planning insights
        6. Monitoring and alerting suggestions

        Focus on actionable insights that will improve system reliability and performance.
        """

        try:
            if self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model="gpt-4", messages=[{"role": "user", "content": prompt}], max_tokens=1000
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM performance analysis failed: {e}")
            return self._fallback_performance_analysis(metrics, summary)

    def _prepare_performance_context(
        self, metrics: List[SLOMetric], summary: Dict[str, Any]
    ) -> str:
        """Prepare context for performance analysis"""
        context = f"""
SYSTEM OVERVIEW:
- Total Services: {summary['total_services']}
- Total Metrics: {summary['total_metrics']}
- Compliance Rate: {summary['compliance_percentage']:.1f}%
- At Risk Metrics: {summary['at_risk_count']}
- Breached SLOs: {summary['breached_count']}
- Overall Health: {summary['health_status']}

DETAILED METRICS:
"""
        for metric in metrics:
            trend_indicator = (
                "📈"
                if metric.trend_data
                and len(metric.trend_data) > 1
                and metric.trend_data[-1] > metric.trend_data[0]
                else "📉"
            )
            context += f"""
- {metric.service_name} {metric.metric_name}:
  Current: {metric.current_value:.2f}{metric.unit} (Target: {metric.slo_target:.2f}{metric.unit})
  Status: {metric.status.upper()}
  Error Budget Used: {metric.error_budget_consumed:.1f}%
  Trend: {trend_indicator}
"""
        return context

    def _fallback_performance_analysis(
        self, metrics: List[SLOMetric], summary: Dict[str, Any]
    ) -> str:
        """Fallback analysis without LLM"""
        analysis = f"""
SYSTEM HEALTH ASSESSMENT:

Overall Status: {summary['health_status']}
- {summary['compliant_count']}/{summary['total_metrics']} metrics are compliant ({summary['compliance_percentage']:.1f}%)
- {summary['breached_count']} critical SLO breaches requiring immediate attention
- {summary['at_risk_count']} metrics at risk of breaching SLO targets

KEY INSIGHTS:
• System shows {"good" if summary['breached_count'] == 0 else "concerning"} reliability patterns
• Performance trends indicate {"stable" if summary['at_risk_count'] < 2 else "degrading"} system behavior
• Error budget consumption {"within acceptable limits" if all(m.error_budget_consumed < 50 for m in metrics) else "approaching critical levels"}

IMMEDIATE ACTIONS NEEDED:
{"• Address critical SLO breaches to prevent service degradation" if summary['breached_count'] > 0 else "• Continue monitoring current performance levels"}
• Review and optimize services with high error budget consumption
• Implement proactive alerting for at-risk metrics
• Consider capacity scaling for services showing performance degradation

STRATEGIC RECOMMENDATIONS:
• Establish automated remediation for common performance issues
• Implement predictive alerting based on trend analysis
• Review SLO targets to ensure they align with business requirements
• Enhance monitoring coverage for early issue detection
"""
        return analysis
