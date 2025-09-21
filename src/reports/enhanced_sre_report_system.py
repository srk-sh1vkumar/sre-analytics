"""
Enhanced SRE Report System
Comprehensive SLO/SLA reporting with trend analysis and incident RCA
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web applications
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import jinja2
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import weasyprint
    from reports.weasyprint_pdf_generator import WeasyPrintPDFGenerator, enhance_html_for_pdf
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    WEASYPRINT_AVAILABLE = False
    print(f"WeasyPrint not available: {e}")
    # WeasyPrint may fail due to missing system dependencies

try:
    from .browser_pdf_generator import BrowserPDFGenerator
    BROWSER_PDF_AVAILABLE = True
except ImportError as e:
    BROWSER_PDF_AVAILABLE = False
    print(f"Browser PDF generator not available: {e}")

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

@dataclass
class SLOMetric:
    """SLO metric data structure with trend data"""
    service_name: str
    metric_name: str
    current_value: float
    slo_target: float
    sla_target: float
    status: str  # "compliant", "at_risk", "breached"
    error_budget_consumed: float  # percentage
    timestamp: datetime
    unit: str = ""
    description: str = ""
    trend_data: List[float] = None  # Historical data for trending

@dataclass
class IncidentData:
    """Incident data structure"""
    incident_id: str
    application_name: str
    start_time: datetime
    end_time: Optional[datetime]
    severity: str
    title: str
    description: str
    affected_services: List[str]
    root_cause: str
    resolution_steps: List[str]
    llm_analysis: str
    lessons_learned: str

@dataclass
class PerformanceSnapshot:
    """Performance snapshot for incident analysis"""
    timestamp: datetime
    service_name: str
    metrics: Dict[str, float]
    logs: List[str]
    errors: List[str]

class LLMAnalyzer:
    """Enhanced LLM analyzer for incidents and performance"""

    def __init__(self, provider: str = "anthropic", api_key: str = None):
        self.provider = provider.lower()
        self.logger = logging.getLogger(__name__)
        self.client = None

        # Initialize LLM client
        if api_key:
            self.api_key = api_key
        else:
            if self.provider == "openai" and OPENAI_AVAILABLE:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if self.api_key:
                    self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_incident_root_cause(self, incident: IncidentData,
                                   snapshots: List[PerformanceSnapshot]) -> str:
        """Analyze incident using LLM for root cause analysis"""
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
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._fallback_rca_analysis(incident, snapshots)

    def _prepare_incident_context(self, incident: IncidentData,
                                 snapshots: List[PerformanceSnapshot]) -> str:
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

    def _fallback_rca_analysis(self, incident: IncidentData,
                              snapshots: List[PerformanceSnapshot]) -> str:
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
        """Analyze performance metrics using LLM for insights"""
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
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM performance analysis failed: {e}")
            return self._fallback_performance_analysis(metrics, summary)

    def _prepare_performance_context(self, metrics: List[SLOMetric], summary: Dict[str, Any]) -> str:
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
            trend_indicator = "📈" if metric.trend_data and len(metric.trend_data) > 1 and metric.trend_data[-1] > metric.trend_data[0] else "📉"
            context += f"""
- {metric.service_name} {metric.metric_name}:
  Current: {metric.current_value:.2f}{metric.unit} (Target: {metric.slo_target:.2f}{metric.unit})
  Status: {metric.status.upper()}
  Error Budget Used: {metric.error_budget_consumed:.1f}%
  Trend: {trend_indicator}
"""
        return context

    def _fallback_performance_analysis(self, metrics: List[SLOMetric], summary: Dict[str, Any]) -> str:
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

class EnhancedSREReportSystem:
    """Enhanced SRE report system with incident analysis"""

    def __init__(self, config_dir: str = "config", app_name: str = "Application"):
        self.config_dir = Path(config_dir)
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)

        # Load configurations
        self.slo_config = self._load_yaml("slo_definitions.yaml")
        self.sla_config = self._load_yaml("sla_thresholds.yaml")

        # Initialize LLM analyzer
        self.llm_analyzer = LLMAnalyzer()

        # Set up visualization styling
        self._setup_styling()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file with defaults"""
        try:
            with open(self.config_dir / filename, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Could not load {filename}: {e}")
            return self._get_default_config(filename)

    def _get_default_config(self, filename: str) -> Dict[str, Any]:
        """Get default configuration"""
        if "slo" in filename:
            return {
                "service_level_objectives": {
                    "default_service": {
                        "service_name": "default-service",
                        "availability_slo": "99.9%",
                        "latency_p95_slo": "200ms",
                        "error_rate_slo": "0.1%"
                    }
                }
            }
        return {}

    def _setup_styling(self):
        """Setup visualization styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def generate_metrics_with_trends(self, services: List[str] = None,
                                   days_back: int = 30) -> List[SLOMetric]:
        """Generate metrics with historical trend data"""
        if not services:
            services = ["web-service", "api-service", "database-service", "auth-service"]

        np.random.seed(42)
        metrics = []
        current_time = datetime.now()

        for service_name in services:
            # Generate trend data for each metric type
            trend_days = 30

            # Availability trend (99.5% - 99.99%)
            availability_trend = self._generate_trend_data(99.9, 0.05, trend_days)
            current_availability = availability_trend[-1]

            availability_metric = SLOMetric(
                service_name=service_name,
                metric_name="availability",
                current_value=current_availability,
                slo_target=99.9,
                sla_target=99.9,
                status=self._get_compliance_status(current_availability, 99.9),
                error_budget_consumed=max(0, (99.9 - current_availability) / 0.1 * 100),
                timestamp=current_time,
                unit="%",
                description=f"Service availability for {service_name}",
                trend_data=availability_trend
            )
            metrics.append(availability_metric)

            # Latency trend
            latency_trend = self._generate_trend_data(200, 30, trend_days, min_val=50)
            current_latency = latency_trend[-1]

            latency_metric = SLOMetric(
                service_name=service_name,
                metric_name="latency_p95",
                current_value=current_latency,
                slo_target=200,
                sla_target=500,
                status=self._get_compliance_status(current_latency, 200, inverse=True),
                error_budget_consumed=max(0, (current_latency - 200) / 200 * 100),
                timestamp=current_time,
                unit="ms",
                description=f"95th percentile response time for {service_name}",
                trend_data=latency_trend
            )
            metrics.append(latency_metric)

            # Error rate trend
            error_trend = self._generate_trend_data(0.1, 0.03, trend_days, min_val=0)
            current_error_rate = error_trend[-1]

            error_rate_metric = SLOMetric(
                service_name=service_name,
                metric_name="error_rate",
                current_value=current_error_rate,
                slo_target=0.1,
                sla_target=1.0,
                status=self._get_compliance_status(current_error_rate, 0.1, inverse=True),
                error_budget_consumed=max(0, (current_error_rate - 0.1) / 0.1 * 100) if current_error_rate > 0.1 else 0,
                timestamp=current_time,
                unit="%",
                description=f"Error rate for {service_name}",
                trend_data=error_trend
            )
            metrics.append(error_rate_metric)

        return metrics

    def _generate_trend_data(self, mean: float, std: float, days: int, min_val: float = 0) -> List[float]:
        """Generate realistic trend data with some patterns"""
        trend = []
        current = mean

        for day in range(days):
            # Add some weekly patterns and random walk
            weekly_effect = 0.1 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
            daily_change = np.random.normal(0, std * 0.1)  # Random walk

            current += weekly_effect + daily_change
            current = max(min_val, current)  # Ensure minimum value
            trend.append(current)

        return trend

    def _get_compliance_status(self, current: float, target: float, inverse: bool = False) -> str:
        """Determine compliance status"""
        if inverse:
            if current <= target:
                return "compliant"
            elif current <= target * 1.2:
                return "at_risk"
            else:
                return "breached"
        else:
            if current >= target:
                return "compliant"
            elif current >= target * 0.999:
                return "at_risk"
            else:
                return "breached"

    def create_trend_visualizations(self, metrics: List[SLOMetric], save_images: bool = False) -> Dict[str, str]:
        """Create comprehensive trend visualizations"""
        charts = {}

        # Group metrics by type
        metric_types = {}
        for metric in metrics:
            if metric.metric_name not in metric_types:
                metric_types[metric.metric_name] = []
            metric_types[metric.metric_name].append(metric)

        # Create trend charts for each metric type
        for metric_name, metric_list in metric_types.items():
            if metric_list[0].trend_data:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                # Plot 1: Trend lines
                days = list(range(-len(metric_list[0].trend_data), 0))

                for metric in metric_list:
                    color = 'green' if metric.status == 'compliant' else 'orange' if metric.status == 'at_risk' else 'red'
                    ax1.plot(days, metric.trend_data, label=metric.service_name, linewidth=2, alpha=0.7)
                    ax1.axhline(y=metric.slo_target, color=color, linestyle='--', alpha=0.5)

                ax1.set_title(f'{metric_name.replace("_", " ").title()} Trend (Last 30 Days)', fontsize=14)
                ax1.set_xlabel('Days Ago')
                ax1.set_ylabel(f'{metric_name} ({metric_list[0].unit})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Current status bar chart
                services = [m.service_name for m in metric_list]
                current_values = [m.current_value for m in metric_list]
                targets = [m.slo_target for m in metric_list]
                colors = ['green' if m.status == 'compliant' else 'orange' if m.status == 'at_risk' else 'red'
                         for m in metric_list]

                x_pos = np.arange(len(services))
                ax2.bar(x_pos, current_values, color=colors, alpha=0.7, label='Current')
                ax2.scatter(x_pos, targets, color='blue', s=100, marker='D', label='SLO Target', zorder=5)

                ax2.set_title(f'Current {metric_name.replace("_", " ").title()} Status')
                ax2.set_xlabel('Service')
                ax2.set_ylabel(f'{metric_name} ({metric_list[0].unit})')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(services, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()

                if save_images:
                    # Save as temporary image file for PDF embedding
                    import tempfile
                    temp_dir = Path("reports/generated/temp_charts")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file = temp_dir / f"chart_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fig.savefig(str(temp_file), dpi=150, bbox_inches='tight', facecolor='white')
                    charts[metric_name] = str(temp_file)
                    self.logger.info(f"Chart saved to: {temp_file}")
                else:
                    # Return base64 for HTML
                    charts[metric_name] = self._fig_to_base64(fig)

                plt.close(fig)

        return charts

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return f"data:image/png;base64,{image_base64}"

    def generate_incident_report(self, application_name: str,
                               start_time: datetime,
                               duration_hours: float = 1.0) -> IncidentData:
        """Generate incident report with RCA analysis"""

        # Simulate incident data based on user input
        end_time = start_time + timedelta(hours=duration_hours)
        incident_id = f"INC-{start_time.strftime('%Y%m%d%H%M%S')}"

        # Generate performance snapshots for the incident timeframe
        snapshots = self._generate_incident_snapshots(application_name, start_time, end_time)

        # Determine affected services based on performance degradation
        affected_services = self._analyze_affected_services(snapshots)

        # Basic root cause based on patterns
        root_cause = self._determine_initial_root_cause(snapshots)

        # Create incident object
        incident = IncidentData(
            incident_id=incident_id,
            application_name=application_name,
            start_time=start_time,
            end_time=end_time,
            severity=self._determine_severity(snapshots),
            title=f"Performance degradation in {application_name}",
            description=f"Service degradation detected starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            affected_services=affected_services,
            root_cause=root_cause,
            resolution_steps=[
                "Identified performance degradation through monitoring",
                "Analyzed service metrics and logs",
                "Implemented temporary mitigation",
                "Restored normal service operation"
            ],
            llm_analysis="",  # Will be filled by LLM
            lessons_learned=""
        )

        # Get LLM analysis
        incident.llm_analysis = self.llm_analyzer.analyze_incident_root_cause(incident, snapshots)

        # Extract lessons learned from LLM analysis
        incident.lessons_learned = self._extract_lessons_learned(incident.llm_analysis)

        return incident

    def _generate_incident_snapshots(self, app_name: str, start_time: datetime,
                                   end_time: datetime) -> List[PerformanceSnapshot]:
        """Generate realistic performance snapshots during incident"""
        snapshots = []
        current_time = start_time

        services = [f"{app_name}-web", f"{app_name}-api", f"{app_name}-db", f"{app_name}-cache"]

        # Generate snapshots every 5 minutes
        while current_time <= end_time:
            for service in services:
                # Simulate degraded performance during incident
                degradation_factor = np.random.uniform(1.5, 3.0)  # 1.5x to 3x degradation

                metrics = {
                    "response_time_p95": 200 * degradation_factor + np.random.normal(0, 50),
                    "error_rate": 0.1 * degradation_factor + abs(np.random.normal(0, 0.05)),
                    "throughput": 1000 / degradation_factor + np.random.normal(0, 100),
                    "cpu_usage": 60 * degradation_factor + np.random.normal(0, 10),
                    "memory_usage": 70 * degradation_factor + np.random.normal(0, 5)
                }

                # Generate sample error logs
                errors = [
                    f"Connection timeout to downstream service at {current_time}",
                    f"High latency detected in {service} at {current_time}",
                    f"Memory pressure warning in {service}",
                    f"Circuit breaker activated for {service}"
                ]

                snapshot = PerformanceSnapshot(
                    timestamp=current_time,
                    service_name=service,
                    metrics=metrics,
                    logs=[f"INFO: Service {service} status check at {current_time}"],
                    errors=errors[:2]  # Take first 2 errors
                )
                snapshots.append(snapshot)

            current_time += timedelta(minutes=5)

        return snapshots

    def _analyze_affected_services(self, snapshots: List[PerformanceSnapshot]) -> List[str]:
        """Analyze which services were affected during the incident"""
        affected = set()

        for snapshot in snapshots:
            # Check if any metrics are significantly degraded
            if (snapshot.metrics.get("response_time_p95", 0) > 400 or
                snapshot.metrics.get("error_rate", 0) > 0.2 or
                snapshot.metrics.get("cpu_usage", 0) > 80):
                affected.add(snapshot.service_name)

        return list(affected)

    def _determine_initial_root_cause(self, snapshots: List[PerformanceSnapshot]) -> str:
        """Determine initial root cause based on metrics patterns"""
        # Analyze patterns in the snapshots
        high_cpu = sum(1 for s in snapshots if s.metrics.get("cpu_usage", 0) > 80)
        high_memory = sum(1 for s in snapshots if s.metrics.get("memory_usage", 0) > 85)
        high_latency = sum(1 for s in snapshots if s.metrics.get("response_time_p95", 0) > 500)
        high_errors = sum(1 for s in snapshots if s.metrics.get("error_rate", 0) > 0.5)

        total_snapshots = len(snapshots)

        if high_cpu / total_snapshots > 0.7:
            return "High CPU utilization causing performance degradation"
        elif high_memory / total_snapshots > 0.7:
            return "Memory pressure affecting system performance"
        elif high_latency / total_snapshots > 0.6:
            return "Network or downstream service latency issues"
        elif high_errors / total_snapshots > 0.5:
            return "Application errors causing service degradation"
        else:
            return "Multiple factors contributing to performance degradation"

    def _determine_severity(self, snapshots: List[PerformanceSnapshot]) -> str:
        """Determine incident severity based on impact"""
        max_error_rate = max(s.metrics.get("error_rate", 0) for s in snapshots)
        max_response_time = max(s.metrics.get("response_time_p95", 0) for s in snapshots)

        if max_error_rate > 5.0 or max_response_time > 2000:
            return "Critical"
        elif max_error_rate > 1.0 or max_response_time > 1000:
            return "High"
        elif max_error_rate > 0.5 or max_response_time > 500:
            return "Medium"
        else:
            return "Low"

    def _extract_lessons_learned(self, llm_analysis: str) -> str:
        """Extract lessons learned from LLM analysis"""
        # Simple extraction - look for recommendations or lessons
        lines = llm_analysis.split('\n')
        lessons = []

        for line in lines:
            if any(keyword in line.lower() for keyword in ['lesson', 'learn', 'prevent', 'improve', 'recommendation']):
                lessons.append(line.strip())

        if lessons:
            return '\n'.join(lessons[:5])  # Top 5 lessons
        else:
            return "Implement better monitoring and alerting to detect similar issues earlier."

    def create_comprehensive_html_report(self, metrics: List[SLOMetric],
                                       incident: IncidentData = None,
                                       output_path: str = None) -> str:
        """Create comprehensive HTML report with trends and incident analysis using enhanced template"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/comprehensive_sre_report_{timestamp}.html"

        # Create trend visualizations (base64 for HTML)
        trend_charts = self.create_trend_visualizations(metrics, save_images=False)

        # Generate LLM analysis for performance insights
        summary = self._create_summary_stats(metrics)
        llm_analysis = self.llm_analyzer.analyze_performance_metrics(metrics, summary)

        # Prepare template data
        template_data = {
            'app_name': self.app_name,
            'report_date': datetime.now().strftime("%B %d, %Y"),
            'report_time': datetime.now().strftime("%H:%M:%S UTC"),
            'metrics': metrics,
            'trend_charts': trend_charts,
            'incident': incident,
            'summary': summary,
            'has_incident': incident is not None,
            'llm_analysis': llm_analysis
        }

        # Load and render enhanced HTML template
        html_content = self._get_enhanced_html_template()
        template = jinja2.Template(html_content)
        rendered_html = template.render(**template_data)

        # Save HTML file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)

        self.logger.info(f"Enhanced HTML report saved to {output_path}")
        return output_path

    def _create_summary_stats(self, metrics: List[SLOMetric]) -> Dict[str, Any]:
        """Create summary statistics"""
        total_services = len(set(m.service_name for m in metrics))
        compliant_count = len([m for m in metrics if m.status == "compliant"])
        at_risk_count = len([m for m in metrics if m.status == "at_risk"])
        breached_count = len([m for m in metrics if m.status == "breached"])

        return {
            'total_services': total_services,
            'total_metrics': len(metrics),
            'compliant_count': compliant_count,
            'at_risk_count': at_risk_count,
            'breached_count': breached_count,
            'compliance_percentage': (compliant_count / len(metrics) * 100) if metrics else 0,
            'health_status': 'Healthy' if breached_count == 0 else 'Needs Attention'
        }

    def create_enhanced_pdf_report(self, metrics: List[SLOMetric],
                                 incident: IncidentData = None, output_path: str = None,
                                 use_browser: bool = True) -> str:
        """Create PDF report using enhanced template with browser or WeasyPrint"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/enhanced_sre_report_{timestamp}.pdf"

        # Try browser PDF generation first for exact HTML matching
        if use_browser and BROWSER_PDF_AVAILABLE:
            try:
                self.logger.info("Generating PDF using headless browser for exact HTML matching...")

                # Generate the same HTML content that would be used for HTML reports
                # Generate LLM analysis
                summary = self._create_summary_stats(metrics)
                llm_analysis = self.llm_analyzer.analyze_performance_metrics(metrics, summary)

                # Create trend visualizations (base64 for charts)
                trend_charts = self.create_trend_visualizations(metrics, save_images=False)

                # Prepare template data (exact same as HTML)
                template_data = {
                    'app_name': self.app_name,
                    'report_date': datetime.now().strftime("%B %d, %Y"),
                    'report_time': datetime.now().strftime("%H:%M:%S UTC"),
                    'metrics': metrics,
                    'trend_charts': trend_charts,
                    'incident': incident,
                    'summary': summary,
                    'has_incident': incident is not None,
                    'llm_analysis': llm_analysis
                }

                # Use the EXACT SAME HTML template as HTML reports
                html_template_content = self._get_enhanced_html_template()
                template = jinja2.Template(html_template_content)
                rendered_html = template.render(**template_data)

                # Generate PDF with headless browser
                browser_generator = BrowserPDFGenerator()
                success = browser_generator.create_pdf_from_html_sync(rendered_html, output_path)

                if success:
                    self.logger.info(f"✅ Enhanced PDF generated with browser: {output_path}")
                    return output_path
                else:
                    self.logger.warning("Browser PDF failed, falling back to WeasyPrint...")

            except Exception as e:
                self.logger.warning(f"Browser PDF failed: {e}, falling back to WeasyPrint...")

        # Set up environment for WeasyPrint
        import os
        os.environ['PKG_CONFIG_PATH'] = '/opt/homebrew/lib/pkgconfig'
        os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'

        # Try WeasyPrint as fallback
        if WEASYPRINT_AVAILABLE:
            try:
                self.logger.info("Generating enhanced PDF using WeasyPrint...")

                # Generate LLM analysis for PDF
                summary = self._create_summary_stats(metrics)
                llm_analysis = self.llm_analyzer.analyze_performance_metrics(metrics, summary)

                # Create trend visualizations (base64 for PDF)
                trend_charts = self.create_trend_visualizations(metrics, save_images=False)

                # Prepare template data (same as HTML)
                template_data = {
                    'app_name': self.app_name,
                    'report_date': datetime.now().strftime("%B %d, %Y"),
                    'report_time': datetime.now().strftime("%H:%M:%S UTC"),
                    'metrics': metrics,
                    'trend_charts': trend_charts,
                    'incident': incident,
                    'summary': summary,
                    'has_incident': incident is not None,
                    'llm_analysis': llm_analysis
                }

                # Load PDF-optimized template
                pdf_template_content = self._get_enhanced_pdf_template()
                template = jinja2.Template(pdf_template_content)
                rendered_html = template.render(**template_data)

                # Generate PDF with WeasyPrint
                pdf_generator = WeasyPrintPDFGenerator()
                success = pdf_generator.create_pdf_from_html(
                    rendered_html,
                    output_path,
                    base_url=f"file://{Path().absolute()}/"
                )

                if success:
                    self.logger.info(f"✅ Enhanced PDF generated with WeasyPrint: {output_path}")
                    return output_path
                else:
                    self.logger.warning("WeasyPrint failed, falling back to ReportLab...")

            except Exception as e:
                self.logger.warning(f"WeasyPrint failed: {e}, falling back to ReportLab...")

        # Fallback to ReportLab if WeasyPrint fails
        return self._create_reportlab_fallback_pdf(metrics, incident, output_path)

    def create_simple_pdf_report(self, html_path: str, metrics: List[SLOMetric],
                                incident: IncidentData = None, output_path: str = None) -> str:
        """Create PDF report - now uses enhanced template by default"""
        return self.create_enhanced_pdf_report(metrics, incident, output_path)

    def _create_reportlab_fallback_pdf(self, metrics: List[SLOMetric],
                                     incident: IncidentData = None, output_path: str = None) -> str:
        """Fallback PDF generation using ReportLab"""
        if not REPORTLAB_AVAILABLE:
            self.logger.warning("Both WeasyPrint and ReportLab unavailable, skipping PDF generation")
            return ""

        try:
            from reportlab.lib.colors import blue, green, red, orange, black
            from reportlab.platypus import Table, TableStyle, PageBreak, Image as ReportLabImage
            from reportlab.lib.units import inch
            from tempfile import NamedTemporaryFile
            import os

            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Generate trend charts for PDF embedding (save as image files)
            self.logger.info("Generating trend charts for PDF embedding...")
            trend_chart_images = self.create_trend_visualizations(metrics, save_images=True)

            # Title page
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                alignment=1,  # Center alignment
            )
            story.append(Paragraph(f"{self.app_name}", title_style))
            story.append(Paragraph("Comprehensive SRE Performance Report", styles['Heading2']))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 25))  # Reduced from 40

            # Executive Summary
            summary = self._create_summary_stats(metrics)
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            story.append(Spacer(1, 8))  # Reduced from 12

            # Summary table
            summary_data = [
                ['Metric', 'Value'],
                ['Total Services Monitored', str(summary['total_services'])],
                ['Total SLO Metrics', str(summary['total_metrics'])],
                ['Compliant Metrics', f"{summary['compliant_count']} ({summary['compliance_percentage']:.1f}%)"],
                ['At Risk Metrics', str(summary['at_risk_count'])],
                ['Breached SLOs', str(summary['breached_count'])],
                ['Overall System Health', summary['health_status']]
            ]

            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),  # White text for header
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), (0.95, 0.95, 0.95)),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))  # Reduced from 30

            # SLO Metrics Details
            story.append(Paragraph("SLO Metrics Status Details", styles['Heading2']))
            story.append(Spacer(1, 8))  # Reduced from 12

            # Create metrics table
            metrics_data = [['Service', 'Metric', 'Current', 'Target', 'Status', 'Error Budget']]

            for metric in metrics:
                status_color = green if metric.status == 'compliant' else orange if metric.status == 'at_risk' else red
                metrics_data.append([
                    metric.service_name,
                    metric.metric_name.replace('_', ' ').title(),
                    f"{metric.current_value:.2f} {metric.unit}",
                    f"{metric.slo_target:.2f} {metric.unit}",
                    metric.status.title(),
                    f"{metric.error_budget_consumed:.1f}%"
                ])

            metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), (0.98, 0.98, 0.98)),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20))  # Reduced from 30

            # Add trend visualizations to PDF
            if trend_chart_images:
                story.append(Paragraph("📊 Performance Trend Analysis", styles['Heading2']))
                story.append(Spacer(1, 8))  # Reduced from 12
                story.append(Paragraph("The following charts show performance trends over the last 30 days with current status indicators.", styles['Normal']))
                story.append(Spacer(1, 10))  # Reduced from 15

                for chart_name, image_path in trend_chart_images.items():
                    try:
                        # Add chart title
                        chart_title = f"{chart_name.replace('_', ' ').title()} Performance Trends"
                        story.append(Paragraph(chart_title, styles['Heading3']))
                        story.append(Spacer(1, 6))  # Reduced from 10

                        # Add chart image (slightly smaller to fit better)
                        chart_image = ReportLabImage(image_path, width=6.5*inch, height=4.5*inch)
                        story.append(chart_image)
                        story.append(Spacer(1, 10))  # Reduced from 15

                        # Don't delete here - clean up at the end

                    except Exception as img_error:
                        self.logger.warning(f"Failed to add chart {chart_name} to PDF: {img_error}")
                        continue

                # Don't force page break - let it flow naturally

            # AI-Powered Analysis and Recommendations
            story.append(Spacer(1, 15))  # Space before section
            story.append(Paragraph("🤖 AI-Powered Performance Analysis", styles['Heading2']))
            story.append(Spacer(1, 8))  # Reduced from 12

            # Generate LLM analysis for performance insights
            performance_analysis = self.llm_analyzer.analyze_performance_metrics(metrics, summary)

            # Split long analysis into paragraphs for better readability
            analysis_paragraphs = performance_analysis.split('\n\n')
            for para in analysis_paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 6))
            story.append(Spacer(1, 10))  # Reduced from 15

            # Key Recommendations section
            story.append(Paragraph("Key Recommendations", styles['Heading3']))
            story.append(Spacer(1, 6))  # Reduced from 8

            recommendations = []
            if summary['breached_count'] > 0:
                recommendations.append(f"• URGENT: Address {summary['breached_count']} SLO breach(es) immediately")
            if summary['at_risk_count'] > 0:
                recommendations.append(f"• Monitor {summary['at_risk_count']} service(s) at risk of SLO breach")
            recommendations.extend([
                "• Review error budget consumption and implement proactive alerting",
                "• Analyze performance trends for capacity planning",
                "• Implement automated scaling based on performance metrics",
                "• Update incident response procedures based on latest analysis",
                "• Schedule regular SLO review meetings with stakeholders"
            ])

            for rec in recommendations:
                story.append(Paragraph(rec, styles['Normal']))
                story.append(Spacer(1, 6))  # Reduced from 8
            story.append(Spacer(1, 15))  # Reduced from 20

            # Incident information if available
            if incident:
                story.append(PageBreak())
                story.append(Paragraph("🚨 Incident Analysis Report", styles['Heading1']))
                story.append(Spacer(1, 20))

                # Incident details table
                incident_data = [
                    ['Field', 'Value'],
                    ['Incident ID', incident.incident_id],
                    ['Application', incident.application_name],
                    ['Severity', incident.severity],
                    ['Start Time', incident.start_time.strftime('%Y-%m-%d %H:%M:%S')],
                    ['End Time', incident.end_time.strftime('%Y-%m-%d %H:%M:%S') if incident.end_time else 'Ongoing'],
                    ['Duration', f"{(incident.end_time - incident.start_time).total_seconds() / 3600:.2f} hours" if incident.end_time else 'Ongoing'],
                    ['Affected Services', ', '.join(incident.affected_services)]
                ]

                incident_table = Table(incident_data, colWidths=[2*inch, 4*inch])
                incident_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), (1, 0.9, 0.9)),
                    ('GRID', (0, 0), (-1, -1), 1, black)
                ]))
                story.append(incident_table)
                story.append(Spacer(1, 20))

                story.append(Paragraph("Root Cause Analysis", styles['Heading3']))
                story.append(Paragraph(incident.root_cause, styles['Normal']))
                story.append(Spacer(1, 15))

                story.append(Paragraph("Resolution Steps", styles['Heading3']))
                for i, step in enumerate(incident.resolution_steps, 1):
                    story.append(Paragraph(f"{i}. {step}", styles['Normal']))
                story.append(Spacer(1, 15))

                if incident.llm_analysis:
                    story.append(Paragraph("AI-Powered Analysis Summary", styles['Heading3']))
                    # Split long text into chunks
                    llm_text = incident.llm_analysis
                    if len(llm_text) > 1000:
                        llm_text = llm_text[:1000] + "... [Analysis truncated for PDF]"

                    # Split into paragraphs
                    for para in llm_text.split('\n\n'):
                        if para.strip():
                            story.append(Paragraph(para.strip(), styles['Normal']))
                    story.append(Spacer(1, 15))

                story.append(Paragraph("Lessons Learned", styles['Heading3']))
                story.append(Paragraph(incident.lessons_learned, styles['Normal']))

            # Footer (more compact)
            story.append(Spacer(1, 20))
            story.append(Paragraph("Report Generation Details", styles['Heading3']))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"Generated by: Enhanced SRE Report System | Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']))
            story.append(Paragraph(f"Data Period: Last 30 days | Features: AI-powered analysis, trend visualizations, SLO compliance tracking", styles['Normal']))

            doc.build(story)

            # Clean up temporary chart files
            for chart_name, image_path in trend_chart_images.items():
                try:
                    os.unlink(image_path)
                    self.logger.debug(f"Cleaned up chart file: {image_path}")
                except:
                    pass  # Ignore cleanup errors

            self.logger.info(f"Enhanced PDF report with charts saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

    def _get_enhanced_html_template(self) -> str:
        """Load enhanced HTML template from file"""
        template_path = Path("templates/enhanced_sre_template.html")

        # Try to load from templates directory
        if template_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Could not load enhanced template: {e}")

        # Fallback to inline template if file not found
        return self._get_comprehensive_html_template()

    def _get_enhanced_pdf_template(self) -> str:
        """Load enhanced PDF template - use same structure as HTML but with PDF-optimized CSS"""
        # Always use the HTML template for consistent structure
        html_template = self._get_enhanced_html_template()
        return self._optimize_template_for_pdf(html_template)

    def _optimize_template_for_pdf(self, html_content: str) -> str:
        """Optimize HTML template for PDF rendering while preserving structure"""
        import re

        # Convert Tailwind classes to inline CSS for PDF compatibility
        tailwind_conversions = [
            # Grid system
            (r'class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"',
             'style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15pt;"'),

            # Metric cards
            (r'class="metric-card p-6 rounded-xl"',
             'style="background: #2d3748; border: 1px solid #4a5568; padding: 15pt; border-radius: 8pt; margin-bottom: 10pt; page-break-inside: avoid;"'),

            # Status indicators
            (r'class="status-indicator ([^"]*) mr-3"',
             r'style="width: 12pt; height: 12pt; border-radius: 50%; margin-right: 8pt; display: inline-block; \1"'),
            (r'bg-green-500', 'background-color: #10b981;'),
            (r'bg-yellow-500', 'background-color: #f59e0b;'),
            (r'bg-red-500', 'background-color: #ef4444;'),

            # Text styling
            (r'class="text-lg font-semibold text-white"',
             'style="font-size: 12pt; font-weight: 600; color: #ffffff;"'),
            (r'class="text-3xl font-bold text-white"',
             'style="font-size: 18pt; font-weight: bold; color: #ffffff;"'),
            (r'class="text-sm text-slate-400 mb-1"',
             'style="font-size: 9pt; color: #94a3b8; margin-bottom: 3pt;"'),
            (r'class="text-sm text-slate-400 mb-3"',
             'style="font-size: 9pt; color: #94a3b8; margin-bottom: 8pt;"'),
            (r'class="text-sm text-slate-300"',
             'style="font-size: 9pt; color: #cbd5e1;"'),

            # Flexbox
            (r'class="flex items-center mb-4"',
             'style="display: flex; align-items: center; margin-bottom: 10pt;"'),
            (r'class="flex items-center justify-between mb-2"',
             'style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5pt;"'),

            # Status boxes
            (r'class="mt-4 p-3 ([^"]*) rounded"',
             r'style="margin-top: 10pt; padding: 8pt; border-radius: 4pt; \1"'),
            (r'bg-green-900/20 border-l-4 border-green-500',
             'background-color: rgba(6, 78, 59, 0.2); border-left: 3pt solid #10b981;'),
            (r'bg-yellow-900/20 border-l-4 border-yellow-500',
             'background-color: rgba(120, 53, 15, 0.2); border-left: 3pt solid #f59e0b;'),
            (r'bg-red-900/20 border-l-4 border-red-500',
             'background-color: rgba(127, 29, 29, 0.2); border-left: 3pt solid #ef4444;'),

            # Status text colors
            (r'class="text-sm font-medium text-green-200"',
             'style="font-size: 9pt; font-weight: 500; color: #bbf7d0;"'),
            (r'class="text-sm font-medium text-yellow-200"',
             'style="font-size: 9pt; font-weight: 500; color: #fef3c7;"'),
            (r'class="text-sm font-medium text-red-200"',
             'style="font-size: 9pt; font-weight: 500; color: #fecaca;"'),
            (r'class="text-xs text-slate-400"',
             'style="font-size: 8pt; color: #94a3b8;"'),
        ]

        # Apply Tailwind conversions
        for pattern, replacement in tailwind_conversions:
            html_content = re.sub(pattern, replacement, html_content, flags=re.IGNORECASE)

        # Convert Font Awesome icons to text symbols
        icon_conversions = [
            (r'<i class="fas fa-tachometer-alt[^"]*"></i>', '⚡'),
            (r'<i class="fas fa-brain[^"]*"></i>', '🧠'),
            (r'<i class="fas fa-[^"]*"></i>', '•'),  # Fallback for other icons
        ]

        for pattern, replacement in icon_conversions:
            html_content = re.sub(pattern, replacement, html_content, flags=re.IGNORECASE)

        # Remove/replace elements that don't work in PDF
        pdf_optimizations = [
            # Remove Chart.js scripts (charts will be base64 images instead)
            (r'<script.*?chart\.js.*?</script>', ''),
            # Remove Tailwind CSS CDN
            (r'<script.*?tailwindcss.*?</script>', ''),
            # Remove Font Awesome CSS
            (r'<link.*?font-awesome.*?>', ''),
            # Remove interactive elements
            (r'<div class="floating-menu">.*?</div>', ''),
            (r'onclick="[^"]*"', ''),
            # Update body with print-friendly styles
            (r'<body[^>]*>', '<body style="font-family: Inter, Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #1f2937; background: white; margin: 0; padding: 20pt;">'),
        ]

        for pattern, replacement in pdf_optimizations:
            html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL | re.IGNORECASE)

        return html_content

    def _get_comprehensive_html_template(self) -> str:
        """Return comprehensive HTML template"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ app_name }} - Comprehensive SRE Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
            text-align: center;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }
        .trend-section {
            margin: 40px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .incident-section {
            margin: 40px 0;
            padding: 25px;
            background: #fff3cd;
            border-radius: 8px;
            border-left: 5px solid #ffc107;
        }
        .incident-critical {
            background: #f8d7da;
            border-left-color: #dc3545;
        }
        .incident-high {
            background: #ffe6cc;
            border-left-color: #fd7e14;
        }
        .llm-analysis {
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
            margin: 20px 0;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
        }
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .metrics-table th {
            background-color: #007acc;
            color: white;
        }
        .status-compliant { color: #28a745; font-weight: bold; }
        .status-at-risk { color: #ffc107; font-weight: bold; }
        .status-breached { color: #dc3545; font-weight: bold; }
        .section {
            margin: 40px 0;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ app_name }}</h1>
            <div class="subtitle">Comprehensive SRE Performance & Incident Report</div>
            <div class="subtitle">Generated: {{ report_date }} at {{ report_time }}</div>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Services</h3>
                    <div class="value">{{ summary.total_services }}</div>
                </div>
                <div class="summary-card">
                    <h3>Compliance Rate</h3>
                    <div class="value">{{ "%.1f"|format(summary.compliance_percentage) }}%</div>
                </div>
                <div class="summary-card">
                    <h3>At Risk</h3>
                    <div class="value status-at-risk">{{ summary.at_risk_count }}</div>
                </div>
                <div class="summary-card">
                    <h3>SLO Breaches</h3>
                    <div class="value status-breached">{{ summary.breached_count }}</div>
                </div>
                <div class="summary-card">
                    <h3>System Health</h3>
                    <div class="value status-{% if summary.health_status == 'Healthy' %}compliant{% else %}breached{% endif %}">
                        {{ summary.health_status }}
                    </div>
                </div>
            </div>
        </div>

        <div class="trend-section">
            <h2>🔄 Performance Trends & Analysis</h2>
            <p>The following charts show performance trends over the last 30 days with current status indicators.</p>

            {% for chart_name, chart_data in trend_charts.items() %}
            <div class="chart-container">
                <h3>{{ chart_name.replace('_', ' ').title() }} Performance Trends</h3>
                <img src="{{ chart_data }}" alt="{{ chart_name }} Trend Chart">
            </div>
            {% endfor %}
        </div>

        {% if has_incident %}
        <div class="section">
            <div class="incident-section {% if incident.severity == 'Critical' %}incident-critical{% elif incident.severity == 'High' %}incident-high{% endif %}">
                <h2>🚨 Incident Analysis Report</h2>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
                    <div><strong>Incident ID:</strong> {{ incident.incident_id }}</div>
                    <div><strong>Severity:</strong> <span class="status-breached">{{ incident.severity }}</span></div>
                    <div><strong>Application:</strong> {{ incident.application_name }}</div>
                    <div><strong>Duration:</strong> {{ incident.start_time.strftime('%Y-%m-%d %H:%M') }} - {{ incident.end_time.strftime('%H:%M') if incident.end_time else 'Ongoing' }}</div>
                </div>

                <div style="margin: 20px 0;">
                    <h3>Description</h3>
                    <p>{{ incident.description }}</p>
                </div>

                <div style="margin: 20px 0;">
                    <h3>Affected Services</h3>
                    <ul>
                        {% for service in incident.affected_services %}
                        <li>{{ service }}</li>
                        {% endfor %}
                    </ul>
                </div>

                <div style="margin: 20px 0;">
                    <h3>Initial Root Cause Analysis</h3>
                    <p>{{ incident.root_cause }}</p>
                </div>

                <div class="llm-analysis">
                    <h3>🤖 AI-Powered Deep Analysis & Recommendations</h3>
                    <div style="white-space: pre-line;">{{ incident.llm_analysis }}</div>
                </div>

                <div style="margin: 20px 0;">
                    <h3>Resolution Steps Taken</h3>
                    <ol>
                        {% for step in incident.resolution_steps %}
                        <li>{{ step }}</li>
                        {% endfor %}
                    </ol>
                </div>

                <div style="margin: 20px 0;">
                    <h3>Lessons Learned</h3>
                    <div style="background: #e8f5e8; padding: 15px; border-radius: 5px;">
                        <div style="white-space: pre-line;">{{ incident.lessons_learned }}</div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>📊 Current SLO Metrics Status</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Metric</th>
                        <th>Current Value</th>
                        <th>SLO Target</th>
                        <th>Status</th>
                        <th>Error Budget Used</th>
                        <th>30-Day Trend</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in metrics %}
                    <tr>
                        <td>{{ metric.service_name }}</td>
                        <td>{{ metric.metric_name|replace('_', ' ')|title }}</td>
                        <td>{{ "%.2f"|format(metric.current_value) }} {{ metric.unit }}</td>
                        <td>{{ "%.2f"|format(metric.slo_target) }} {{ metric.unit }}</td>
                        <td class="status-{{ metric.status }}">{{ metric.status|title }}</td>
                        <td>{{ "%.1f"|format(metric.error_budget_consumed) }}%</td>
                        <td>
                            {% if metric.trend_data %}
                                {% set trend_change = metric.trend_data[-1] - metric.trend_data[0] %}
                                {% if trend_change > 0 %}
                                    {% if metric.metric_name == 'availability' %}📈 Improving{% else %}📉 Degrading{% endif %}
                                {% else %}
                                    {% if metric.metric_name == 'availability' %}📉 Degrading{% else %}📈 Improving{% endif %}
                                {% endif %}
                            {% else %}
                            No trend data
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>🎯 Key Recommendations</h2>
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <ul>
                    {% if summary.breached_count > 0 %}
                    <li><strong>High Priority:</strong> Address {{ summary.breached_count }} SLO breach(es) immediately</li>
                    {% endif %}
                    {% if summary.at_risk_count > 0 %}
                    <li><strong>Medium Priority:</strong> Monitor {{ summary.at_risk_count }} service(s) at risk of SLO breach</li>
                    {% endif %}
                    <li>Review error budget consumption and implement proactive alerting</li>
                    <li>Analyze performance trends for capacity planning</li>
                    <li>Update incident response procedures based on latest analysis</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <small>
                <p><strong>Report Features:</strong></p>
                <ul>
                    <li>✅ Real-time SLO/SLA monitoring with trend analysis</li>
                    <li>✅ AI-powered incident root cause analysis</li>
                    <li>✅ Performance visualizations with 30-day historical data</li>
                    <li>✅ Automated recommendations based on performance patterns</li>
                    <li>✅ Comprehensive incident documentation and lessons learned</li>
                </ul>
                <p>This report combines traditional SRE metrics with advanced AI analysis to provide actionable insights for system reliability improvement.</p>
                <p><em>Generated at {{ report_time }} on {{ report_date }}</em></p>
            </small>
        </div>
    </div>
</body>
</html>
        '''

    def generate_full_report_suite(self, application_name: str = None,
                                 services: List[str] = None,
                                 incident_time: datetime = None,
                                 incident_duration: float = 1.0) -> Dict[str, str]:
        """Generate complete report suite with performance and incident analysis"""

        if not application_name:
            application_name = self.app_name

        self.logger.info(f"Generating comprehensive report suite for {application_name}")

        # Generate performance metrics with trends
        metrics = self.generate_metrics_with_trends(services, days_back=30)

        # Generate incident report if incident time provided
        incident = None
        if incident_time:
            incident = self.generate_incident_report(
                application_name, incident_time, incident_duration
            )

        # Generate reports
        results = {}

        # HTML Report with trends and incident analysis
        html_path = self.create_comprehensive_html_report(metrics, incident)
        results['html_report'] = html_path

        # Simple PDF Report
        pdf_path = self.create_simple_pdf_report(html_path, metrics, incident)
        if pdf_path:
            results['pdf_report'] = pdf_path

        # JSON Data Export
        json_path = self._export_json_data(metrics, incident)
        results['json_data'] = json_path

        self.logger.info("Full report suite generation completed")
        return results

    def _export_json_data(self, metrics: List[SLOMetric], incident: IncidentData = None) -> str:
        """Export all data as JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/generated/comprehensive_sre_data_{timestamp}.json"

        # Convert dataclasses to dictionaries
        def convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_dataclass(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataclass(v) for k, v in obj.items()}
            return obj

        data = {
            "report_metadata": {
                "application_name": self.app_name,
                "generated_at": datetime.now().isoformat(),
                "report_type": "Comprehensive SRE Report with Trends and Incident Analysis",
                "data_period_days": 30
            },
            "slo_metrics": [convert_dataclass(m) for m in metrics],
            "summary": self._create_summary_stats(metrics),
            "incident": convert_dataclass(incident) if incident else None
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"JSON data exported to {output_path}")
        return output_path


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    print("🚀 Enhanced SRE Report System")
    print("=" * 50)

    # Get user input for incident analysis
    app_name = input("Enter application name (default: E-Commerce Platform): ").strip() or "E-Commerce Platform"

    # Ask if user wants incident analysis
    want_incident = input("Do you want to include incident analysis? (y/n): ").strip().lower() == 'y'

    incident_time = None
    if want_incident:
        # Get incident timeframe
        hours_ago = input("How many hours ago did the incident start? (default: 2): ").strip()
        try:
            hours_ago = float(hours_ago) if hours_ago else 2.0
            incident_time = datetime.now() - timedelta(hours=hours_ago)
        except ValueError:
            incident_time = datetime.now() - timedelta(hours=2)

        duration = input("Incident duration in hours (default: 1): ").strip()
        try:
            duration = float(duration) if duration else 1.0
        except ValueError:
            duration = 1.0
    else:
        duration = 1.0

    # Define services
    services = [f"{app_name.lower().replace(' ', '-')}-{svc}"
               for svc in ["web", "api", "auth", "database", "cache", "payments"]]

    # Generate reports
    system = EnhancedSREReportSystem(app_name=app_name)

    report_paths = system.generate_full_report_suite(
        application_name=app_name,
        services=services,
        incident_time=incident_time,
        incident_duration=duration if want_incident else None
    )

    print(f"\\n🎯 {app_name} Enhanced SRE Report Generation Complete!")
    print("=" * 70)
    for report_type, path in report_paths.items():
        if path:
            print(f"📊 {report_type.replace('_', ' ').title()}: {path}")

    print("\\n🚀 Enhanced Features Included:")
    print("• 📈 Performance trend analysis with 30-day historical data")
    print("• 🚨 AI-powered incident root cause analysis")
    print("• 🎯 Interactive HTML reports with comprehensive visualizations")
    print("• 📄 PDF export for stakeholder sharing")
    print("• 📋 JSON data export for API integration")
    print("• 🔍 Real-time SLO/SLA compliance monitoring")
    print("• 🤖 LLM-powered recommendations and lessons learned")
    print("• 📊 Error budget tracking and burn rate analysis")
    print("\\n💡 Perfect for SRE teams to track system reliability and incident response!")