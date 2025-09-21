"""
Generic SLO/SLA Report Generator with LLM Integration
Generates comprehensive Service Level Objective and Service Level Agreement compliance reports
for any application or system with intelligent error analysis and recommendations.
"""

import pandas as pd
import numpy as np
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
import weasyprint
from openai import OpenAI
import anthropic

@dataclass
class SLOMetric:
    """SLO metric data structure"""
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

@dataclass
class ErrorAnalysis:
    """Error analysis data structure"""
    service_name: str
    error_type: str
    frequency: int
    severity: str
    root_cause: str
    impact: str
    llm_recommendation: str

@dataclass
class SLACompliance:
    """SLA compliance data structure"""
    service_name: str
    availability_percent: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    compliance_status: str
    penalties_applicable: bool
    credit_percentage: float

class LLMAnalyzer:
    """LLM-powered error analysis and recommendations"""

    def __init__(self, provider: str = "anthropic", api_key: str = None):
        self.provider = provider.lower()
        self.logger = logging.getLogger(__name__)

        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
                if self.api_key:
                    self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if self.api_key:
                    self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_errors_and_recommend(self, metrics: List[SLOMetric],
                                   error_patterns: List[Dict]) -> List[ErrorAnalysis]:
        """Analyze errors using LLM and generate recommendations"""
        if not hasattr(self, 'client'):
            self.logger.warning("No LLM client configured. Using fallback analysis.")
            return self._fallback_analysis(metrics, error_patterns)

        try:
            # Prepare context for LLM
            context = self._prepare_analysis_context(metrics, error_patterns)

            if self.provider == "anthropic":
                return self._analyze_with_anthropic(context, metrics)
            elif self.provider == "openai":
                return self._analyze_with_openai(context, metrics)
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(metrics, error_patterns)

    def _prepare_analysis_context(self, metrics: List[SLOMetric],
                                error_patterns: List[Dict]) -> str:
        """Prepare context string for LLM analysis"""
        context = "System Performance Analysis\\n\\n"
        context += "=== SLO Metrics Status ===\\n"

        breached_metrics = [m for m in metrics if m.status == "breached"]
        at_risk_metrics = [m for m in metrics if m.status == "at_risk"]

        if breached_metrics:
            context += "CRITICAL - SLO Breaches:\\n"
            for metric in breached_metrics:
                context += f"- {metric.service_name}: {metric.metric_name} = {metric.current_value:.2f} (target: {metric.slo_target:.2f})\\n"

        if at_risk_metrics:
            context += "\\nWARNING - At Risk Services:\\n"
            for metric in at_risk_metrics:
                context += f"- {metric.service_name}: {metric.metric_name} = {metric.current_value:.2f} (target: {metric.slo_target:.2f})\\n"

        if error_patterns:
            context += "\\n=== Error Patterns ===\\n"
            for pattern in error_patterns:
                context += f"- {pattern.get('type', 'Unknown')}: {pattern.get('count', 0)} occurrences\\n"

        return context

    def _analyze_with_anthropic(self, context: str, metrics: List[SLOMetric]) -> List[ErrorAnalysis]:
        """Analyze using Anthropic Claude"""
        prompt = f"""
        As an experienced Site Reliability Engineer, analyze the following system performance data and provide actionable recommendations.

        {context}

        Please provide:
        1. Root cause analysis for each issue
        2. Specific technical recommendations
        3. Priority levels (Critical, High, Medium, Low)
        4. Implementation steps

        Format your response as a structured analysis focusing on practical solutions.
        """

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis_text = response.content[0].text
            return self._parse_llm_response(analysis_text, metrics)
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return self._fallback_analysis(metrics, [])

    def _analyze_with_openai(self, context: str, metrics: List[SLOMetric]) -> List[ErrorAnalysis]:
        """Analyze using OpenAI GPT"""
        prompt = f"""
        As an experienced Site Reliability Engineer, analyze the following system performance data and provide actionable recommendations.

        {context}

        Please provide:
        1. Root cause analysis for each issue
        2. Specific technical recommendations
        3. Priority levels (Critical, High, Medium, Low)
        4. Implementation steps

        Format your response as a structured analysis focusing on practical solutions.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )

            analysis_text = response.choices[0].message.content
            return self._parse_llm_response(analysis_text, metrics)
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._fallback_analysis(metrics, [])

    def _parse_llm_response(self, response_text: str, metrics: List[SLOMetric]) -> List[ErrorAnalysis]:
        """Parse LLM response into structured ErrorAnalysis objects"""
        error_analyses = []

        # Simple parsing - in production, you might want more sophisticated NLP
        breached_metrics = [m for m in metrics if m.status in ["breached", "at_risk"]]

        for i, metric in enumerate(breached_metrics):
            error_analysis = ErrorAnalysis(
                service_name=metric.service_name,
                error_type=f"{metric.metric_name}_issue",
                frequency=1,
                severity="High" if metric.status == "breached" else "Medium",
                root_cause=f"LLM Analysis: {metric.metric_name} performance degradation",
                impact=f"SLO breach affecting {metric.service_name}",
                llm_recommendation=self._extract_recommendation_for_service(
                    response_text, metric.service_name, metric.metric_name
                )
            )
            error_analyses.append(error_analysis)

        return error_analyses

    def _extract_recommendation_for_service(self, response_text: str,
                                          service_name: str, metric_name: str) -> str:
        """Extract specific recommendation for a service from LLM response"""
        # Simple extraction - in production, use more sophisticated parsing
        lines = response_text.split('\\n')
        recommendation = f"Review {metric_name} performance for {service_name}. "

        # Look for relevant recommendations in the response
        for line in lines:
            if any(keyword in line.lower() for keyword in [service_name.lower(), metric_name.lower(), 'recommend']):
                recommendation += line.strip() + " "

        if len(recommendation.strip()) < 50:  # Fallback if no specific recommendation found
            if metric_name == "availability":
                recommendation = "Investigate service health checks, implement circuit breakers, and review deployment practices."
            elif metric_name == "latency_p95":
                recommendation = "Optimize database queries, implement caching, and review resource allocation."
            elif metric_name == "error_rate":
                recommendation = "Review error logs, implement better error handling, and add monitoring for error patterns."

        return recommendation.strip()

    def _fallback_analysis(self, metrics: List[SLOMetric],
                          error_patterns: List[Dict]) -> List[ErrorAnalysis]:
        """Fallback analysis when LLM is not available"""
        error_analyses = []

        breached_metrics = [m for m in metrics if m.status in ["breached", "at_risk"]]

        for metric in breached_metrics:
            recommendation = self._get_fallback_recommendation(metric)

            error_analysis = ErrorAnalysis(
                service_name=metric.service_name,
                error_type=f"{metric.metric_name}_issue",
                frequency=1,
                severity="High" if metric.status == "breached" else "Medium",
                root_cause=f"Rule-based analysis: {metric.metric_name} threshold exceeded",
                impact=f"SLO breach affecting {metric.service_name}",
                llm_recommendation=recommendation
            )
            error_analyses.append(error_analysis)

        return error_analyses

    def _get_fallback_recommendation(self, metric: SLOMetric) -> str:
        """Get fallback recommendations based on metric type"""
        recommendations = {
            "availability": "1. Check service health endpoints 2. Review deployment logs 3. Implement circuit breakers 4. Scale resources if needed",
            "latency_p95": "1. Profile application performance 2. Optimize database queries 3. Implement caching 4. Review resource allocation",
            "latency_p99": "1. Identify performance outliers 2. Optimize slow operations 3. Implement request timeouts 4. Review system capacity",
            "error_rate": "1. Analyze error logs 2. Implement better error handling 3. Add input validation 4. Review dependent services"
        }

        return recommendations.get(metric.metric_name, "Review service performance and implement monitoring improvements")

class GenericSLOSLAReportGenerator:
    """Generic SLO/SLA report generator for any application"""

    def __init__(self, config_dir: str = "config", app_name: str = "Application"):
        self.config_dir = Path(config_dir)
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)

        # Load configurations
        self.slo_config = self._load_yaml("slo_definitions.yaml")
        self.sla_config = self._load_yaml("sla_thresholds.yaml")

        # Initialize LLM analyzer
        self.llm_analyzer = LLMAnalyzer()

        # Set up Jinja2 for HTML templating
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_dir / filename, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Could not load {filename}: {e}")
            return self._get_default_config(filename)

    def _get_default_config(self, filename: str) -> Dict[str, Any]:
        """Get default configuration when file is not available"""
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
        elif "sla" in filename:
            return {
                "service_level_agreements": {
                    "production": {
                        "external_availability": "99.9%",
                        "api_response_time_sla": "500ms"
                    }
                }
            }
        return {}

    def generate_demo_metrics(self, services: List[str] = None, days_back: int = 30) -> List[SLOMetric]:
        """Generate realistic demo metrics for any application"""
        if not services:
            services = ["web-service", "api-service", "database-service", "auth-service"]

        np.random.seed(42)
        metrics = []
        current_time = datetime.now()

        for service_name in services:
            # Generate availability metrics (99.8% - 99.99%)
            base_availability = 99.9 + np.random.normal(0, 0.05)
            current_availability = max(99.5, min(99.99, base_availability))

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
                description=f"Service availability for {service_name}"
            )
            metrics.append(availability_metric)

            # Generate latency metrics (100ms - 800ms)
            base_latency = 200 + np.random.normal(0, 50)
            current_latency = max(50, base_latency)

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
                description=f"95th percentile response time for {service_name}"
            )
            metrics.append(latency_metric)

            # Generate error rate metrics (0% - 2%)
            base_error_rate = 0.1 + abs(np.random.normal(0, 0.05))
            current_error_rate = min(2.0, base_error_rate)

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
                description=f"Error rate for {service_name}"
            )
            metrics.append(error_rate_metric)

        return metrics

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

    def create_html_report(self, metrics: List[SLOMetric],
                          error_analyses: List[ErrorAnalysis],
                          output_path: str = None) -> str:
        """Generate comprehensive HTML report"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/slo_sla_report_{timestamp}.html"

        # Create charts as base64 images
        charts = self._create_report_charts(metrics)

        # Prepare template data
        template_data = {
            'app_name': self.app_name,
            'report_date': datetime.now().strftime("%B %d, %Y"),
            'report_time': datetime.now().strftime("%H:%M:%S UTC"),
            'metrics': metrics,
            'error_analyses': error_analyses,
            'charts': charts,
            'summary': self._create_summary_stats(metrics),
            'recommendations': self._create_prioritized_recommendations(error_analyses)
        }

        # Create HTML template
        html_template = self._get_html_template()
        template = jinja2.Template(html_template)
        html_content = template.render(**template_data)

        # Save HTML file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"HTML report saved to {output_path}")
        return output_path

    def create_pdf_report(self, html_path: str, output_path: str = None) -> str:
        """Convert HTML report to PDF"""
        if not output_path:
            output_path = html_path.replace('.html', '.pdf')

        try:
            # Read HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Convert to PDF using WeasyPrint
            weasyprint.HTML(string=html_content).write_pdf(output_path)
            self.logger.info(f"PDF report saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            # Fallback: try with minimal HTML
            return self._create_simple_pdf_report(output_path)

    def _create_simple_pdf_report(self, output_path: str) -> str:
        """Create a simple PDF report as fallback"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
            )
            story.append(Paragraph(f"{self.app_name} SLO/SLA Compliance Report", title_style))
            story.append(Spacer(1, 12))

            # Date
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Content
            story.append(Paragraph("Report Summary", styles['Heading2']))
            story.append(Paragraph("This is a simplified PDF version of the SLO/SLA compliance report.", styles['Normal']))
            story.append(Paragraph("For full interactive features and detailed analysis, please refer to the HTML version.", styles['Normal']))

            doc.build(story)
            self.logger.info(f"Simple PDF report saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Simple PDF generation also failed: {e}")
            return ""

    def _create_report_charts(self, metrics: List[SLOMetric]) -> Dict[str, str]:
        """Create charts and return as base64 encoded images"""
        charts = {}

        # Service availability chart
        availability_metrics = [m for m in metrics if m.metric_name == "availability"]
        if availability_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            services = [m.service_name for m in availability_metrics]
            values = [m.current_value for m in availability_metrics]
            targets = [m.slo_target for m in availability_metrics]

            colors = ['green' if m.status == 'compliant' else 'orange' if m.status == 'at_risk' else 'red'
                     for m in availability_metrics]

            ax.bar(services, values, color=colors, alpha=0.7, label='Current')
            ax.plot(services, targets, 'bo-', label='SLO Target')
            ax.set_ylabel('Availability (%)')
            ax.set_title('Service Availability Status')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            charts['availability'] = self._fig_to_base64(fig)
            plt.close(fig)

        # Response time chart
        latency_metrics = [m for m in metrics if m.metric_name == "latency_p95"]
        if latency_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            services = [m.service_name for m in latency_metrics]
            values = [m.current_value for m in latency_metrics]
            targets = [m.slo_target for m in latency_metrics]

            colors = ['green' if m.status == 'compliant' else 'orange' if m.status == 'at_risk' else 'red'
                     for m in latency_metrics]

            ax.bar(services, values, color=colors, alpha=0.7, label='Current')
            ax.plot(services, targets, 'bo-', label='SLO Target')
            ax.set_ylabel('Response Time (ms)')
            ax.set_title('Service Response Time Status')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()

            charts['latency'] = self._fig_to_base64(fig)
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

    def _create_prioritized_recommendations(self, error_analyses: List[ErrorAnalysis]) -> List[Dict[str, Any]]:
        """Create prioritized list of recommendations"""
        recommendations = []

        # Sort by severity
        severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        sorted_analyses = sorted(error_analyses,
                               key=lambda x: severity_order.get(x.severity, 99))

        for i, analysis in enumerate(sorted_analyses, 1):
            recommendations.append({
                'priority': i,
                'service': analysis.service_name,
                'severity': analysis.severity,
                'issue': analysis.error_type.replace('_', ' ').title(),
                'root_cause': analysis.root_cause,
                'recommendation': analysis.llm_recommendation,
                'impact': analysis.impact
            })

        return recommendations

    def _get_html_template(self) -> str:
        """Return HTML template for the report"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ app_name }} - SLO/SLA Compliance Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
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
        .header .subtitle {
            color: #666;
            margin: 10px 0;
            font-size: 1.2em;
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
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }
        .status-compliant { color: #28a745; }
        .status-at-risk { color: #ffc107; }
        .status-breached { color: #dc3545; }

        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
        .metrics-table tr:hover {
            background-color: #f5f5f5;
        }

        .recommendations {
            margin: 30px 0;
        }
        .recommendation-item {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #007acc;
        }
        .recommendation-item.high-priority {
            border-left-color: #dc3545;
        }
        .recommendation-item.medium-priority {
            border-left-color: #ffc107;
        }

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
            <div class="subtitle">SLO/SLA Compliance Report</div>
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

        {% if charts %}
        <div class="section">
            <h2>Performance Charts</h2>
            {% if charts.availability %}
            <div class="chart-container">
                <h3>Service Availability</h3>
                <img src="{{ charts.availability }}" alt="Service Availability Chart">
            </div>
            {% endif %}
            {% if charts.latency %}
            <div class="chart-container">
                <h3>Response Time Performance</h3>
                <img src="{{ charts.latency }}" alt="Response Time Chart">
            </div>
            {% endif %}
        </div>
        {% endif %}

        <div class="section">
            <h2>SLO Metrics Detail</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Metric</th>
                        <th>Current Value</th>
                        <th>SLO Target</th>
                        <th>Status</th>
                        <th>Error Budget Used</th>
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
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if recommendations %}
        <div class="section">
            <h2>LLM-Powered Recommendations</h2>
            {% for rec in recommendations %}
            <div class="recommendation-item {% if rec.severity == 'High' %}high-priority{% elif rec.severity == 'Medium' %}medium-priority{% endif %}">
                <h3>Priority {{ rec.priority }}: {{ rec.issue }} ({{ rec.service }})</h3>
                <p><strong>Severity:</strong> {{ rec.severity }}</p>
                <p><strong>Root Cause:</strong> {{ rec.root_cause }}</p>
                <p><strong>Impact:</strong> {{ rec.impact }}</p>
                <p><strong>Recommendation:</strong> {{ rec.recommendation }}</p>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2>Error Analysis Summary</h2>
            {% if error_analyses %}
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Error Type</th>
                        <th>Severity</th>
                        <th>Frequency</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {% for error in error_analyses %}
                    <tr>
                        <td>{{ error.service_name }}</td>
                        <td>{{ error.error_type|replace('_', ' ')|title }}</td>
                        <td class="status-{% if error.severity == 'High' %}breached{% elif error.severity == 'Medium' %}at-risk{% else %}compliant{% endif %}">
                            {{ error.severity }}
                        </td>
                        <td>{{ error.frequency }}</td>
                        <td>{{ error.impact }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p>No critical errors detected in the current monitoring period.</p>
            {% endif %}
        </div>

        <div class="section">
            <small>
                <p><strong>Note:</strong> This report is generated automatically using advanced analytics and LLM-powered insights.
                For questions or concerns about specific recommendations, please consult with your SRE team.</p>
                <p>Report generated at {{ report_time }} on {{ report_date }}</p>
            </small>
        </div>
    </div>
</body>
</html>
        '''

    def generate_comprehensive_report(self, services: List[str] = None,
                                    days_back: int = 30,
                                    include_llm_analysis: bool = True) -> Dict[str, str]:
        """Generate complete SLO/SLA report with all formats"""
        self.logger.info(f"Starting comprehensive SLO/SLA report generation for {self.app_name}")

        # Generate metrics
        metrics = self.generate_demo_metrics(services, days_back)

        # Generate error analysis with LLM
        error_analyses = []
        if include_llm_analysis:
            try:
                error_patterns = []  # In production, this would come from your error tracking system
                error_analyses = self.llm_analyzer.analyze_errors_and_recommend(metrics, error_patterns)
            except Exception as e:
                self.logger.warning(f"LLM analysis failed, using fallback: {e}")
                error_analyses = self.llm_analyzer._fallback_analysis(metrics, [])

        # Generate reports
        results = {}

        # HTML Report
        html_path = self.create_html_report(metrics, error_analyses)
        results['html_report'] = html_path

        # PDF Report
        pdf_path = self.create_pdf_report(html_path)
        if pdf_path:
            results['pdf_report'] = pdf_path

        # JSON Data Export
        json_path = self._export_json_data(metrics, error_analyses)
        results['json_data'] = json_path

        self.logger.info("Comprehensive SLO/SLA report generation completed")
        return results

    def _export_json_data(self, metrics: List[SLOMetric],
                         error_analyses: List[ErrorAnalysis]) -> str:
        """Export all data as JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/generated/slo_sla_data_{timestamp}.json"

        data = {
            "report_metadata": {
                "application_name": self.app_name,
                "generated_at": datetime.now().isoformat(),
                "report_type": "SLO/SLA Compliance Report",
                "data_period_days": 30
            },
            "slo_metrics": [asdict(m) for m in metrics],
            "error_analyses": [asdict(e) for e in error_analyses],
            "summary": self._create_summary_stats(metrics)
        }

        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj

        data = convert_datetime(data)

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

    # Example usage - can be customized for any application
    app_name = "Multi-Service Application"
    services = ["web-frontend", "api-gateway", "user-service", "payment-service", "notification-service"]

    generator = GenericSLOSLAReportGenerator(app_name=app_name)
    report_paths = generator.generate_comprehensive_report(
        services=services,
        days_back=30,
        include_llm_analysis=True
    )

    print(f"🎯 {app_name} SLO/SLA Report Generation Complete!")
    print("=" * 60)
    for report_type, path in report_paths.items():
        if path:  # Only show successful generations
            print(f"📊 {report_type.replace('_', ' ').title()}: {path}")

    print("\n🚀 Features Included:")
    print("• Generic application support (configurable)")
    print("• HTML report with interactive charts")
    print("• PDF export capability")
    print("• LLM-powered error analysis and recommendations")
    print("• JSON data export for API integration")
    print("• Comprehensive SLO/SLA compliance tracking")
    print("• Error budget analysis")
    print("• Prioritized actionable recommendations")