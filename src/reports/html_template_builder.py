"""
HTML Template Builder Module

Handles generation of modular HTML templates for SRE reports.
"""

import logging
from typing import Any, Dict


class HTMLTemplateBuilder:
    """Builds modular HTML templates for SRE reports"""

    def __init__(self):
        """Initialize HTML template builder"""
        self.logger = logging.getLogger(__name__)

    def get_comprehensive_html_template(self) -> str:
        """
        Return comprehensive HTML template

        Composed from modular template sections for better maintainability.
        Each section is extracted to its own method for easier testing and modification.
        """
        return f"""<!DOCTYPE html>
<html lang="en">
{self._get_html_header_and_styles()}
<body>
    <div class="container">
        <div class="header">
            <h1>{{{{ app_name }}}}</h1>
            <div class="subtitle">Comprehensive SRE Performance & Incident Report</div>
            <div class="subtitle">Generated: {{{{ report_date }}}} at {{{{ report_time }}}}</div>
        </div>

{self._get_html_executive_summary()}

{self._get_html_trend_charts()}

{self._get_html_incident_analysis()}

{self._get_html_metrics_table()}

{self._get_html_recommendations()}

{self._get_html_footer()}
    </div>
</body>
</html>
        """

    def _get_html_header_and_styles(self) -> str:
        """Generate HTML header with embedded styles"""
        return """<head>
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
</head>"""

    def _get_html_executive_summary(self) -> str:
        """Generate executive summary section"""
        return """        <div class="section">
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
        </div>"""

    def _get_html_trend_charts(self) -> str:
        """Generate trend charts section"""
        return """        <div class="trend-section">
            <h2>🔄 Performance Trends & Analysis</h2>
            <p>The following charts show performance trends over the last 30 days with current status indicators.</p>

            {% for chart_name, chart_data in trend_charts.items() %}
            <div class="chart-container">
                <h3>{{ chart_name.replace('_', ' ').title() }} Performance Trends</h3>
                <img src="{{ chart_data }}" alt="{{ chart_name }} Trend Chart">
            </div>
            {% endfor %}
        </div>"""

    def _get_html_incident_analysis(self) -> str:
        """Generate incident analysis section"""
        return """        {% if has_incident %}
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
        {% endif %}"""

    def _get_html_metrics_table(self) -> str:
        """Generate metrics table section"""
        return """        <div class="section">
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
        </div>"""

    def _get_html_recommendations(self) -> str:
        """Generate recommendations section"""
        return """        <div class="section">
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
        </div>"""

    def _get_html_footer(self) -> str:
        """Generate report footer"""
        return """        <div class="section">
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
        </div>"""
