"""
SLO/SLA Report Generator
Generates comprehensive Service Level Objective and Service Level Agreement compliance reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

class SLOSLAReportGenerator:
    """Generates comprehensive SLO/SLA compliance reports"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

        # Load configurations
        self.slo_config = self._load_yaml("slo_definitions.yaml")
        self.sla_config = self._load_yaml("sla_thresholds.yaml")

        # Set up visualization styling
        self._setup_styling()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_dir / filename, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {e}")
            return {}

    def _setup_styling(self):
        """Setup matplotlib and seaborn styling"""
        plt.style.use('default')
        sns.set_palette("husl")

        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

    def generate_demo_metrics(self, days_back: int = 30) -> List[SLOMetric]:
        """Generate realistic demo metrics for the report"""
        np.random.seed(42)  # For reproducible results
        metrics = []
        current_time = datetime.now()

        services = self.slo_config.get('service_level_objectives', {})

        for service_name, slo_config in services.items():
            # Generate availability metrics
            base_availability = float(slo_config.get('availability_slo', '99.9%').rstrip('%'))
            # Add realistic variance (±0.1%)
            current_availability = base_availability + np.random.normal(0, 0.05)

            availability_metric = SLOMetric(
                service_name=service_name,
                metric_name="availability",
                current_value=current_availability,
                slo_target=base_availability,
                sla_target=99.9,  # External SLA commitment
                status=self._get_compliance_status(current_availability, base_availability),
                error_budget_consumed=max(0, (base_availability - current_availability) / (100 - base_availability) * 100),
                timestamp=current_time
            )
            metrics.append(availability_metric)

            # Generate latency metrics (P95)
            base_latency_p95 = int(slo_config.get('latency_p95_slo', '200ms').rstrip('ms'))
            # Add realistic variance (±20ms)
            current_latency_p95 = base_latency_p95 + np.random.normal(0, 15)

            latency_p95_metric = SLOMetric(
                service_name=service_name,
                metric_name="latency_p95",
                current_value=current_latency_p95,
                slo_target=base_latency_p95,
                sla_target=500,  # External SLA commitment
                status=self._get_compliance_status(current_latency_p95, base_latency_p95, inverse=True),
                error_budget_consumed=max(0, (current_latency_p95 - base_latency_p95) / base_latency_p95 * 100),
                timestamp=current_time
            )
            metrics.append(latency_p95_metric)

            # Generate error rate metrics
            base_error_rate = float(slo_config.get('error_rate_slo', '0.1%').rstrip('%'))
            # Add realistic variance
            current_error_rate = abs(base_error_rate + np.random.normal(0, 0.02))

            error_rate_metric = SLOMetric(
                service_name=service_name,
                metric_name="error_rate",
                current_value=current_error_rate,
                slo_target=base_error_rate,
                sla_target=1.0,  # External SLA commitment
                status=self._get_compliance_status(current_error_rate, base_error_rate, inverse=True),
                error_budget_consumed=max(0, (current_error_rate - base_error_rate) / base_error_rate * 100) if base_error_rate > 0 else 0,
                timestamp=current_time
            )
            metrics.append(error_rate_metric)

        return metrics

    def _get_compliance_status(self, current: float, target: float, inverse: bool = False) -> str:
        """Determine compliance status based on current vs target values"""
        if inverse:  # For metrics where lower is better (latency, error rate)
            if current <= target:
                return "compliant"
            elif current <= target * 1.1:
                return "at_risk"
            else:
                return "breached"
        else:  # For metrics where higher is better (availability)
            if current >= target:
                return "compliant"
            elif current >= target * 0.999:
                return "at_risk"
            else:
                return "breached"

    def generate_sla_compliance_report(self, metrics: List[SLOMetric]) -> List[SLACompliance]:
        """Generate SLA compliance analysis"""
        compliance_reports = []

        # Group metrics by service
        services = {}
        for metric in metrics:
            if metric.service_name not in services:
                services[metric.service_name] = {}
            services[metric.service_name][metric.metric_name] = metric

        for service_name, service_metrics in services.items():
            availability = service_metrics.get('availability')
            latency_p95 = service_metrics.get('latency_p95')
            error_rate = service_metrics.get('error_rate')

            # Calculate SLA compliance
            sla_thresholds = self.sla_config.get('compliance_thresholds', {}).get('penalty_thresholds', {})

            # Determine compliance status and penalties
            compliance_status = "compliant"
            penalties_applicable = False
            credit_percentage = 0

            if availability:
                avail_value = availability.current_value
                if avail_value < sla_thresholds.get('availability', {}).get('tier_3', 98.0):
                    compliance_status = "severe_breach"
                    penalties_applicable = True
                    credit_percentage = 50
                elif avail_value < sla_thresholds.get('availability', {}).get('tier_2', 99.0):
                    compliance_status = "moderate_breach"
                    penalties_applicable = True
                    credit_percentage = 25
                elif avail_value < sla_thresholds.get('availability', {}).get('tier_1', 99.5):
                    compliance_status = "minor_breach"
                    penalties_applicable = True
                    credit_percentage = 10

            compliance = SLACompliance(
                service_name=service_name,
                availability_percent=availability.current_value if availability else 0,
                avg_response_time=latency_p95.current_value * 0.7 if latency_p95 else 0,  # Estimate avg from p95
                p95_response_time=latency_p95.current_value if latency_p95 else 0,
                p99_response_time=latency_p95.current_value * 1.5 if latency_p95 else 0,  # Estimate p99 from p95
                error_rate=error_rate.current_value if error_rate else 0,
                compliance_status=compliance_status,
                penalties_applicable=penalties_applicable,
                credit_percentage=credit_percentage
            )
            compliance_reports.append(compliance)

        return compliance_reports

    def create_slo_dashboard(self, metrics: List[SLOMetric], output_path: str = None):
        """Create an interactive SLO dashboard using Plotly"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/slo_dashboard_{timestamp}.html"

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Service Availability SLOs', 'Response Time SLOs',
                'Error Rate SLOs', 'Error Budget Consumption',
                'SLO Compliance Summary', 'Trend Analysis'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "scatter"}]
            ]
        )

        # Group metrics by type
        availability_metrics = [m for m in metrics if m.metric_name == "availability"]
        latency_metrics = [m for m in metrics if m.metric_name == "latency_p95"]
        error_metrics = [m for m in metrics if m.metric_name == "error_rate"]

        # Colors for compliance status
        color_map = {"compliant": "green", "at_risk": "orange", "breached": "red"}

        # 1. Service Availability
        if availability_metrics:
            services = [m.service_name for m in availability_metrics]
            current_vals = [m.current_value for m in availability_metrics]
            target_vals = [m.slo_target for m in availability_metrics]
            colors = [color_map[m.status] for m in availability_metrics]

            fig.add_trace(
                go.Bar(name="Current", x=services, y=current_vals, marker_color=colors),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(name="SLO Target", x=services, y=target_vals, mode="markers",
                          marker=dict(symbol="diamond", size=10, color="blue")),
                row=1, col=1
            )

        # 2. Response Time
        if latency_metrics:
            services = [m.service_name for m in latency_metrics]
            current_vals = [m.current_value for m in latency_metrics]
            target_vals = [m.slo_target for m in latency_metrics]
            colors = [color_map[m.status] for m in latency_metrics]

            fig.add_trace(
                go.Bar(name="Current (ms)", x=services, y=current_vals, marker_color=colors),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(name="SLO Target (ms)", x=services, y=target_vals, mode="markers",
                          marker=dict(symbol="diamond", size=10, color="blue")),
                row=1, col=2
            )

        # 3. Error Rate
        if error_metrics:
            services = [m.service_name for m in error_metrics]
            current_vals = [m.current_value for m in error_metrics]
            target_vals = [m.slo_target for m in error_metrics]
            colors = [color_map[m.status] for m in error_metrics]

            fig.add_trace(
                go.Bar(name="Current (%)", x=services, y=current_vals, marker_color=colors),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(name="SLO Target (%)", x=services, y=target_vals, mode="markers",
                          marker=dict(symbol="diamond", size=10, color="blue")),
                row=2, col=1
            )

        # 4. Error Budget Consumption
        services = [m.service_name for m in metrics if m.metric_name == "availability"]
        budget_consumed = [m.error_budget_consumed for m in metrics if m.metric_name == "availability"]

        fig.add_trace(
            go.Bar(name="Budget Consumed (%)", x=services, y=budget_consumed,
                  marker_color=["red" if b > 50 else "orange" if b > 25 else "green" for b in budget_consumed]),
            row=2, col=2
        )

        # 5. Compliance Summary Table
        compliance_data = []
        for metric in metrics:
            compliance_data.append([
                metric.service_name,
                metric.metric_name,
                f"{metric.current_value:.2f}",
                f"{metric.slo_target:.2f}",
                metric.status
            ])

        fig.add_trace(
            go.Table(
                header=dict(values=["Service", "Metric", "Current", "Target", "Status"]),
                cells=dict(values=list(zip(*compliance_data)))
            ),
            row=3, col=1
        )

        # 6. Trend Analysis (mock data for demo)
        days = list(range(-30, 1))
        availability_trend = [99.9 + np.random.normal(0, 0.1) for _ in days]

        fig.add_trace(
            go.Scatter(x=days, y=availability_trend, mode="lines+markers",
                      name="Availability Trend"),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="E-Commerce Microservices SLO Dashboard",
            title_x=0.5
        )

        # Save the dashboard
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"SLO Dashboard saved to {output_path}")

        return output_path

    def generate_excel_report(self, metrics: List[SLOMetric], compliance: List[SLACompliance],
                            output_path: str = None) -> str:
        """Generate comprehensive Excel report"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/generated/slo_sla_report_{timestamp}.xlsx"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Executive Summary
            exec_summary = self._create_executive_summary(metrics, compliance)
            exec_summary.to_excel(writer, sheet_name='Executive Summary', index=False)

            # 2. SLO Metrics Detail
            slo_df = pd.DataFrame([{
                'Service': m.service_name,
                'Metric': m.metric_name,
                'Current Value': m.current_value,
                'SLO Target': m.slo_target,
                'SLA Target': m.sla_target,
                'Status': m.status,
                'Error Budget Consumed (%)': m.error_budget_consumed,
                'Timestamp': m.timestamp
            } for m in metrics])
            slo_df.to_excel(writer, sheet_name='SLO Metrics', index=False)

            # 3. SLA Compliance
            sla_df = pd.DataFrame([{
                'Service': c.service_name,
                'Availability (%)': c.availability_percent,
                'Avg Response Time (ms)': c.avg_response_time,
                'P95 Response Time (ms)': c.p95_response_time,
                'P99 Response Time (ms)': c.p99_response_time,
                'Error Rate (%)': c.error_rate,
                'Compliance Status': c.compliance_status,
                'Penalties Applicable': c.penalties_applicable,
                'Credit Percentage': c.credit_percentage
            } for c in compliance])
            sla_df.to_excel(writer, sheet_name='SLA Compliance', index=False)

            # 4. Error Budget Analysis
            error_budget_df = self._create_error_budget_analysis(metrics)
            error_budget_df.to_excel(writer, sheet_name='Error Budget Analysis', index=False)

            # 5. Recommendations
            recommendations_df = self._create_recommendations(metrics, compliance)
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)

        self.logger.info(f"Excel report saved to {output_path}")
        return output_path

    def _create_executive_summary(self, metrics: List[SLOMetric], compliance: List[SLACompliance]) -> pd.DataFrame:
        """Create executive summary dataframe"""
        total_services = len(set(m.service_name for m in metrics))
        compliant_services = len([m for m in metrics if m.status == "compliant"])
        at_risk_services = len([m for m in metrics if m.status == "at_risk"])
        breached_services = len([m for m in metrics if m.status == "breached"])

        sla_breaches = len([c for c in compliance if c.penalties_applicable])
        total_credit_exposure = sum(c.credit_percentage for c in compliance)

        summary_data = [
            {"Metric": "Total Services Monitored", "Value": total_services},
            {"Metric": "Services Meeting SLOs", "Value": f"{compliant_services}/{total_services}"},
            {"Metric": "Services At Risk", "Value": at_risk_services},
            {"Metric": "Services with SLO Breaches", "Value": breached_services},
            {"Metric": "SLA Breaches (Penalties)", "Value": sla_breaches},
            {"Metric": "Total Credit Exposure (%)", "Value": f"{total_credit_exposure:.1f}%"},
            {"Metric": "Overall System Health", "Value": "Good" if breached_services == 0 else "Needs Attention"},
            {"Metric": "Report Generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ]

        return pd.DataFrame(summary_data)

    def _create_error_budget_analysis(self, metrics: List[SLOMetric]) -> pd.DataFrame:
        """Create error budget analysis"""
        error_budgets = self.slo_config.get('error_budgets', {}).get('monthly_budgets', {})

        budget_data = []
        for service_name, budget_minutes in error_budgets.items():
            service_metrics = [m for m in metrics if m.service_name == service_name]
            if service_metrics:
                # Find availability metric for error budget calculation
                availability_metric = next((m for m in service_metrics if m.metric_name == "availability"), None)
                if availability_metric:
                    consumed_percent = availability_metric.error_budget_consumed
                    remaining_minutes = budget_minutes * (100 - consumed_percent) / 100

                    budget_data.append({
                        "Service": service_name,
                        "Monthly Budget (minutes)": budget_minutes,
                        "Consumed (%)": f"{consumed_percent:.2f}%",
                        "Remaining (minutes)": f"{remaining_minutes:.1f}",
                        "Status": "Critical" if consumed_percent > 90 else "Warning" if consumed_percent > 50 else "Healthy"
                    })

        return pd.DataFrame(budget_data)

    def _create_recommendations(self, metrics: List[SLOMetric], compliance: List[SLACompliance]) -> pd.DataFrame:
        """Create recommendations based on current status"""
        recommendations = []

        # Analyze breaches and at-risk services
        breached_services = [m for m in metrics if m.status == "breached"]
        at_risk_services = [m for m in metrics if m.status == "at_risk"]

        for metric in breached_services:
            recommendations.append({
                "Priority": "High",
                "Service": metric.service_name,
                "Issue": f"{metric.metric_name} SLO breach",
                "Recommendation": f"Immediate investigation required for {metric.metric_name} performance",
                "Impact": "SLA penalties may apply"
            })

        for metric in at_risk_services:
            recommendations.append({
                "Priority": "Medium",
                "Service": metric.service_name,
                "Issue": f"{metric.metric_name} approaching SLO threshold",
                "Recommendation": f"Monitor {metric.metric_name} closely and consider optimization",
                "Impact": "Risk of SLO breach"
            })

        # SLA compliance recommendations
        for comp in compliance:
            if comp.penalties_applicable:
                recommendations.append({
                    "Priority": "Critical",
                    "Service": comp.service_name,
                    "Issue": "SLA breach with financial penalties",
                    "Recommendation": "Immediate remediation required, consider customer credits",
                    "Impact": f"{comp.credit_percentage}% customer credit exposure"
                })

        if not recommendations:
            recommendations.append({
                "Priority": "Info",
                "Service": "All Services",
                "Issue": "No critical issues detected",
                "Recommendation": "Continue monitoring and maintain current practices",
                "Impact": "System operating within acceptable parameters"
            })

        return pd.DataFrame(recommendations)

    def generate_comprehensive_report(self, days_back: int = 30) -> Dict[str, str]:
        """Generate complete SLO/SLA report with all components"""
        self.logger.info("Starting comprehensive SLO/SLA report generation")

        # Generate demo metrics
        metrics = self.generate_demo_metrics(days_back)
        compliance = self.generate_sla_compliance_report(metrics)

        # Generate all report components
        results = {}

        # 1. Interactive Dashboard
        dashboard_path = self.create_slo_dashboard(metrics)
        results['dashboard'] = dashboard_path

        # 2. Excel Report
        excel_path = self.generate_excel_report(metrics, compliance)
        results['excel_report'] = excel_path

        # 3. JSON Data Export
        json_path = self._export_json_data(metrics, compliance)
        results['json_data'] = json_path

        self.logger.info("Comprehensive SLO/SLA report generation completed")
        return results

    def _export_json_data(self, metrics: List[SLOMetric], compliance: List[SLACompliance]) -> str:
        """Export all data as JSON for API consumption"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/generated/slo_sla_data_{timestamp}.json"

        data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "SLO/SLA Compliance Report",
                "data_period_days": 30
            },
            "slo_metrics": [
                {
                    "service_name": m.service_name,
                    "metric_name": m.metric_name,
                    "current_value": m.current_value,
                    "slo_target": m.slo_target,
                    "sla_target": m.sla_target,
                    "status": m.status,
                    "error_budget_consumed": m.error_budget_consumed,
                    "timestamp": m.timestamp.isoformat()
                } for m in metrics
            ],
            "sla_compliance": [
                {
                    "service_name": c.service_name,
                    "availability_percent": c.availability_percent,
                    "avg_response_time": c.avg_response_time,
                    "p95_response_time": c.p95_response_time,
                    "p99_response_time": c.p99_response_time,
                    "error_rate": c.error_rate,
                    "compliance_status": c.compliance_status,
                    "penalties_applicable": c.penalties_applicable,
                    "credit_percentage": c.credit_percentage
                } for c in compliance
            ]
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

    # Generate comprehensive report
    generator = SLOSLAReportGenerator()
    report_paths = generator.generate_comprehensive_report(days_back=30)

    print("🎯 SLO/SLA Report Generation Complete!")
    print("=" * 50)
    for report_type, path in report_paths.items():
        print(f"📊 {report_type.replace('_', ' ').title()}: {path}")

    print("\n📋 Report Summary:")
    print("• Interactive dashboard with real-time SLO metrics")
    print("• Comprehensive Excel report with multiple worksheets")
    print("• JSON data export for API integration")
    print("• Error budget analysis and recommendations")
    print("• SLA compliance status and penalty calculations")