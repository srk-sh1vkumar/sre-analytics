"""
Chart Generator Module

Handles creation of trend visualizations and charts for SLO metrics.
"""

import logging
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

from src.config.constants import (
    CHART_WIDTH,
    CHART_HEIGHT,
    CHART_DPI,
    STATUS_COMPLIANT,
    STATUS_AT_RISK,
    STATUS_BREACHED
)
from src.reports.llm_analyzer import SLOMetric


class ChartGenerator:
    """Generates visualizations and charts for SLO metrics"""

    def __init__(self):
        """Initialize chart generator"""
        self.logger = logging.getLogger(__name__)

    def create_trend_visualizations(self, metrics: List[SLOMetric], save_images: bool = False) -> Dict[str, str]:
        """
        Create comprehensive trend visualizations for metrics

        Args:
            metrics: List of SLO metrics with trend data
            save_images: If True, save as files; if False, return base64

        Returns:
            Dict[str, str]: Map of metric names to chart data (file paths or base64)
        """
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
                fig = self._create_metric_chart(metric_name, metric_list)

                if save_images:
                    # Save as temporary image file for PDF embedding
                    chart_path = self._save_chart_to_file(fig, metric_name)
                    charts[metric_name] = chart_path
                    self.logger.info(f"Chart saved to: {chart_path}")
                else:
                    # Return base64 for HTML
                    charts[metric_name] = self._fig_to_base64(fig)

                plt.close(fig)

        return charts

    def _create_metric_chart(self, metric_name: str, metric_list: List[SLOMetric]):
        """
        Create a dual-panel chart for a metric type

        Args:
            metric_name: Name of the metric
            metric_list: List of metrics of this type

        Returns:
            matplotlib.figure.Figure: Created chart
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Trend lines
        days = list(range(-len(metric_list[0].trend_data), 0))

        for metric in metric_list:
            color = self._get_status_color(metric.status)
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
        colors = [self._get_status_color(m.status) for m in metric_list]

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

        return fig

    def _get_status_color(self, status: str) -> str:
        """
        Get color for status

        Args:
            status: Compliance status

        Returns:
            str: Color name
        """
        if status == STATUS_COMPLIANT:
            return 'green'
        elif status == STATUS_AT_RISK:
            return 'orange'
        elif status == STATUS_BREACHED:
            return 'red'
        else:
            return 'gray'

    def _save_chart_to_file(self, fig, metric_name: str) -> str:
        """
        Save chart to temporary file

        Args:
            fig: Matplotlib figure
            metric_name: Name of metric for filename

        Returns:
            str: Path to saved file
        """
        temp_dir = Path("reports/generated/temp_charts")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"chart_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(str(temp_file), dpi=CHART_DPI, bbox_inches='tight', facecolor='white')
        return str(temp_file)

    def _fig_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 string for HTML embedding

        Args:
            fig: Matplotlib figure

        Returns:
            str: Base64-encoded image data URI
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=CHART_DPI, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return f"data:image/png;base64,{image_base64}"

    def create_summary_chart(self, summary: Dict[str, any]) -> str:
        """
        Create a summary dashboard chart

        Args:
            summary: Metrics summary dictionary

        Returns:
            str: Base64-encoded chart
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Chart 1: Compliance overview pie chart
        labels = ['Compliant', 'At Risk', 'Breached']
        sizes = [summary['compliant_count'], summary['at_risk_count'], summary['breached_count']]
        colors = ['green', 'orange', 'red']
        explode = (0.1, 0, 0) if summary['compliant_count'] > 0 else (0, 0, 0.1)

        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title('SLO Compliance Status')

        # Chart 2: Compliance percentage bar
        ax2.barh(['Compliance Rate'], [summary['compliance_percentage']], color='blue', alpha=0.7)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Percentage (%)')
        ax2.set_title('Overall Compliance Rate')
        ax2.grid(True, alpha=0.3, axis='x')

        # Chart 3: Error budget consumption
        ax3.barh(['Error Budget Used'], [summary['avg_error_budget_consumed']],
                color='orange' if summary['avg_error_budget_consumed'] > 50 else 'green', alpha=0.7)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel('Percentage (%)')
        ax3.set_title('Average Error Budget Consumed')
        ax3.grid(True, alpha=0.3, axis='x')

        # Chart 4: Health status indicator
        health_colors = {'Healthy': 'green', 'Degraded': 'orange', 'Unhealthy': 'red'}
        health_color = health_colors.get(summary['health_status'], 'gray')
        ax4.text(0.5, 0.5, summary['health_status'], ha='center', va='center',
                fontsize=36, fontweight='bold', color=health_color)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('System Health Status', fontsize=14)

        plt.tight_layout()

        chart_base64 = self._fig_to_base64(fig)
        plt.close(fig)

        return chart_base64
