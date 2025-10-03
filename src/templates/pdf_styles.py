"""
PDF Styling Module
Centralized CSS styles for PDF report generation
"""

from typing import Dict

class PDFStyles:
    """Centralized PDF styling configuration"""

    # Page configuration
    PAGE_SIZE = "A4 portrait"
    PAGE_MARGIN = "1.5cm 2cm"

    # Typography
    FONT_FAMILY = "Arial, Helvetica, sans-serif"
    BASE_FONT_SIZE = "9pt"

    # Colors
    COLORS = {
        'primary': '#667eea',
        'secondary': '#1e40af',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#3b82f6',
        'text_dark': '#1f2937',
        'text_muted': '#64748b',
        'border': '#e2e8f0',
        'background': '#f8fafc',
    }

    @staticmethod
    def get_base_css() -> str:
        """Get base PDF CSS styles"""
        return """
        @page {
            size: A4 portrait;
            margin: 1.5cm 2cm;
            @top-center {
                content: "SRE Performance Report";
                font-family: Arial, sans-serif;
                font-size: 9pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-family: Arial, sans-serif;
                font-size: 9pt;
                color: #666;
            }
        }

        /* Override web fonts with system fonts */
        * {
            font-family: Arial, Helvetica, sans-serif !important;
        }

        body {
            font-family: Arial, Helvetica, sans-serif !important;
            font-size: 9pt;
            line-height: 1.5;
            color: #1f2937 !important;
            background: white !important;
            margin: 0;
            padding: 0;
        }
        """

    @staticmethod
    def get_theme_overrides() -> str:
        """Get dark theme to light theme overrides for PDF"""
        return """
        /* Override dark theme for PDF */
        .glass-card, .metric-card {
            background: white !important;
            border: 1pt solid #e2e8f0 !important;
            box-shadow: none !important;
            transform: none !important;
        }

        .container {
            max-width: 100% !important;
            background: white;
            padding: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            margin: 0 !important;
        }
        """

    @staticmethod
    def get_tailwind_overrides() -> str:
        """Get Tailwind CSS utility overrides for PDF"""
        return """
        /* Tailwind utility overrides */
        .text-slate-300, .text-slate-400, .text-slate-500 {
            color: #64748b !important;
        }

        .text-white {
            color: #1f2937 !important;
        }

        .text-5xl {
            font-size: 24pt !important;
        }

        .text-3xl {
            font-size: 18pt !important;
        }

        .text-2xl {
            font-size: 14pt !important;
        }

        .text-xl {
            font-size: 12pt !important;
        }

        .text-lg {
            font-size: 10pt !important;
        }

        .text-sm {
            font-size: 8pt !important;
        }

        .text-xs {
            font-size: 7pt !important;
        }

        /* Grid layouts */
        .grid {
            display: table !important;
            width: 100%;
        }

        .grid-cols-1, .grid-cols-2, .grid-cols-3, .grid-cols-4, .grid-cols-5 {
            display: table-row !important;
        }

        .grid > div {
            display: table-cell !important;
            padding: 8pt;
            vertical-align: top;
            border: 1pt solid #f1f5f9;
        }

        /* Flex to block */
        .flex {
            display: block !important;
        }

        .flex-wrap {
            display: block !important;
        }

        /* Hide icons in PDF */
        .fas, .far, .fab, i[class*="fa-"] {
            display: none !important;
        }

        /* SVG progress rings - hide */
        svg, .progress-ring {
            display: none !important;
        }

        /* Border width utilities */
        .border-l-4 {
            border-left-width: 4pt !important;
        }

        /* Space-y utility */
        .space-y-4 > * {
            margin-bottom: 10pt;
        }

        .space-y-4 > *:last-child {
            margin-bottom: 0;
        }
        """

    @staticmethod
    def get_component_styles() -> str:
        """Get component-specific styles"""
        return """
        /* Header styling */
        .header {
            background: #667eea !important;
            color: white !important;
            padding: 15pt !important;
            margin-bottom: 20pt;
            text-align: center;
            border-radius: 4pt;
            page-break-after: avoid;
        }

        .header h1 {
            font-size: 20pt !important;
            margin-bottom: 8pt !important;
            font-weight: bold;
            page-break-after: avoid;
        }

        .header .subtitle {
            font-size: 12pt !important;
        }

        .header .meta {
            display: block !important;
            margin-top: 12pt;
        }

        .header .meta-item {
            display: inline-block;
            margin: 0 15pt;
            text-align: center;
        }

        .header .meta-label {
            font-size: 9pt;
            display: block;
        }

        .header .meta-value {
            font-weight: bold;
            display: block;
        }

        /* Section titles */
        .section {
            margin-bottom: 20pt;
            page-break-inside: avoid;
        }

        .section-title {
            font-size: 14pt !important;
            font-weight: bold !important;
            color: #1e293b;
            margin-bottom: 12pt !important;
            padding-bottom: 4pt !important;
            border-bottom: 2pt solid #667eea !important;
            page-break-after: avoid;
        }

        .section h2 {
            font-size: 14pt;
            color: #1e293b;
            border-bottom: 2pt solid #667eea;
            padding-bottom: 4pt;
            margin-top: 15pt;
            margin-bottom: 10pt;
            page-break-after: avoid;
        }

        .section h3 {
            font-size: 12pt;
            margin-top: 12pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
        }

        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 8pt;
            height: 8pt;
            border-radius: 50%;
            margin-right: 4pt;
        }

        .bg-green-500 { background: #10b981 !important; }
        .bg-yellow-500 { background: #f59e0b !important; }
        .bg-red-500 { background: #ef4444 !important; }
        .bg-blue-500 { background: #3b82f6 !important; }

        .text-green-400, .text-green-500 { color: #10b981 !important; }
        .text-yellow-400, .text-yellow-500 { color: #f59e0b !important; }
        .text-red-400, .text-red-500 { color: #ef4444 !important; }
        .text-blue-400, .text-blue-500 { color: #3b82f6 !important; }

        /* Gradient text to normal */
        .gradient-text {
            -webkit-text-fill-color: initial !important;
            background: none !important;
            color: #667eea !important;
        }
        """

    @staticmethod
    def get_card_styles() -> str:
        """Get card and grid styles"""
        return """
        /* Summary cards */
        .summary-grid, .stats-grid, .metrics-grid {
            display: table !important;
            width: 100%;
            margin: 12pt 0;
            page-break-inside: avoid;
        }

        .summary-card, .stat-card, .metric-card, .kpi-card {
            display: table-cell !important;
            width: 33.33%;
            background: #f8fafc;
            padding: 10pt;
            border: 1pt solid #e2e8f0;
            text-align: center;
            vertical-align: top;
            page-break-inside: avoid;
        }

        .summary-card .label, .stat-card .label, .metric-card .label, .kpi-card .label {
            font-size: 9pt;
            color: #64748b;
            display: block;
            margin-bottom: 4pt;
        }

        .summary-card .value, .stat-card .value, .metric-card .value, .kpi-card .value {
            font-size: 16pt;
            font-weight: bold;
            color: #667eea;
            display: block;
        }
        """

    @staticmethod
    def get_table_styles() -> str:
        """Get table styles"""
        return """
        /* Tables */
        .metrics-table, table {
            width: 100%;
            border-collapse: collapse;
            margin: 12pt 0;
            font-size: 8pt;
            page-break-inside: auto;
        }

        .metrics-table th,
        .metrics-table td,
        table th,
        table td {
            padding: 6pt;
            border: 1pt solid #e2e8f0;
            text-align: left;
            vertical-align: top;
        }

        .metrics-table th, table th {
            background-color: #f1f5f9;
            font-weight: bold;
            color: #1e293b;
        }

        .metrics-table tbody tr:nth-child(even),
        table tbody tr:nth-child(even) {
            background-color: #f8fafc;
        }

        /* Charts */
        .chart-container {
            margin: 12pt 0;
            text-align: center;
            page-break-inside: avoid;
        }

        .chart-container img {
            max-width: 100%;
            max-height: 250pt;
            height: auto;
        }
        """

    @staticmethod
    def get_incident_styles() -> str:
        """Get incident and analysis styles"""
        return """
        /* Incident sections */
        .incident-card, .incident-section {
            margin: 15pt 0;
            padding: 12pt;
            background: #fef3c7;
            border-left: 4pt solid #f59e0b;
            page-break-inside: avoid;
        }

        .incident-critical {
            background: #fee2e2 !important;
            border-left-color: #ef4444 !important;
        }

        /* LLM Analysis - Enhanced with border outline */
        .llm-analysis, .analysis-card, .ai-insight {
            background: #eff6ff !important;
            padding: 12pt !important;
            border: 2pt solid #3b82f6 !important;
            border-left: 4pt solid #1e40af !important;
            border-radius: 4pt;
            margin: 12pt 0 !important;
            font-size: 8pt;
            page-break-inside: avoid;
            box-shadow: 0 0 0 1pt #93c5fd;
        }

        .llm-analysis h3, .analysis-card h3, .ai-insight h3 {
            color: #1e40af !important;
            font-size: 10pt !important;
            font-weight: bold;
            margin-bottom: 8pt;
            padding-bottom: 4pt;
            border-bottom: 1pt solid #bfdbfe;
        }

        .llm-analysis p, .analysis-card p, .ai-insight p {
            margin-bottom: 6pt;
            line-height: 1.6;
        }
        """

    @staticmethod
    def get_recommendation_styles() -> str:
        """Get recommendation styles"""
        return """
        /* Recommendations - Enhanced with border outline */
        .recommendation-item, .recommendation-card {
            background: #f0fdf4 !important;
            border: 2pt solid #10b981 !important;
            border-left: 4pt solid #059669 !important;
            border-radius: 4pt;
            padding: 12pt !important;
            margin: 10pt 0 !important;
            page-break-inside: avoid;
            box-shadow: 0 0 0 1pt #86efac;
        }

        .recommendation-item h3, .recommendation-card h3,
        .recommendation-item h4, .recommendation-card h4 {
            color: #065f46 !important;
            font-size: 10pt !important;
            font-weight: bold;
            margin-bottom: 6pt;
        }

        .recommendation-item strong, .recommendation-card strong {
            color: #047857 !important;
        }

        /* Key Recommendations Section - Tailwind pattern */
        [class*="bg-red-900"],
        [class*="bg-blue-900"],
        [class*="bg-green-900"],
        [class*="bg-yellow-900"] {
            border: 2pt solid !important;
            border-radius: 4pt !important;
            padding: 12pt !important;
            margin: 10pt 0 !important;
            page-break-inside: avoid;
        }

        /* Critical recommendations - Red */
        [class*="bg-red-900"] {
            background: #fee2e2 !important;
            border-color: #dc2626 !important;
            border-left: 4pt solid #991b1b !important;
        }

        [class*="border-red-500"] {
            border-left-color: #dc2626 !important;
        }

        [class*="text-red-200"],
        [class*="text-red-300"] {
            color: #7f1d1d !important;
        }

        /* Medium priority - Blue */
        [class*="bg-blue-900"] {
            background: #dbeafe !important;
            border-color: #2563eb !important;
            border-left: 4pt solid #1e40af !important;
        }

        [class*="border-blue-500"] {
            border-left-color: #2563eb !important;
        }

        [class*="text-blue-200"],
        [class*="text-blue-300"] {
            color: #1e3a8a !important;
        }

        /* Low priority - Green */
        [class*="bg-green-900"] {
            background: #d1fae5 !important;
            border-color: #059669 !important;
            border-left: 4pt solid #047857 !important;
        }

        [class*="border-green-500"] {
            border-left-color: #059669 !important;
        }

        [class*="text-green-200"],
        [class*="text-green-300"] {
            color: #065f46 !important;
        }

        /* Warning/Info - Yellow */
        [class*="bg-yellow-900"] {
            background: #fef3c7 !important;
            border-color: #d97706 !important;
            border-left: 4pt solid #b45309 !important;
        }

        [class*="border-yellow-500"] {
            border-left-color: #d97706 !important;
        }

        [class*="text-yellow-200"],
        [class*="text-yellow-300"] {
            color: #78350f !important;
        }
        """

    @staticmethod
    def get_status_styles() -> str:
        """Get status indicator styles"""
        return """
        /* Status indicators */
        .status-compliant, .status-success { color: #10b981 !important; font-weight: bold; }
        .status-at-risk, .status-warning { color: #f59e0b !important; font-weight: bold; }
        .status-breached, .status-danger { color: #ef4444 !important; font-weight: bold; }
        .status-critical { color: #dc2626 !important; font-weight: bold; }

        /* Badges */
        .badge {
            padding: 2pt 6pt;
            border-radius: 3pt;
            font-size: 8pt;
            font-weight: bold;
        }

        .badge-success { background: #d1fae5; color: #065f46; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-danger { background: #fee2e2; color: #991b1b; }

        /* Progress bars */
        .progress {
            background: #e5e7eb;
            height: 8pt;
            border-radius: 4pt;
            overflow: hidden;
        }

        .progress-bar {
            background: #667eea;
            height: 100%;
        }
        """

    @staticmethod
    def get_utility_styles() -> str:
        """Get utility styles"""
        return """
        /* Hide interactive elements */
        .interactive-element, .no-print, button, .button {
            display: none !important;
        }

        /* Page break rules */
        .page-break {
            page-break-before: always;
        }

        .avoid-break, .no-break {
            page-break-inside: avoid;
        }

        /* Links */
        a {
            color: #1e293b !important;
            text-decoration: none;
        }

        /* Ensure images render properly */
        img {
            max-width: 100%;
            height: auto;
            page-break-inside: avoid;
        }
        """

    @classmethod
    def get_complete_pdf_css(cls) -> str:
        """Get complete CSS for PDF generation"""
        return (
            cls.get_base_css() +
            cls.get_theme_overrides() +
            cls.get_tailwind_overrides() +
            cls.get_component_styles() +
            cls.get_card_styles() +
            cls.get_table_styles() +
            cls.get_incident_styles() +
            cls.get_recommendation_styles() +
            cls.get_status_styles() +
            cls.get_utility_styles()
        )
