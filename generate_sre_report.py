#!/usr/bin/env python3
"""
Enhanced SRE Report Generator
Interactive script to generate comprehensive SLO/SLA and incident reports
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append('src')

try:
    from reports.enhanced_sre_report_system import EnhancedSREReportSystem
    from collectors.oauth_appdynamics_collector import OAuthAppDynamicsCollector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reports/sre_report.log'),
            logging.StreamHandler()
        ]
    )

def get_user_input():
    """Get user input for report generation"""
    print("🎯 Enhanced SRE Report Generator")
    print("=" * 50)

    # Application name
    app_name = input("Enter application name (default: E-Commerce Platform): ").strip()
    if not app_name:
        app_name = "E-Commerce Platform"

    # Services
    print("\nEnter services (comma-separated, or press Enter for defaults):")
    services_input = input("Services: ").strip()
    if services_input:
        services = [s.strip() for s in services_input.split(',')]
    else:
        services = [f"{app_name.lower().replace(' ', '-')}-{svc}"
                   for svc in ["web", "api", "auth", "database", "cache", "payments"]]

    # Incident analysis
    include_incident = input("\nInclude incident analysis? (y/n, default: y): ").strip().lower()
    include_incident = include_incident != 'n'

    incident_time = None
    incident_duration = 1.0

    if include_incident:
        # Incident timeframe
        hours_ago_input = input("Hours ago when incident started (default: 2): ").strip()
        try:
            hours_ago = float(hours_ago_input) if hours_ago_input else 2.0
        except ValueError:
            hours_ago = 2.0

        incident_time = datetime.now() - timedelta(hours=hours_ago)

        # Incident duration
        duration_input = input("Incident duration in hours (default: 1): ").strip()
        try:
            incident_duration = float(duration_input) if duration_input else 1.0
        except ValueError:
            incident_duration = 1.0

    return {
        'app_name': app_name,
        'services': services,
        'include_incident': include_incident,
        'incident_time': incident_time,
        'incident_duration': incident_duration
    }

def test_appdynamics_connection():
    """Test AppDynamics connection"""
    print("\n🔐 Testing AppDynamics Connection...")

    try:
        collector = OAuthAppDynamicsCollector()
        test_results = collector.test_connection()

        print(f"• OAuth Authentication: {'✅' if test_results['oauth_authentication'] else '❌'}")
        print(f"• API Access: {'✅' if test_results['applications_access'] else '❌'}")
        print(f"• Applications Found: {test_results['applications_count']}")

        if test_results['error_message']:
            print(f"• Note: {test_results['error_message']}")
            print("• Will use demo data for report generation")

        return test_results['oauth_authentication'] and test_results['applications_access']

    except Exception as e:
        print(f"• Connection failed: {e}")
        print("• Will use demo data for report generation")
        return False

def generate_reports(config):
    """Generate comprehensive reports"""
    print(f"\n📊 Generating Reports for {config['app_name']}...")
    print("-" * 60)

    # Initialize report system
    system = EnhancedSREReportSystem(app_name=config['app_name'])

    # Generate reports
    try:
        report_paths = system.generate_full_report_suite(
            application_name=config['app_name'],
            services=config['services'],
            incident_time=config['incident_time'] if config['include_incident'] else None,
            incident_duration=config['incident_duration'] if config['include_incident'] else None
        )

        print("\n✅ Report Generation Complete!")
        print("=" * 50)

        for report_type, path in report_paths.items():
            if path and os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"📄 {report_type.replace('_', ' ').title()}: {path} ({file_size:.1f} KB)")
            else:
                print(f"⚠️  {report_type.replace('_', ' ').title()}: Generation failed")

        # Show HTML report path prominently
        html_report = report_paths.get('html_report')
        if html_report and os.path.exists(html_report):
            print(f"\n🌐 Open this file in your browser for interactive report:")
            print(f"   file://{os.path.abspath(html_report)}")

        return report_paths

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def print_features():
    """Print feature summary"""
    print("\n🚀 Enhanced SRE Report Features:")
    print("=" * 50)
    features = [
        "📈 Performance trends with 30-day historical analysis",
        "🚨 AI-powered incident root cause analysis",
        "🎯 Interactive HTML reports with comprehensive charts",
        "📄 PDF export capability (when dependencies available)",
        "📊 Real-time SLO/SLA compliance monitoring",
        "🤖 LLM-powered recommendations and insights",
        "📋 JSON data export for API integration",
        "🔐 OAuth-based AppDynamics integration",
        "⚡ Error budget tracking and burn rate analysis",
        "🔍 Automated anomaly detection and alerting"
    ]

    for feature in features:
        print(f"   {feature}")

def print_next_steps():
    """Print next steps for users"""
    print("\n🎯 Next Steps:")
    print("=" * 30)
    print("1. 🔐 Configure OAuth credentials in .env file for live data")
    print("2. 🤖 Add LLM API keys for enhanced AI analysis")
    print("3. 📊 Schedule regular report generation (cron job)")
    print("4. 📧 Integrate with alerting systems (Slack, email)")
    print("5. 📈 Set up historical data collection for trending")
    print("6. 🔄 Customize SLO/SLA thresholds in config files")

def main():
    """Main execution function"""
    # Ensure reports directory exists
    Path("reports/generated").mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging()

    # Get user configuration
    config = get_user_input()

    # Test AppDynamics (optional)
    appdynamics_available = test_appdynamics_connection()

    # Generate reports
    report_paths = generate_reports(config)

    # Show features and next steps
    if report_paths:
        print_features()
        print_next_steps()

        print(f"\n🎉 SRE Report Generation Complete for {config['app_name']}!")
        print("Your comprehensive SLO/SLA and incident analysis reports are ready.")
    else:
        print("\n❌ Report generation failed. Check logs for details.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Report generation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()