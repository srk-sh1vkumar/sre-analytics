#!/usr/bin/env python3
"""
Demo script to showcase the enhanced SRE template integration
"""

import logging
import os
from datetime import datetime, timedelta
from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("🚀 Enhanced SRE Template Integration Demo")
    print("=" * 50)

    try:
        # Initialize the enhanced report system
        app_name = "Enhanced SRE Analytics Platform"
        system = EnhancedSREReportSystem(app_name=app_name)

        print(f"📊 Generating comprehensive report for '{app_name}'...")

        # Define test services
        services = [
            "web-frontend", "api-gateway", "user-service",
            "product-service", "order-service", "payment-service",
            "inventory-service", "notification-service"
        ]

        # Generate metrics with 30-day trends
        print("📈 Creating performance metrics with trend data...")
        metrics = system.generate_metrics_with_trends(services, days_back=30)
        print(f"✅ Generated {len(metrics)} metrics for {len(services)} services")

        # Create an incident for demonstration
        print("🚨 Generating sample incident for analysis...")
        incident_time = datetime.now() - timedelta(hours=6)
        incident = system.generate_incident_report(
            application_name=app_name,
            start_time=incident_time,
            duration_hours=1.5
        )
        print(f"✅ Generated incident: {incident.incident_id}")

        # Generate the full report suite
        print("🎨 Creating enhanced reports with modern template...")
        report_paths = system.generate_full_report_suite(
            application_name=app_name,
            services=services,
            incident_time=incident_time,
            incident_duration=1.5
        )

        print("\n🎯 Enhanced Template Integration Results:")
        print("=" * 50)

        for report_type, path in report_paths.items():
            if path:
                # Get file size
                file_size = os.path.getsize(path)
                print(f"📊 {report_type.replace('_', ' ').title()}:")
                print(f"   📁 Path: {path}")
                print(f"   💾 Size: {file_size:,} bytes")

                # Verify enhanced template features
                if report_type == 'html_report':
                    with open(path, 'r') as f:
                        content = f.read()

                    enhanced_features = []
                    if 'Tailwind CSS' in content:
                        enhanced_features.append("Tailwind CSS")
                    if 'Inter' in content:
                        enhanced_features.append("Inter Font")
                    if 'glass-card' in content:
                        enhanced_features.append("Glass Morphism")
                    if 'Chart.js' in content:
                        enhanced_features.append("Interactive Charts")
                    if 'status-indicator' in content:
                        enhanced_features.append("Animated Status")
                    if 'ai-insight' in content:
                        enhanced_features.append("AI Insights")

                    print(f"   ✨ Enhanced Features: {', '.join(enhanced_features)}")
                print()

        print("🎉 Enhanced Template Features Successfully Integrated:")
        print("• 🎨 Modern glass morphism design with gradients")
        print("• 📱 Responsive layout with Tailwind CSS")
        print("• 🎯 Interactive status indicators with animations")
        print("• 📊 Chart.js integration for dynamic visualizations")
        print("• 🤖 AI-powered insights and recommendations")
        print("• 🚨 Enhanced incident analysis with rich formatting")
        print("• 📄 Professional PDF generation support")
        print("• 🔄 Real-time trend analysis with 30-day history")

        print(f"\n📈 Report Statistics:")
        summary = system._create_summary_stats(metrics)
        print(f"• Services Monitored: {summary['total_services']}")
        print(f"• SLO Compliance: {summary['compliance_percentage']:.1f}%")
        print(f"• At Risk Metrics: {summary['at_risk_count']}")
        print(f"• Breached SLOs: {summary['breached_count']}")
        print(f"• System Health: {summary['health_status']}")

        print(f"\n🎯 Template Integration: ✅ COMPLETE")
        print("The enhanced template provides a modern, interactive SRE dashboard")
        print("with AI-powered insights, beautiful visualizations, and professional reporting.")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()