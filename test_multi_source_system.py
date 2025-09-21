"""
Test Script for Multi-Source Analytics System
Demonstrates the generic analytics capabilities with multiple data sources
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_sources.base import DataSourceRegistry, QueryParams, MetricType, DataSourceType
from data_sources.file_adapter import FileAdapter
from data_sources.appdynamics_adapter import AppDynamicsAdapter
from data_sources.prometheus_adapter import PrometheusAdapter
from analytics.generic_metrics_engine import GenericMetricsEngine, SLOTarget
from analytics.enhanced_recommendation_system import EnhancedRecommendationSystem, RecommendationContext
from config.multi_source_config import ConfigurationManager


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_multi_source.log')
        ]
    )


def test_configuration_system():
    """Test the configuration system"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION SYSTEM")
    print("="*60)

    config_manager = ConfigurationManager("config/test_multi_source_config.yaml")

    # Load or create default config
    config = config_manager.load_config()
    print(f"✅ Configuration loaded with {len(config.data_sources)} data sources")

    # Get config summary
    summary = config_manager.get_config_summary()
    print(f"📊 Config Summary: {summary}")

    # Validate configuration
    issues = config_manager.validate_config()
    if issues:
        print(f"⚠️  Configuration issues: {issues}")
    else:
        print("✅ Configuration validation passed")

    return config_manager


def test_file_adapter():
    """Test the file adapter with CSV data"""
    print("\n" + "="*60)
    print("TESTING FILE ADAPTER")
    print("="*60)

    from data_sources.base import DataSourceConfig

    # Create file adapter configuration
    file_config = DataSourceConfig(
        source_type=DataSourceType.CSV_FILE,
        name="test_csv_source",
        connection_params={
            "file_path": "data/sample_metrics.csv",
            "file_type": "csv"
        },
        enabled=True
    )

    # Create and test adapter
    file_adapter = FileAdapter(file_config)

    # Test connection
    connected = file_adapter.connect()
    print(f"📁 File connection: {'✅ Connected' if connected else '❌ Failed'}")

    if connected:
        # Get available services
        services = file_adapter.get_available_services()
        print(f"🔍 Available services: {services}")

        # Get available metrics for first service
        if services:
            metrics = file_adapter.get_available_metrics(services[0])
            print(f"📈 Available metrics for {services[0]}: {metrics}")

        # Query metrics
        query_params = QueryParams(
            start_time=datetime(2024, 1, 20, 9, 0, 0),
            end_time=datetime(2024, 1, 20, 11, 0, 0),
            services=services[:2] if services else None,
            metric_types=[MetricType.RESPONSE_TIME, MetricType.ERROR_RATE]
        )

        metrics_data = file_adapter.query_metrics(query_params)
        print(f"📊 Retrieved {len(metrics_data)} metric data points")

        # Show sample metrics
        for i, metric in enumerate(metrics_data[:5]):
            print(f"   {i+1}. {metric.service_name} - {metric.metric_type.value}: {metric.value} {metric.unit}")

        # Get health status
        health = file_adapter.get_health_status()
        print(f"💚 Health status: {health['status']}")

    return file_adapter if connected else None


def test_data_source_registry():
    """Test the data source registry"""
    print("\n" + "="*60)
    print("TESTING DATA SOURCE REGISTRY")
    print("="*60)

    registry = DataSourceRegistry()

    # Register file adapter
    file_adapter = test_file_adapter()
    if file_adapter:
        registry.register_adapter(file_adapter)
        print("✅ File adapter registered")

    # Test registry operations
    adapters = registry.list_adapters()
    print(f"📋 Registered adapters: {adapters}")

    enabled_adapters = registry.get_enabled_adapters()
    print(f"🟢 Enabled adapters: {len(enabled_adapters)}")

    # Get registry status
    status = registry.get_registry_status()
    print("🔍 Registry status:")
    for name, adapter_status in status.items():
        print(f"   {name}: {'✅' if adapter_status['connected'] else '❌'} {adapter_status.get('error', '')}")

    return registry


def test_metrics_engine():
    """Test the generic metrics engine"""
    print("\n" + "="*60)
    print("TESTING GENERIC METRICS ENGINE")
    print("="*60)

    # Setup registry
    registry = test_data_source_registry()

    # Create metrics engine
    engine = GenericMetricsEngine(registry)

    # Set custom SLO targets
    custom_targets = [
        SLOTarget(MetricType.RESPONSE_TIME, 200, "less_than", "ms", "Response time under 200ms"),
        SLOTarget(MetricType.ERROR_RATE, 1.0, "less_than", "%", "Error rate under 1%"),
        SLOTarget(MetricType.CPU_UTILIZATION, 80, "less_than", "%", "CPU under 80%"),
        SLOTarget(MetricType.MEMORY_UTILIZATION, 85, "less_than", "%", "Memory under 85%")
    ]

    engine.set_slo_targets("user-service", custom_targets)
    engine.set_slo_targets("product-service", custom_targets)
    engine.set_slo_targets("order-service", custom_targets)

    print("🎯 SLO targets configured")

    # Query and analyze
    query_params = QueryParams(
        start_time=datetime(2024, 1, 20, 9, 0, 0),
        end_time=datetime(2024, 1, 20, 11, 0, 0)
    )

    analysis_results = engine.collect_and_analyze(query_params)
    print(f"📊 Analysis completed for {len(analysis_results)} services")

    # Display results
    for service_name, result in analysis_results.items():
        print(f"\n🔍 Analysis for {service_name}:")
        print(f"   Overall Health: {result.overall_health}")
        print(f"   SLO Results: {len(result.slo_results)} metrics evaluated")

        for slo in result.slo_results:
            status_emoji = "✅" if slo.status == "compliant" else "⚠️" if slo.status == "at_risk" else "❌"
            print(f"   {status_emoji} {slo.metric_type.value}: {slo.current_value:.1f} "
                  f"(target: {slo.target_value}, compliance: {slo.compliance_percentage:.1f}%)")

        print(f"   Key Insights:")
        for insight in result.key_insights[:3]:
            print(f"     • {insight}")

        print(f"   Recommendations:")
        for rec in result.recommendations[:2]:
            print(f"     • {rec}")

    # Export results
    json_export = engine.export_analysis_results(analysis_results, "json")
    with open("reports/generated/multi_source_analysis.json", "w") as f:
        f.write(json_export)
    print("💾 Analysis results exported to JSON")

    return engine, analysis_results


def test_recommendation_system():
    """Test the enhanced recommendation system"""
    print("\n" + "="*60)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*60)

    # Get analysis results from previous test
    engine, analysis_results = test_metrics_engine()

    # Create recommendation system
    rec_system = EnhancedRecommendationSystem()

    # Create recommendation contexts
    contexts = []
    for service_name, analysis_result in analysis_results.items():
        context = RecommendationContext(
            service_name=service_name,
            analysis_result=analysis_result,
            historical_data=[],  # Would contain historical metrics in real scenario
            business_context={"criticality": "high", "sla_tier": "premium"}
        )
        contexts.append(context)

    # Generate recommendations
    recommendations = rec_system.generate_comprehensive_recommendations(contexts)
    print(f"💡 Generated recommendations for {len(recommendations)} services")

    # Display recommendations
    for service_name, service_recs in recommendations.items():
        print(f"\n🎯 Recommendations for {service_name}:")

        for i, rec in enumerate(service_recs[:3], 1):  # Show top 3
            priority_emoji = "🔥" if rec.priority == "critical" else "🟡" if rec.priority == "high" else "🔵"
            print(f"   {priority_emoji} {i}. {rec.title}")
            print(f"      Priority: {rec.priority}, Category: {rec.category}")
            print(f"      Description: {rec.description}")
            print(f"      Expected Impact: {rec.expected_impact}")
            print(f"      Implementation Effort: {rec.implementation_effort}")
            print(f"      Time to Implement: {rec.time_to_implement}")
            print(f"      Confidence: {rec.confidence_score:.1f}")

            print(f"      Action Items:")
            for action in rec.action_items[:2]:  # Show first 2 actions
                print(f"        • {action}")
            print()

    # Export recommendations
    markdown_export = rec_system.export_recommendations(recommendations, "markdown")
    with open("reports/generated/multi_source_recommendations.md", "w") as f:
        f.write(markdown_export)
    print("💾 Recommendations exported to Markdown")

    return rec_system, recommendations


def test_complete_workflow():
    """Test the complete workflow from configuration to recommendations"""
    print("\n" + "="*80)
    print("TESTING COMPLETE MULTI-SOURCE WORKFLOW")
    print("="*80)

    # Step 1: Configuration
    config_manager = test_configuration_system()

    # Step 2: Data Sources
    registry = test_data_source_registry()

    # Step 3: Analytics Engine
    engine, analysis_results = test_metrics_engine()

    # Step 4: Recommendations
    rec_system, recommendations = test_recommendation_system()

    # Summary
    print("\n" + "="*60)
    print("WORKFLOW SUMMARY")
    print("="*60)

    total_services = len(analysis_results)
    total_recommendations = sum(len(recs) for recs in recommendations.values())

    critical_issues = 0
    for service_name, result in analysis_results.items():
        critical_issues += sum(1 for slo in result.slo_results if slo.status == "breached")

    print(f"📊 Services Analyzed: {total_services}")
    print(f"🎯 Total Recommendations: {total_recommendations}")
    print(f"🔥 Critical Issues Found: {critical_issues}")

    # Health overview
    health_summary = {}
    for service_name, result in analysis_results.items():
        health = result.overall_health
        health_summary[health] = health_summary.get(health, 0) + 1

    print(f"💚 System Health Overview:")
    for health_status, count in health_summary.items():
        emoji = "✅" if health_status == "healthy" else "⚠️" if health_status == "degraded" else "❌"
        print(f"   {emoji} {health_status.title()}: {count} services")

    print("\n🎉 Multi-source analytics system test completed successfully!")
    return True


def main():
    """Main test function"""
    print("🚀 Starting Multi-Source Analytics System Tests")
    print(f"📅 Test started at: {datetime.now()}")

    # Setup
    setup_logging()

    # Create directories
    os.makedirs("config", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports/generated", exist_ok=True)

    try:
        # Run complete workflow test
        success = test_complete_workflow()

        if success:
            print("\n✅ All tests completed successfully!")
            print("\n📁 Generated files:")
            print("   • config/test_multi_source_config.yaml - Configuration file")
            print("   • reports/generated/multi_source_analysis.json - Analysis results")
            print("   • reports/generated/multi_source_recommendations.md - Recommendations")
            print("   • test_multi_source.log - Test execution log")
        else:
            print("\n❌ Some tests failed!")

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()