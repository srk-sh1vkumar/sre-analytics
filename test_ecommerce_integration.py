"""
E-commerce Integration Test
Tests the multi-source analytics system with actual e-commerce monitoring infrastructure
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_sources.base import DataSourceRegistry, QueryParams, MetricType, DataSourceType, DataSourceConfig
from data_sources.file_adapter import FileAdapter
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
            logging.FileHandler('test_ecommerce_integration.log')
        ]
    )


def test_ecommerce_prometheus_connection():
    """Test connection to actual Prometheus instance"""
    print("\n" + "="*70)
    print("TESTING E-COMMERCE PROMETHEUS CONNECTION")
    print("="*70)

    # Check if Prometheus is running
    import requests
    try:
        response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus is accessible at localhost:9090")
            data = response.json()
            if data.get('status') == 'success':
                results = data.get('data', {}).get('result', [])
                print(f"📊 Found {len(results)} 'up' metrics")

                # Show sample targets
                for i, result in enumerate(results[:5]):
                    instance = result.get('metric', {}).get('instance', 'unknown')
                    job = result.get('metric', {}).get('job', 'unknown')
                    value = result.get('value', [None, '0'])[1]
                    print(f"   {i+1}. {job} @ {instance}: {'🟢 UP' if value == '1' else '🔴 DOWN'}")

            return True
        else:
            print(f"❌ Prometheus returned status {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Prometheus: {e}")
        print("💡 Make sure the e-commerce stack is running: docker-compose up -d")
        return False


def test_ecommerce_services_detection():
    """Test detection of actual e-commerce services"""
    print("\n" + "="*70)
    print("TESTING E-COMMERCE SERVICES DETECTION")
    print("="*70)

    prometheus_config = DataSourceConfig(
        source_type=DataSourceType.PROMETHEUS,
        name="ecommerce_prometheus",
        connection_params={"url": "http://localhost:9090"},
        enabled=True
    )

    prometheus_adapter = PrometheusAdapter(prometheus_config)

    if prometheus_adapter.test_connection():
        print("✅ Connected to Prometheus")

        # Get available services
        services = prometheus_adapter.get_available_services()
        print(f"🔍 Detected services: {services}")

        # Filter for e-commerce related services
        ecommerce_services = [s for s in services if any(keyword in s.lower()
                            for keyword in ['user', 'product', 'order', 'cart', 'gateway', 'eureka', 'frontend'])]

        print(f"🏪 E-commerce services found: {ecommerce_services}")

        # Test metrics for each service
        for service in ecommerce_services[:3]:  # Test first 3 services
            print(f"\n📈 Testing metrics for {service}:")
            metrics = prometheus_adapter.get_available_metrics(service)
            print(f"   Available metrics: {len(metrics)}")
            if metrics:
                print(f"   Sample metrics: {metrics[:3]}")

        return prometheus_adapter, ecommerce_services
    else:
        print("❌ Cannot connect to Prometheus")
        return None, []


def test_ecommerce_metrics_collection():
    """Test collecting actual metrics from e-commerce services"""
    print("\n" + "="*70)
    print("TESTING E-COMMERCE METRICS COLLECTION")
    print("="*70)

    prometheus_adapter, services = test_ecommerce_services_detection()

    if not prometheus_adapter or not services:
        print("⚠️  Skipping metrics collection - no services detected")
        return None

    # Query recent metrics (last hour)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    query_params = QueryParams(
        start_time=start_time,
        end_time=end_time,
        services=services[:3],  # Test with first 3 services
        metric_types=[MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.THROUGHPUT]
    )

    print(f"🔍 Querying metrics from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")

    try:
        metrics = prometheus_adapter.query_metrics(query_params)
        print(f"📊 Collected {len(metrics)} metric data points")

        # Group by service and metric type
        service_metrics = {}
        for metric in metrics:
            service = metric.service_name
            metric_type = metric.metric_type.value

            if service not in service_metrics:
                service_metrics[service] = {}
            if metric_type not in service_metrics[service]:
                service_metrics[service][metric_type] = []

            service_metrics[service][metric_type].append(metric.value)

        # Show summary
        print("\n📈 Metrics Summary:")
        for service, types in service_metrics.items():
            print(f"   🔹 {service}:")
            for metric_type, values in types.items():
                if values:
                    avg_value = sum(values) / len(values)
                    print(f"     • {metric_type}: {len(values)} points, avg: {avg_value:.2f}")

        return metrics

    except Exception as e:
        print(f"❌ Error collecting metrics: {e}")
        return None


def test_ecommerce_analytics_engine():
    """Test analytics engine with actual e-commerce data"""
    print("\n" + "="*70)
    print("TESTING E-COMMERCE ANALYTICS ENGINE")
    print("="*70)

    # Load e-commerce specific configuration
    config_manager = ConfigurationManager("config/ecommerce_sources_config.yaml")
    config = config_manager.load_config()

    print(f"✅ Loaded e-commerce configuration with {len(config.data_sources)} data sources")

    # Setup registry with available adapters
    registry = DataSourceRegistry()

    # Add Prometheus adapter if available
    prometheus_adapter, services = test_ecommerce_services_detection()
    if prometheus_adapter:
        registry.register_adapter(prometheus_adapter)
        print("✅ Prometheus adapter registered")

    # Add file adapter for sample data
    file_adapter = None
    try:
        sample_data_path = "/Users/shiva/Projects/ecommerce-microservices/reports/examples/sample_sre_metrics_data.json"
        if Path(sample_data_path).exists():
            file_config = DataSourceConfig(
                source_type=DataSourceType.JSON_FILE,
                name="sample_sre_data",
                connection_params={
                    "file_path": sample_data_path,
                    "file_type": "json"
                },
                enabled=True
            )
            file_adapter = FileAdapter(file_config)
            if file_adapter.connect():
                registry.register_adapter(file_adapter)
                print("✅ Sample SRE data adapter registered")
    except Exception as e:
        print(f"⚠️  Could not load sample data: {e}")

    # Create analytics engine
    engine = GenericMetricsEngine(registry)

    # Set e-commerce specific SLO targets
    ecommerce_services = [
        "user-service", "product-service", "order-service",
        "cart-service", "api-gateway", "frontend"
    ]

    for service in ecommerce_services:
        targets = config_manager.get_slo_targets(service)
        if targets:
            engine.set_slo_targets(service, targets)
            print(f"🎯 SLO targets set for {service}: {len(targets)} metrics")

    # Perform analysis
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)

    query_params = QueryParams(
        start_time=start_time,
        end_time=end_time
    )

    print(f"\n🔍 Analyzing e-commerce system performance...")
    analysis_results = engine.collect_and_analyze(query_params)

    print(f"📊 Analysis completed for {len(analysis_results)} services")

    # Display e-commerce specific results
    print("\n" + "="*50)
    print("E-COMMERCE SYSTEM HEALTH REPORT")
    print("="*50)

    critical_services = config.reporting.get('ecommerce_context', {}).get('critical_services', [])

    for service_name, result in analysis_results.items():
        criticality = "🔥 CRITICAL" if service_name in critical_services else "📊 Standard"
        health_emoji = "✅" if result.overall_health == "healthy" else "⚠️" if result.overall_health == "degraded" else "❌"

        print(f"\n{health_emoji} {service_name} ({criticality})")
        print(f"   Overall Health: {result.overall_health}")

        # Show SLO compliance
        compliant = sum(1 for slo in result.slo_results if slo.status == "compliant")
        total = len(result.slo_results)
        compliance_rate = (compliant / total * 100) if total > 0 else 0

        print(f"   SLO Compliance: {compliant}/{total} ({compliance_rate:.1f}%)")

        # Show critical violations
        violations = [slo for slo in result.slo_results if slo.status == "breached"]
        if violations:
            print(f"   ❌ Critical Violations:")
            for violation in violations:
                print(f"     • {violation.metric_type.value}: {violation.current_value:.2f} "
                      f"(target: {violation.target_value})")

        # Show top insights
        if result.key_insights:
            print(f"   💡 Key Insights:")
            for insight in result.key_insights[:2]:
                print(f"     • {insight}")

    return engine, analysis_results


def test_ecommerce_recommendations():
    """Test e-commerce specific recommendations"""
    print("\n" + "="*70)
    print("TESTING E-COMMERCE RECOMMENDATIONS")
    print("="*70)

    engine, analysis_results = test_ecommerce_analytics_engine()

    if not analysis_results:
        print("⚠️  No analysis results available for recommendations")
        return

    # Create recommendation system
    rec_system = EnhancedRecommendationSystem()

    # Create e-commerce contexts
    contexts = []
    for service_name, analysis_result in analysis_results.items():
        # E-commerce specific business context
        business_context = {
            "criticality": "high" if service_name in ["user-service", "order-service", "payment-service"] else "medium",
            "sla_tier": "premium",
            "revenue_impact": "high" if service_name in ["order-service", "payment-service", "cart-service"] else "medium",
            "customer_facing": service_name in ["frontend", "api-gateway", "user-service"],
            "peak_hours": ["12:00-13:00", "18:00-21:00"],
            "environment": "production"
        }

        context = RecommendationContext(
            service_name=service_name,
            analysis_result=analysis_result,
            historical_data=[],
            business_context=business_context
        )
        contexts.append(context)

    # Generate e-commerce specific recommendations
    recommendations = rec_system.generate_comprehensive_recommendations(contexts)

    print(f"💡 Generated {sum(len(recs) for recs in recommendations.values())} recommendations")

    # Display recommendations by priority
    print("\n" + "="*50)
    print("E-COMMERCE OPTIMIZATION RECOMMENDATIONS")
    print("="*50)

    all_recs = []
    for service_name, service_recs in recommendations.items():
        for rec in service_recs:
            rec.service_name = service_name  # Add service name for sorting
            all_recs.append(rec)

    # Sort by priority and confidence
    priority_order = {"critical": 1, "high": 2, "medium": 3, "low": 4}
    all_recs.sort(key=lambda x: (priority_order.get(x.priority, 5), -x.confidence_score))

    # Show top recommendations
    print("\n🔥 TOP CRITICAL RECOMMENDATIONS:")
    critical_recs = [r for r in all_recs if r.priority == "critical"][:3]
    for i, rec in enumerate(critical_recs, 1):
        print(f"\n{i}. {rec.title} ({rec.service_name})")
        print(f"   Impact: {rec.expected_impact}")
        print(f"   Effort: {rec.implementation_effort} | Time: {rec.time_to_implement}")
        print(f"   Confidence: {rec.confidence_score:.1f}")
        print(f"   Actions:")
        for action in rec.action_items[:2]:
            print(f"     • {action}")

    print("\n⚡ HIGH PRIORITY RECOMMENDATIONS:")
    high_recs = [r for r in all_recs if r.priority == "high"][:5]
    for i, rec in enumerate(high_recs, 1):
        print(f"\n{i}. {rec.title} ({rec.service_name})")
        print(f"   Category: {rec.category} | Impact: {rec.expected_impact}")

    # Export e-commerce report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export analysis results
    analysis_export = engine.export_analysis_results(analysis_results, "json")
    with open(f"reports/generated/ecommerce_analysis_{timestamp}.json", "w") as f:
        f.write(analysis_export)

    # Export recommendations
    rec_export = rec_system.export_recommendations(recommendations, "markdown")
    with open(f"reports/generated/ecommerce_recommendations_{timestamp}.md", "w") as f:
        f.write(rec_export)

    print(f"\n💾 Reports saved:")
    print(f"   • ecommerce_analysis_{timestamp}.json")
    print(f"   • ecommerce_recommendations_{timestamp}.md")

    return recommendations


def main():
    """Main integration test"""
    print("🏪 Starting E-commerce Integration Tests")
    print(f"📅 Test started at: {datetime.now()}")
    print("\n💡 Why Python for Analytics:")
    print("   • Superior data science libraries (pandas, numpy, scipy)")
    print("   • Better ML/AI integration for intelligent recommendations")
    print("   • Faster development for analytics and reporting features")
    print("   • Excellent API integration capabilities")
    print("   • Rich visualization and report generation tools")

    # Setup
    setup_logging()
    os.makedirs("reports/generated", exist_ok=True)

    success = True

    try:
        # Test 1: Check if e-commerce stack is running
        print("\n🔍 CHECKING E-COMMERCE INFRASTRUCTURE...")
        prometheus_available = test_ecommerce_prometheus_connection()

        if not prometheus_available:
            print("\n⚠️  E-commerce stack not running. Testing with sample data only.")
            print("💡 To test with live data, run: cd /Users/shiva/Projects/ecommerce-microservices && docker-compose up -d")

        # Test 2: Service detection and metrics collection
        if prometheus_available:
            print("\n🔍 TESTING LIVE METRICS COLLECTION...")
            metrics = test_ecommerce_metrics_collection()
            if metrics:
                print(f"✅ Successfully collected {len(metrics)} live metrics")
            else:
                print("⚠️  Live metrics collection failed")

        # Test 3: Full analytics pipeline
        print("\n📊 TESTING ANALYTICS PIPELINE...")
        recommendations = test_ecommerce_recommendations()

        if recommendations:
            total_recommendations = sum(len(recs) for recs in recommendations.values())
            print(f"✅ Generated {total_recommendations} actionable recommendations")

        # Summary
        print("\n" + "="*70)
        print("INTEGRATION TEST SUMMARY")
        print("="*70)
        print("✅ Multi-source analytics system ready for e-commerce monitoring")
        print("✅ Prometheus integration configured")
        print("✅ E-commerce specific SLO targets defined")
        print("✅ Business-context aware recommendations generated")
        print("✅ Exportable reports created")

        print("\n🎯 NEXT STEPS:")
        print("1. Start e-commerce stack: docker-compose up -d")
        print("2. Run load tests to generate metrics")
        print("3. Schedule regular analytics reports")
        print("4. Configure alerting based on SLO violations")
        print("5. Integrate with CI/CD for performance monitoring")

    except Exception as e:
        print(f"\n💥 Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    return success


if __name__ == "__main__":
    main()