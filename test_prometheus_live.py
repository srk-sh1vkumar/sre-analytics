#!/usr/bin/env python3
"""
Live Test Script for Prometheus Integration

Tests the Prometheus integration against a running Prometheus instance.
Connects to localhost:9090 (ecommerce-microservices Prometheus).
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_sources.prometheus_integration import PrometheusIntegration
from src.data_sources.base import DataSourceConfig, DataSourceType


def main():
    print("=" * 70)
    print("PROMETHEUS INTEGRATION - LIVE TEST")
    print("=" * 70)
    print()

    # Create configuration
    config = DataSourceConfig(
        source_type=DataSourceType.PROMETHEUS,
        name="Local Prometheus",
        connection_params={"url": "http://localhost:9090"},
        enabled=True
    )

    # SLO targets for ecommerce services
    slo_targets = {
        "_default": {
            "response_time": 200.0,
            "error_rate": 0.01,
            "availability": 0.999,
            "cpu_utilization": 80.0,
            "memory_utilization": 85.0
        },
        "product-service": {
            "response_time": 150.0,
            "error_rate": 0.005,
            "availability": 0.995
        },
        "order-service": {
            "response_time": 300.0,
            "error_rate": 0.005,
            "availability": 0.9995
        }
    }

    print("📊 Initializing Prometheus Integration...")
    integration = PrometheusIntegration(
        config=config,
        slo_targets=slo_targets,
        rate_limit_calls=100,
        cache_ttl=60
    )

    # Test 1: Connection
    print("\n🔌 Test 1: Testing Prometheus Connection...")
    connected = integration.connect()
    if connected:
        print("   ✅ Successfully connected to Prometheus at http://localhost:9090")
    else:
        print("   ❌ Failed to connect to Prometheus")
        print("   💡 Make sure Prometheus is running: docker ps | grep prometheus")
        return 1

    # Test 2: Get available services
    print("\n📋 Test 2: Discovering Available Services...")
    services = integration.get_available_services()
    if services:
        print(f"   ✅ Found {len(services)} services:")
        for service in services[:10]:  # Show first 10
            print(f"      - {service}")
        if len(services) > 10:
            print(f"      ... and {len(services) - 10} more")
    else:
        print("   ⚠️  No services found (Prometheus may be empty)")

    # Test 3: Get Prometheus info
    print("\n📈 Test 3: Getting Prometheus Server Info...")
    info = integration.get_prometheus_info()
    if "error" not in info:
        print("   ✅ Prometheus server info:")
        print(f"      Base URL: {info.get('base_url')}")
        conn_status = info.get('connection_status', {})
        print(f"      Status: {conn_status.get('status')}")
        print(f"      Available Services: {conn_status.get('available_services', 0)}")
    else:
        print(f"   ⚠️  Error getting info: {info['error']}")

    # Test 4: Query specific services (if available)
    print("\n🎯 Test 4: Querying SLO Metrics for Services...")
    if services:
        # Try to query up to 3 services
        test_services = services[:3]
        print(f"   Querying services: {', '.join(test_services)}")

        try:
            slo_metrics = integration.get_slo_metrics_for_services(
                services=test_services,
                use_cache=False
            )

            if slo_metrics:
                print(f"   ✅ Retrieved {len(slo_metrics)} SLO metrics")
                print("\n   📊 Sample Metrics:")
                for metric in slo_metrics[:5]:  # Show first 5
                    print(f"      • {metric.service_name} - {metric.metric_name}:")
                    print(f"        Current: {metric.current_value:.2f}{metric.unit}")
                    print(f"        Target: {metric.slo_target:.2f}{metric.unit}")
                    print(f"        Status: {metric.status}")
                    print(f"        Error Budget: {metric.error_budget_consumed:.1f}%")
            else:
                print("   ⚠️  No SLO metrics retrieved (services may not have matching metrics)")
        except Exception as e:
            print(f"   ⚠️  Error querying metrics: {e}")
    else:
        print("   ⏭️  Skipping (no services available)")

    # Test 5: Health Report
    print("\n💚 Test 5: Generating Service Health Report...")
    if services:
        test_service = services[0]
        print(f"   Generating health report for: {test_service}")

        try:
            health_report = integration.get_service_health_report(
                test_service,
                hours_back=1
            )

            print(f"   ✅ Health Report Generated:")
            health = health_report.get('health_score', {})
            print(f"      Health Score: {health.get('health_score', 'N/A')}/100")
            print(f"      Status: {health.get('status', 'N/A')}")
            print(f"      Metrics Count: {health.get('metrics_count', 0)}")

            insights = health_report.get('insights', [])
            if insights:
                print(f"\n      💡 Insights:")
                for insight in insights:
                    print(f"         {insight}")

        except Exception as e:
            print(f"   ⚠️  Error generating health report: {e}")
    else:
        print("   ⏭️  Skipping (no services available)")

    # Test 6: Custom PromQL Query
    print("\n🔍 Test 6: Testing Custom PromQL Query...")
    try:
        result = integration.query_promql("up")
        if result.get("status") == "success":
            data = result.get("data", {})
            result_type = data.get("resultType")
            results = data.get("result", [])
            print(f"   ✅ PromQL query executed successfully")
            print(f"      Result Type: {result_type}")
            print(f"      Results Count: {len(results)}")
            if results:
                print(f"      Sample: {results[0].get('metric', {})} = {results[0].get('value', ['N/A', 'N/A'])[1]}")
        else:
            print(f"   ⚠️  Query failed: {result}")
    except Exception as e:
        print(f"   ⚠️  Error executing PromQL: {e}")

    # Test 7: Statistics
    print("\n📊 Test 7: Integration Statistics...")
    stats = integration.get_statistics()
    print("   ✅ Statistics:")
    print(f"      Total Queries: {stats['total_queries']}")
    print(f"      API Calls: {stats['api_calls']}")
    print(f"      Cache Hits: {stats['cache_hits']}")
    print(f"      Cache Misses: {stats['cache_misses']}")
    print(f"      Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
    print(f"      Errors: {stats['errors']}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ PROMETHEUS INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Review config/prometheus_example.yaml for configuration options")
    print("   2. Customize SLO targets for your services")
    print("   3. Generate SRE reports from Prometheus data")
    print("   4. Build real-time dashboard with Prometheus metrics")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
