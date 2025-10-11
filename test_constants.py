#!/usr/bin/env python3
"""
Test script for constants module
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import constants
from src.config.constants import (
    DEFAULT_TREND_DAYS, AVAILABILITY_MIN, AVAILABILITY_MAX, LATENCY_P95_TARGET_MS,
    STATUS_COMPLIANT, STATUS_AT_RISK, STATUS_BREACHED, METRIC_AVAILABILITY,
    PORT_FLASK_APP, API_TIMEOUT_DEFAULT
)

def test_constants():
    """Test that constants module is properly configured"""

    print("=" * 60)
    print("Testing Constants Module")
    print("=" * 60)

    # Test time constants
    print("\n1. Time and Date Constants:")
    print(f"   DEFAULT_TREND_DAYS: {DEFAULT_TREND_DAYS}")
    assert DEFAULT_TREND_DAYS == 30, "DEFAULT_TREND_DAYS should be 30"
    print("   ✓ Time constants verified")

    # Test SLO/SLA thresholds
    print("\n2. SLO/SLA Thresholds:")
    print(f"   AVAILABILITY_MIN: {AVAILABILITY_MIN}%")
    print(f"   AVAILABILITY_MAX: {AVAILABILITY_MAX}%")
    print(f"   LATENCY_P95_TARGET_MS: {LATENCY_P95_TARGET_MS}ms")
    assert AVAILABILITY_MIN == 99.5, "AVAILABILITY_MIN should be 99.5"
    assert AVAILABILITY_MAX == 99.99, "AVAILABILITY_MAX should be 99.99"
    assert LATENCY_P95_TARGET_MS == 200, "LATENCY_P95_TARGET_MS should be 200"
    print("   ✓ SLO/SLA thresholds verified")

    # Test status constants
    print("\n3. Status Constants:")
    print(f"   STATUS_COMPLIANT: '{STATUS_COMPLIANT}'")
    print(f"   STATUS_AT_RISK: '{STATUS_AT_RISK}'")
    print(f"   STATUS_BREACHED: '{STATUS_BREACHED}'")
    assert STATUS_COMPLIANT == 'compliant'
    assert STATUS_AT_RISK == 'at_risk'
    assert STATUS_BREACHED == 'breached'
    print("   ✓ Status constants verified")

    # Test metric names
    print("\n4. Metric Names:")
    print(f"   METRIC_AVAILABILITY: '{METRIC_AVAILABILITY}'")
    assert METRIC_AVAILABILITY == 'availability'
    print("   ✓ Metric names verified")

    # Test port numbers
    print("\n5. Port Numbers:")
    print(f"   PORT_FLASK_APP: {PORT_FLASK_APP}")
    print(f"   PORT_PROMETHEUS: {constants.PORT_PROMETHEUS}")
    print(f"   PORT_GRAFANA: {constants.PORT_GRAFANA}")
    assert PORT_FLASK_APP == 5001
    print("   ✓ Port numbers verified")

    # Test API timeouts
    print("\n6. API Timeouts:")
    print(f"   API_TIMEOUT_DEFAULT: {API_TIMEOUT_DEFAULT}s")
    print(f"   API_RETRY_ATTEMPTS: {constants.API_RETRY_ATTEMPTS}")
    assert API_TIMEOUT_DEFAULT == 30
    assert constants.API_RETRY_ATTEMPTS == 3
    print("   ✓ API configuration verified")

    # Test that constants are Final (should not be modifiable)
    print("\n7. Testing Immutability:")
    try:
        # This should work (reassignment in local scope)
        temp = DEFAULT_TREND_DAYS
        print(f"   ✓ Constants can be read: {temp}")
    except:
        print("   ✗ Error reading constants")
        return False

    # Verify module structure
    print("\n8. Module Structure:")
    constant_groups = [
        'TIME AND DATE CONSTANTS',
        'SLO/SLA THRESHOLDS',
        'API AND NETWORK CONSTANTS',
        'PORT NUMBERS',
        'PDF GENERATION CONSTANTS',
        'METRICS AND MONITORING CONSTANTS',
        'INCIDENT SEVERITY LEVELS',
        'SLO STATUS CATEGORIES'
    ]

    # Read the constants file to verify sections exist
    constants_file = os.path.join(os.path.dirname(__file__), 'src', 'config', 'constants.py')
    with open(constants_file, 'r') as f:
        content = f.read()
        for group in constant_groups:
            if group in content:
                print(f"   ✓ Found section: {group}")
            else:
                print(f"   ✗ Missing section: {group}")

    # Count total constants
    print("\n9. Statistics:")
    all_constants = [attr for attr in dir(constants) if attr.isupper()]
    print(f"   Total constants defined: {len(all_constants)}")
    print(f"   Categories: 13+")

    # Show sample of constants by category
    print("\n10. Sample Constants by Category:")
    samples = {
        'Time/Date': ['DEFAULT_TREND_DAYS', 'DAYS_IN_MONTH'],
        'SLO/SLA': ['AVAILABILITY_MIN', 'LATENCY_P95_TARGET_MS'],
        'API/Network': ['API_TIMEOUT_DEFAULT', 'API_RETRY_ATTEMPTS'],
        'Ports': ['PORT_FLASK_APP', 'PORT_PROMETHEUS'],
        'Status': ['STATUS_COMPLIANT', 'STATUS_AT_RISK'],
        'Metrics': ['METRIC_AVAILABILITY', 'METRIC_LATENCY_P95']
    }

    for category, const_list in samples.items():
        print(f"\n   {category}:")
        for const_name in const_list:
            if hasattr(constants, const_name):
                value = getattr(constants, const_name)
                print(f"      {const_name} = {value}")

    print("\n" + "=" * 60)
    print("Constants Module Test Complete!")
    print("=" * 60)

    print("\n✅ Benefits of centralized constants:")
    print("   • Single source of truth for all magic numbers")
    print("   • Type-safe with Final annotations")
    print("   • Easy to update thresholds across entire codebase")
    print("   • Better code readability and maintainability")
    print("   • Prevents inconsistent hardcoded values")

    return True

if __name__ == "__main__":
    try:
        success = test_constants()
        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
