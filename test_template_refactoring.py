#!/usr/bin/env python3
"""
Test script for refactored HTML template generation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.reports.enhanced_sre_report_system import EnhancedSREReportSystem

def test_template_refactoring():
    """Test that refactored template methods work correctly"""

    print("=" * 60)
    print("Testing Refactored HTML Template Generation")
    print("=" * 60)

    # Initialize system
    print("\n1. Initializing SRE report system...")
    sre_system = EnhancedSREReportSystem(app_name="Test Application")
    print("   ✓ System initialized")

    # Test individual template sections
    print("\n2. Testing individual template sections:")

    sections = {
        'Header & Styles': sre_system._get_html_header_and_styles,
        'Executive Summary': sre_system._get_html_executive_summary,
        'Trend Charts': sre_system._get_html_trend_charts,
        'Incident Analysis': sre_system._get_html_incident_analysis,
        'Metrics Table': sre_system._get_html_metrics_table,
        'Recommendations': sre_system._get_html_recommendations,
        'Footer': sre_system._get_html_footer,
    }

    section_stats = {}
    for section_name, section_method in sections.items():
        content = section_method()
        lines = content.count('\n')
        chars = len(content)
        section_stats[section_name] = {'lines': lines, 'chars': chars}
        print(f"   ✓ {section_name}: {lines} lines, {chars} characters")

    # Test complete template
    print("\n3. Testing complete template composition:")
    complete_template = sre_system._get_comprehensive_html_template()
    total_lines = complete_template.count('\n')
    total_chars = len(complete_template)
    print(f"   Complete template: {total_lines} lines, {total_chars} characters")

    # Verify all sections are present
    print("\n4. Verifying all sections are included:")
    required_elements = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        'Executive Summary',
        'Performance Trends',
        'Incident Analysis',
        'Current SLO Metrics',
        'Key Recommendations',
        'Report Features',
        '</html>'
    ]

    for element in required_elements:
        if element in complete_template:
            print(f"   ✓ Found: {element}")
        else:
            print(f"   ✗ Missing: {element}")
            return False

    # Verify Jinja2 template variables are preserved
    print("\n5. Verifying Jinja2 template variables:")
    template_vars = [
        '{{ app_name }}',
        '{{ report_date }}',
        '{{ report_time }}',
        '{{ summary.total_services }}',
        '{% for metric in metrics %}',
        '{% if has_incident %}'
    ]

    for var in template_vars:
        if var in complete_template:
            print(f"   ✓ Found: {var}")
        else:
            print(f"   ✗ Missing: {var}")
            return False

    # Compare with old template (if it exists)
    print("\n6. Comparing new vs old template structure:")
    try:
        old_template = sre_system._get_comprehensive_html_template_old()
        old_lines = old_template.count('\n')
        print(f"   Old template: {old_lines} lines")
        print(f"   New template: {total_lines} lines")
        print(f"   Difference: {total_lines - old_lines} lines")
    except:
        print("   Old template not available for comparison")

    # Summary statistics
    print("\n7. Section Statistics:")
    print(f"   {'Section':<25} {'Lines':<10} {'Characters':<15}")
    print(f"   {'-' * 50}")
    for section_name, stats in section_stats.items():
        print(f"   {section_name:<25} {stats['lines']:<10} {stats['chars']:<15}")
    print(f"   {'-' * 50}")
    print(f"   {'TOTAL':<25} {total_lines:<10} {total_chars:<15}")

    print("\n" + "=" * 60)
    print("Template Refactoring Test Complete!")
    print("=" * 60)

    print("\n✅ Benefits of refactored approach:")
    print("   • Each section is now independently testable")
    print("   • Easier to modify individual sections")
    print("   • Better code organization and maintainability")
    print("   • Follows Single Responsibility Principle")
    print("   • Reduced cognitive complexity")

    return True

if __name__ == "__main__":
    try:
        success = test_template_refactoring()
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
