#!/usr/bin/env python3
"""
AppDynamics OAuth Troubleshooting Tool
Comprehensive diagnostics for AppDynamics connectivity and authentication issues
"""

import sys
import os
from pathlib import Path
import logging
import requests
from datetime import datetime

# Add src to path
sys.path.append('src')

try:
    from collectors.oauth_appdynamics_collector import OAuthAppDynamicsCollector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def setup_detailed_logging():
    """Setup comprehensive logging"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/appdynamics_troubleshooting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def check_environment_variables():
    """Check if required environment variables are set"""
    print("🔧 Environment Variables Check")
    print("=" * 40)

    required_vars = [
        'APPDYNAMICS_CONTROLLER_HOST',
        'APPDYNAMICS_CLIENT_ID',
        'APPDYNAMICS_CLIENT_SECRET'
    ]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask secret values
            if 'SECRET' in var or 'KEY' in var:
                display_value = '*' * (len(value) - 4) + value[-4:] if len(value) > 4 else '*' * len(value)
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)

    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or environment configuration")
        return False

    return True

def test_basic_connectivity():
    """Test basic network connectivity to AppDynamics controller"""
    print("\n🌐 Basic Connectivity Test")
    print("=" * 40)

    controller_host = os.getenv('APPDYNAMICS_CONTROLLER_HOST')
    if not controller_host:
        print("❌ Controller host not configured")
        return False

    controller_url = f"https://{controller_host}"

    try:
        print(f"Testing connection to: {controller_url}")
        response = requests.get(f"{controller_url}/controller", timeout=10, verify=True)

        print(f"✅ Controller reachable")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Headers: {dict(response.headers)}")

        return True

    except requests.exceptions.SSLError as e:
        print(f"❌ SSL Error: {e}")
        print("🔧 Try disabling SSL verification or check certificate")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("🔧 Check URL, firewall settings, and network connectivity")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout Error: {e}")
        print("🔧 Check network latency and increase timeout")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_oauth_endpoint():
    """Test if OAuth endpoint is available"""
    print("\n🔐 OAuth Endpoint Test")
    print("=" * 40)

    controller_host = os.getenv('APPDYNAMICS_CONTROLLER_HOST')
    controller_url = f"https://{controller_host}"
    oauth_url = f"{controller_url}/controller/api/oauth/access_token"

    try:
        print(f"Testing OAuth endpoint: {oauth_url}")

        # Try a basic GET request to see if endpoint exists
        response = requests.get(f"{controller_url}/controller/api/oauth", timeout=10)

        if response.status_code == 404:
            print("❌ OAuth endpoint not found (404)")
            print("🔧 Possible issues:")
            print("   1. OAuth not enabled on this AppDynamics controller")
            print("   2. Incorrect AppDynamics version (OAuth requires newer versions)")
            print("   3. Incorrect endpoint URL")
            return False
        else:
            print("✅ OAuth endpoint appears to be available")
            print(f"   Status Code: {response.status_code}")
            return True

    except Exception as e:
        print(f"❌ OAuth endpoint test failed: {e}")
        return False

def test_oauth_authentication():
    """Test OAuth token request with detailed logging"""
    print("\n🎫 OAuth Authentication Test")
    print("=" * 40)

    try:
        collector = OAuthAppDynamicsCollector()

        # The constructor already attempts OAuth, so check if it worked
        if collector.token:
            print("✅ OAuth authentication successful!")
            print(f"   Token Type: {collector.token.token_type}")
            print(f"   Expires At: {collector.token.expires_at}")
            return True
        else:
            print("❌ OAuth authentication failed")
            return False

    except Exception as e:
        print(f"❌ OAuth collector initialization failed: {e}")
        return False

def test_api_access():
    """Test API access with obtained token"""
    print("\n📊 API Access Test")
    print("=" * 40)

    try:
        collector = OAuthAppDynamicsCollector()

        if not collector.token:
            print("❌ No OAuth token available, skipping API test")
            return False

        # Test applications API
        applications = collector.get_applications()

        if applications:
            print(f"✅ API access successful!")
            print(f"   Found {len(applications)} applications:")
            for app in applications[:5]:  # Show first 5
                print(f"   - {app.get('name', 'Unknown')} (ID: {app.get('id', 'N/A')})")
            return True
        else:
            print("⚠️  API access works but no applications found")
            print("🔧 Check if your OAuth client has permissions to access applications")
            return False

    except Exception as e:
        print(f"❌ API access test failed: {e}")
        return False

def run_comprehensive_troubleshooting():
    """Run all troubleshooting tests"""
    print("🔍 AppDynamics OAuth Comprehensive Troubleshooting")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        'env_vars': False,
        'connectivity': False,
        'oauth_endpoint': False,
        'oauth_auth': False,
        'api_access': False
    }

    # Run all tests
    results['env_vars'] = check_environment_variables()

    if results['env_vars']:
        results['connectivity'] = test_basic_connectivity()

        if results['connectivity']:
            results['oauth_endpoint'] = test_oauth_endpoint()

            if results['oauth_endpoint']:
                results['oauth_auth'] = test_oauth_authentication()

                if results['oauth_auth']:
                    results['api_access'] = test_api_access()

    # Print summary
    print("\n📋 Troubleshooting Summary")
    print("=" * 40)
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        test_name = test.replace('_', ' ').title()
        print(f"{test_name}: {status}")

    # Print recommendations
    print("\n💡 Recommendations")
    print("=" * 40)

    if not results['env_vars']:
        print("1. 🔧 Fix environment variables in .env file")
    elif not results['connectivity']:
        print("1. 🌐 Check network connectivity and controller URL")
    elif not results['oauth_endpoint']:
        print("1. 🔐 Enable OAuth on AppDynamics controller or check version compatibility")
    elif not results['oauth_auth']:
        print("1. 🎫 Verify OAuth client credentials are correct")
        print("2. 🔑 Check if OAuth client has proper permissions")
        print("3. 📞 Contact AppDynamics administrator to verify OAuth setup")
    elif not results['api_access']:
        print("1. 👤 Check OAuth client permissions for application access")
        print("2. 📱 Verify applications exist in the controller")
    else:
        print("🎉 All tests passed! AppDynamics integration should be working.")

    return results

def main():
    """Main troubleshooting function"""
    # Setup logging
    setup_detailed_logging()

    # Run troubleshooting
    results = run_comprehensive_troubleshooting()

    # Generate report
    all_passed = all(results.values())

    print(f"\n🎯 Troubleshooting {'COMPLETED SUCCESSFULLY' if all_passed else 'FOUND ISSUES'}")
    print(f"Detailed logs saved in: logs/appdynamics_troubleshooting_*.log")

    if not all_passed:
        print("\n🔧 For additional help:")
        print("1. Check the generated log file for detailed error messages")
        print("2. Verify AppDynamics controller version supports OAuth")
        print("3. Contact your AppDynamics administrator")
        print("4. Review AppDynamics OAuth documentation")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Troubleshooting cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error during troubleshooting: {e}")
        import traceback
        traceback.print_exc()