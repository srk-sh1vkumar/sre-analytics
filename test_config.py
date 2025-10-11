#!/usr/bin/env python3
"""
Test script for centralized configuration system
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.app_config import get_config, Config

def test_config():
    """Test configuration loading and validation"""

    print("=" * 60)
    print("Testing Centralized Configuration System")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = get_config()
    print("   ✓ Configuration loaded successfully")

    # Test AppDynamics config
    print("\n2. AppDynamics Configuration:")
    print(f"   Controller Host: {config.appdynamics.controller_host or '(not set)'}")
    print(f"   Client ID: {'***' if config.appdynamics.client_id else '(not set)'}")
    print(f"   Client Secret: {'***' if config.appdynamics.client_secret else '(not set)'}")
    print(f"   Account: {config.appdynamics.account or '(not set)'}")
    print(f"   Access Key: {'***' if config.appdynamics.access_key else '(not set)'}")
    print(f"   Primary App: {config.appdynamics.primary_app or '(not set)'}")

    # Test LLM config
    print("\n3. LLM Configuration:")
    print(f"   OpenAI API Key: {'✓ Configured' if config.llm.openai_api_key else '✗ Not configured'}")
    print(f"   Anthropic API Key: {'✓ Configured' if config.llm.anthropic_api_key else '✗ Not configured'}")

    # Test Report config
    print("\n4. Report Configuration:")
    print(f"   Output Path: {config.report.output_path}")

    # Test Flask config
    print("\n5. Flask Configuration:")
    print(f"   Secret Key: {'✓ Set' if config.flask.secret_key else '✗ Not set'}")
    print(f"   Environment: {config.flask.env}")

    # Test System config
    print("\n6. System Configuration:")
    print(f"   PKG_CONFIG_PATH: {config.system.pkg_config_path or '(not set)'}")
    print(f"   DYLD_LIBRARY_PATH: {config.system.dyld_library_path or '(not set)'}")

    # Validation
    print("\n7. Configuration Validation:")
    validation = config.validate()
    for section, is_valid in validation.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"   {section}: {status}")

    # Get validation errors
    errors = config.get_validation_errors()
    if errors:
        print("\n8. Validation Errors:")
        for error in errors:
            print(f"   ⚠️  {error}")
    else:
        print("\n8. Validation Errors: None")

    # Test to_dict() method (masks secrets)
    print("\n9. Configuration Dictionary (secrets masked):")
    config_dict = config.to_dict()
    import json
    print(json.dumps(config_dict, indent=2))

    print("\n" + "=" * 60)
    print("Configuration Test Complete!")
    print("=" * 60)

    return config

if __name__ == "__main__":
    try:
        config = test_config()
        print("\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
