#!/usr/bin/env python3
"""
Test Alternative AppDynamics Authentication Methods
"""

import requests
import json
import base64
import os
from dotenv import load_dotenv

load_dotenv()

def test_basic_auth():
    """Test with basic authentication (username/password)"""
    print("🔐 Testing Basic Authentication")
    print("=" * 40)

    # For basic auth, you might need username@account:password
    controller_url = f"https://{os.getenv('APPDYNAMICS_CONTROLLER_HOST')}"

    # Try different authentication formats
    auth_formats = [
        f"{os.getenv('APPDYNAMICS_CLIENT_ID')}:{os.getenv('APPDYNAMICS_CLIENT_SECRET')}",
        f"api_access@bny-ucf:{os.getenv('APPDYNAMICS_CLIENT_SECRET')}",
        f"{os.getenv('APPDYNAMICS_CLIENT_ID')}@bny-ucf:{os.getenv('APPDYNAMICS_CLIENT_SECRET')}"
    ]

    for i, auth_string in enumerate(auth_formats, 1):
        print(f"\nTrying format {i}: {auth_string.split(':')[0]}:****")

        # Create basic auth header
        encoded_credentials = base64.b64encode(auth_string.encode()).decode()
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.get(
                f"{controller_url}/controller/rest/applications",
                headers=headers,
                timeout=10
            )

            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                apps = response.json()
                print(f"   ✅ Success! Found {len(apps)} applications")
                return True
            elif response.status_code == 401:
                print("   ❌ Still unauthorized")
            else:
                print(f"   ⚠️ Unexpected status: {response.text[:100]}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False

def test_oauth_variations():
    """Test OAuth with different configurations"""
    print("\n🎫 Testing OAuth Variations")
    print("=" * 40)

    controller_url = f"https://{os.getenv('APPDYNAMICS_CONTROLLER_HOST')}"
    token_url = f"{controller_url}/controller/api/oauth/access_token"

    # Try different content types and data formats
    variations = [
        {
            'name': 'Form-encoded with Basic Auth',
            'headers': {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': f"Basic {base64.b64encode((os.getenv('APPDYNAMICS_CLIENT_ID') + ':' + os.getenv('APPDYNAMICS_CLIENT_SECRET')).encode()).decode()}"
            },
            'data': 'grant_type=client_credentials'
        },
        {
            'name': 'JSON with credentials in body',
            'headers': {
                'Content-Type': 'application/json'
            },
            'data': json.dumps({
                'grant_type': 'client_credentials',
                'client_id': os.getenv('APPDYNAMICS_CLIENT_ID'),
                'client_secret': os.getenv('APPDYNAMICS_CLIENT_SECRET')
            })
        },
        {
            'name': 'Form-encoded with client scope',
            'headers': {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            'data': f"grant_type=client_credentials&client_id={os.getenv('APPDYNAMICS_CLIENT_ID')}&client_secret={os.getenv('APPDYNAMICS_CLIENT_SECRET')}&scope=read"
        }
    ]

    for i, variation in enumerate(variations, 1):
        print(f"\nTrying variation {i}: {variation['name']}")

        try:
            response = requests.post(
                token_url,
                headers=variation['headers'],
                data=variation['data'],
                timeout=10
            )

            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")

            if response.status_code == 200:
                try:
                    token_data = response.json()
                    print(f"   ✅ Success! Got token: {token_data.get('access_token', 'No token')[:20]}...")
                    return token_data
                except:
                    print(f"   ⚠️ Success but non-JSON response: {response.text[:100]}")
            else:
                print(f"   ❌ Failed: {response.text[:200]}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return None

def test_api_versions():
    """Test different API versions"""
    print("\n🔄 Testing Different API Versions")
    print("=" * 40)

    controller_url = f"https://{os.getenv('APPDYNAMICS_CONTROLLER_HOST')}"

    # Try different API endpoints
    endpoints = [
        "/controller/api/oauth/access_token",
        "/controller/api/v1/oauth/access_token",
        "/controller/oauth/access_token",
        "/controller/auth/oauth/access_token"
    ]

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'client_credentials',
        'client_id': os.getenv('APPDYNAMICS_CLIENT_ID'),
        'client_secret': os.getenv('APPDYNAMICS_CLIENT_SECRET')
    }

    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")

        try:
            response = requests.post(
                f"{controller_url}{endpoint}",
                headers=headers,
                data=data,
                timeout=10
            )

            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                print(f"   ✅ This endpoint works!")
                return endpoint
            elif response.status_code == 404:
                print(f"   ❌ Endpoint not found")
            else:
                print(f"   ⚠️ Different error: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return None

def main():
    """Run all authentication tests"""
    print("🧪 AppDynamics Alternative Authentication Tests")
    print("=" * 60)

    # Test basic authentication
    basic_auth_success = test_basic_auth()

    if basic_auth_success:
        print("\n✅ Basic authentication works! Use this instead of OAuth.")
        return

    # Test OAuth variations
    oauth_token = test_oauth_variations()

    if oauth_token:
        print("\n✅ OAuth variation works!")
        return

    # Test different API versions
    working_endpoint = test_api_versions()

    if working_endpoint:
        print(f"\n✅ Found working OAuth endpoint: {working_endpoint}")
        return

    print("\n❌ All authentication methods failed.")
    print("\n🔧 Final recommendations:")
    print("1. Contact AppDynamics administrator to verify OAuth setup")
    print("2. Check if OAuth is enabled on the controller")
    print("3. Verify client credentials in AppDynamics admin panel")
    print("4. Try using username/password authentication instead")
    print("5. Check AppDynamics version compatibility")

if __name__ == "__main__":
    main()