#!/usr/bin/env python3
"""
Debug AppDynamics applications API response format
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

controller_host = os.getenv('APPDYNAMICS_CONTROLLER_HOST')
client_id = os.getenv('APPDYNAMICS_CLIENT_ID')
client_secret = os.getenv('APPDYNAMICS_CLIENT_SECRET')

controller_url = f"https://{controller_host}"
token_url = f"{controller_url}/controller/api/oauth/access_token"

print("🔍 Debugging AppDynamics Applications API Response")
print("=" * 60)

# Get OAuth token
headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
}

data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret
}

try:
    print("1. Getting OAuth token...")
    response = requests.post(token_url, headers=headers, data=data, timeout=30)

    if response.status_code != 200:
        print(f"   ❌ OAuth failed: {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)

    token_data = response.json()
    access_token = token_data['access_token']
    print(f"   ✅ OAuth token obtained")

    # Test different application API endpoints
    test_endpoints = [
        "/controller/rest/applications",
        "/controller/rest/applications?output=JSON",
        "/controller/restui/applications/getAllApplicationsData",
        "/controller/api/applications",
        "/controller/api/v1/applications"
    ]

    for endpoint in test_endpoints:
        print(f"\n2. Testing endpoint: {endpoint}")

        api_headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        try:
            api_response = requests.get(
                f"{controller_url}{endpoint}",
                headers=api_headers,
                timeout=30
            )

            print(f"   Status: {api_response.status_code}")
            print(f"   Headers: {dict(api_response.headers)}")
            print(f"   Content-Type: {api_response.headers.get('Content-Type', 'Unknown')}")
            print(f"   Content-Length: {len(api_response.text)}")
            print(f"   Response (first 200 chars): {api_response.text[:200]}")

            # Try different parsing approaches
            if api_response.status_code == 200:
                # Try JSON
                try:
                    json_data = api_response.json()
                    print(f"   ✅ JSON parsing successful")
                    print(f"   Data type: {type(json_data)}")
                    if isinstance(json_data, list):
                        print(f"   Applications found: {len(json_data)}")
                        if json_data:
                            print(f"   First app: {json_data[0] if json_data else 'None'}")
                    elif isinstance(json_data, dict):
                        print(f"   Dict keys: {list(json_data.keys())}")
                    break
                except Exception as e:
                    print(f"   ❌ JSON parsing failed: {e}")

                # Try XML
                if 'xml' in api_response.headers.get('Content-Type', '').lower():
                    print(f"   📄 Response appears to be XML")
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(api_response.text)
                        print(f"   XML root tag: {root.tag}")
                        applications = root.findall('.//application')
                        print(f"   Applications found: {len(applications)}")
                        for app in applications[:3]:  # Show first 3
                            name = app.get('name') or app.find('name')
                            if name is not None:
                                name = name.text if hasattr(name, 'text') else str(name)
                            print(f"   App: {name}")
                        if applications:
                            break
                    except Exception as e:
                        print(f"   ❌ XML parsing failed: {e}")

                # Show raw content if small enough
                if len(api_response.text) < 1000:
                    print(f"   📄 Full response: {api_response.text}")

        except Exception as e:
            print(f"   ❌ Request failed: {e}")

    print(f"\n" + "=" * 60)
    print("🔧 Summary:")
    print("- OAuth authentication is working")
    print("- Need to find the correct API endpoint and format")
    print("- Will update the collector based on findings")

except Exception as e:
    print(f"❌ Failed to get OAuth token: {e}")