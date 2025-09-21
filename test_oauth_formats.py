#!/usr/bin/env python3
"""
Test different OAuth formats for AppDynamics authentication
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

print("🧪 Testing Different OAuth Client ID Formats")
print("=" * 60)

# Test different client ID formats for BNY Mellon
test_formats = [
    f"{client_id}",                           # api_access
    f"{client_id}@bny-ucf",                   # api_access@bny-ucf
    f"{client_id}@bny",                       # api_access@bny
    f"{client_id}@customer1",                 # api_access@customer1
    f"bny-ucf\\{client_id}",                  # bny-ucf\api_access
    f"bny\\{client_id}",                      # bny\api_access
]

for i, test_client_id in enumerate(test_formats, 1):
    print(f"\n🔑 Test {i}: Client ID = '{test_client_id}'")

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'client_credentials',
        'client_id': test_client_id,
        'client_secret': client_secret
    }

    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=30)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            print(f"   ✅ SUCCESS! OAuth token obtained")
            try:
                token_data = response.json()
                print(f"   Access Token: {token_data.get('access_token', 'N/A')[:20]}...")
                print(f"   Token Type: {token_data.get('token_type', 'N/A')}")
                print(f"   Expires In: {token_data.get('expires_in', 'N/A')} seconds")
                break
            except:
                print(f"   ✅ SUCCESS! Response: {response.text[:100]}...")
                break
        elif response.status_code == 401:
            print(f"   ❌ Unauthorized - trying next format")
        else:
            print(f"   ⚠️  Unexpected status: {response.text[:100]}...")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("🔍 Additional OAuth Troubleshooting")
print("=" * 60)

# Test if we can access the OAuth endpoint at all
print("\n1. Testing OAuth endpoint accessibility...")
try:
    response = requests.get(f"{controller_url}/controller/api/oauth", timeout=10)
    print(f"   GET /controller/api/oauth: {response.status_code}")
    if response.status_code == 405:
        print("   ✅ Endpoint exists (405 = Method Not Allowed for GET)")
    elif response.status_code == 401:
        print("   ⚠️  Endpoint requires authentication")
    else:
        print(f"   Response: {response.text[:200]}...")
except Exception as e:
    print(f"   ❌ Error accessing OAuth endpoint: {e}")

# Test for AppDynamics version/info
print("\n2. Testing AppDynamics version info...")
try:
    response = requests.get(f"{controller_url}/controller/rest/serverstatus", timeout=10)
    print(f"   Server status endpoint: {response.status_code}")
    if response.status_code == 200:
        print(f"   Server appears to be running")
except Exception as e:
    print(f"   Could not get server status: {e}")

print("\n💡 Recommendations:")
print("1. Contact BNY Mellon AppDynamics admin to:")
print("   - Verify OAuth is enabled")
print("   - Confirm correct client_id format")
print("   - Check if account name is required")
print("   - Verify client permissions")
print("2. Try username/password authentication if OAuth not available")
print("3. Check AppDynamics documentation for your specific version")