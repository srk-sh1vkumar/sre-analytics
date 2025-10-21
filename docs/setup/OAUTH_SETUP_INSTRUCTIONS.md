# 🔐 AppDynamics OAuth Setup Guide

## Current Issue
OAuth authentication is failing with 401 Unauthorized errors. The credentials need to be properly configured in AppDynamics.

## 🎯 Step-by-Step Solution

### Step 1: Access AppDynamics Controller
1. Navigate to: https://bny-ucf.saas.appdynamics.com/controller
2. Log in with your AppDynamics administrator credentials

### Step 2: Configure OAuth API Client

#### Navigate to API Clients
1. Go to **Administration** → **API Clients**
2. Click **"Create API Client"** or **"New"**

#### Create OAuth Client
Fill in the following details:
- **Client Name:** `SRE Analytics System`
- **Client ID:** `sre_analytics_client` (or your preferred ID)
- **Client Type:** `Confidential`
- **Grant Types:**
  - ✅ `Client Credentials`
- **Scopes/Permissions:** Select the following:
  - ✅ `applications:read`
  - ✅ `metrics:read`
  - ✅ `analytics:read`
  - ✅ `events:read`
  - ✅ `dashboards:read`
  - ✅ `alerts:read`

#### Generate Client Secret
1. After creating the client, click **"Generate Secret"**
2. **IMPORTANT:** Copy both the Client ID and Client Secret immediately
3. The secret will only be shown once

### Step 3: Update Environment Variables

Once you have the new credentials, update your `.env` file:

```bash
# AppDynamics OAuth Configuration (UPDATE THESE)
APPDYNAMICS_CONTROLLER_HOST=bny-ucf.saas.appdynamics.com
APPDYNAMICS_CLIENT_ID=your_new_client_id_here
APPDYNAMICS_CLIENT_SECRET=your_new_client_secret_here
```

### Step 4: Test OAuth Connection

Run the troubleshooting tool to verify:
```bash
python3 troubleshoot_appdynamics.py
```

## 🔄 Alternative: Username/Password Authentication

If OAuth setup is not possible, you can use username/password authentication:

```bash
# Alternative: Basic Authentication
# APPDYNAMICS_USERNAME=your_username@account_name
# APPDYNAMICS_PASSWORD=your_password
```

## 🚨 Common Issues & Solutions

### Issue 1: "OAuth not enabled"
**Solution:** Contact AppDynamics administrator to enable OAuth on the controller

### Issue 2: "Insufficient permissions"
**Solution:** Ensure the OAuth client has all required API scopes

### Issue 3: "Invalid client credentials"
**Solution:**
1. Verify Client ID and Secret are correct
2. Ensure no extra spaces or characters
3. Check if client is active/enabled

### Issue 4: "Account name required"
**Solution:** Try using format `client_id@account_name`

## 📞 Need Help?

If you continue having issues:
1. Contact your AppDynamics administrator
2. Verify your AppDynamics controller version supports OAuth
3. Check AppDynamics documentation for your specific version

## ✅ Verification Steps

After configuring OAuth:
1. Run `python3 troubleshoot_appdynamics.py` - should show ✅ OAuth successful
2. Generate a report - should show "Connected to AppDynamics successfully"
3. Check reports for real metrics instead of demo data

## 🎯 Expected Success Output

When working correctly, you should see:
```
🔐 Testing AppDynamics Connection...
• OAuth Authentication: ✅
• API Access: ✅
• Applications Found: [number] > 0
• Connected to AppDynamics successfully
```
