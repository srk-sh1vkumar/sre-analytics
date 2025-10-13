#!/usr/bin/env python3
"""
Start SRE Analytics API Server

Starts FastAPI server with uvicorn and initializes default admin API key.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.app import app
from src.api.auth import key_manager, Role
import uvicorn


def initialize_default_keys():
    """Create default API keys for testing"""
    print("🔑 Initializing API keys...")

    # Create admin key
    admin_key, admin_obj = key_manager.generate_key(
        name="Default Admin Key",
        role=Role.ADMIN,
        rate_limit=1000,
        metadata={"created_by": "system", "purpose": "default_admin"}
    )
    print(f"   ✅ Admin API Key: {admin_key}")
    print(f"      Key ID: {admin_obj.key_id}")
    print(f"      Rate Limit: 1000 req/min")

    # Create read-only key for testing
    read_key, read_obj = key_manager.generate_key(
        name="Default Read-Only Key",
        role=Role.READ,
        rate_limit=100,
        metadata={"created_by": "system", "purpose": "default_readonly"}
    )
    print(f"   ✅ Read-Only API Key: {read_key}")
    print(f"      Key ID: {read_obj.key_id}")
    print(f"      Rate Limit: 100 req/min")

    print()
    print("⚠️  Save these API keys! They cannot be retrieved later.")
    print()


def main():
    """Start API server"""
    print("=" * 70)
    print("SRE ANALYTICS API SERVER")
    print("=" * 70)
    print()

    # Initialize default API keys
    initialize_default_keys()

    # Server configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print(f"🚀 Starting API server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print(f"   ReDoc: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/redoc")
    print()

    # Start uvicorn server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
