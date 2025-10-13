"""
Tests for API Authentication and Authorization

Tests API key management, validation, and rate limiting.
"""

import pytest
from datetime import datetime, timedelta
from src.api.auth import (
    APIKeyManager, APIKey, RateLimiter,
    Role, has_permission
)


@pytest.fixture
def key_manager():
    """Create fresh API key manager"""
    return APIKeyManager()


@pytest.fixture
def rate_limiter():
    """Create fresh rate limiter"""
    return RateLimiter()


class TestAPIKeyManager:
    """Test API key management"""

    def test_generate_key(self, key_manager):
        """Test generating new API key"""
        raw_key, api_key = key_manager.generate_key(
            name="Test Key",
            role=Role.READ,
            rate_limit=100
        )

        assert raw_key.startswith("sre_")
        assert len(raw_key) > 20
        assert api_key.name == "Test Key"
        assert api_key.role == Role.READ
        assert api_key.rate_limit == 100
        assert api_key.enabled is True
        assert isinstance(api_key.created_at, datetime)

    def test_validate_key_success(self, key_manager):
        """Test validating valid API key"""
        raw_key, created_key = key_manager.generate_key("Test", Role.READ)

        validated_key = key_manager.validate_key(raw_key)

        assert validated_key is not None
        assert validated_key.key_id == created_key.key_id
        assert validated_key.name == "Test"
        assert validated_key.last_used is not None

    def test_validate_key_invalid(self, key_manager):
        """Test validating invalid API key"""
        validated_key = key_manager.validate_key("sre_invalid_key")

        assert validated_key is None

    def test_validate_key_wrong_prefix(self, key_manager):
        """Test validating key with wrong prefix"""
        validated_key = key_manager.validate_key("wrong_prefix_key")

        assert validated_key is None

    def test_revoke_key(self, key_manager):
        """Test revoking API key"""
        raw_key, api_key = key_manager.generate_key("Test", Role.READ)

        # Should work before revocation
        validated = key_manager.validate_key(raw_key)
        assert validated is not None

        # Revoke
        success = key_manager.revoke_key(api_key.key_id)
        assert success is True

        # Should not work after revocation
        validated = key_manager.validate_key(raw_key)
        assert validated is None

    def test_revoke_nonexistent_key(self, key_manager):
        """Test revoking non-existent key"""
        success = key_manager.revoke_key("nonexistent_id")
        assert success is False

    def test_list_keys(self, key_manager):
        """Test listing all API keys"""
        key_manager.generate_key("Key 1", Role.READ)
        key_manager.generate_key("Key 2", Role.WRITE)
        key_manager.generate_key("Key 3", Role.ADMIN)

        keys = key_manager.list_keys()

        assert len(keys) == 3
        assert all("key_id" in k for k in keys)
        assert all("name" in k for k in keys)
        assert all("role" in k for k in keys)
        # Should not expose key_hash
        assert all("key_hash" not in k for k in keys)

    def test_get_key_by_id(self, key_manager):
        """Test getting key by ID"""
        raw_key, api_key = key_manager.generate_key("Test", Role.READ)

        found_key = key_manager.get_key_by_id(api_key.key_id)

        assert found_key is not None
        assert found_key.key_id == api_key.key_id
        assert found_key.name == "Test"

    def test_get_key_by_id_not_found(self, key_manager):
        """Test getting non-existent key by ID"""
        found_key = key_manager.get_key_by_id("nonexistent_id")
        assert found_key is None

    def test_different_roles(self, key_manager):
        """Test creating keys with different roles"""
        read_key, read_obj = key_manager.generate_key("Read", Role.READ)
        write_key, write_obj = key_manager.generate_key("Write", Role.WRITE)
        admin_key, admin_obj = key_manager.generate_key("Admin", Role.ADMIN)

        assert read_obj.role == Role.READ
        assert write_obj.role == Role.WRITE
        assert admin_obj.role == Role.ADMIN

    def test_custom_metadata(self, key_manager):
        """Test adding custom metadata to keys"""
        metadata = {"team": "platform", "environment": "production"}
        raw_key, api_key = key_manager.generate_key(
            "Test",
            Role.READ,
            metadata=metadata
        )

        assert api_key.metadata == metadata


class TestRateLimiter:
    """Test rate limiting"""

    def test_check_rate_limit_within_limit(self, rate_limiter):
        """Test requests within rate limit"""
        raw_key, api_key = APIKeyManager().generate_key("Test", Role.READ, rate_limit=5)

        # First 5 requests should succeed
        for i in range(5):
            allowed, info = rate_limiter.check_rate_limit(api_key)
            assert allowed is True
            assert info["allowed"] is True
            assert info["limit"] == 5
            assert info["remaining"] == 5 - i - 1

    def test_check_rate_limit_exceeded(self, rate_limiter):
        """Test requests exceeding rate limit"""
        raw_key, api_key = APIKeyManager().generate_key("Test", Role.READ, rate_limit=3)

        # First 3 requests should succeed
        for _ in range(3):
            allowed, info = rate_limiter.check_rate_limit(api_key)
            assert allowed is True

        # 4th request should fail
        allowed, info = rate_limiter.check_rate_limit(api_key)
        assert allowed is False
        assert info["allowed"] is False
        assert info["current"] == 3
        assert "reset_in" in info

    def test_get_rate_limit_info(self, rate_limiter):
        """Test getting rate limit info without checking"""
        raw_key, api_key = APIKeyManager().generate_key("Test", Role.READ, rate_limit=10)

        # Get info without making request
        info = rate_limiter.get_rate_limit_info(api_key)

        assert info["limit"] == 10
        assert info["remaining"] == 10
        assert info["reset_in"] == 60

        # Make 3 requests
        for _ in range(3):
            rate_limiter.check_rate_limit(api_key)

        # Check info again
        info = rate_limiter.get_rate_limit_info(api_key)
        assert info["remaining"] == 7  # 10 - 3

    def test_rate_limit_window_reset(self, rate_limiter):
        """Test rate limit window reset"""
        raw_key, api_key = APIKeyManager().generate_key("Test", Role.READ, rate_limit=2)

        # Use up rate limit
        rate_limiter.check_rate_limit(api_key)
        rate_limiter.check_rate_limit(api_key)

        # Should be rate limited
        allowed, _ = rate_limiter.check_rate_limit(api_key)
        assert allowed is False

        # Note: Cannot easily test time-based reset without mocking time


class TestPermissions:
    """Test role-based permissions"""

    def test_has_permission_read(self):
        """Test READ role permissions"""
        key_manager = APIKeyManager()
        raw_key, api_key = key_manager.generate_key("Test", Role.READ)

        assert has_permission(api_key, Role.READ) is True
        assert has_permission(api_key, Role.WRITE) is False
        assert has_permission(api_key, Role.ADMIN) is False

    def test_has_permission_write(self):
        """Test WRITE role permissions"""
        key_manager = APIKeyManager()
        raw_key, api_key = key_manager.generate_key("Test", Role.WRITE)

        assert has_permission(api_key, Role.READ) is True
        assert has_permission(api_key, Role.WRITE) is True
        assert has_permission(api_key, Role.ADMIN) is False

    def test_has_permission_admin(self):
        """Test ADMIN role permissions"""
        key_manager = APIKeyManager()
        raw_key, api_key = key_manager.generate_key("Test", Role.ADMIN)

        assert has_permission(api_key, Role.READ) is True
        assert has_permission(api_key, Role.WRITE) is True
        assert has_permission(api_key, Role.ADMIN) is True

    def test_permission_hierarchy(self):
        """Test role hierarchy"""
        key_manager = APIKeyManager()

        # Create keys with different roles
        read_key, read_obj = key_manager.generate_key("Read", Role.READ)
        write_key, write_obj = key_manager.generate_key("Write", Role.WRITE)
        admin_key, admin_obj = key_manager.generate_key("Admin", Role.ADMIN)

        # READ can only do READ operations
        assert has_permission(read_obj, Role.READ) is True
        assert has_permission(read_obj, Role.WRITE) is False

        # WRITE can do READ and WRITE
        assert has_permission(write_obj, Role.READ) is True
        assert has_permission(write_obj, Role.WRITE) is True
        assert has_permission(write_obj, Role.ADMIN) is False

        # ADMIN can do everything
        assert has_permission(admin_obj, Role.READ) is True
        assert has_permission(admin_obj, Role.WRITE) is True
        assert has_permission(admin_obj, Role.ADMIN) is True
