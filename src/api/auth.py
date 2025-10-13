"""
Authentication and Authorization for API

Provides API key-based authentication with role-based access control.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    """API access roles"""
    ADMIN = "admin"  # Full access
    WRITE = "write"  # Read + Write operations
    READ = "read"  # Read-only access


@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    key_hash: str
    name: str
    role: Role
    created_at: datetime
    last_used: Optional[datetime] = None
    rate_limit: int = 100  # Requests per minute
    enabled: bool = True
    metadata: Dict = field(default_factory=dict)


class APIKeyManager:
    """
    Manage API keys for authentication

    In production, this should be backed by a database.
    For now, uses in-memory storage.
    """

    def __init__(self):
        self.keys: Dict[str, APIKey] = {}
        self._key_prefix = "sre_"

    def generate_key(
        self,
        name: str,
        role: Role = Role.READ,
        rate_limit: int = 100,
        metadata: Optional[Dict] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key

        Args:
            name: Descriptive name for the key
            role: Access role (ADMIN, WRITE, READ)
            rate_limit: Requests per minute limit
            metadata: Optional metadata dict

        Returns:
            Tuple of (raw_key, api_key_object)
        """
        # Generate random key
        raw_key = f"{self._key_prefix}{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        key_id = secrets.token_hex(8)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            role=role,
            created_at=datetime.now(),
            rate_limit=rate_limit,
            enabled=True,
            metadata=metadata or {}
        )

        self.keys[key_hash] = api_key
        return raw_key, api_key

    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key

        Args:
            raw_key: Raw API key string

        Returns:
            APIKey object if valid, None otherwise
        """
        if not raw_key.startswith(self._key_prefix):
            return None

        key_hash = self._hash_key(raw_key)
        api_key = self.keys.get(key_hash)

        if api_key and api_key.enabled:
            # Update last used timestamp
            api_key.last_used = datetime.now()
            return api_key

        return None

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key by ID

        Args:
            key_id: API key ID to revoke

        Returns:
            True if revoked, False if not found
        """
        for api_key in self.keys.values():
            if api_key.key_id == key_id:
                api_key.enabled = False
                return True
        return False

    def list_keys(self) -> List[Dict]:
        """
        List all API keys (without hashes)

        Returns:
            List of API key metadata
        """
        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "role": key.role.value,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used.isoformat() if key.last_used else None,
                "rate_limit": key.rate_limit,
                "enabled": key.enabled,
                "metadata": key.metadata
            }
            for key in self.keys.values()
        ]

    def get_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """
        Get API key by ID

        Args:
            key_id: API key ID

        Returns:
            APIKey object if found, None otherwise
        """
        for api_key in self.keys.values():
            if api_key.key_id == key_id:
                return api_key
        return None

    def _hash_key(self, raw_key: str) -> str:
        """Hash API key using SHA256"""
        return hashlib.sha256(raw_key.encode()).hexdigest()


class RateLimiter:
    """
    Rate limiter for API requests

    Tracks requests per API key with sliding window.
    """

    def __init__(self):
        # key_hash -> list of timestamps
        self.requests: Dict[str, List[float]] = {}

    def check_rate_limit(self, api_key: APIKey) -> tuple[bool, Dict]:
        """
        Check if request is within rate limit

        Args:
            api_key: API key to check

        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        key_hash = api_key.key_hash
        now = datetime.now().timestamp()
        window_start = now - 60  # 1 minute window

        # Initialize if not exists
        if key_hash not in self.requests:
            self.requests[key_hash] = []

        # Remove old requests outside window
        self.requests[key_hash] = [
            ts for ts in self.requests[key_hash]
            if ts > window_start
        ]

        current_count = len(self.requests[key_hash])
        limit = api_key.rate_limit

        if current_count >= limit:
            return False, {
                "allowed": False,
                "limit": limit,
                "current": current_count,
                "reset_in": int(self.requests[key_hash][0] + 60 - now)
            }

        # Add current request
        self.requests[key_hash].append(now)

        return True, {
            "allowed": True,
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset_in": 60
        }

    def get_rate_limit_info(self, api_key: APIKey) -> Dict:
        """
        Get current rate limit status without checking

        Args:
            api_key: API key to check

        Returns:
            Rate limit info dict
        """
        key_hash = api_key.key_hash
        now = datetime.now().timestamp()
        window_start = now - 60

        if key_hash not in self.requests:
            return {
                "limit": api_key.rate_limit,
                "remaining": api_key.rate_limit,
                "reset_in": 60
            }

        # Count requests in current window
        recent_requests = [
            ts for ts in self.requests[key_hash]
            if ts > window_start
        ]

        current_count = len(recent_requests)
        remaining = max(0, api_key.rate_limit - current_count)

        reset_in = 60
        if recent_requests:
            reset_in = int(recent_requests[0] + 60 - now)

        return {
            "limit": api_key.rate_limit,
            "remaining": remaining,
            "reset_in": reset_in
        }


# Global instances (in production, use dependency injection)
key_manager = APIKeyManager()
rate_limiter = RateLimiter()


def has_permission(api_key: APIKey, required_role: Role) -> bool:
    """
    Check if API key has required permission level

    Args:
        api_key: API key to check
        required_role: Minimum required role

    Returns:
        True if has permission, False otherwise
    """
    role_hierarchy = {
        Role.READ: 1,
        Role.WRITE: 2,
        Role.ADMIN: 3
    }

    return role_hierarchy[api_key.role] >= role_hierarchy[required_role]
