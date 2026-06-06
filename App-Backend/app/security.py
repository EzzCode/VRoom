"""
API key authentication dependency.

The key is loaded from the ``VROOM_API_KEY`` environment variable (via
``settings.api_key``).  If the variable is absent at startup, an ephemeral
random key is generated and logged as a warning.

TODO(security): Migrate to OAuth 2.0 / JWT for production multi-tenant use.
TODO(security): Add rate limiting middleware before production deployment.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer_scheme = HTTPBearer()


async def require_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """Validate the bearer token against the configured API key."""
    if credentials.credentials != settings.vroom_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return credentials.credentials
