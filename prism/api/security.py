"""API authentication for the PRISM REST service.

Implements simple bearer-token auth backed by the ``PRISM_API_KEYS`` setting.
In production this should be replaced/augmented with an OAuth2 / JWT provider,
but the dependency keeps the surface minimal and secure-by-default (every data
endpoint requires a valid key).
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from prism.config import get_settings

_bearer = HTTPBearer(auto_error=True)


def _is_valid_key(candidate: str) -> bool:
    # Constant-time comparison against each configured key.
    return any(hmac.compare_digest(candidate, key) for key in get_settings().api_keys)


def require_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """FastAPI dependency that validates the bearer token."""
    token = credentials.credentials
    if not _is_valid_key(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
