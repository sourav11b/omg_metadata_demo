"""
Atlas Admin API authentication helper.

Supports both authentication methods:
  • Service Account (OAuth2 Client Credentials) — keys start with mdb_sa_id_ / mdb_sa_sk_
  • Programmatic API Key (HTTP Digest Auth) — legacy public/private key pair
"""

import time
import requests
from requests.auth import HTTPDigestAuth, AuthBase

from config.settings import ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY

_OAUTH_TOKEN_URL = "https://cloud.mongodb.com/api/oauth/token"

# ── Cached OAuth2 token state ─────────────────────────────────────────────────
_oauth_token: str | None = None
_oauth_expires_at: float = 0.0


def _is_service_account() -> bool:
    """Detect whether the credentials are a Service Account (OAuth2)."""
    return ATLAS_PUBLIC_KEY.startswith("mdb_sa_id_")


def _get_oauth_token() -> str:
    """Obtain or refresh an OAuth2 bearer token using client credentials."""
    global _oauth_token, _oauth_expires_at

    if _oauth_token and time.time() < _oauth_expires_at - 30:
        return _oauth_token

    resp = requests.post(
        _OAUTH_TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(_ATLAS_PUBLIC_KEY_raw(), _ATLAS_PRIVATE_KEY_raw()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    data = resp.json()
    _oauth_token = data["access_token"]
    _oauth_expires_at = time.time() + data.get("expires_in", 3600)
    return _oauth_token


def _ATLAS_PUBLIC_KEY_raw() -> str:
    return ATLAS_PUBLIC_KEY


def _ATLAS_PRIVATE_KEY_raw() -> str:
    return ATLAS_PRIVATE_KEY


# ── Bearer Auth adapter for requests ──────────────────────────────────────────

class _BearerAuth(AuthBase):
    """Attaches a fresh OAuth2 bearer token to every request."""

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {_get_oauth_token()}"
        return r


# ── Public API ────────────────────────────────────────────────────────────────

def get_atlas_auth():
    """Return the correct auth object for the Atlas Admin API."""
    if _is_service_account():
        return _BearerAuth()
    return HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
