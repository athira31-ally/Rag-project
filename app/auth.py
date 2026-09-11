"""Google OAuth2 login + JWT session issuance.

Real Google OAuth needs a browser round-trip and live credentials, which
isn't something a test suite (or a reviewer without your Google client
secret) can exercise. So, same pattern as the rest of the repo: a real
GoogleOAuthProvider for production, and a deterministic FakeOAuthProvider
(default) that issues a session for a fixed dev user without any network
call -- useful for local development and for tests that need an
authenticated request without standing up real OAuth.
"""
from __future__ import annotations

import time
from typing import Protocol

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.config import settings

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/dev-token", auto_error=False)


class User(BaseModel):
    sub: str
    email: str
    name: str


class OAuthProvider(Protocol):
    def build_authorize_url(self, redirect_uri: str, state: str) -> str: ...

    def exchange_code_for_user(self, code: str, redirect_uri: str) -> User: ...


class GoogleOAuthProvider:
    _AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    _TOKEN_URL = "https://oauth2.googleapis.com/token"
    _USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

    def __init__(self, client_id: str | None, client_secret: str | None):
        if not client_id or not client_secret:
            raise RuntimeError("GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET are required for the 'google' auth provider")
        self._client_id = client_id
        self._client_secret = client_secret

    def build_authorize_url(self, redirect_uri: str, state: str) -> str:
        from urllib.parse import urlencode

        params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
        }
        return f"{self._AUTH_URL}?{urlencode(params)}"

    def exchange_code_for_user(self, code: str, redirect_uri: str) -> User:
        import httpx

        with httpx.Client(timeout=10.0) as client:
            token_response = client.post(
                self._TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]

            userinfo_response = client.get(
                self._USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}
            )
            userinfo_response.raise_for_status()
            info = userinfo_response.json()

        return User(sub=info["sub"], email=info["email"], name=info.get("name", info["email"]))


class FakeOAuthProvider:
    """Deterministic dev/test provider -- no network call, no real Google
    credentials required. `exchange_code_for_user` accepts any code and
    returns a fixed user, so /auth flows are fully testable offline."""

    def build_authorize_url(self, redirect_uri: str, state: str) -> str:
        return f"/auth/callback?code=dev-code&state={state}&redirect_uri={redirect_uri}"

    def exchange_code_for_user(self, code: str, redirect_uri: str) -> User:
        return User(sub="dev-user-1", email="dev@example.com", name="Dev User")


def get_oauth_provider() -> OAuthProvider:
    if settings.auth_provider == "fake":
        return FakeOAuthProvider()
    if settings.auth_provider == "google":
        return GoogleOAuthProvider(client_id=settings.google_client_id, client_secret=settings.google_client_secret)
    raise ValueError(f"Unknown AUTH_PROVIDER: {settings.auth_provider!r}")


def create_access_token(user: User) -> str:
    now = int(time.time())
    payload = {
        "sub": user.sub,
        "email": user.email,
        "name": user.name,
        "iat": now,
        "exp": now + settings.jwt_expire_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> User:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc
    return User(sub=payload["sub"], email=payload["email"], name=payload["name"])


def get_current_user(token: str | None = Depends(_oauth2_scheme)) -> User:
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return decode_access_token(token)
