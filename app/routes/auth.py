from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth import User, create_access_token, get_current_user, get_oauth_provider

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthorizeUrlResponse(BaseModel):
    authorize_url: str
    state: str


class CallbackRequest(BaseModel):
    code: str
    redirect_uri: str = "http://localhost:8000/auth/callback"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.get("/login", response_model=AuthorizeUrlResponse)
def login(redirect_uri: str = "http://localhost:8000/auth/callback") -> AuthorizeUrlResponse:
    provider = get_oauth_provider()
    state = secrets.token_urlsafe(16)
    return AuthorizeUrlResponse(authorize_url=provider.build_authorize_url(redirect_uri, state), state=state)


@router.post("/callback", response_model=TokenResponse)
def callback(payload: CallbackRequest) -> TokenResponse:
    provider = get_oauth_provider()
    user = provider.exchange_code_for_user(payload.code, payload.redirect_uri)
    return TokenResponse(access_token=create_access_token(user))


@router.post("/dev-token", response_model=TokenResponse)
def dev_token() -> TokenResponse:
    """Shortcut for local dev/tests with the default 'fake' auth provider --
    skips the redirect round-trip entirely and issues a session directly."""
    provider = get_oauth_provider()
    user = provider.exchange_code_for_user(code="dev-code", redirect_uri="http://localhost")
    return TokenResponse(access_token=create_access_token(user))


@router.get("/me", response_model=User)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
