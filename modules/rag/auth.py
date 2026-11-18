"""JWT verification dependency — consumes the same tokens issued by the
Pyramid `api` service (pyramid_jwt, HS512, shared JWT_SECRET).

Authorization header format: `Authorization: JWT <token>` (not Bearer).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, status


ALGORITHM = "HS512"

# pyramid_jwt (api side) stores the numeric user ID in `sub`, but PyJWT >= 2
# enforces RFC 7519's "sub must be string" rule. We skip that one validation
# — signature + `client` scoping are the real guarantees we need.
DECODE_OPTIONS = {"verify_sub": False}


@dataclass
class JwtClaims:
    client: str
    login: str
    raw: dict


def _secret() -> str:
    s = os.environ.get("JWT_SECRET")
    if not s:
        raise RuntimeError("JWT_SECRET is not set in the mtia container environment")
    return s


def verify_jwt(authorization: str | None = Header(default=None)) -> JwtClaims:
    """FastAPI dependency: decode + validate `Authorization: JWT <token>`."""
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.upper() != "JWT" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "expected 'Authorization: JWT <token>'")

    try:
        claims = jwt.decode(token, _secret(), algorithms=[ALGORITHM], options=DECODE_OPTIONS)
    except jwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}")

    client = claims.get("client")
    login = claims.get("login")
    if not client or not login:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token missing client/login claim")

    return JwtClaims(client=client, login=login, raw=claims)


def require_client_match(requested_client: str, claims: JwtClaims) -> None:
    """Reject cross-tenant access: the client in the URL/body must match the token."""
    if requested_client != claims.client:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"token is scoped to client={claims.client!r}, cannot access {requested_client!r}")


def require_feature(name: str, detail: str | None = None):
    """FastAPI dependency factory: the token must carry the client flag `name`.

    Client flags are claims the v3 `api` puts in the token at login
    (`api/features.py`): a boolean key of `j_client.data`, or one derived from
    the client's commercial plan — `plantAgent` comes with Ultimate AI
    (MTS-2613). Only a literal `true` counts. A plan change takes effect on
    the user's next login, because the claim lives in the token.

    Returns the verified claims, so an endpoint needs this dependency alone.
    """
    def dependency(claims: JwtClaims = Depends(verify_jwt)) -> JwtClaims:
        if claims.raw.get(name) is not True:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail or f"token for client={claims.client!r} lacks the {name!r} feature")
        return claims

    return dependency
