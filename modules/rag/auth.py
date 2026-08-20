"""JWT verification dependency — consumes the same tokens issued by the
Pyramid `api` service (pyramid_jwt, HS512).

mtia only validates; the `api` is the issuer. Both therefore read the same
four variables and must resolve every token the same way — a mtia that
disagrees rejects tokens the `api` accepts, and the symptom that reaches
support is "the chat is down", with no visible link to the key rotation.

Authorization header format: `Authorization: JWT <token>` (not Bearer).

The key-ring (MTS-1669)
-----------------------
The `api` used to sign and validate with the literal key `secret`, which is
public. Rotating it outright would cut off every open session, so validation
walks a ring: the primary key from `JWT_SECRET`, then the legacy keys.

`JWT_LEGACY_MODE` decides how permissive the legacy step is:

* ``all`` — the general window. Any token signed with a legacy key is
  accepted. Lasts days, and while it lasts the public key can still forge.
* ``allowlist`` — only the baked tokens whose sha256 is listed. A forged token
  has a valid signature and an unknown hash, so forgery is closed even though
  the legacy key is still accepted.

Everything ambiguous falls to the safe side: an unknown mode is treated as
``allowlist``, an unreadable date as already expired, and any action other
than ``warn`` as ``enforce``.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import os
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException, status

log = logging.getLogger(__name__)

ALGORITHM = "HS512"

MODE_ALL = "all"
MODE_ALLOWLIST = "allowlist"

ENFORCE = "enforce"
WARN = "warn"

# pyramid_jwt (api side) stores the numeric user ID in `sub`, but PyJWT >= 2
# enforces RFC 7519's "sub must be string" rule. We skip that one validation
# — signature + `client` scoping are the real guarantees we need.
DECODE_OPTIONS = {"verify_sub": False}


@dataclass
class JwtClaims:
    client: str
    login: str
    raw: dict


@dataclass(frozen=True)
class AllowlistEntry:
    """Until when a baked token is admitted, and what to do once it is not."""

    expires: datetime.date | None
    action: str


def token_hash(token: str) -> str:
    """sha256 of the **bare** token, which is what the allowlist carries.

    The bare token: no `JWT ` header prefix, no trailing newline. Getting this
    wrong is invisible when configuring and takes down every baked consumer at
    once, so the `api` has a dedicated test for it (T-20) and this mirrors it.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_allowlist(raw: str | None) -> dict[str, AllowlistEntry]:
    """Parse ``hash:date[:warn]`` entries separated by commas."""
    entries: dict[str, AllowlistEntry] = {}
    for chunk in (raw or "").split(","):
        parts = [p.strip() for p in chunk.strip().split(":")]
        digest = parts[0]
        if not digest:
            continue
        try:
            expires = datetime.datetime.strptime(parts[1], "%Y-%m-%d").date()
        except (IndexError, ValueError):
            log.warning(
                "jwt_legacy_entry_unparsable hash=%s: treated as expired",
                digest[:12])
            expires = None
        action = WARN if len(parts) > 2 and parts[2] == WARN else ENFORCE
        entries[digest] = AllowlistEntry(expires=expires, action=action)
    return entries


def _split(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _keyring() -> tuple[str, list[str], str, dict[str, AllowlistEntry]]:
    """Read the ring from the environment. Missing primary key is fatal."""
    primary = os.environ.get("JWT_SECRET")
    if not primary:
        raise RuntimeError("JWT_SECRET is not set in the mtia container environment")
    mode = (os.environ.get("JWT_LEGACY_MODE") or "").strip().lower()
    if mode not in (MODE_ALL, MODE_ALLOWLIST):
        mode = MODE_ALLOWLIST
    return (
        primary,
        _split(os.environ.get("JWT_SECRET_LEGACY")),
        mode,
        parse_allowlist(os.environ.get("JWT_LEGACY_ALLOWLIST")),
    )


def _decode(token: str, key: str) -> dict | None:
    try:
        return jwt.decode(token, key, algorithms=[ALGORITHM],
                          options=DECODE_OPTIONS)
    except jwt.PyJWTError:
        return None


def _legacy_allowed(token: str, mode: str,
                    allowlist: dict[str, AllowlistEntry],
                    today: datetime.date) -> bool:
    if mode == MODE_ALL:
        return True
    entry = allowlist.get(token_hash(token))
    if entry is None:
        return False
    if entry.expires is None or entry.expires < today:
        log.warning("jwt_legacy_entry_expired hash=%s expired=%s action=%s",
                    token_hash(token)[:12], entry.expires, entry.action)
        return entry.action == WARN
    return True


def claims_from_keyring(token: str, primary: str, legacy: list[str], mode: str,
                        allowlist: dict[str, AllowlistEntry],
                        today: datetime.date | None = None) -> dict | None:
    """Claims of the first ring key that validates the token, or ``None``."""
    claims = _decode(token, primary)
    if claims is not None:
        return claims
    today = today or datetime.date.today()
    for key in legacy:
        claims = _decode(token, key)
        if claims is None:
            continue
        if not _legacy_allowed(token, mode, allowlist, today):
            return None
        log.warning("jwt_legacy_key_accepted login=%s client=%s modo=%s",
                    claims.get("login"), claims.get("client"), mode)
        return claims
    return None


def verify_jwt(authorization: str | None = Header(default=None)) -> JwtClaims:
    """FastAPI dependency: decode + validate `Authorization: JWT <token>`."""
    if not authorization:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.upper() != "JWT" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "expected 'Authorization: JWT <token>'")

    claims = claims_from_keyring(token, *_keyring())
    if claims is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")

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
