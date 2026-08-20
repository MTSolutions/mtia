"""Key-ring de validación JWT en mtia (MTS-1669).

mtia solo valida: los tokens los emite el `api`. Por eso tiene que aceptar
exactamente los mismos que el `api` acepta — mismas variables, mismos modos y
la misma allowlist. Si se desincroniza, mtia rechaza tokens válidos y el
síntoma que llega es "el chat no anda", sin relación aparente con la rotación.

Cada test lleva el caso aprobado de `specs/mts-1669-tests.md`.
"""
from __future__ import annotations

import datetime

import jwt
import pytest
from fastapi import HTTPException

from modules.rag.auth import (
    claims_from_keyring,
    parse_allowlist,
    token_hash,
    verify_jwt,
)

PRIMARIA = "clave-fuerte-de-este-servidor"
HEREDADA = "secret"
AJENA = "clave-fuerte-de-OTRO-servidor"

CLAIMS = {"sub": 2240, "login": "qroma", "client": "qroma"}


def firmar(key: str, claims: dict | None = None) -> str:
    return jwt.encode(claims or CLAIMS, key, algorithm="HS512")


def entorno(monkeypatch, *, legacy="", mode="allowlist", allowlist=""):
    monkeypatch.setenv("JWT_SECRET", PRIMARIA)
    monkeypatch.setenv("JWT_SECRET_LEGACY", legacy)
    monkeypatch.setenv("JWT_LEGACY_MODE", mode)
    monkeypatch.setenv("JWT_LEGACY_ALLOWLIST", allowlist)


def test_t18_acepta_la_primaria(monkeypatch):
    """T-18 / CE-1: el camino normal, el de todos los tokens nuevos.

    Riesgo: rotar el servidor deja a mtia rechazando a todo el mundo.
    """
    entorno(monkeypatch)
    claims = verify_jwt(f"JWT {firmar(PRIMARIA)}")
    assert claims.client == "qroma"
    assert claims.login == "qroma"


def test_t18_acepta_la_heredada_solo_en_modo_all(monkeypatch):
    """T-18 / CE-3: la ventana general vale igual en mtia que en el api.

    Riesgo: el paso 1 del rollout corta mtia mientras el api sigue sirviendo.
    """
    entorno(monkeypatch, legacy=HEREDADA, mode="all")
    assert verify_jwt(f"JWT {firmar(HEREDADA)}").client == "qroma"


def test_t18_en_allowlist_la_heredada_no_listada_se_rechaza(monkeypatch):
    """T-18 / CE-1: cerrada la ventana, un token forjado no entra.

    Riesgo: mtia queda como la puerta de atrás — el api cierra la forja y el
    mismo token forjado sigue sirviendo contra mtia.
    """
    entorno(monkeypatch, legacy=HEREDADA, mode="allowlist")
    with pytest.raises(HTTPException) as exc:
        verify_jwt(f"JWT {firmar(HEREDADA)}")
    assert exc.value.status_code == 401


def test_t18_en_allowlist_el_token_horneado_listado_se_acepta(monkeypatch):
    """T-18 / CE-4: el consumidor que no se puede reconstruir sigue vivo.

    Riesgo: cerrar la forja deja sin servicio a los terminales.
    """
    horneado = firmar(HEREDADA)
    entorno(monkeypatch, legacy=HEREDADA, mode="allowlist",
            allowlist=f"{token_hash(horneado)}:2099-01-01")
    assert verify_jwt(f"JWT {horneado}").client == "qroma"


def test_t18_rechaza_la_clave_de_otro_servidor(monkeypatch):
    """T-18 / CE-2: aislamiento entre servidores, también en mtia.

    Riesgo: un token de un servidor sirve contra el mtia de otro.
    """
    entorno(monkeypatch, legacy=HEREDADA, mode="all")
    with pytest.raises(HTTPException) as exc:
        verify_jwt(f"JWT {firmar(AJENA)}")
    assert exc.value.status_code == 401


def test_t18_la_caducidad_de_la_entrada_se_respeta():
    """T-18 / CE-4: vencida en enforce no pasa; en warn sí.

    Riesgo: que la allowlist de mtia sea eterna mientras la del api caduca,
    dejando viva una credencial que el api ya dio por muerta.
    """
    horneado = firmar(HEREDADA)
    hoy = datetime.date(2026, 9, 1)
    h = token_hash(horneado)

    vencida = parse_allowlist(f"{h}:2026-08-31")
    assert claims_from_keyring(horneado, PRIMARIA, [HEREDADA], "allowlist",
                               vencida, today=hoy) is None

    indulgente = parse_allowlist(f"{h}:2026-08-31:warn")
    claims = claims_from_keyring(horneado, PRIMARIA, [HEREDADA], "allowlist",
                                 indulgente, today=hoy)
    assert claims["client"] == "qroma"


def test_t19_sin_jwt_secret_falla_al_arrancar(monkeypatch):
    """T-19 / CE-6: falla cerrada, también fuera del api.

    Riesgo: el key-ring introduce por accidente un default permisivo y mtia
    termina validando contra una clave vacía.
    """
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError) as exc:
        verify_jwt(f"JWT {firmar(PRIMARIA)}")
    assert "JWT_SECRET" in str(exc.value)


def test_t19_modo_desconocido_cae_en_el_lado_seguro(monkeypatch):
    """T-19 / CE-1: un tipeo en el modo no puede abrir la ventana.

    Riesgo: JWT_LEGACY_MODE mal escrito dejaría mtia aceptando cualquier
    token firmado con la clave pública.
    """
    entorno(monkeypatch, legacy=HEREDADA, mode="ALL_")
    with pytest.raises(HTTPException):
        verify_jwt(f"JWT {firmar(HEREDADA)}")
