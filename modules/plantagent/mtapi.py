"""Thin wrapper over the mtapi2 XML-RPC service — the source of truth for
official plant indicators.

The JWT-derived client is always injected as the first positional argument;
callers (and the LLM) never supply it. mtapi2's dispatcher returns string
sentinels instead of raising when a function or client isn't wired up, so we
surface those as typed errors — a missing indicator must never reach the user
as a bogus figure.
"""
from __future__ import annotations

import os
import xmlrpc.client


DEFAULT_MTAPI2_URL = "http://mtapi2:7777/api/xmlrpc"

# Sentinels returned by mtapi2's clientcall() dispatcher (see mtapi2 views.py).
_NOT_IMPLEMENTED = "Method Not Implemented"
_CLIENT_NOT_IMPLEMENTED = "Client Not Implemented"


class MtapiError(RuntimeError):
    """An mtapi2 call failed at the transport / XML-RPC layer."""


class MtapiUnavailable(MtapiError):
    """The requested function/client is not implemented in mtapi2.

    Subclass of MtapiError so callers can catch broadly or specifically.
    """


def _url() -> str:
    return os.environ.get("MTAPI2_URL", DEFAULT_MTAPI2_URL)


def _proxy() -> xmlrpc.client.ServerProxy:
    # allow_none/use_datetime mirror how `api` connects to mtapi2.
    return xmlrpc.client.ServerProxy(_url(), allow_none=True, use_datetime=True)


def call(fn: str, client: str, *args, proxy: object | None = None):
    """Invoke mtapi2 ``fn(client, *args)`` and return its result.

    Args:
        fn: mtapi2 method name (e.g. "oee", "getplants", "time_det").
        client: the JWT-derived client name, always passed first.
        *args: remaining positional arguments for the mtapi2 function.
        proxy: optional ServerProxy override (tests inject a stub).

    Raises:
        MtapiUnavailable: mtapi2 has no such function/client.
        MtapiError: transport or XML-RPC fault.
    """
    p = proxy if proxy is not None else _proxy()
    method = getattr(p, fn)
    try:
        result = method(client, *args)
    except xmlrpc.client.Fault as e:
        raise MtapiError("mtapi2 {!r} faulted: {}".format(fn, e)) from e
    except (xmlrpc.client.ProtocolError, OSError) as e:
        raise MtapiError("mtapi2 {!r} unreachable: {}".format(fn, e)) from e

    if result in (_NOT_IMPLEMENTED, _CLIENT_NOT_IMPLEMENTED):
        raise MtapiUnavailable(
            "mtapi2 has no {!r} for client {!r}".format(fn, client))
    return result
