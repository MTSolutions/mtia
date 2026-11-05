"""Unit tests for the mtapi2 client wrapper — stubbed ServerProxy, no network."""
from __future__ import annotations

import datetime as dt

import pytest
import xmlrpc.client

from modules.plantagent import mtapi


class _FakeProxy:
    """Records calls; returns a canned result or raises a canned exception.

    XML-RPC proxies resolve any attribute to a callable method, so we mimic that
    via __getattr__ (instance attrs like `calls` resolve normally and are not
    intercepted).
    """

    def __init__(self, result=None, raises=None):
        self.calls: list[tuple] = []
        self._result = result
        self._raises = raises

    def __getattr__(self, name):
        def method(*args):
            self.calls.append((name, args))
            if self._raises is not None:
                raise self._raises
            return self._result
        return method


def test_call_injects_client_as_first_arg():
    proxy = _FakeProxy(result=0.87)
    start, end = dt.datetime(2026, 6, 1), dt.datetime(2026, 6, 2)

    result = mtapi.call("oee", "cic", start, end, 1079, proxy=proxy)

    assert result == 0.87
    name, args = proxy.calls[0]
    assert name == "oee"
    assert args == ("cic", start, end, 1079)
    assert args[0] == "cic"  # client is always first


def test_method_not_implemented_raises_unavailable():
    proxy = _FakeProxy(result="Method Not Implemented")
    with pytest.raises(mtapi.MtapiUnavailable):
        mtapi.call("absorcion", "cic", proxy=proxy)


def test_client_not_implemented_raises_unavailable():
    proxy = _FakeProxy(result="Client Not Implemented")
    with pytest.raises(mtapi.MtapiUnavailable):
        mtapi.call("oee", "unknownclient", proxy=proxy)


def test_xmlrpc_fault_raises_mtapierror():
    proxy = _FakeProxy(raises=xmlrpc.client.Fault(1, "boom"))
    with pytest.raises(mtapi.MtapiError):
        mtapi.call("oee", "cic", proxy=proxy)


def test_transport_error_raises_mtapierror():
    proxy = _FakeProxy(raises=OSError("connection refused"))
    with pytest.raises(mtapi.MtapiError):
        mtapi.call("oee", "cic", proxy=proxy)


def test_unavailable_is_subclass_of_mtapierror():
    # Callers can catch MtapiError broadly, or MtapiUnavailable specifically.
    assert issubclass(mtapi.MtapiUnavailable, mtapi.MtapiError)
