"""Fail-closed safety guards shared by discovery and direct test-file runs.

Import and install this module before importing any application module. Only
directories created by TestTemporaryDirectory are eligible for SQLite access.
Unexpected attempts remain recorded even when application code catches errors.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_LIVE_DATA = (_ROOT / "data").resolve()
_ALLOWED_ROOTS = set()
_VIOLATIONS = []
_EXPECTED_VIOLATIONS = []
_INSTALLED = False
_CONNECT = sqlite3.connect
_MKDIR = Path.mkdir
_RUN = unittest.TestCase.run


def _deny(message):
    if _EXPECTED_VIOLATIONS:
        _EXPECTED_VIOLATIONS[-1].append(message)
    else:
        _VIOLATIONS.append(message)
    raise AssertionError(message)


def _guarded_connect(database, *args, **kwargs):
    # URI connections are rejected rather than risk unvalidated URI semantics.
    value = os.fsdecode(os.fspath(database))
    if value == ":memory:":
        return _CONNECT(database, *args, **kwargs)
    if value.startswith("file:") or kwargs.get("uri"):
        _deny("SQLite URI connections are forbidden in tests")
    path = Path(value).resolve()
    if not any(path.is_relative_to(root) for root in _ALLOWED_ROOTS):
        _deny(f"SQLite access outside a registered test directory: {path}")
    return _CONNECT(database, *args, **kwargs)


def _guarded_mkdir(path, *args, **kwargs):
    if path.resolve().is_relative_to(_LIVE_DATA):
        _deny(f"Production data directory initialization in tests: {path}")
    return _MKDIR(path, *args, **kwargs)


def _blocked_inference(*args, **kwargs):
    _deny("Unmocked Ollama inference attempted during tests")


def _guarded_run(case, result=None):
    result = _RUN(case, result)
    if _VIOLATIONS:
        try:
            raise AssertionError("Test isolation violations: " + "; ".join(_VIOLATIONS))
        except AssertionError:
            result.addFailure(case, sys.exc_info())
    return result


def install_guards():
    global _INSTALLED
    if _INSTALLED:
        return
    sqlite3.connect = _guarded_connect
    sqlite3.dbapi2.connect = _guarded_connect
    Path.mkdir = _guarded_mkdir
    unittest.TestCase.run = _guarded_run
    import ollama
    ollama.chat = _blocked_inference
    ollama.Client.chat = _blocked_inference
    ollama.AsyncClient.chat = _blocked_inference
    # Also protect aliases if application modules were imported by another test
    # harness before this support module was installed.
    for name in ("llm", "task_controller", "reflection", "self_state", "relationship"):
        module = sys.modules.get(name)
        if module is not None:
            module.ollama_chat = _blocked_inference
    _INSTALLED = True


class TestTemporaryDirectory(tempfile.TemporaryDirectory):
    """Register precisely this fixture's directory for its lifetime."""

    def __init__(self, *args, **kwargs):
        directory = kwargs.get("dir", args[2] if len(args) > 2 else None)
        parent = Path(directory if directory is not None else tempfile.gettempdir()).resolve()
        if parent.is_relative_to(_LIVE_DATA):
            _deny("Test temporary directory must not be inside production data")
        super().__init__(*args, **kwargs)
        self._registered_root = Path(self.name).resolve()
        if self._registered_root.is_relative_to(_LIVE_DATA):
            _deny("Test temporary directory must not be inside production data")
        _ALLOWED_ROOTS.add(self._registered_root)

    def cleanup(self):
        try:
            super().cleanup()
        finally:
            _ALLOWED_ROOTS.discard(self._registered_root)


@contextmanager
def expected_violation():
    """Acknowledge one deliberate guard probe without clearing real failures."""
    messages = []
    _EXPECTED_VIOLATIONS.append(messages)
    try:
        yield messages
    finally:
        _EXPECTED_VIOLATIONS.pop()
        if len(messages) != 1:
            raise AssertionError("Expected exactly one isolation guard violation")
