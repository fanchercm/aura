"""
Shared pytest fixtures for the Aura test suite.

The ``engine`` fixture is the **single integration swap point** between the
reference (oracle) engine and a production engine. Today it yields
:class:`aura.reference.RefEngine`; once a production engine exists it will be
parametrized over both (``params=[RefEngine, ProductionEngine]``) so every
physics invariant runs against both implementations.
"""

from __future__ import annotations

import pytest

from aura.reference import RefEngine


@pytest.fixture(scope="module")
def engine():
    """The engine under test. Swap/extend this to add the production engine."""
    return RefEngine()
