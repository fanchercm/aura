"""
Pluggable physics contributions (the spec's DomainModule path).

Domain modules add structured physics to the calculated pattern without the
forward model special-casing them. They are configured on the engine and read
their refinable coefficients from the refinement state's parameters, so they
participate in refinement like any other parameter.

* :class:`~aura.domains.background.ChebyshevBackground` — a compact background
  basis (one of the dominant parameter-explosion risks; MOD-3): a handful of
  Chebyshev coefficients replace a per-channel background.
* :class:`~aura.domains.texture.MarchDollase` — preferred-orientation intensity
  correction (a per-reflection, angle-dependent contribution; SCI-3/SCI-6).
"""

from __future__ import annotations

from aura.domains.background import ChebyshevBackground
from aura.domains.texture import MarchDollase

__all__ = ["ChebyshevBackground", "MarchDollase"]
