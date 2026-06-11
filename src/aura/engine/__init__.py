"""
Aura refinement engine (production implementation of the spec Protocols).

Built incrementally across the roadmap:

* Phase 3 — :mod:`~aura.engine.symmetry`, :mod:`~aura.engine.scattering`,
  :mod:`~aura.engine.positions`, and :class:`~aura.engine.forward.ProductionForward`
  (the real, multi-phase, all-data-type forward model).
* Phase 4 — the scipy least-squares minimizer.
* Phase 5 — the real parametric engine.
* Phase 6 — the JAX/autodiff backend.

:class:`ProductionEngine` composes these and is the object the physics-invariant
suite runs against (swapped in via the ``engine`` fixture).
"""

from __future__ import annotations

__all__ = ["ProductionEngine", "ProductionForward"]


def __getattr__(name: str):
    # Lazy export so submodules (symmetry/scattering/positions) can be imported
    # before forward.py exists, and to avoid importing the full engine eagerly.
    if name in __all__:
        from aura.engine import forward

        return getattr(forward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
