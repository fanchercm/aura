"""
Preferred-orientation (texture) as a per-reflection intensity correction.

Texture is angle-dependent and multiplicative *per reflection*, so — unlike an
additive background — it cannot be expressed through the ``contribute(y_calc)``
interface (the summed pattern has lost per-reflection identity). It is therefore
a **reflection-level** correction the forward model applies inside its reflection
loop: ``intensity *= MarchDollase.factor(hkl, cell, ratio)``.

The March–Dollase model multiplies each reflection by

    P(α) = (r² cos²α + sin²α / r)^(-3/2)

where α is the angle between the reflection's reciprocal-lattice vector and the
preferred-orientation axis, and r is the refinable March ratio
(``phase:<name>:march.ratio``). r = 1 is the random (texture-free) limit, where
P ≡ 1 for every reflection. This is the compact, directional parameterization the
requirements call for (one coefficient instead of per-slice intensity terms) and
the headline beneficiary of retained angular resolution (SCI-3/SCI-6).
"""

from __future__ import annotations

import numpy as np

from aura.spec import UnitCell, reciprocal_metric_tensor


class MarchDollase:
    """March–Dollase preferred-orientation correction about a fixed axis."""

    name = "march-dollase"

    def __init__(self, phase_name: str, axis: tuple[int, int, int] = (0, 0, 1)) -> None:
        if axis == (0, 0, 0):
            raise ValueError("Preferred-orientation axis must be a nonzero hkl.")
        self.phase_name = phase_name
        self.axis = tuple(float(v) for v in axis)

    def factor(self, hkl, cell: UnitCell, ratio: float) -> float:
        """March–Dollase multiplier for one reflection (1.0 at ratio=1)."""
        if ratio == 1.0:
            return 1.0
        gstar = reciprocal_metric_tensor(*cell.as_tuple())
        h = np.asarray(hkl, dtype=float)
        a = np.asarray(self.axis, dtype=float)
        hh = float(h @ gstar @ h)
        aa = float(a @ gstar @ a)
        if hh <= 0 or aa <= 0:
            return 1.0
        cos2 = (float(h @ gstar @ a) ** 2) / (hh * aa)
        cos2 = min(max(cos2, 0.0), 1.0)
        sin2 = 1.0 - cos2
        return (ratio * ratio * cos2 + sin2 / ratio) ** (-1.5)
