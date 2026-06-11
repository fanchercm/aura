"""
Differentiable (JAX) forward model — autodiff Jacobians.

A JAX reimplementation of the production CW forward model whose Jacobian comes
from automatic differentiation instead of finite differences. This is the
substrate the spec commits to (JAXFit / JAX-COSMO): AD removes the finite-
difference fragility of legacy Rietveld codes, and the same code runs on GPU.

Design:

* Reflections (hkl + multiplicity) are **pre-enumerated** with gemmi at the
  current cell — a discrete, non-differentiable set that is fixed under the
  infinitesimal parameter perturbations AD takes. Their d-spacings, positions,
  structure factors, and profiles are all computed in JAX, so the model is
  differentiable w.r.t. cell, atom, and histogram parameters.
* Scattering factors use the same physics as the numpy engine: X-ray IT92
  Gaussian coefficients ``f(s²)=Σ aᵢexp(-bᵢs²)+c`` (≈Z at s=0) and the constant
  neutron scattering length b — both extracted from gemmi once.
* The profile is summed over the **full grid** (no windowing), so the model is a
  single smooth differentiable function. The numpy engine's windowing is a
  performance optimization; for tight agreement compare against a full-grid numpy
  evaluation (``ProductionForward`` with a very large ``peak_window``).

Scope: CW (X-ray / neutron), single phase. TOF/EDD and multi-phase raise
``NotImplementedError`` here — the numpy backend covers them; extending the JAX
backend is incremental. **Requires 64-bit JAX** (enabled on import); float32
would break the AD-vs-finite-difference agreement and numpy parity.
"""

from __future__ import annotations

import functools

import gemmi
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (must follow the x64 config update)

from aura.engine import positions, symmetry  # noqa: E402
from aura.engine.forward import ProductionForward  # noqa: E402
from aura.spec import DataType, Histogram, RefinementState  # noqa: E402

_CW = (DataType.CW_XRAY, DataType.CW_NEUTRON)


@functools.lru_cache(maxsize=512)
def _xray_it92(element: str) -> tuple[float, ...]:
    return tuple(gemmi.Element(element).it92.get_coefs())  # [a1..a4, b1..b4, c]


@functools.lru_cache(maxsize=512)
def _neutron_b(element: str) -> float:
    return float(gemmi.Element(element).neutron92.get_coefs()[0])


def _reciprocal_metric(a, b, c, al, be, ga):
    """Reciprocal metric tensor G* = inv(G), in JAX (differentiable in cell params)."""
    al, be, ga = jnp.radians(al), jnp.radians(be), jnp.radians(ga)
    g = jnp.array(
        [
            [a * a, a * b * jnp.cos(ga), a * c * jnp.cos(be)],
            [a * b * jnp.cos(ga), b * b, b * c * jnp.cos(al)],
            [a * c * jnp.cos(be), b * c * jnp.cos(al), c * c],
        ]
    )
    return jnp.linalg.inv(g)


def _pseudo_voigt(x, center, fwhm, eta):
    """Normalized pseudo-Voigt (matches aura.spec.pseudo_voigt), vectorized in JAX."""
    dx = x - center
    sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    gauss = jnp.exp(-0.5 * (dx / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))
    gamma = fwhm / 2.0
    lorentz = (gamma / jnp.pi) / (dx * dx + gamma * gamma)
    return eta * lorentz + (1.0 - eta) * gauss


class JaxForward:
    """Differentiable CW forward model with autodiff Jacobians."""

    name = "production-jax"

    def calculate(self, state: RefinementState, histogram: Histogram) -> np.ndarray:
        x0, fn = self._build(state, histogram)
        return np.asarray(fn(x0))

    def jacobian(self, state: RefinementState, histogram: Histogram) -> np.ndarray:
        x0, fn = self._build(state, histogram)
        if x0.shape[0] == 0:
            return np.zeros((len(histogram.x), 0))
        return np.asarray(jax.jacfwd(fn)(x0))

    # ------------------------------------------------------------------
    def _build(self, state: RefinementState, histogram: Histogram):
        """Return (x0, fn) where fn(x)->y is a pure differentiable JAX function
        of the varied-parameter vector x and x0 is its current value."""
        if histogram.data_type not in _CW:
            raise NotImplementedError(
                f"JAX backend supports CW data only; got {histogram.data_type}. "
                "Use the numpy backend for TOF/EDD."
            )
        if len(state.phases) != 1:
            raise NotImplementedError(
                "JAX backend supports a single phase; use the numpy backend for "
                "multi-phase refinement."
            )
        phase = state.phases[0]
        is_neutron = histogram.data_type is DataType.CW_NEUTRON
        wl = float(histogram.wavelength)
        x_grid = jnp.asarray(np.asarray(histogram.x, dtype=float))

        # Non-differentiable: enumerate reflections at the current cell.
        eff = ProductionForward._effective_phase(
            phase, {p.name: p.value for p in state.parameters}
        )
        d_min, d_max = positions.d_range_for_histogram(histogram, pad=0.02)
        refl = symmetry.generate_reflections(eff, round(d_min, 4), round(d_max, 4))
        if not refl:
            const = float(_param_value(state, f"hist:{histogram.id}:bkg", 0.0))
            return jnp.zeros(0), (lambda _x, c=const: jnp.full(x_grid.shape, c))
        hkl = jnp.asarray(np.array([r.hkl for r in refl], dtype=float))  # (R,3)
        mult = jnp.asarray(np.array([r.multiplicity for r in refl], dtype=float))

        # Atom arrays + per-atom scattering coefficients (static).
        atoms = phase.atoms
        if is_neutron:
            bcoh = jnp.asarray([_neutron_b(at.element) for at in atoms])  # (A,)
            it92a = it92b = it92c = None
        else:
            coefs = np.array([_xray_it92(at.element) for at in atoms])  # (A,9)
            it92a = jnp.asarray(coefs[:, 0:4])  # (A,4)
            it92b = jnp.asarray(coefs[:, 4:8])
            it92c = jnp.asarray(coefs[:, 8])
            bcoh = None

        # Resolver: each physics input is either a varied x-slot or a constant.
        spec_inputs, x0 = _input_spec(state, phase, histogram.id)
        hkl_T = hkl  # (R,3)

        def fn(x):
            g = lambda key: spec_inputs[key](x)  # noqa: E731
            a, b, c = g("cell.a"), g("cell.b"), g("cell.c")
            al, be, ga = g("cell.alpha"), g("cell.beta"), g("cell.gamma")
            # Cubic shim: mirror a->b,c when the base cell is cubic and only a is free.
            if spec_inputs["_cubic_shim"]:
                b = c = a
            gstar = _reciprocal_metric(a, b, c, al, be, ga)
            dstar2 = jnp.einsum("ri,ij,rj->r", hkl_T, gstar, hkl_T)  # (R,)
            d = 1.0 / jnp.sqrt(dstar2)
            stol2 = 1.0 / (4.0 * d * d)  # (R,)
            two_theta = jnp.degrees(
                2.0 * jnp.arcsin(jnp.clip(wl / (2.0 * d), -1.0, 1.0))
            )

            xyz = jnp.stack(
                [
                    jnp.stack([g(f"atom{i}.x"), g(f"atom{i}.y"), g(f"atom{i}.z")])
                    for i in range(len(atoms))
                ]
            )  # (A,3)
            occ = jnp.stack([g(f"atom{i}.occ") for i in range(len(atoms))])  # (A,)
            biso = jnp.stack([g(f"atom{i}.b_iso") for i in range(len(atoms))])  # (A,)

            if is_neutron:
                f_ra = jnp.broadcast_to(bcoh, (hkl_T.shape[0], len(atoms)))  # (R,A)
            else:
                # f(s2) = sum_k a_k exp(-b_k s2) + c, per atom, per reflection.
                expo = jnp.exp(-it92b[None, :, :] * stol2[:, None, None])  # (R,A,4)
                f_ra = (
                    jnp.sum(it92a[None, :, :] * expo, axis=2) + it92c[None, :]
                )  # (R,A)
            dw = jnp.exp(-biso[None, :] * stol2[:, None])  # (R,A)
            phase_arg = 2.0 * jnp.pi * (hkl_T @ xyz.T)  # (R,A)
            amp = occ[None, :] * f_ra * dw  # (R,A)
            fr = jnp.sum(amp * jnp.cos(phase_arg), axis=1)
            fi = jnp.sum(amp * jnp.sin(phase_arg), axis=1)
            fsq = fr * fr + fi * fi  # (R,)

            theta = jnp.radians(two_theta) / 2.0
            s, co = jnp.sin(theta), jnp.cos(theta)
            lorentz = 1.0 / (s * s * co)
            if is_neutron:
                lp = lorentz
            else:
                lp = lorentz * (1.0 + jnp.cos(jnp.radians(two_theta)) ** 2) / 2.0

            scale, bkg, fwhm, eta = g("scale"), g("bkg"), g("fwhm"), g("eta")
            intensity = scale * mult * fsq * lp  # (R,)
            profiles = _pseudo_voigt(
                x_grid[None, :], two_theta[:, None], fwhm, eta
            )  # (R,N)
            return bkg + jnp.sum(intensity[:, None] * profiles, axis=0)

        return x0, fn


def _param_value(state: RefinementState, name: str, default: float) -> float:
    for p in state.parameters:
        if p.name == name:
            return p.value
    return default


def _input_spec(state: RefinementState, phase, hist_id: str):
    """Build, for every physics input, a closure x -> value (varied slot or const),
    plus the x0 vector of varied values in slot order.

    Returns (inputs, x0) where ``inputs[key](x)`` yields a JAX scalar.
    """
    varied = [p for p in state.parameters if p.vary]
    index = {p.name: i for i, p in enumerate(varied)}
    values = {p.name: p.value for p in state.parameters}
    x0 = jnp.asarray(np.array([p.value for p in varied], dtype=float))

    def resolver(param_name: str, default: float):
        if param_name in index:
            i = index[param_name]
            return lambda x: x[i]
        const = float(values.get(param_name, default))
        return lambda _x, c=const: jnp.asarray(c)

    c = phase.cell
    inputs = {
        "cell.a": resolver(f"phase:{phase.name}:cell.a", c.a),
        "cell.b": resolver(f"phase:{phase.name}:cell.b", c.b),
        "cell.c": resolver(f"phase:{phase.name}:cell.c", c.c),
        "cell.alpha": resolver(f"phase:{phase.name}:cell.alpha", c.alpha),
        "cell.beta": resolver(f"phase:{phase.name}:cell.beta", c.beta),
        "cell.gamma": resolver(f"phase:{phase.name}:cell.gamma", c.gamma),
        "scale": resolver(f"hist:{hist_id}:scale", 1.0),
        "bkg": resolver(f"hist:{hist_id}:bkg", 0.0),
        "fwhm": resolver(f"hist:{hist_id}:fwhm", 0.1),
        "eta": resolver(f"hist:{hist_id}:eta", 0.5),
    }
    for i, at in enumerate(phase.atoms):
        pre = f"phase:{phase.name}:atom{i}"
        inputs[f"atom{i}.x"] = resolver(f"{pre}.x", at.x)
        inputs[f"atom{i}.y"] = resolver(f"{pre}.y", at.y)
        inputs[f"atom{i}.z"] = resolver(f"{pre}.z", at.z)
        inputs[f"atom{i}.occ"] = resolver(f"{pre}.occ", at.occ)
        inputs[f"atom{i}.b_iso"] = resolver(f"{pre}.b_iso", at.b_iso)

    pre = f"phase:{phase.name}:cell."
    is_cubic = c.a == c.b == c.c and c.alpha == c.beta == c.gamma == 90.0
    inputs["_cubic_shim"] = (
        is_cubic
        and f"{pre}a" in values
        and not (f"{pre}b" in values or f"{pre}c" in values)
    )
    return inputs, x0
