"""
Production minimizer: bounded least squares over the full ensemble.

Implements :class:`aura.spec.Minimizer` on :func:`scipy.optimize.least_squares`
(trust-region reflective, with bounds), minimizing the weighted residual
*stacked across all histograms simultaneously* — the parametric/surface objective
that makes shared parameters benefit every pattern at once.

The Jacobian is assembled from the forward model's own ``jacobian`` (finite
difference today; autodiff in Phase 6), so the minimizer is agnostic to how
derivatives are produced. Outputs a full :class:`~aura.spec.RefinementResult`:
Rwp, reduced χ², parameter covariance → σ, the normal-matrix condition number,
profiling diagnostics, and a complete provenance manifest.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from aura import spec
from aura.spec import RefinementResult, RefinementState, UnitCell


def _peak_rss_mb() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / 1024.0 if maxrss < 1e9 else maxrss / (1024.0 * 1024.0)


class ProductionMinimizer:
    """Bounded least-squares refinement of the stacked ensemble residual."""

    name = "production-trf"

    def refine(
        self,
        state: RefinementState,
        forward,
        parametric,
        max_iter: int = 100,
        tol: float = 1e-8,
        seed: int | None = None,
    ) -> RefinementResult:
        t0 = time.perf_counter()
        counter = {"n": 0}
        varied = [p for p in state.parameters if p.vary]
        names = [p.name for p in varied]

        yo = np.concatenate([h.y_obs for h in state.histograms])
        w = np.concatenate([h.weights for h in state.histograms])

        def set_x(xv: np.ndarray) -> RefinementState:
            pm = dict(zip(names, xv, strict=False))
            params = tuple(
                replace(p, value=pm.get(p.name, p.value)) for p in state.parameters
            )
            return replace(state, parameters=params)

        def y_calc(cur: RefinementState) -> np.ndarray:
            blocks = []
            for hist in state.histograms:
                expanded = parametric.expand(cur, hist)
                blocks.append(forward.calculate(expanded, hist))
            counter["n"] += len(state.histograms)
            return np.concatenate(blocks)

        sw_all = np.sqrt(w)

        def residual(xv: np.ndarray) -> np.ndarray:
            yc = y_calc(set_x(xv))
            return sw_all * (yo - yc)

        def jac(xv: np.ndarray) -> np.ndarray:
            # Central finite difference at the *residual* level over the free
            # parameters. Done here (not via forward.jacobian) so that a free
            # parameter which is a parametric *coefficient* — not a quantity the
            # forward model reads directly — still gets a correct derivative:
            # perturbing it re-runs expand, which propagates to every driven
            # histogram (the parametric chain rule). Phase 6 replaces this with
            # autodiff through expand.
            cols = []
            for j in range(len(xv)):
                step = 1e-6 * max(abs(xv[j]), 1.0)
                xp = xv.copy()
                xp[j] += step
                xm = xv.copy()
                xm[j] -= step
                yp = y_calc(set_x(xp))
                ym = y_calc(set_x(xm))
                cols.append(-sw_all * (yp - ym) / (2.0 * step))
            return np.stack(cols, axis=1)

        # No free parameters: just evaluate at the seed.
        if not varied:
            return self._result(
                state,
                state,
                yo,
                w,
                y_calc(state),
                0,
                True,
                None,
                counter["n"],
                t0,
                seed,
                max_iter,
                tol,
            )

        x0 = np.array([p.value for p in varied], dtype=float)
        lower = np.array([p.lower for p in varied])
        upper = np.array([p.upper for p in varied])
        # Nudge seeds off the bounds so TRF has an interior start.
        x0 = np.clip(x0, lower + 1e-12, upper - 1e-12)

        sol = least_squares(
            residual,
            x0,
            jac=jac,
            bounds=(lower, upper),
            method="trf",
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max_iter * (len(varied) + 1),
        )

        final = set_x(sol.x)
        sigma, cov, cond = self._uncertainties(
            sol.jac, yo, y_calc(final), w, len(varied)
        )
        sig_map = dict(zip(names, sigma, strict=False))
        final = replace(
            final,
            parameters=tuple(
                replace(p, sigma=sig_map[p.name]) if p.name in sig_map else p
                for p in final.parameters
            ),
        )
        return self._result(
            state,
            final,
            yo,
            w,
            y_calc(final),
            int(sol.nfev),
            bool(sol.success),
            cov,
            counter["n"],
            t0,
            seed,
            max_iter,
            tol,
            cond=cond,
        )

    @staticmethod
    def _uncertainties(res_jac, yo, yc, w, n_varied):
        """Parameter σ, covariance, and condition number from the residual Jacobian."""
        jtj = res_jac.T @ res_jac
        gof = spec.reduced_chi_square(yo, yc, w, n_varied)
        cond = float(np.linalg.cond(jtj)) if jtj.size else float("nan")
        try:
            cov = gof * np.linalg.inv(jtj)
            sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        except np.linalg.LinAlgError:
            cov = None
            sigma = np.full(n_varied, np.nan)
        return sigma, cov, cond

    def _result(
        self,
        seed_state,
        final,
        yo,
        w,
        yc,
        nfev,
        success,
        cov,
        n_forward,
        t0,
        seed,
        max_iter,
        tol,
        cond=float("nan"),
    ) -> RefinementResult:
        from aura import provenance

        gof = spec.reduced_chi_square(yo, yc, w, final.n_varied or 1)
        # SPD guard: flag if any refined cell went non-physical.
        physical = all(_cell_physical(ph.cell) for ph in final.phases)
        msg = (
            f"refined: {final.n_varied} params, {nfev} nfev, "
            f"converged={success}, physical={physical}"
        )
        result_state = final.with_log(msg)
        manifest = provenance.build_manifest(
            seed_state,
            kernel_backend=self.name,
            optimizer={"method": "trf", "max_iter": max_iter, "tol": tol},
            random_seed=seed,
        )
        diagnostics = {
            "condition_number": cond,
            "wall_time_s": time.perf_counter() - t0,
            "n_forward_evals": float(n_forward),
            "peak_rss_mb": _peak_rss_mb(),
            "n_points": float(len(yo)),
        }
        return RefinementResult(
            state=result_state,
            rwp=spec.rwp(yo, yc, w),
            reduced_chi2=gof,
            converged=success and physical,
            n_iterations=nfev,
            covariance=cov,
            diagnostics=diagnostics,
            provenance=manifest,
            seed_quality_ok=physical,
        )


def _cell_physical(cell: UnitCell) -> bool:
    return cell.is_physical()
