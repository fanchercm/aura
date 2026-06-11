"""
Chunked, bounded-memory minimizer for campaign-scale refinement.

The Phase-4/5 minimizer hands the full stacked residual (and its Jacobian) to
SciPy, which is fine for one state but not for ~10⁵ patterns: the stacked
Jacobian is ``N_points_total × N_params``. This Gauss–Newton minimizer instead
**accumulates the normal-equations matrices** ``JᵀJ`` (``N_params × N_params``)
and ``Jᵀr`` (``N_params``) one histogram at a time, folding each histogram's
block in and discarding it. Peak memory is therefore ``O(N_params²)`` plus the
largest single histogram — independent of the number of patterns (PERF-3).

Each histogram's residual-Jacobian block is computed by finite difference over
the free parameters with ``expand`` applied *inside* the perturbation, so shared
parametric coefficients get correct derivatives (the same chain-rule treatment as
Phase 5) — and because the blocks are independent, they parallelize across
histograms (PERF-4) via a thread pool.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from aura import spec
from aura.spec import RefinementResult, RefinementState


def _peak_rss_mb() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / 1024.0 if maxrss < 1e9 else maxrss / (1024.0 * 1024.0)


class ChunkedMinimizer:
    """Gauss–Newton with histogram-chunked normal-equations accumulation."""

    name = "production-chunked"

    def __init__(self, parallel: bool = False, max_workers: int | None = None) -> None:
        self.parallel = parallel
        self.max_workers = max_workers

    def refine(
        self,
        state: RefinementState,
        forward,
        parametric,
        max_iter: int = 50,
        tol: float = 1e-8,
        seed: int | None = None,
    ) -> RefinementResult:
        t0 = time.perf_counter()
        counter = {"n": 0}
        varied = [p for p in state.parameters if p.vary]
        names = [p.name for p in varied]
        n = len(varied)
        lower = np.array([p.lower for p in varied])
        upper = np.array([p.upper for p in varied])
        x = np.clip(np.array([p.value for p in varied], dtype=float), lower, upper)

        def set_x(xv):
            pm = dict(zip(names, xv, strict=False))
            return replace(
                state,
                parameters=tuple(
                    replace(p, value=pm.get(p.name, p.value)) for p in state.parameters
                ),
            )

        converged = False
        it = 0
        jtj = np.zeros((n, n))
        for it in range(1, max_iter + 1):  # noqa: B007 (it is the final iter count)
            cur = set_x(x)
            jtj, jtr, _chi2 = self._accumulate(
                cur, forward, parametric, names, x, counter
            )
            if n == 0:
                break
            lam = 1e-6 * np.trace(jtj) / max(n, 1)
            try:
                step = -np.linalg.solve(jtj + lam * np.eye(n), jtr)
            except np.linalg.LinAlgError:
                break
            x_new = np.clip(x + step, lower, upper)
            if np.max(np.abs(x_new - x)) < tol:
                x = x_new
                converged = True
                break
            x = x_new

        final = set_x(x)
        return self._result(
            state,
            final,
            forward,
            parametric,
            jtj,
            n,
            converged,
            it,
            counter["n"],
            t0,
            seed,
            max_iter,
            tol,
        )

    # ------------------------------------------------------------------
    def _accumulate(self, cur, forward, parametric, names, x, counter):
        """Sum JᵀJ, Jᵀr, χ² over histograms — each block built and discarded."""
        hists = cur.histograms
        n = len(names)

        def block(hist):
            return _histogram_normal_eqs(cur, hist, forward, parametric, names, x)

        if self.parallel and len(hists) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                results = list(ex.map(block, hists))
        else:
            results = [block(h) for h in hists]

        jtj = np.zeros((n, n))
        jtr = np.zeros(n)
        chi2 = 0.0
        for bjtj, bjtr, bchi2, nfwd in results:
            jtj += bjtj
            jtr += bjtr
            chi2 += bchi2
            counter["n"] += nfwd
        return jtj, jtr, chi2

    def _result(
        self,
        seed_state,
        final,
        forward,
        parametric,
        jtj,
        n,
        converged,
        it,
        n_forward,
        t0,
        seed,
        max_iter,
        tol,
    ) -> RefinementResult:
        from aura import provenance

        yo = np.concatenate([h.y_obs for h in final.histograms])
        w = np.concatenate([h.weights for h in final.histograms])
        yc = np.concatenate(
            [
                forward.calculate(parametric.expand(final, h), h)
                for h in final.histograms
            ]
        )
        gof = spec.reduced_chi_square(yo, yc, w, max(n, 1))
        cond = float(np.linalg.cond(jtj)) if n else float("nan")
        try:
            cov = gof * np.linalg.inv(jtj) if n else None
            sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None)) if cov is not None else []
        except np.linalg.LinAlgError:
            cov, sigma = None, np.full(n, np.nan)
        names = [p.name for p in final.parameters if p.vary]
        sig_map = dict(zip(names, sigma, strict=False))
        final = replace(
            final,
            parameters=tuple(
                replace(p, sigma=sig_map[p.name]) if p.name in sig_map else p
                for p in final.parameters
            ),
        )
        physical = all(c.is_physical() for c in (ph.cell for ph in final.phases))
        result_state = final.with_log(
            f"chunked-refined: {n} params, {it} iters, converged={converged}"
        )
        manifest = provenance.build_manifest(
            seed_state,
            kernel_backend=self.name,
            optimizer={
                "method": "chunked-gauss-newton",
                "max_iter": max_iter,
                "tol": tol,
            },
            random_seed=seed,
        )
        diagnostics = {
            "condition_number": cond,
            "wall_time_s": time.perf_counter() - t0,
            "n_forward_evals": float(n_forward),
            "peak_rss_mb": _peak_rss_mb(),
            "n_points": float(len(yo)),
            "n_histograms": float(len(final.histograms)),
        }
        return RefinementResult(
            state=result_state,
            rwp=spec.rwp(yo, yc, w),
            reduced_chi2=gof,
            converged=converged and physical,
            n_iterations=it,
            covariance=cov,
            diagnostics=diagnostics,
            provenance=manifest,
            seed_quality_ok=physical,
        )


def _histogram_normal_eqs(cur, hist, forward, parametric, names, x):
    """One histogram's (JᵀJ, Jᵀr, χ², n_forward) via residual-FD over free params.

    Builds the residual-Jacobian block for this histogram only (npts × nparams),
    folds it into the local normal-equations matrices, and returns those — the
    block itself is never retained beyond this call (bounded memory).
    """
    sw = np.sqrt(hist.weights)
    n = len(names)

    def residual(xv):
        st = replace(
            cur,
            parameters=tuple(
                replace(
                    p, value=dict(zip(names, xv, strict=False)).get(p.name, p.value)
                )
                for p in cur.parameters
            ),
        )
        yc = forward.calculate(parametric.expand(st, hist), hist)
        return sw * (hist.y_obs - yc)

    r0 = residual(x)
    nfwd = 1
    jblock = np.zeros((len(r0), n))
    for j in range(n):
        step = 1e-6 * max(abs(x[j]), 1.0)
        xp = x.copy()
        xp[j] += step
        xm = x.copy()
        xm[j] -= step
        jblock[:, j] = (residual(xp) - residual(xm)) / (2.0 * step)
        nfwd += 2
    return jblock.T @ jblock, jblock.T @ r0, float(r0 @ r0), nfwd
