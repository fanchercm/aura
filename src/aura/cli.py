"""
Command-line interface for Aura.

Scriptable entry points for high-throughput Rietveld workflows (UX-1):

    $ aura --version
    $ aura identify POWDER --candidate a.cif --candidate b.cif [--wavelength W]
    $ aura refine POWDER PHASE.cif [--instrument I] [--wavelength W]
                  [--background-order N] [--bank K] [--plot fit.png] [--out ckpt.pkl]

These wire the importer registry → bridge → engine → diagnostics together; the
Python API remains the primary interface for anything non-routine.
"""

from __future__ import annotations

import click

from aura import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Aura - Prototype framework for large-volume Rietveld analysis."""


@main.command()
@click.argument("powder", type=click.Path(exists=True))
@click.option(
    "--candidate",
    "candidates",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Candidate phase CIF (repeatable).",
)
@click.option(
    "--wavelength",
    type=float,
    default=None,
    help="CW wavelength (Å), if not in the file.",
)
@click.option("--bank", type=int, default=0, help="Histogram/bank index to use.")
def identify(powder, candidates, wavelength, bank):
    """Propose confidence-ranked candidate phases for POWDER (AI proposes; the
    engine disposes — refine the top candidate to validate)."""
    import aura.io as io
    from aura.ai.identify import PhaseIdentifier
    from aura.bridge import state_to_histograms

    state = io.read(powder, "powder")
    hist = state_to_histograms(state, wavelength=wavelength)[bank]
    lib = [io.read(c, "phase") for c in candidates]
    ranked = PhaseIdentifier(lib).propose(hist)
    click.echo(f"Candidates for {hist.id} (propose-only; validate by refinement):")
    for phase, conf in ranked:
        click.echo(f"  {conf:6.3f}  {phase.name}  [{phase.space_group}]")


@main.command()
@click.argument("powder", type=click.Path(exists=True))
@click.argument("phase_cif", type=click.Path(exists=True))
@click.option("--instrument", type=click.Path(exists=True), default=None)
@click.option("--wavelength", type=float, default=None, help="CW wavelength (Å).")
@click.option(
    "--background-order",
    type=int,
    default=6,
    help="Chebyshev background order (0 = flat).",
)
@click.option("--bank", type=int, default=0, help="Histogram/bank index to refine.")
@click.option("--max-iter", type=int, default=80)
@click.option(
    "--plot", "plot_path", type=click.Path(), default=None, help="Save a fit plot here."
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default=None,
    help="Save the refined state here.",
)
def refine(
    powder,
    phase_cif,
    instrument,
    wavelength,
    background_order,
    bank,
    max_iter,
    plot_path,
    out_path,
):
    """Refine PHASE_CIF against POWDER (lattice + scale + width + background)."""
    import numpy as np

    import aura.io as io
    from aura.bridge import state_to_histograms
    from aura.diagnostics import classify_parameters, summarize
    from aura.domains.background import ChebyshevBackground
    from aura.engine.forward import ProductionEngine
    from aura.io.instrument import apply_instprm
    from aura.spec import RefinementState

    state = io.read(powder, "powder")
    if instrument:
        apply_instprm(state, io.read(instrument, "instrument"))
    hist = state_to_histograms(state, wavelength=wavelength)[bank]
    phase = io.read(phase_cif, "phase")

    params = _default_params(
        phase, hist, background_order, float(np.median(hist.y_obs))
    )
    modules = (
        [ChebyshevBackground(order=background_order)] if background_order > 0 else []
    )
    engine = ProductionEngine(domain_modules=modules)
    rstate = RefinementState((phase,), (hist,), tuple(params))

    rwp0 = _seed_rwp(engine, rstate, hist)
    result = engine.refine(rstate, engine, engine, max_iter=max_iter, seed=1)

    s = summarize(result)
    click.echo(f"Refined {phase.name} against {hist.id} ({hist.data_type.value}):")
    click.echo(
        f"  Rwp {rwp0:.4f} -> {result.rwp:.4f}   GoF {s['reduced_chi2']:.2f}   "
        f"converged={s['converged']}   cond={s['condition_number']:.2e}"
    )
    statuses = classify_parameters(result)
    for p in result.state.parameters:
        if p.vary:
            sig = f"{p.sigma:.3g}" if p.sigma is not None else "n/a"
            click.echo(f"    {statuses[p.name]:14s} {p.name} = {p.value:.5g} ± {sig}")

    if plot_path:
        from aura.viz import plot_fit

        plot_fit(result.state, engine, hist.id, path=plot_path)
        click.echo(f"  fit plot -> {plot_path}")
    if out_path:
        from aura.checkpoint import save_refinement

        save_refinement(result.state, out_path)
        click.echo(f"  checkpoint -> {out_path}")


def _default_params(phase, hist, background_order, bkg_level):
    from aura.spec import DataType, Parameter, ParamKind

    c = phase.cell
    cubic = c.a == c.b == c.c and c.alpha == c.beta == c.gamma == 90.0
    params = []
    axes = ["a"] if cubic else ["a", "b", "c"]
    for ax in axes:
        v = getattr(c, ax)
        params.append(
            Parameter(
                f"phase:{phase.name}:cell.{ax}",
                ParamKind.PHASE,
                v,
                vary=True,
                lower=v * 0.95,
                upper=v * 1.05,
            )
        )
    is_tof = hist.data_type is DataType.TOF
    fwhm0, fwmax = (30.0, 300.0) if is_tof else (0.3, 3.0)
    hid = hist.id
    params += [
        Parameter(
            f"hist:{hid}:scale",
            ParamKind.HISTOGRAM,
            1e-3,
            vary=True,
            lower=1e-12,
            upper=1e8,
        ),
        Parameter(
            f"hist:{hid}:fwhm",
            ParamKind.HISTOGRAM,
            fwhm0,
            vary=True,
            lower=fwhm0 * 0.1,
            upper=fwmax,
        ),
        Parameter(f"hist:{hid}:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
    ]
    if background_order > 0:
        for k in range(background_order + 1):
            params.append(
                Parameter(
                    f"hist:{hid}:bkg_c{k}",
                    ParamKind.HISTOGRAM,
                    bkg_level if k == 0 else 0.0,
                    vary=True,
                    lower=-1e6,
                    upper=1e6,
                )
            )
    else:
        params.append(
            Parameter(
                f"hist:{hid}:bkg",
                ParamKind.HISTOGRAM,
                bkg_level,
                vary=True,
                lower=0.0,
                upper=1e6,
            )
        )
    return params


def _seed_rwp(engine, rstate, hist):
    from aura import spec

    return spec.rwp(hist.y_obs, engine.calculate(rstate, hist), hist.weights)


@main.command()
def gui():
    """Launch the Aura desktop workbench (requires PySide6)."""
    from aura.gui import GUINotAvailable, launch

    try:
        raise SystemExit(launch())
    except GUINotAvailable as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
