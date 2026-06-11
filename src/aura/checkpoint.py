"""
Checkpoint / restart of a refinement state (UX-5).

A large analysis should survive interruption. Because ``RefinementState`` is an
immutable dataclass of plain data + numpy arrays, it serializes cleanly. The one
exception is ``parametric_models``, which carry Python callables (``func``) that
do not serialize portably; they are *model definitions* the caller owns, so they
are dropped on save and re-attached on load (a warning is issued if any were
present). The data — phases, histograms, parameters, constraints, provenance log
— round-trips exactly.

Serialization uses :mod:`pickle`. Checkpoints are trusted, self-produced files;
do not load a checkpoint from an untrusted source.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import replace
from pathlib import Path

from aura.spec import ParametricModel, RefinementState


def save_refinement(state: RefinementState, path: str | Path) -> Path:
    """Write *state* to *path*. Parametric-model callables are dropped (see module
    docstring); their targets/coeff names are preserved for re-attachment."""
    path = Path(path)
    to_save = state
    model_specs: list[tuple[str, tuple[str, ...]]] = []
    if state.parametric_models:
        warnings.warn(
            "parametric_models carry callables and are not serialized; re-attach "
            "them on load via restore_models().",
            stacklevel=2,
        )
        model_specs = [
            (m.target, tuple(m.coeff_names)) for m in state.parametric_models
        ]
        to_save = replace(state, parametric_models=())
    with path.open("wb") as f:
        pickle.dump({"state": to_save, "model_specs": model_specs}, f)
    return path


def load_refinement(path: str | Path) -> RefinementState:
    """Load a refinement state written by :func:`save_refinement`."""
    with Path(path).open("rb") as f:
        payload = pickle.load(f)  # noqa: S301 - trusted, self-produced checkpoint
    return payload["state"]


def saved_model_specs(path: str | Path) -> list[tuple[str, tuple[str, ...]]]:
    """The (target, coeff_names) of parametric models present when *path* was saved,
    so the caller can rebuild and re-attach the matching ParametricModels."""
    with Path(path).open("rb") as f:
        payload = pickle.load(f)  # noqa: S301
    return payload["model_specs"]


def restore_models(
    state: RefinementState, models: tuple[ParametricModel, ...]
) -> RefinementState:
    """Re-attach parametric *models* to a loaded *state*."""
    return replace(state, parametric_models=tuple(models))


def resume(engine, path: str | Path, *, models=(), **refine_kwargs):
    """Load a checkpoint and continue refining it with *engine*.

    *models* re-attaches parametric models if the checkpoint had any.
    """
    state = load_refinement(path)
    if models:
        state = restore_models(state, models)
    return engine.refine(state, engine, engine, **refine_kwargs)
