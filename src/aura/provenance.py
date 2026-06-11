"""
Builders and (de)serialization for :class:`aura.spec.ProvenanceManifest`.

The manifest itself is a pure frozen dataclass in :mod:`aura.spec` (part of the
state contract). This module does the impure work of *constructing* one from a
``RefinementState`` plus run configuration: hashing inputs and model topology,
reading the git commit and environment lock, and round-tripping to YAML.

Design: hashes are computed from canonical, sorted string representations so the
same inputs always produce the same digest (reproducibility), while any change
to data, model topology, phases, or instrument terms changes the corresponding
digest (change detection).
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import yaml

from aura import __version__
from aura.spec import Histogram, ProvenanceManifest, RefinementState

_REQUIRED_FIELDS = (
    "input_data_hash",
    "software_version",
    "git_commit",
    "kernel_backend",
    "optimizer",
    "parameter_graph_hash",
    "phase_model_hash",
    "instrument_model_hash",
    "environment_lock",
)


# ---------------------------------------------------------------------------
# Hash builders
# ---------------------------------------------------------------------------


def hash_inputs(histograms: Iterable[Histogram]) -> str:
    """sha256 over every histogram's (id, data_type, x, y_obs, weights)."""
    h = hashlib.sha256()
    for hist in histograms:
        h.update(hist.id.encode())
        h.update(hist.data_type.value.encode())
        for arr in (hist.x, hist.y_obs, hist.weights):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def parameter_graph_hash(state: RefinementState) -> str:
    """Hash the model *topology*: parameter identity/role/vary/bounds + ties.

    Deliberately excludes parameter *values* — this digest answers "did the model
    structure change", not "did the fit move". Values live in ``input``/result.
    """
    params = sorted(
        (p.name, p.kind.value, bool(p.vary), float(p.lower), float(p.upper))
        for p in state.parameters
    )
    models = sorted((m.target, tuple(m.coeff_names)) for m in state.parametric_models)
    constraints = tuple(sorted(state.constraints))
    return _hash_repr(("params", params, "models", models, "constraints", constraints))


def phase_model_hash(state: RefinementState) -> str:
    """Hash the crystallographic content: each phase's cell and atom sites."""
    phases = []
    for ph in state.phases:
        atoms = tuple((a.element, a.x, a.y, a.z, a.occ, a.b_iso) for a in ph.atoms)
        phases.append((ph.name, ph.space_group, ph.cell.as_tuple(), atoms))
    return _hash_repr(("phases", tuple(phases)))


def instrument_model_hash(state: RefinementState) -> str:
    """Hash per-histogram instrument terms (wavelength/DIFC/DIFA/zero/2theta)."""
    terms = sorted(
        (
            h.id,
            h.data_type.value,
            h.wavelength,
            h.difc,
            h.difa,
            h.zero,
            h.two_theta_fixed,
        )
        for h in state.histograms
    )
    return _hash_repr(("instrument", terms))


def _hash_repr(obj: object) -> str:
    return hashlib.sha256(repr(obj).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Environment / VCS
# ---------------------------------------------------------------------------


def software_version() -> str:
    return __version__


def git_commit(start: Path | None = None) -> str:
    """Return the repo HEAD commit, or ``"unknown"`` if unavailable."""
    repo = _find_up(".git", start)
    if repo is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo.parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_lock(start: Path | None = None) -> str:
    """sha256 of ``pixi.lock`` (the env lock), or ``"unknown"``."""
    lock = _find_up("pixi.lock", start)
    if lock is None or not lock.is_file():
        return "unknown"
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _find_up(name: str, start: Path | None = None) -> Path | None:
    """Walk upward from *start* (default: this file) looking for *name*."""
    base = (start or Path(__file__)).resolve()
    for parent in (base, *base.parents):
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Manifest construction + (de)serialization
# ---------------------------------------------------------------------------


def build_manifest(
    state: RefinementState,
    *,
    kernel_backend: str,
    optimizer: Mapping[str, object],
    random_seed: int | None = None,
    agent_patch_id: str | None = None,
    human_review_state: str | None = None,
) -> ProvenanceManifest:
    """Assemble a complete manifest for a refinement of *state*."""
    return ProvenanceManifest(
        input_data_hash=hash_inputs(state.histograms),
        software_version=software_version(),
        git_commit=git_commit(),
        kernel_backend=kernel_backend,
        optimizer=dict(optimizer),
        random_seed=random_seed,
        parameter_graph_hash=parameter_graph_hash(state),
        phase_model_hash=phase_model_hash(state),
        instrument_model_hash=instrument_model_hash(state),
        environment_lock=environment_lock(),
        agent_patch_id=agent_patch_id,
        human_review_state=human_review_state,
    )


def to_yaml(manifest: ProvenanceManifest) -> str:
    """Serialize a manifest to YAML."""
    return yaml.safe_dump(
        {
            "input_data_hash": manifest.input_data_hash,
            "software_version": manifest.software_version,
            "git_commit": manifest.git_commit,
            "kernel_backend": manifest.kernel_backend,
            "optimizer": dict(manifest.optimizer),
            "random_seed": manifest.random_seed,
            "parameter_graph_hash": manifest.parameter_graph_hash,
            "phase_model_hash": manifest.phase_model_hash,
            "instrument_model_hash": manifest.instrument_model_hash,
            "environment_lock": manifest.environment_lock,
            "agent_patch_id": manifest.agent_patch_id,
            "human_review_state": manifest.human_review_state,
        },
        sort_keys=True,
    )


def from_yaml(text: str) -> ProvenanceManifest:
    """Reconstruct a manifest from YAML produced by :func:`to_yaml`."""
    data = yaml.safe_load(text)
    return ProvenanceManifest(**data)


def validate(manifest: ProvenanceManifest) -> None:
    """Raise if any required field is missing/empty (oracle §13 BLOCKING rule)."""
    for name in _REQUIRED_FIELDS:
        value = getattr(manifest, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"Provenance manifest missing required field: {name!r}")
