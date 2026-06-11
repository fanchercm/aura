"""
Provenance manifest contract (oracle §13 — BLOCKING).

A refinement result without complete provenance is not a trustworthy scientific
result. These tests assert the manifest is fully populated after a refine, that
identical inputs hash identically (reproducibility), that perturbing inputs or
model topology changes the right digest (change detection), and that the
manifest round-trips through YAML.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aura import provenance
from aura.provenance import _REQUIRED_FIELDS
from aura.reference import RefEngine
from aura.spec import (
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParamKind,
    Phase,
    ProvenanceManifest,
    RefinementState,
    UnitCell,
)

WL = 1.5406


def _phase(a: float = 5.431) -> Phase:
    return Phase("Si", "Fd-3m", UnitCell(a, a, a), (AtomSite("Si", 0, 0, 0),))


def _histogram(hid: str = "h0", n: int = 80) -> Histogram:
    x = np.linspace(10.0, 80.0, n)
    y = 100.0 + np.zeros(n)
    w = 1.0 / np.clip(y, 1.0, None)
    return Histogram(hid, DataType.CW_XRAY, x, y, w, {"T": 300.0}, wavelength=WL)


def _params(hid: str = "h0") -> tuple[Parameter, ...]:
    return (
        Parameter(
            "phase:Si:cell.a", ParamKind.PHASE, 5.431, vary=True, lower=5.0, upper=6.0
        ),
        Parameter(
            f"hist:{hid}:scale",
            ParamKind.HISTOGRAM,
            1.0,
            vary=True,
            lower=0.0,
            upper=1e6,
        ),
        Parameter(f"hist:{hid}:bkg", ParamKind.HISTOGRAM, 5.0, vary=False),
        Parameter(f"hist:{hid}:fwhm", ParamKind.HISTOGRAM, 0.3, vary=False),
        Parameter(f"hist:{hid}:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
    )


def _state() -> RefinementState:
    h = _histogram()
    return RefinementState((_phase(),), (h,), _params())


def _refine():
    eng = RefEngine()
    return eng.refine(_state(), eng, eng, max_iter=5, seed=12345)


# --- Manifest completeness -----------------------------------------------------


class TestManifestPresent:

    def test_refine_attaches_a_manifest(self):
        res = _refine()
        assert isinstance(res.provenance, ProvenanceManifest)

    def test_all_required_fields_present_and_nonempty(self):
        res = _refine()
        provenance.validate(res.provenance)  # raises if any required field empty
        for name in _REQUIRED_FIELDS:
            value = getattr(res.provenance, name)
            assert value is not None
            if isinstance(value, str):
                assert value.strip()

    def test_seed_and_backend_recorded(self):
        res = _refine()
        assert res.provenance.random_seed == 12345
        assert res.provenance.kernel_backend == "reference-numpy"
        assert res.provenance.optimizer["max_iter"] == 5

    def test_validate_rejects_missing_field(self):
        res = _refine()
        broken = replace(res.provenance, input_data_hash="")
        with pytest.raises(ValueError, match="input_data_hash"):
            provenance.validate(broken)


# --- Profiling diagnostics (PERF-6) -------------------------------------------


class TestDiagnostics:

    def test_profiling_hooks_populated(self):
        res = _refine()
        for key in ("wall_time_s", "n_forward_evals", "peak_rss_mb", "n_points"):
            assert key in res.diagnostics
        assert res.diagnostics["n_forward_evals"] > 0
        assert res.diagnostics["n_points"] == 80
        assert res.diagnostics["wall_time_s"] >= 0.0


# --- Reproducibility & change detection ---------------------------------------


class TestHashing:

    def test_same_inputs_same_hashes(self):
        a = _refine().provenance
        b = _refine().provenance
        assert a.input_data_hash == b.input_data_hash
        assert a.parameter_graph_hash == b.parameter_graph_hash
        assert a.phase_model_hash == b.phase_model_hash
        assert a.instrument_model_hash == b.instrument_model_hash

    def test_perturbed_data_changes_data_hash_only(self):
        base = _state()
        h = base.histograms[0]
        bumped = replace(h, y_obs=h.y_obs + 1.0)
        perturbed = replace(base, histograms=(bumped,))
        assert provenance.hash_inputs(perturbed.histograms) != provenance.hash_inputs(
            base.histograms
        )
        # Model topology unchanged.
        assert provenance.parameter_graph_hash(
            perturbed
        ) == provenance.parameter_graph_hash(base)

    def test_topology_change_changes_graph_hash(self):
        base = _state()
        # Free a previously-fixed parameter => different model topology.
        newp = tuple(
            replace(p, vary=True) if p.name.endswith(":bkg") else p
            for p in base.parameters
        )
        changed = replace(base, parameters=newp)
        assert provenance.parameter_graph_hash(
            changed
        ) != provenance.parameter_graph_hash(base)

    def test_phase_change_changes_phase_hash(self):
        base = _state()
        changed = replace(base, phases=(_phase(a=5.50),))
        assert provenance.phase_model_hash(changed) != provenance.phase_model_hash(base)


# --- Serialization -------------------------------------------------------------


class TestYamlRoundTrip:

    def test_to_from_yaml_identity(self):
        m = _refine().provenance
        text = provenance.to_yaml(m)
        restored = provenance.from_yaml(text)
        assert restored == m
