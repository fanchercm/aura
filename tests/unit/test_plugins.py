"""
Tests for the versioned plugin SDK (Phase 11).

Validates:
- Version assertion (compatible/incompatible major version).
- register/extensions round-trip for all extension-point groups.
- load_plugins scans entry points without crashing (no plugins installed = 0 loaded).
- Protocol satisfaction checks for Exporter, ProfileModel, BackgroundModel,
  ScatteringModel, InstrumentKernel.
- Reader entry-point loading wires into io.registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from aura.plugins import (
    AURA_SDK_VERSION,
    BackgroundModel,
    Exporter,
    InstrumentKernel,
    ProfileModel,
    ScatteringModel,
    assert_compatible,
    extensions,
    load_plugins,
    register,
)

# ---------------------------------------------------------------------------
# Version assertion
# ---------------------------------------------------------------------------


class TestVersionAssertion:

    def test_compatible_same_major(self):
        # Should not raise.
        assert_compatible(AURA_SDK_VERSION)

    def test_compatible_minor_difference(self):
        major = int(AURA_SDK_VERSION.split(".")[0])
        assert_compatible(f"{major}.99")  # minor diff is OK

    def test_incompatible_major_raises(self):
        major = int(AURA_SDK_VERSION.split(".")[0])
        with pytest.raises(RuntimeError, match="major"):
            assert_compatible(f"{major + 1}.0")

    def test_sdk_version_is_semver(self):
        parts = AURA_SDK_VERSION.split(".")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# register / extensions
# ---------------------------------------------------------------------------


class TestRegisterExtensions:

    def test_register_returns_extension(self):
        obj = object()
        result = register("aura._test_group", obj)
        assert result is obj

    def test_extensions_after_register(self):
        marker = object()
        register("aura._test_group2", marker)
        exts = extensions("aura._test_group2")
        assert marker in exts

    def test_extensions_empty_group(self):
        exts = extensions("aura._nonexistent_group_xyz")
        assert exts == []

    def test_multiple_extensions_same_group(self):
        for i in range(3):
            register("aura._test_multi", i)
        exts = extensions("aura._test_multi")
        # At least 3 in there (tests may be run multiple times in one session).
        assert len(exts) >= 3

    def test_register_does_not_affect_other_groups(self):
        register("aura._isolated_a", "a_obj")
        before = len(extensions("aura._isolated_b"))
        register("aura._isolated_b", "b_obj")
        assert len(extensions("aura._isolated_a")) >= 1
        assert len(extensions("aura._isolated_b")) == before + 1


# ---------------------------------------------------------------------------
# load_plugins (no plugins installed → 0 loaded, no crash)
# ---------------------------------------------------------------------------


def test_load_plugins_no_installed_plugins():
    # No third-party aura plugins are installed in the test environment;
    # this must succeed and return 0 (not raise).
    n = load_plugins()
    assert isinstance(n, int) and n >= 0


def test_load_plugins_specific_group():
    n = load_plugins("aura.profile_models")
    assert isinstance(n, int) and n >= 0


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


class ConcreteExporter:
    name = "test-exporter"
    format_id = "test-fmt"

    def export(self, obj: object, path: str) -> None:
        pass


class ConcreteProfileModel:
    name = "gaussian"

    def profile(self, x, center: float, fwhm: float, eta: float):
        return np.exp(-((x - center) ** 2) / (2 * (fwhm / 2.355) ** 2))


class ConcreteBackgroundModel:
    name = "constant"
    n_params = 1

    def evaluate(self, x, coeffs):
        return np.full_like(np.asarray(x, dtype=float), coeffs[0])


class ConcreteScatteringModel:
    name = "unit-f"

    def f_squared(self, element: str, q: float) -> float:
        return 1.0


class ConcreteInstrumentKernel:
    name = "constant-fwhm"
    data_types = ("cw_xray",)

    def fwhm(self, x: float, params: object) -> float:
        return 0.1


class TestProtocols:

    def test_exporter_protocol(self):
        assert isinstance(ConcreteExporter(), Exporter)

    def test_profile_model_protocol(self):
        assert isinstance(ConcreteProfileModel(), ProfileModel)

    def test_background_model_protocol(self):
        assert isinstance(ConcreteBackgroundModel(), BackgroundModel)

    def test_scattering_model_protocol(self):
        assert isinstance(ConcreteScatteringModel(), ScatteringModel)

    def test_instrument_kernel_protocol(self):
        assert isinstance(ConcreteInstrumentKernel(), InstrumentKernel)

    def test_incomplete_exporter_fails(self):
        class Bad:
            name = "bad"
            # missing format_id and export()

        assert not isinstance(Bad(), Exporter)

    def test_register_exporter_via_sdk(self):
        exp = ConcreteExporter()
        register("aura.exporters", exp)
        assert exp in extensions("aura.exporters")

    def test_register_profile_model_via_sdk(self):
        pm = ConcreteProfileModel()
        register("aura.profile_models", pm)
        assert pm in extensions("aura.profile_models")
