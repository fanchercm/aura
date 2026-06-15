"""
2D image ingest, detector geometry, and azimuthal integration (Phase 10).

Validated synthetically (no real 2D data ships): project Debye rings for a known
phase onto a detector, integrate, and recover the pattern / cell. Also checks the
geometry round-trip, sector↔full consistency (SCI-4), and the .npy image reader.
"""

from __future__ import annotations

import numpy as np

import aura.io as io
from aura.engine import symmetry
from aura.engine.forward import ProductionEngine
from aura.integrate import integrate_full, integrate_sectors
from aura.io.image.geometry import DetectorGeometry
from aura.io.image.readers import Image
from aura.spec import (
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParamKind,
    Phase,
    RefinementState,
    UnitCell,
)

WL = 0.4
NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
)


def _geometry(n=512):
    return DetectorGeometry(
        distance=0.10,
        poni1=n / 2 * 100e-6,
        poni2=n / 2 * 100e-6,
        pixel1=100e-6,
        pixel2=100e-6,
        wavelength=WL,
        shape=(n, n),
    )


def _synthetic_rings(geom, phase, ring_width=0.08, bkg=5.0):
    """Paint Debye rings (∝ multiplicity·|F|²-ish) for phase onto the detector."""
    tt_pix = geom.two_theta_array()
    tt_max = float(tt_pix.max())
    d_min = WL / (2 * np.sin(np.radians(tt_max) / 2))
    refl = symmetry.generate_reflections(phase, round(d_min, 4), 40.0)
    img = np.full(geom.shape, bkg)
    ring_tts = []
    for r in refl:
        tt0 = np.degrees(2 * np.arcsin(WL / (2 * r.d)))
        if tt0 > tt_max - 0.3 or tt0 < 0.6:
            continue
        ring_tts.append(tt0)
        img += (
            1000.0 * r.multiplicity * np.exp(-0.5 * ((tt_pix - tt0) / ring_width) ** 2)
        )
    return Image(data=img, source="synthetic"), sorted(ring_tts)


class TestGeometry:

    def test_pixel_angle_round_trip(self):
        geom = _geometry()
        for r1, r2 in [(100.0, 380.0), (256.0, 256.0), (10.0, 500.0)]:
            tt, az = geom.pixel_to_angles(r1, r2)
            br1, br2 = geom.angles_to_pixel(tt, az)
            assert abs(br1 - r1) < 1e-6 and abs(br2 - r2) < 1e-6

    def test_beam_center_is_zero_2theta(self):
        geom = _geometry()
        tt = geom.two_theta_array()
        assert tt.min() < 0.05  # near the PONI 2θ ≈ 0

    def test_save_load(self, tmp_path):
        geom = _geometry()
        p = geom.save(tmp_path / "geom.json")
        loaded = DetectorGeometry.load(p)
        assert loaded == geom


class TestImageReader:

    def test_npy_round_trip(self, tmp_path):
        arr = np.random.default_rng(0).random((64, 64))
        p = tmp_path / "frame.npy"
        np.save(p, arr)
        img = io.read(p, "image")
        assert isinstance(img, Image)
        assert np.allclose(img.data, arr)

    def test_image_readers_registered(self):
        names = {r.name for r in io.registry.readers("image")}
        assert "NumPy image" in names


class TestIntegration:

    def test_full_integration_peaks_at_ring_positions(self):
        from scipy.signal import find_peaks

        geom = _geometry()
        image, rings = _synthetic_rings(geom, NABR)
        result = integrate_full(image, geom, n_bins=800)
        pk, _ = find_peaks(
            result.mean_intensity, height=np.median(result.mean_intensity) * 1.5
        )
        found = np.sort(result.two_theta[pk])
        for ring in rings[:5]:
            assert np.any(
                np.abs(found - ring) < 0.1
            ), f"no integrated peak near {ring:.2f}"

    def test_sectors_recombine_to_full(self):
        geom = _geometry()
        image, _ = _synthetic_rings(geom, NABR)
        full = integrate_full(image, geom, n_bins=800)
        state = integrate_sectors(image, geom, n_sectors=8, n_bins=800)
        assert len(state.slices) == 8
        recombined = sum(s.y for s in state.slices)
        assert np.allclose(recombined, full.intensity_sum)  # SCI-4: degrade-to-1D

    def test_image_integrate_refine_recovers_cell(self):
        """Synthetic image → integrate → refine recovers the injected cell."""
        geom = _geometry()
        image, _ = _synthetic_rings(geom, NABR)
        full = integrate_full(image, geom, n_bins=1200)
        y = full.mean_intensity
        h = Histogram(
            "img",
            DataType.CW_XRAY,
            full.two_theta,
            y,
            1.0 / np.clip(y, 1.0, None),
            {},
            wavelength=WL,
        )
        eng = ProductionEngine()
        seed = (
            Parameter(
                "phase:NaBr:cell.a",
                ParamKind.PHASE,
                5.95,
                vary=True,
                lower=5.8,
                upper=6.1,
            ),
            Parameter(
                "hist:img:scale",
                ParamKind.HISTOGRAM,
                1e-2,
                vary=True,
                lower=1e-8,
                upper=1e4,
            ),
            Parameter(
                "hist:img:bkg",
                ParamKind.HISTOGRAM,
                float(np.median(y)),
                vary=True,
                lower=0.0,
                upper=1e6,
            ),
            Parameter(
                "hist:img:fwhm",
                ParamKind.HISTOGRAM,
                0.2,
                vary=True,
                lower=0.02,
                upper=1.0,
            ),
            Parameter("hist:img:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
        )
        res = eng.refine(
            RefinementState((NABR,), (h,), seed), eng, eng, max_iter=80, seed=1
        )
        a = next(p.value for p in res.state.parameters if p.name == "phase:NaBr:cell.a")
        assert abs(a - 5.9738) < 5e-3, f"recovered a={a:.4f}"
