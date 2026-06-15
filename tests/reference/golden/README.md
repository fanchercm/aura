# tests/reference/golden — GSAS-II golden files (release gate)

This directory holds one-time captures of reference GSAS-II refinements that
serve as cross-validation ground-truth for the aura engine. The comparison
asserts:

    |Δ| < k · sqrt(σ_aura² + σ_gsasii²),    k ≈ 2–3

at the cell-parameter level, and a few percent on peak positions and intensities.

## Capture procedure

1. Refine the target dataset in GSAS-II (instrument: instprm/prm on disk).
2. Export the refined .lst file and extract cell parameters + esds.
3. Store as `<dataset>_gsasii.json`:

    ```json
    {
        "dataset": "PBSO4_CWN",
        "software": "GSAS-II",
        "git_commit": "<gsasii_commit>",
        "date": "YYYY-MM-DD",
        "cell": {"a": 6.9591, "b": 8.4779, "c": 5.3962,
                 "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        "cell_esd": {"a": 0.0003, "b": 0.0003, "c": 0.0003,
                     "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        "rwp": 0.078,
        "gof": 1.24
    }
    ```

## Datasets targeted for golden-file capture

| File | Phase | DataType | Status |
|------|-------|----------|--------|
| PBSO4.CWN | PbSO4 | CW neutron (D1A) | pending — needs GSAS-II run |
| PBSO4.XRA | PbSO4 | CW X-ray | pending |
| 11BM_NAC.fxye | NAC | CW X-ray (synchrotron) | pending |
| SNAP + instprm | NaBr+Pb | TOF (SNAP) | pending |

## CI integration

Golden-file tests in `tests/reference/` are **not** run on every PR.
They run only at release tags or on manual `workflow_dispatch`. See
`.github/workflows/tests.yml` for the `reference` marker filter.
