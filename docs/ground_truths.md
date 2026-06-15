# Ground Truths

This file captures key findings, decisions, and verified facts discovered during development. It serves as a persistent knowledge base that AI assistants (like GitHub Copilot) and developers can reference across sessions.

**Why this matters:** AI assistants don't remember previous conversations. By recording important discoveries here, you ensure that context isn't lost between sessions. When Copilot reads this file, it can make better suggestions based on what's already been learned about your project.

## How to Use This File

- **Add entries as you discover important facts** — things like API quirks, configuration requirements, performance constraints, or design decisions.
- **Include the date and context** so future-you (or Copilot) understands why something was noted.
- **Link to relevant code or docs** when helpful.
- Copilot is instructed to update this file automatically when it discovers key findings during development.

## Findings

<!-- Add your key findings below. Use the format: -->
<!-- ### YYYY-MM-DD: Brief title -->
<!-- Description of the finding, why it matters, and any relevant links. -->

### 2026-04-09: Phase 0 complete — template customized to Aura

- `pyproject.toml` updated: name=`aura`, Python >=3.11, core deps (numpy, scipy, matplotlib, click, tqdm, pyyaml)
- `pixi.toml` created for environment management. All deps via conda-forge (ORNL network TLS issues block pypi-dependencies in pixi). Aura itself installed editable via `pixi run pip install -e . --no-deps`.
- CLI entry point `aura` registered and working (`aura --version` → `0.1.0`). Uses `click.group()` for subcommand architecture.
- All stale `package_name` references removed from source, tests, and tool configs.
- 2 tests passing, 100% coverage on existing code.

### 2026-04-10: Phase 1 — Core data model and GSAS loaders

- **Data model** (`src/aura/models.py`): `Campaign` → `MeasurementState` → `DiffractionSlice` hierarchy implemented as dataclasses. `ParameterScope` enum (GLOBAL, STATE, SLICE) defines scoping levels.
- **GSAS loaders** (`src/aura/loaders.py`): `load_gsa_file()` parses a `.gsa` column file into a `MeasurementState` with one `DiffractionSlice` per bank. `load_instprm()` parses `.instprm` files into per-bank parameter dicts. `load_campaign_from_directory()` loads a full directory of `.gsa` files into a `Campaign`.
- **Test data**: 19 SNAP runs (67702–67720) in `tests/testDataGsas/`, 6 banks each = 114 histograms. Two-phase sample: NaBr + Pb. Runs are sequential in increasing pressure (room temperature). CIF files for both phases included.
- **GSA format notes**: Header is JSON-ish dict (lines before first `#`). Each bank block: metadata comment (`# Total flight path ...m, tth ...deg, DIFC ...`), spectrum comment, `BANK N` line, then columnar TOF/intensity/error. Some banks have leading NaN values (masked region).
- **instprm format**: `#Bank N:` comment headers, then `key:value` pairs per bank. Key parameters: `difC`, `2-theta`, `fltPath`, profile coefficients (sig-0/1/2, alpha, beta-0/1).
- 41 tests passing, 96% overall coverage.

### 2026-04-10: Phase 1 — Parameter system and RefinementModel

- **Parameter system** (`src/aura/parameters.py`): `Parameter` (single refinable value with scope, bounds, fixed/free, initial_value/reset), `ParameterSet` (ordered collection with vector I/O, bulk fix/free, scope/name filtering), `Constraint` (functional relationship between params — equality, linear, or arbitrary callable), `RefinementModel` (ties Campaign + ParameterSet + Constraints; provides independent free-parameter vector excluding fixed and constrained-dependent params).
- **Design decisions**: (1) Parameter trajectory/history stored *externally* in refinement engine, not on Parameter — keeps the class lightweight and serializable. Parameter only stores `initial_value` for reset. (2) Crystallographic constraints (bond-length, symmetry-aware) deferred — require space-group machinery (gemmi/diffpy). Phase 1 constraints: equality ties, linear ties, callable ties. (3) `param_id` is an opaque unique key — scope qualifiers encoded by convention (e.g. `"NaBr:a"`, `"SNAP067702:bank1:bkg_c0"`) but not parsed.
- **Realistic SNAP test**: 436 parameters (2 lattice + 36 profile + 18 calibration + 38 state-scales + 342 backgrounds), 382 free, profile/calibration fixed by default. Constraint reduces independent count correctly.
- 92 tests passing, 97% overall coverage. `parameters.py` at 100%.

### 2026-06-11: Phase 0 (next-gen rebuild) — oracle adopted as package core

Start of the next-generation rebuild per the approved plan
([using-the-requirments-planning-humble-parasol.md](../../.claude/plans/using-the-requirments-planning-humble-parasol.md)).
The executable specification is now the authoritative core; the old mutable
`parameters.py` layer is slated for deletion in Phase 1.

- **`docs/spec.py` → `src/aura/spec.py`**: the canonical core (numpy oracle
  kernels, immutable `RefinementState`/`Phase`/`Histogram`/`Parameter`,
  `@runtime_checkable` Protocols `ForwardModel`/`ParametricEngine`/`Minimizer`/
  `DomainModule`/`PhaseIdentifier`, and the `ACCEPTANCE` tolerance dict).
  `python -m aura.spec` runs the self-consistency check (PASS).
- **`RefEngine` → `src/aura/reference.py`**: the reference numpy engine
  (single cubic phase, scale + flat bkg, pseudo-Voigt/reflection). It is the
  *oracle* the physics suite runs green against today, and the reference a
  production engine is checked against. Conforms to all three Protocols.
- **Oracle suite → `tests/physics/test_invariants.py`**: imports rewritten to
  `aura.spec`/`aura.reference`. The `engine` fixture moved to `tests/conftest.py`
  — this is the **single swap point** to point the suite at a production engine
  (will become `params=[RefEngine, ProductionEngine]` from Phase 3).
- **Tier layout**: `tests/{unit,physics,campaign,reference,ai,performance}/`,
  each with a `README.md` of intended invariants. Existing tests moved to
  `tests/unit/` (data-dir paths fixed to `parent.parent`; the 19-run campaign
  load marked `@pytest.mark.integration`). 92 unit tests pass.
- **CI/config fixes**: `tests.yml` matrix `3.9/3.10` (violated
  `requires-python>=3.11`) → `[3.11,3.12,3.13]` + a single macOS smoke leg;
  stale `--cov=src/package_name` → `src/aura`; added a `python -m aura.spec`
  step. `pyproject` mypy `python_version` `3.9`→`3.11`; added `physics` marker.
- **Verified green**: 92 unit tests (4.9s) + 33 physics invariants (**943s ≈
  15m43s**) all pass; `python -m aura.spec` PASS; `RefEngine` conforms to all
  three Protocols.
- **CI implication**: at ~16 min the *full* physics tier is too slow to block
  every PR. The fast non-refining subset (crystallography, state, conformance,
  forward purity, Jacobian, AI guardrails — 21 tests) runs in **0.64s** and is
  the right per-PR gate; the slow FD-refinement invariants belong nightly until
  the Phase 6 AD backend cuts their cost. The reference `Minimizer` uses
  finite-difference Gauss–Newton that runs to `max_iter` — this is the
  fragility/cost that motivates Phase 6.

### 2026-06-11: Phase 1 — native importer registry + readers + bridge

A native, pluggable importer registry (`src/aura/io/`) modeled on GSAS-II's
reader-class design, **with no GSAS-II dependency** (decision: native
reimplementation, all four importer domains). The old mutable `parameters.py`
is deleted; the spec model is now the only parameter model.

- **Registry** (`io/registry.py`): a `@runtime_checkable Reader` Protocol
  (`name`, `domain`, `extensions`, `contents_validator`, `read`) + a
  `ReaderRegistry` that selects by **content sniffing**, not just extension
  (so the shared `.gsa` extension routes to the TOF column reader vs CONST
  reader correctly, and `.cif` resolves powder-vs-phase). `aura.readers`
  entry-point hook seeds the Phase-11 plugin SDK. Long-tail formats are
  registered as explicit `NotImplementedError` stubs so coverage gaps are
  testable, not silent.
- **gemmi** added as a dependency (`>=0.6.0`) for CIF parsing (pip-installable
  here; network worked despite the pixi/PyPI note). `CIFReader` handles
  GSAS-II-flavored CIFs (`_space_group_name_H-M_alt` + symop loop, esd notation
  `5.9738(7)`); NaBr/Pb/NAC all parse to physical cells.

**Hard-won GSAS format decodings** (verified empirically against the data —
record so we never re-derive them):
- **GSAS CONST raw** (`.XRA`, `.CWN`): points are packed `(I2,I6)` per 8-char
  field. `I2` = **number of detectors contributing** to that point, `I6` =
  **summed counts**. So `y = counts/ndet`, `esd = sqrt(counts)/ndet`. The D1A
  CWN data has `I2` varying 1→10 (detector overlap across the 2θ scan); lab XRA
  has `I2` blank ⇒ ndet=1. Abscissa from the BANK record's `CONST <start>
  <step>` in **centidegrees** (÷100 → 2θ°). The title line's start/step/end is
  human-readable only and can disagree (FAP title says end 90° but
  nchan×step ⇒ 130°; the BANK record is authoritative).
- **FXYE** (11-BM): free-format `X Y E` with **X in centidegrees** (÷100). The
  54000-step header is nominal; read all rows (≈59.5k here), step ≈0.001°.
- **TOF `.gsa`** (SNAP): SLOG banks, handled by the original `load_gsa_file`
  (wrapped as `GSASColumnReader`); distinguished from CONST by `SLOG`/`DIFC`.
- **Old `.prm`** (D1A): strict fixed columns — `INS`(1-3), blank(4), bank(5-6),
  key(7+). `ICONS[0]` = CW wavelength (1.909 Å); `HTYPE PNCR` ⇒ CW neutron.
  Getting the column offsets wrong silently misparses the key as the bank digit.
- **Note**: any uniform error in the CONST `ndet` interpretation would be a pure
  scale factor (peak *positions* — the physics — are unaffected), so the
  refinement scale absorbs it; cross-check the absolute intensity scale against
  GSAS-II at the Phase-11 reference gate.

- **Bridge** (`bridge.py`): one-way `Campaign/MeasurementState → tuple[Histogram]`.
  Drops invalid points via `valid_mask` (SNAP bank1 3386→3369, leading NaNs),
  weights `1/clip(σ²,1,∞)` (Poisson), routes data-type fields
  (wavelength/difc/two_theta_fixed), derives `driving` from state metadata
  (run_number as the SNAP stimulus proxy). Raises clearly if a CW slice lacks a
  wavelength (supply via `apply_instprm` or `wavelength=`).
- **Coverage now**: TOF (SNAP) ✓, CW neutron (PbSO₄/D1A) ✓, CW X-ray
  (PbSO₄/FAP/NAC) ✓ — three of four `DataType`s on real data. EDD + 2D images +
  single-crystal remain (Phase 10).
- **Tests**: 69 unit tests pass (was 92; the ~50-test `test_parameters.py` for
  the retired mutable layer was replaced by leaner `test_parameters_spec.py` +
  the new `test_io.py`). ruff + black clean across `src/`+`tests/`. Phase 1
  full-suite confirmation: **102 passed** (69 unit + 33 physics) in 15m.

### 2026-06-11: Phase 2 — provenance manifest (BLOCKING contract)

Landed early, before the production engine, because it changes `RefinementResult`'s
shape and `Minimizer.refine`'s signature — retrofitting later would touch every
call site (oracle §13 makes provenance BLOCKING).

- **`ProvenanceManifest`** (frozen dataclass in `spec.py`, kept pure — no I/O):
  input_data_hash, software_version, git_commit, kernel_backend, optimizer,
  random_seed, parameter_graph_hash, phase_model_hash, instrument_model_hash,
  environment_lock, agent_patch_id, human_review_state. Added as an optional
  `provenance` field on `RefinementResult`.
- **`src/aura/provenance.py`** does the impure construction: `hash_inputs`
  (sha256 over histogram x/y/weights bytes), `parameter_graph_hash` (sorted
  name/kind/vary/bounds + model ties — **topology, not values**),
  `phase_model_hash`, `instrument_model_hash`, `git_commit` (subprocess, walks
  up for `.git`, "unknown" fallback), `environment_lock` (sha256 of `pixi.lock`),
  `build_manifest`, `to_yaml`/`from_yaml`, `validate`. Placed in a separate
  module to keep `spec.py` import-light and dependency-free (avoids a
  spec→provenance cycle).
- **`Minimizer.refine` gained `seed: int | None = None`** (recorded for
  determinism; default keeps the physics suite's `refine(st, eng, eng, ...)`
  calls working). `RefEngine.refine` now populates the manifest + profiling
  diagnostics (PERF-6): `wall_time_s`, `n_forward_evals`, `peak_rss_mb`
  (coarse, via `resource.getrusage`), `n_points` — alongside the existing
  `condition_number`.
- **Hash design**: canonical sorted `repr` → sha256, so identical inputs hash
  identically (reproducibility) and any change to data / model topology / phase /
  instrument changes the *corresponding* digest (change detection). Param-graph
  hash deliberately excludes parameter *values* — it answers "did the model
  structure change", not "did the fit move".
- **Cost note**: building a manifest each `refine` calls `git rev-parse`
  (subprocess) and reads `pixi.lock`. Fine for the reference oracle; a production
  engine should cache the env/git lookups across a campaign's many refines.
- 79 unit tests pass (10 new BLOCKING provenance tests in `test_provenance.py`);
  ruff + black clean.

### 2026-06-11: Phase 3 — production forward model (all 4 data types)

The first real, non-stub engine component: `src/aura/engine/` with a multi-phase,
all-data-type forward model on real crystallography. **"First real Rietveld
works" prerequisite.**

- **`engine/symmetry.py`** — gemmi-backed `generate_reflections(phase, d_min,
  d_max)`. **Vectorized**: computes every index's d-spacing via the reciprocal
  metric tensor (`spec.reciprocal_metric_tensor`, numpy einsum) and range-filters
  in bulk, then calls gemmi only per *observable* reflection for absence +
  symmetry orbit. NAC d≥0.41 Å: 1486 reflections in **0.25s** (a naive per-hkl
  gemmi loop took minutes). Multiplicity = symmetry+Friedel orbit size.
- **Crystallographic correctness validated**: FCC multiplicities 111→8, 200→6,
  220→12, 311→24; F-centring absences (100/110/210 absent); and the subtle
  **Laue m-3 vs m-3m** distinction — NAC (I2₁3, point group 23 → Laue m-3)
  correctly *splits* {310}/{301} into two orbits (mult 12 each), which only
  happens with the true point group, not an assumed m-3m.
- **`engine/scattering.py`** — real factors via gemmi: X-ray `Element.it92.calculate_sf(stol2)`
  (Q-dependent, f(0)≈Z), neutron `Element.neutron92.calculate_sf` (constant b,
  Na=3.63 fm). `stol2 = 1/(4d²)`.
- **`engine/positions.py`** — `position(d, hist)` dispatches to the spec CW/TOF/EDD
  laws; `d_range_for_histogram` inverts over the x-range to bound generation
  (caps the low-angle d→∞ edge).
- **`engine/forward.py`** — `ProductionForward` (multi-phase; intensity = scale ·
  phase_scale · multiplicity · |F|² · LP; flat bkg) with **windowed** peak
  evaluation (±12·FWHM via `searchsorted` → O(points × nearby_peaks), the Phase-8
  invariant) and FD Jacobian. Reflection generation cached per (hashable Phase,
  rounded d-range). Parameter convention: `phase:<name>:cell.{a..gamma}`,
  `phase:<name>:atom<i>.{x,y,z,occ,b_iso}`, `phase_scale:<phase>:<hist>`,
  `hist:<id>:{scale,bkg,fwhm,eta}` — absent keys fall back to the phase/default.
  `ProductionEngine` composes it (minimizer/parametric/JAX attach in Phases 4-6).
- **Validation**: forward purity / non-negativity / Jacobian shape; **peak
  positions exact** — all in-range predicted peaks have the calc maximum at the
  predicted angle, equal to the analytic Bragg 2θ to 1e-9. **Real TOF data**:
  NaBr+Pb forward correlates +0.315 with observed SNAP — predicted Bragg peaks
  fall where the data has intensity.
- **Fixture note**: the shared `engine` fixture stays on `RefEngine`;
  `ProductionForward` is tested directly in `test_production_forward.py`. Fixture
  parametrization over `[RefEngine, ProductionEngine]` waits for Phase 4 (when
  `refine` exists), since the shared fixture feeds refine-based invariants.
- **11-BM `NAC.fxye` calibrated wavelength = 0.413909 Å** (user-supplied; not
  stored in the file). At this λ, NAC reflections land exactly on the observed
  strong peaks — (211)→5.668°, (310)→7.320°, (222)→8.020° — and the NAC forward
  pattern correlates **+0.48** with the observed 11-BM data. So **CW X-ray
  real-data alignment is validated** (NAC), alongside TOF (SNAP). My earlier
  λ-scan missed this because the synchrotron peaks are razor-sharp (~0.006°) and a
  0.01-Å scan grid stepped over the true value — a lesson that sharp-peak
  correlation is hypersensitive to λ. X-unit confirmed centidegrees (×100 →
  impossible >180°). Constant lives in `test_production_forward.py::NAC_11BM_WAVELENGTH`.
- **PbSO₄ + FAP CIFs added** (PbSO₄ *Pbnm* a=6.955 b=8.472 c=5.397, 5 atoms;
  FAP *P6₃/m* a=9.370 c=6.880, 7 atoms). Real-data alignment now validated on
  **every available pairing**:
  - TOF — SNAP NaBr+Pb: +0.32
  - CW neutron — **PbSO₄/D1A (the IUCr round-robin): +0.26** ← CW-neutron closed
  - CW X-ray — PbSO₄/Cu: +0.51 · NAC/11-BM: +0.48 · FAP/Cu: +0.20
  FAP is lower because of a ~0.3° zero/lattice offset visible in its predicted-vs-
  observed peaks (predicted (201) 25.47° vs observed 25.80°) — exactly what a
  Phase-4 zero/cell refinement fixes; positions are otherwise correct.
- **Only EDD remains** without real data (synthetic round-trip only).
- 88 unit + 22 forward/symmetry tests pass (TOF + CW-neutron + 3× CW-xray real-
  data checks); `test_invariants` unaffected; ruff + black clean.
- **Phase 3 COMPLETE**: forward model correct for all 4 data types; every real
  dataset (3 of 4 modalities) validated.

### 2026-06-11: Phase 4 — production minimizer ("first real Rietveld fit")

`src/aura/engine/minimize.py` — `ProductionMinimizer.refine` on
`scipy.optimize.least_squares` (trust-region reflective, bounded), minimizing the
weighted residual **stacked across all histograms** (the parametric/surface
objective). Jacobian assembled from the forward model's own `jacobian` (so the
minimizer is agnostic to FD-vs-AD). Outputs Rwp, reduced χ², covariance→σ,
condition number, profiling diagnostics, and the full provenance manifest.

- **`ProductionEngine` now conforms to all three Protocols** (ForwardModel +
  ParametricEngine + Minimizer): `refine` delegates to the minimizer, `expand`
  is identity until Phase 5. So one object can be passed as forward/parametric/
  minimizer (matching the oracle's `engine.refine(st, engine, engine)` call).
- **Cubic cell shim** in `forward._effective_phase`: if the base cell is cubic
  and only `cell.a` is refined (no explicit b/c), keep a=b=c — so single-parameter
  cubic refinement works (the common case + the oracle's convention). Lower-
  symmetry coupling is via parametric ties (Phase 5).
- **Validated (categories C/E/F/G) on production-self-generated data**: fixpoint
  at truth (Rwp=0, exact), parameter recovery within σ, GoF calibration ∈[0.8,1.5]
  on noisy data, determinism, refined-cell-physical, conditioning reported,
  zero-weight masking respected, grossly-wrong fixed model → poor Rwp, provenance
  recorded. 13 tests in `test_production_refine.py`.
- **Data-coupling insight**: the oracle's `_make_histogram` bakes in `RefEngine`,
  so the production engine can't be naively dropped into the shared fixture (it
  would be fitting a *different* model's data). Production refinement tests
  therefore generate their own self-consistent fixtures. **Full-oracle fixture
  parametrization over `[RefEngine, ProductionEngine]` is deferred to Phase 5**,
  when the real `expand` exists (category D needs it).
- **FIRST REAL RIETVELD FIT** — PbSO₄/D1A CW-neutron (the IUCr round-robin):
  refines in 2.5s (14 nfev), **Rwp 0.661 → 0.522**, orthorhombic lattice recovered
  to **a=6.9544, b=8.4727, c=5.3937 Å** (reference 6.955/8.472/5.397) within
  ~0.001 Å with realistic σ, physical cell. **GoF ≈ 103** is high *and that is
  correct*: the flat-background / single-width / approximate-LP model is
  inadequate for real data — the R-factor-non-oracle rule in action (don't trust
  Rwp alone; GoF exposes model inadequacy). A *good* Rwp awaits the Phase-7
  background basis + better profile/intensity models. The lattice recovery is the
  real, verifiable result.
- `test_invariants` unaffected (no spec/reference change). 114 tests pass (88 unit
  + 13 forward + 13 refine); ruff + black clean.
- **Process lesson**: the Phase-4 commit briefly shipped a black-unclean
  `forward.py` (a `# noqa` edit after the last black run). **Always run `black`
  as the final step before committing**, after any post-lint code edits.

### 2026-06-11: Phase 5 — real ParametricEngine (the load-bearing wall)

The core scientific contribution: a single evolving model fit across an ensemble,
so shared coefficients replace one-parameter-per-pattern (Stinton & Evans 2007).

- **`engine/parametric.py`** — `ProductionParametric.expand(state, hist)`: pure
  `RefinementState → RefinementState` that evaluates each `ParametricModel.func(
  coeffs, hist.driving)` and writes the result into the target parameter (identity
  when no models). **`engine/models_parametric.py`** — `identity_model` (=
  degenerate per-histogram, reproduces independent refinement), `linear_model`
  (a0 + slope·driving[var]), `polynomial_model`, `from_callable`.
  `ProductionEngine.expand` now delegates here.
- **Minimizer Jacobian refactor (key subtlety)**: with parametric models the
  *free* parameters are the coefficients (`param:a0`, …), but the forward model
  reads the *target* (`phase:…:cell.a`). The Phase-4 Jacobian (assembled from
  `forward.jacobian` over varied params) gave **zero** for coefficients — they
  don't appear in `calculate`. Fixed by computing the residual Jacobian as a
  **central FD over the free parameters at the residual level**, with `expand`
  *inside* the perturbation: perturbing a coefficient re-runs expand and
  propagates to every driven histogram (the parametric chain rule, numerically).
  Phase 6 replaces this with autodiff *through* expand. Phase-4 (non-parametric)
  refinement tests still pass with the new Jacobian.
- **Category D validated** (`test_production_parametric.py`, 6 tests, ~16s):
  degenerate identity models == independent refinement to 1e-4 (the strict-
  generalization claim); a(P)=a0+αP recovered across 5 pressure states
  (a0→5.970, α→0.020 with σ); **parametric fit reduces per-state RMS scatter vs
  independent fits** (Stinton–Evans benefit); determinism; expand identity-without-
  models and driving-resolution + immutability.
- **Engine-swap (`test_engine_swap.py`, 8 tests)**: both RefEngine and
  ProductionEngine satisfy the engine-agnostic oracle invariants (Protocol
  conformance, forward purity, non-negativity, Jacobian-vs-FD self-consistency)
  via a parametrized fixture — the sound expression of the spec's "swap the
  engine" goal.
- **Fixture-parametrization decision (Phase 5d)**: full parametrization of the
  *refinement* oracle over both engines was **not** done. The oracle's
  `_make_histogram` generates data with `RefEngine`, so the production engine
  can't fit it to truth (data/model mismatch) — "one-line fixture swap" is
  insufficient; engine-self-consistent data is required. Categories C/D/E/F/G are
  therefore validated for the production engine by the dedicated
  `test_production_{forward,refine,parametric}.py` suites (self-generated data),
  and the engine-agnostic subset by `test_engine_swap.py`. Documented rather than
  forcing a brittle refactor + 2× ~15-min runtime.
- `test_invariants` unaffected (no spec/reference change). 128 tests pass (88 unit
  + 40 production/swap physics) + 33 oracle; ruff + black clean. Phase 5 complete.

### 2026-06-11: Phase 6 — JAX/autodiff backend

`src/aura/engine/forward_jax.py` — `JaxForward`, a differentiable reimplementation
of the CW forward model whose Jacobian comes from **autodiff** (`jax.jacfwd`),
not finite differences (JAXFit/JAX-COSMO substrate). `jax>=0.4` optional dep
(`pip install -e ".[jax]"`); CPU.

- **Design**: reflections (hkl + multiplicity) are pre-enumerated with gemmi at
  the current cell (a discrete set, fixed under AD's infinitesimal perturbations);
  everything continuous — d via the reciprocal metric, CW positions, structure
  factors (IT92 X-ray Gaussians `Σaᵢexp(-bᵢs²)+c`, constant neutron b, coefs from
  gemmi), full-grid pseudo-Voigt — is JAX, so the model differentiates w.r.t.
  cell / atom / scale / bkg / fwhm / eta. The param→input gather (`_input_spec`)
  maps each physics input to a varied x-slot or a constant, including the cubic
  shim, so `jacfwd` gives `(n_points, n_varied)`.
- **MUST be 64-bit**: `jax.config.update("jax_enable_x64", True)` on import.
  float32 would blow the AD-vs-FD and numpy-parity tolerances.
- **Validated** (`test_jax_backend.py`, 6 tests): **AD Jacobian vs independent FD
  of the same JAX model agrees to `jacobian_rtol`=1e-4** (actual ~1e-7) on every
  varied parameter for CW X-ray AND CW neutron, *including `cell.a`* — the
  GSAS-II metric-tensor failure class differentiates cleanly through the direct
  G* (no A-tensor layer). JAX vs numpy full-grid forward agree to **1.2e-7**
  (same physics). `ProductionEngine(backend="jax")` conforms to ForwardModel.
- **`ProductionEngine(backend="numpy"|"jax")`** switch (default numpy, backward-
  compatible). `expand`/`refine` unchanged.
- **Scope**: JAX backend is CW single-phase. TOF/EDD and multi-phase raise
  `NotImplementedError` (numpy backend covers them). Wiring the *minimizer* to
  consume AD Jacobians directly (vs the Phase-5 residual-FD) for speed, and
  extending JAX to TOF/EDD/multi-phase, are incremental follow-ups — the Phase-6
  deliverable is the differentiable substrate + the AD-vs-FD invariant made real.
- 46 production+JAX physics tests pass (incl. 6 JAX); `test_invariants` unaffected;
  ruff + black clean. Phase 6 complete.

### 2026-06-11: Phase 7 — domain modules (background-first), texture, QPA

`src/aura/domains/` — pluggable physics contributions, configured on the engine,
reading their refinable coefficients from `state.parameters`.

- **`ChebyshevBackground`** (the priority — background is the dominant parameter-
  explosion risk, MOD-3): a spec `DomainModule` whose `contribute(state, hist,
  y_calc)` adds `Σ c_k T_k(t)` (t = x mapped to [-1,1]), reading `hist:<id>:bkg_c{k}`.
  A handful of coefficients replace a per-channel background. Wired into
  `ProductionForward(domain_modules=...)` / `ProductionEngine(domain_modules=...)`,
  applied after the peak sum.
- **CONCRETE WIN (closes the Phase-4 loop)**: refining real PbSO₄/D1A CW-neutron
  with a Chebyshev(6) background drops **Rwp 0.52 → 0.39** (GoF 103 → 58) while the
  lattice stays recovered (6.955/8.470/5.393). Remaining gap is the single-FWHM
  profile / approximate LP — profile-coefficient refinement is a later improvement.
- **`MarchDollase`** texture: a **per-reflection** preferred-orientation
  multiplier `P(α)=(r²cos²α+sin²α/r)^(-3/2)` (α = angle between reflection and PO
  axis via the reciprocal metric). It is *not* a `DomainModule` — texture is
  multiplicative per reflection, which the `contribute(y_calc)` interface can't
  express (the summed pattern has lost per-reflection identity) — so it's a
  reflection-level correction applied inside the forward loop, gated on
  `phase:<name>:march.ratio`. **r=1 is the random/texture-free limit (P≡1)**;
  r<1 enhances along-axis reflections (P=r⁻³) and suppresses perpendicular
  (P=r^1.5) — the platy-habit convention.
- **Interface lesson**: the spec `DomainModule.contribute(y_calc)` fits *additive*
  contributions (background, diffuse) but not *per-reflection multiplicative* ones
  (texture, extinction). Aura uses two hooks: additive `DomainModule`s and a
  reflection-level `texture` correction.
- **QPA closure** (`aura/quantify.py`): `phase_weight_fractions(phases, scales)` =
  `s·ZMV` normalized; closure (Σ=1, positive) holds for any positive ZMV (the
  GSAS-II non-closure bug guard). ZMV uses a cell-volume × asymmetric-unit-mass
  proxy (exact site multiplicities deferred; they don't affect closure).
- **Deferred**: full spherical-harmonic ODF texture and anisotropic-microstrain
  tensor broadening (March-Dollase + the single FWHM cover the demonstrators).
  Background/texture in the JAX backend (numpy-only for now).
- 44 production+domain tests pass (10 new in `test_domains.py`); `test_invariants`
  unaffected; ruff + black clean. Phase 7 complete.

### 2026-06-11: Phase 8 — scale, chunking, checkpointing

PERF requirements: ~10⁵ patterns, chunked not dense (PERF-1/2/3), parallel forward
eval (PERF-4), checkpoint/restart (UX-5), profiling hooks (PERF-6).

- **O(points × nearby_peaks) scaling invariant**: `ProductionForward.peak_point_ops`
  counts windowed profile-point writes per `calculate`. On a 4000-pt NaBr pattern,
  windowed ops are <20% of the naive `n_refl × n_points`; doubling grid density
  ~doubles ops (linear, not quadratic) — confirming each peak touches a fixed
  angular window regardless of total reflections. (`tests/performance/test_scale.py`.)
- **`ChunkedMinimizer`** (`engine/chunked.py`): Gauss–Newton that **accumulates the
  normal equations** `JᵀJ` (N_params²) and `Jᵀr` one histogram at a time, folding
  each block in and discarding it — peak memory `O(N_params²)` + the largest single
  histogram, **independent of the number of patterns** (PERF-3, the "no dense
  all-at-once" requirement). Per-histogram residual-Jacobian via FD with `expand`
  inside (handles parametric, like Phase 5); blocks are independent so they
  parallelize over histograms via a `ThreadPoolExecutor` (PERF-4). Validated to
  recover the **same cell as the SciPy minimizer** (1e-4) on a single state, and to
  refine a 12-histogram shared-cell ensemble.
- **Checkpoint/restart** (`checkpoint.py`): `save_refinement`/`load_refinement`
  (pickle; trusted self-produced files) round-trip the immutable state exactly;
  `resume(engine, path)` continues refining. `parametric_models` carry callables
  (not portably serializable) so they're dropped on save with their
  target/coeff-names preserved, and re-attached via `restore_models` on load.
- **Memory stability** (`@pytest.mark.slow`): 400 sequential `calculate`s grow RSS
  <5% — no per-pattern accumulation.
- **Deliberately demonstrated, not brute-forced**: the 10⁴-pattern campaign claim
  is evidenced by the scaling counter (per-peak cost is grid-local) + the
  bounded-memory accumulation + the memory-stability loop, rather than an actual
  10⁴-pattern refinement (too slow for CI). **Deferred**: Dask/MPI out-of-core +
  cluster execution (the chunked accumulator is the interface they'd slot behind);
  staged/block (fit-by-block) scheduling beyond the per-histogram chunking.
- 8 scale tests pass; full production suite unaffected; ruff + black clean.
  Phase 8 complete.

### 2026-06-12: Phase 9 — AI triage (propose-only), diagnostics, visualization, CLI

- **`aura.ai.PhaseIdentifier`** (propose-only): ranks candidate phases by a fast,
  forward-only peak-position match score (predicted strong peaks = mult·|F|²·LP,
  weighted by the SAME LP as the forward model — without that weighting TOF
  rankings invert, since the d⁴ Lorentz factor makes large-d/large-TOF peaks the
  strong ones). Returns `(Phase, confidence∈[0,1])` ranked; **never refines**. A
  heuristic stand-in for a trained model (CPICANN-style) — the *contract* is the
  point.
- **Category H validated** (`tests/ai/test_identify.py`): propose returns only
  ranked bounded candidates; **candidate round-trips through the engine — the
  acceptance signal is the engine's Rwp/GoF, never AI confidence**; chemistry
  filter; ranks the right phase #1 on clean synthetic single-phase data.
  Hallucination containment (§11.3): negative counts rejected; an unparseable
  space-group candidate scores 0, not fabricated. Provenance (§11.1): `AIProposal`
  carries source/model_version/confidence, `human_acceptance=None` (AI never
  self-accepts). NOTE: on the pressure-shifted multi-phase SNAP data with ambient
  CIFs the heuristic is unreliable — by design the engine disambiguates, not the AI.
- **`aura.diagnostics`** (UX-3/UX-4): `summarize`, `classify_parameter`
  (converged / at_bound / ill_determined / fixed — the "which converged / diverged
  / not converging" view), `convergence_counts`, `histogram_misfits` (worst-first
  drill-down), `campaign_summary` (per-state rollup).
- **`aura.viz`** (matplotlib Agg, headless): `plot_fit` (obs/calc/diff),
  `plot_parameter_convergence` (relative-σ bars colored by status),
  `plot_campaign_trajectory` (param ± σ vs driving variable).
- **`aura.quantify`** carried from Phase 7 (weight fractions).
- **CLI** (`aura identify`, `aura refine`): wire registry→bridge→engine→
  diagnostics; `refine` builds a default param set (cubic-aware cell + scale +
  width + Chebyshev background), prints Rwp/GoF + per-parameter status, optional
  `--plot`/`--out`. Tested via `CliRunner` on real PbSO₄.
- **Deferred**: trained ML identifier (the heuristic is the placeholder);
  interactive PySide6 workbench (Phase 11); plotly dashboards.
- 108 unit+AI tests pass (diagnostics 7, viz 3, AI 7, CLI 5 + existing); ruff +
  black clean. Phase 9 complete.

### 2026-06-15: Phase 10 — images, 2D integration, single-crystal ingest

Completes GSAS-II importer parity and builds the **partially-integrated data
pipeline** that is the project's scientific motivation (2D image → N angular 1D
slices, SCI-1/3). No real 2D/sfact data ships, so it is validated synthetically.

- **`aura.io.image.DetectorGeometry`** — flat-detector PONI geometry (distance,
  PONI, pixel sizes, wavelength, shape); native per-pixel 2θ/azimuth/d arrays;
  `pixel_to_angles`/`angles_to_pixel` round-trip exactly; JSON save/load.
- **Image readers** (`domain="image"`): `NpyImageReader` (native) +
  `FabioImageReader` (TIFF/CBF/EDF/GE/Mar via **fabio**, which is installed).
- **`aura.integrate`**: `integrate_full` (degrade-to-1D, SCI-4) and
  `integrate_sectors(image, geom, n_sectors)` → a `MeasurementState` of N
  directionally-resolved 1D slices. Bin values are **summed** intensity, so the
  sectors partition the pixels and **recombine exactly into the full pattern**
  (the SCI-4 consistency test). Synthetic Debye-ring image → integrate → peaks at
  the analytic ring 2θ; → refine recovers the injected NaBr cell to <5e-3 Å.
- **Single-crystal** (`domain="sfact"`): `ShelxHklReader` (HKLF4, stops at the
  `0 0 0` terminator) + `CifReflnReader` (gemmi `_refln` loop) → validated
  `StructureFactors` table. Represent+validate only (powder-profile engine does
  not refine single-crystal). Domain separation means the same `.cif` serves the
  `phase` and `sfact` domains without conflict.
- **Bug fixed (hardens all phases)**: `positions.d_range_for_histogram` padded
  `d_min` by `pad·span`; an image-integrated pattern includes a near-beam-center
  bin (2θ≈0) → d_max≈480 Å → span huge → d_min floored to 1e-3 → reflection
  enumeration tried an 11903³ meshgrid (12 TiB). Now each bound is padded
  **relative to itself** (`d_min·(1-pad)`, `d_max·(1+pad)`); production-forward/
  refine/domains tests still pass.
- pyFAI is NOT installed → the optional pyFAI cross-check is omitted; native
  integration is validated against analytic ring positions + exact sector
  recombination instead (rigorous without it). fabio added (available); pyFAI
  remains an optional cross-check for later.
- 14 new tests (imaging 8, sfact 6); 114 unit+imaging pass; ruff + black clean.
  Phase 10 complete — **all 11 importer/data domains live**; only Phase 11
  (facility runway) remains.
