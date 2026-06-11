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
