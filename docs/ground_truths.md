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
