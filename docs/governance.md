# Aura Governance

This document defines how the Aura project is governed: who makes decisions,
how changes to scientific defaults and extension APIs are proposed, and how the
community can participate.

---

## Technical Steering Committee (TSC)

The TSC is the ultimate authority on technical direction, scientific defaults,
and stable extension-point APIs. It meets quarterly and operates by rough
consensus; a simple majority vote resolves ties.

**Representation** — The TSC should include at minimum:

| Role | Responsibility |
|------|----------------|
| Synchrotron workflow lead | CW X-ray / high-resolution / 2D detector workflows |
| Neutron TOF/event-data lead | TOF / event-mode / facility-scale workflows |
| Texture/stress user lead | Combined analysis, preferred orientation, residual stress |
| Software engineering lead | CI/CD, plugin SDK, API stability, performance |
| Scientist-at-large | Represents general scientist / instrument-scientist users |

Current TSC members are listed in `docs/TSC.md` (not tracked in this file to
avoid stale names).

---

## RFC Process

Any change that could break third-party plugins or alter published results
**must** go through an RFC (Request for Comments) before merge.

### Triggers for an RFC

- Breaking changes to any Protocol in `aura.spec` or `aura.plugins`.
- Changes to scientific defaults (e.g., default profile function, default
  background model, default extinction correction, ACCEPTANCE tolerances).
- Changes to the parameter naming convention (e.g., adding a new dotted key).
- Changes to the HDF5/NeXus layout version (bumps `AURA_FORMAT_VERSION`).
- New official entry-point groups (new SDK extension points).
- Changes to the CI blocking/gating rules.

### RFC lifecycle

1. **Draft** — Author opens a GitHub Discussion titled `RFC: <short description>`.
   The body must include: motivation, proposed change, affected components,
   migration path for downstream code, and test strategy.
2. **Comment period** — Minimum 14 days. TSC and community members may comment.
   Major objections must be resolved before advancement.
3. **TSC review** — TSC member tags the discussion `rfc-approved` or
   `rfc-rejected` with a brief rationale.
4. **Implementation** — Approved RFCs are implemented in a PR that references
   the discussion. The PR description must include a "Migration notes" section.
5. **Changelog** — `CHANGELOG.md` records the RFC number and a one-line summary.

### What does NOT require an RFC

- Bug fixes that don't change user-visible behaviour.
- New readers for file formats (new entry-point registrations).
- New optional parameters with documented defaults.
- Documentation, tests, and CI changes.
- Performance improvements that preserve numerical output within tolerance.

---

## Extension stability policy

The SDK version (`AURA_SDK_VERSION` in `aura.plugins`) follows **semantic
versioning**:

| Change | Version bump |
|--------|-------------|
| Breaking Protocol change (removed/renamed method, changed signature) | Major |
| New optional method on a Protocol (backward-compatible) | Minor |
| Bug fix, documentation, no API change | Patch |

Third-party plugins should pin to a major version and test on each minor bump.
The SDK changelog documents every Protocol change since `1.0`.

### Deprecation policy

Before removing or renaming a Protocol method:
1. Mark it `@deprecated` in the docstring with a target removal version.
2. Keep it functional for at least **two minor releases**.
3. Issue a `DeprecationWarning` on first use.
4. Remove in the next major version after the deprecation period.

---

## Scientific defaults and ACCEPTANCE tolerances

The `ACCEPTANCE` object in `aura.spec` encodes correctness thresholds the
engine must satisfy. Changes to these values require an RFC and a rationale
citing the physical basis (instrument resolution, counting statistics, etc.).

Current defaults (not to be changed without RFC):

| Parameter | Value | Basis |
|-----------|-------|-------|
| `jacobian_rtol` | `1e-4` | AD vs FD finite-difference tolerance |
| `recovery_n_sigma` | `3.0` | Parameter recovery within 3σ |
| `gof_low` | `0.8` | Lower GoF bound for a well-determined model |
| `gof_high` | `1.5` | Upper GoF bound for a well-determined model |
| `phase_fraction_sum_atol` | `1e-6` | Phase-fraction closure |
| `parametric_equiv_rtol` | `1e-5` | Parametric vs independent equivalence |

---

## Benchmark and cross-validation gate

At release tags, CI runs the reference tier (`pytest -m reference`), which
includes the PbSO₄ CPD round-robin check and any GSAS-II golden-file
comparisons captured in `tests/reference/golden/`. A release is blocked if any
reference test fails.

Benchmark results (peak-point throughput, memory stability) are recorded in
`docs/benchmark_history.md` at each release so regressions are visible over time.

---

## Community participation

- **Bug reports** — GitHub Issues; use the provided template.
- **Feature requests** — GitHub Issues tagged `enhancement`; may become an RFC.
- **Pull requests** — Require one TSC member or delegated reviewer sign-off.
  Physics-touching PRs require the `physics-reviewer` sign-off.
- **Instrument models** — New instrument kernels are contributed as plugins
  (via the `aura.instrument_kernels` entry-point group) unless they require
  changes to `spec.py`, in which case an RFC is needed.
- **Code of Conduct** — Contributors are expected to follow the
  [Contributor Covenant](https://www.contributor-covenant.org/) v2.1.
