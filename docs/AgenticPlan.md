# Agentic Oracle: Invariant Test Suite
## Scientific Validation Framework for a Next-Generation Rietveld Analysis Platform

**Version:** 1.0  
**Status:** Foundational Scientific Specification  
**Purpose:** Define the immutable scientific, numerical, and operational invariants that every code change, AI-generated patch, optimization, refactor, and architectural modification must satisfy.

---

# Guiding Principle

> A patch is invalid if it improves Rwp but worsens physical truth, uncertainty calibration, provenance, reproducibility, or diagnostic honesty.

The oracle exists to prevent optimization toward visually attractive fits rather than scientifically correct diffraction analysis.

---

# Oracle Architecture

Every proposed code change is evaluated against four progressively more expensive validation layers.

```text
fast/unit        every commit
physics/core     every PR
campaign         nightly
reference        release/blocking
```

Each test produces:

```yaml
status: pass/fail/xpass
tolerance_class: exact | numerical | statistical | scientific

artifact:
  input_digest
  output_digest
  parameter_vector
  covariance
  residual_map
  provenance_manifest
```

No merge may occur unless all required gates pass.

---

# 1. Crystallographic Invariants

## 1.1 Symmetry Closure

### Purpose

Guarantee correctness of all space-group operations.

### Validation

For every supported space group:

```text
identity exists
inverse exists
closure under composition
equivalent positions preserved
fractional coordinates wrapped correctly
```

### Acceptance

```text
max coordinate error < 1e-12
```

### Priority

BLOCKING

---

## 1.2 Reflection Equivalence

### Validation

For all symmetry-equivalent reflections:

```text
F²(hkl) == F²(h'k'l')
d(hkl) == d(h'k'l')
multiplicity preserved
systematic absences remain absent
```

### Acceptance

```text
relative F² error < 1e-10
d-spacing error < 1e-12 Å
forbidden intensity < 1e-14 strongest peak
```

### Priority

BLOCKING

---

## 1.3 Metric Tensor Consistency

### Validation

For all crystal systems:

```text
G positive definite
G* = inverse(G)
d-spacing matches analytical formula
volume consistency maintained
```

### Acceptance

```text
relative volume error < 1e-12
relative d-spacing error < 1e-12
```

### Priority

BLOCKING

---

## 1.4 Coordinate Round Trips

### Validation

```text
fractional → Cartesian → fractional
Uiso ↔ Biso
ADP transforms
```

### Acceptance

```text
coordinate error < 1e-12
B=8π²U error < 1e-14
positive-definite ADPs preserved
```

### Priority

BLOCKING

---

# 2. Scattering and Intensity Invariants

## 2.1 Debye-Waller Zero Limit

### Validation

As U → 0:

```text
DW factor → 1
intensity increases monotonically
```

### Acceptance

```text
relative error < 1e-12
```

### Priority

BLOCKING

---

## 2.2 Scale Linearity

### Validation

```text
I(scale=kS) = k × I(scale=S)
```

### Acceptance

```text
relative profile error < 1e-12
```

### Priority

BLOCKING

---

## 2.3 Occupancy Behavior

### Validation

```text
F ∝ occupancy
I ∝ occupancy²
```

### Acceptance

```text
relative error < 1e-10
```

### Priority

BLOCKING

---

## 2.4 Neutron Scattering-Length Swap

### Validation

Changing only scattering length:

```text
peak positions invariant
intensities transform correctly
```

### Acceptance

```text
position error < 1e-12
intensity error < 1e-8
```

### Priority

PHYSICS CORE

---

# 3. Peak Shape and Instrument Invariants

## 3.1 Profile Normalization

### Validation

For every profile function:

```text
∫P(x)dx = 1
P(x) ≥ 0
```

### Acceptance

```text
area error < 1e-8
negative density forbidden
```

### Priority

BLOCKING

---

## 3.2 Translation Invariance

### Validation

```text
P(x−Δ,c)=P(x,c+Δ)
```

### Acceptance

```text
relative L2 error < 1e-10
```

### Priority

BLOCKING

---

## 3.3 Width Positivity

### Validation

```text
FWHM > 0
σ² > 0
Lorentz width > 0
```

### Priority

BLOCKING

---

## 3.4 CW Bragg Position

### Validation

```text
2d sinθ = λ
```

### Acceptance

```text
error < 1e-12 Å
```

### Priority

BLOCKING

---

## 3.5 TOF Instrument Law

### Validation

```text
TOF = DIFC*d + DIFA*d² + ZERO
```

### Acceptance

```text
absolute error < 1e-9
```

### Priority

BLOCKING

---

## 3.6 EDD Energy Relation

### Validation

```text
E = hc/(2d sinθ)
```

### Acceptance

```text
relative error < 1e-10
```

### Priority

BLOCKING

---

# 4. Background Invariants

## 4.1 Deterministic Background Models

### Validation

```text
same coefficients
same grid
same output
```

### Acceptance

```text
bitwise equality on reference path
```

### Priority

BLOCKING

---

## 4.2 Background-Peak Separation

### Synthetic Challenge

```text
broad background
sharp Bragg peaks
```

### Acceptance

```text
major peak bias < 0.5%
weak peak bias < 3%
```

### Priority

PHYSICS CORE

---

# 5. Parameter Graph Invariants

## 5.1 Parameter Scope Integrity

Supported scopes:

```text
local
shared
global
functional
```

### Validation

Ensure updates propagate only according to graph definition.

### Priority

BLOCKING

---

## 5.2 Jacobian Consistency

### Validation

```text
finite difference
vs
analytic/autodiff
```

### Acceptance

```text
relative error < 1e-6
```

### Priority

BLOCKING

---

## 5.3 Bound Enforcement

### Validation

```text
parameter bounds
fixed parameters
tied parameters
```

### Acceptance

```text
no violations > 1e-12
```

### Priority

BLOCKING

---

## 5.4 Gauge Freedom Detection

### Validation

Known degenerate models:

```text
collinear scales
phase fraction/scale coupling
background degeneracy
```

### Required Behavior

```text
rank deficiency detected
uncertainties inflated
false precision prevented
```

### Priority

BLOCKING

---

# 6. Optimizer Invariants

## 6.1 Accepted-Step Monotonicity

### Validation

```text
accepted χ² must not increase
```

### Acceptance

```text
χ²new ≤ χ²old + 1e-12
```

### Priority

BLOCKING

---

## 6.2 Analytic Minimum Recovery

### Synthetic Problems

```text
Gaussian
Voigt
Pseudo-Voigt
```

### Acceptance

```text
position error < 1e-5
area error < 1e-4
width error < 1e-4
```

### Priority

BLOCKING

---

## 6.3 Synthetic Rietveld Recovery

### Recover

```text
lattice
scale
zero
background
width
phase fraction
```

### Acceptance

```text
lattice error < 1e-5
phase fraction < 0.002
```

### Priority

PHYSICS CORE

---

## 6.4 Randomized Initialization Robustness

### Validation

100 random starts

### Acceptance

```text
easy problems: ≥95% convergence
hard problems: ≥80% success or ambiguity detection
```

### Priority

CAMPAIGN

---

# 7. Sequential and Parametric Invariants

## 7.1 Sequential Ordering Independence

### Validation

```text
forward
reverse
parallel
```

Must agree when mathematically equivalent.

### Acceptance

```text
relative difference < 1e-8
```

### Priority

BLOCKING

---

## 7.2 Shared Parameter Benefit

### Validation

Shared instrument parameter across N patterns.

### Requirement

```text
uncertainty decreases
bias unchanged
```

### Priority

PHYSICS CORE

---

## 7.3 Parametric Model Superiority

Example:

```text
a(T)=a₀+αT
```

### Requirement

```text
parametric fit outperforms independent fit
```

### Priority

PHYSICS CORE

---

## 7.4 Phase Appearance/Disappearance

### Validation

```text
absent → present → absent
```

### Acceptance

```text
negative fractions forbidden
false positives < 0.003
```

### Priority

CAMPAIGN

---

## 7.5 Dynamic Strain Recovery

### Acceptance

```text
clean data < 25 µε
noisy data < 100 µε
```

### Priority

PHYSICS CORE

---

# 8. Texture and Multibank Invariants

## 8.1 Random Texture Limit

### Validation

Uniform ODF.

### Acceptance

```text
integrated intensity spread < 1e-3
```

### Priority

PHYSICS CORE

---

## 8.2 ODF Normalization

### Validation

```text
∫ODF dg = 1
```

### Acceptance

```text
error < 1e-6
```

### Priority

BLOCKING

---

## 8.3 Bank Permutation Invariance

### Validation

Reorder detector banks.

### Requirement

Physical parameters unchanged.

### Priority

BLOCKING

---

# 9. Data and I/O Invariants

## 9.1 NeXus/HDF5/Zarr Round Trip

### Validation

```text
read → model → write → read
```

### Must Preserve

```text
counts
axes
uncertainties
metadata
units
provenance
```

### Priority

BLOCKING

---

## 9.2 Unit Consistency

### Validation

```text
μs/ns
keV/eV
deg/rad
MPa/GPa
```

### Acceptance

```text
scientific quantities invariant
```

### Priority

BLOCKING

---

## 9.3 Missing Metadata Protection

### Required Behavior

```text
explicit validation error
machine-readable diagnostics
```

### Priority

BLOCKING

---

# 10. Uncertainty and Statistical Invariants

## 10.1 Poisson Weighting Calibration

### Acceptance

```text
mean residual < 0.05
σ within [0.9,1.1]
```

### Priority

PHYSICS CORE

---

## 10.2 Coverage Validation

### Requirement

```text
68% truth coverage for 1σ
95% truth coverage for 2σ
```

### Acceptance

```text
1σ: 60–76%
2σ: 90–98%
```

### Priority

CAMPAIGN

---

## 10.3 R-factor Non-Oracle Rule

A low Rwp must never override:

```text
rank deficiency
boundary solutions
unphysical values
structured residuals
missing uncertainties
```

### Priority

BLOCKING

---

# 11. AI-Specific Invariants

## 11.1 AI Provenance

Every AI recommendation must store:

```yaml
source: ai_initialization
model_version:
confidence:
training_digest:
human_acceptance:
```

### Priority

BLOCKING

---

## 11.2 AI Utility Validation

### Requirement

AI must improve:

```text
convergence rate
time-to-first-fit
```

without changing validated scientific results.

### Priority

CAMPAIGN

---

## 11.3 Hallucination Containment

Impossible inputs:

```text
invalid space group
negative counts
missing wavelength
unsupported modality
```

### Requirement

```text
repair suggestions allowed
fabrication forbidden
```

### Priority

BLOCKING

---

# 12. Performance Invariants

## 12.1 Computational Scaling

### Requirement

```text
O(points × nearby_peaks)
```

not

```text
O(points × all_peaks)
```

### Priority

NIGHTLY

---

## 12.2 Sequential Memory Stability

### Validation

```text
10,000-pattern campaign
```

### Acceptance

```text
memory growth < 5%
```

### Priority

NIGHTLY

---

## 12.3 Backend Consistency

Compare:

```text
CPU
SIMD
GPU
JAX
```

### Acceptance

```text
parameter differences < 0.1σ
```

### Priority

RELEASE

---

# 13. Provenance Invariants

Every refinement must record:

```yaml
input_data_hash:
software_version:
git_commit:
kernel_backend:
optimizer:
random_seed:
parameter_graph_hash:
phase_model_hash:
instrument_model_hash:
environment_lock:
agent_patch_id:
human_review_state:
```

Missing provenance fields cause immediate failure.

### Priority

BLOCKING

---

# 14. Cross-Code Reference Oracle

Reference implementations:

```text
CrysFML2008
GSAS-II
MAUD
FullProf
Mantid
DiffPy-CMI
TOPAS (benchmark only)
```

### Validation

```text
peak positions
integrated intensities
refined physical parameters
```

### Requirement

Agreement within combined uncertainty.

### Priority

RELEASE

---

# Repository Layout

```text
tests/
├── unit/
├── physics/
├── campaign/
├── reference/
├── ai/
└── performance/
```

```text
tests/
  unit/
    test_symmetry.py
    test_metric_tensor.py
    test_profiles.py
    test_units.py

  physics/
    test_structure_factors.py
    test_cw_positions.py
    test_tof_positions.py
    test_edd_positions.py
    test_texture.py
    test_uncertainty.py

  campaign/
    test_sequential_ordering.py
    test_parametric_lattice.py
    test_phase_onset.py
    test_dynamic_strain.py
    test_multibank_texture.py

  reference/
    test_crysfml_symmetry.py
    test_gsasii_reference.py
    test_maud_reference.py
    test_fullprof_reference.py

  ai/
    test_ai_initialization.py
    test_ai_guardrails.py

  performance/
    test_scaling.py
    test_memory_leak.py
    test_backend_consistency.py
```

---

# Scientific Success Criterion

The software is considered scientifically trustworthy only when:

```text
Crystallographic correctness
AND
Physical correctness
AND
Statistical calibration
AND
Reproducibility
AND
Provenance completeness
AND
Performance stability
AND
AI transparency

all pass simultaneously.
```

No single metric—including Rwp, χ², speed, or benchmark score—is sufficient by itself.
