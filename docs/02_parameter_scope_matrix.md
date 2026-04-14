# Parameter Scope Matrix for a Prototype Framework for Partially Integrated, High-Throughput Diffraction Analysis

## Purpose

This document defines a parameter-scope matrix for a prototype framework intended to analyse partially integrated diffraction datasets at scales beyond the assumptions of conventional Rietveld software.

The purpose of the matrix is to make explicit:

- what classes of parameters are needed
- which parameters are expected to be shared and which are expected to vary
- where parameter-count growth is likely to occur
- which parts of the model are physically meaningful and which are primarily nuisance terms
- which classes of parameters are likely to dominate architectural decisions

This matrix is intended to complement the higher-level requirements matrix by giving a more concrete view of the likely structure of real refinement problems.

---

## Parameter scope matrix

| Parameter class | Example parameters | Scope options | Expected variability | Typical count | Physical or nuisance? | Notes |
|---|---|---|---|---:|---|---|
| Crystal structure | lattice parameters, atomic positions, occupancies, ADPs | global, per phase, per state | low to moderate | 5–100 | physical | Typically shared across angular slices within a measurement state |
| Phase abundance / scale | phase scale factors, relative phase fractions, absorption-like scale terms | per state, per slice, structured across states | low to moderate | 1–20 | hybrid | May be physically meaningful, but may also absorb geometry and experimental effects |
| Peak-shape / profile | Gaussian/Lorentzian terms, asymmetry terms, broadening coefficients | global, per detector group, per state, per slice | low to moderate | 3–30 | hybrid | Often partly instrumental and partly sample-driven |
| Microstructure | crystallite size, strain broadening, anisotropic broadening terms | per phase, per state, angular model | moderate | 2–20 | physical | May require angle-dependent or tensor-like parameterization |
| Texture / orientation | preferred orientation terms, spherical harmonic coefficients, ODF parameters | per phase, per state | moderate to high | 3–50 | physical | Likely one of the main motivations for retaining angular resolution |
| Stress / strain state | anisotropic lattice strain, stress-model coefficients | per phase, per state, structured across states | moderate | 1–30 | physical | Especially relevant when directional information is preserved |
| Instrument geometry | zero offset, detector offsets, wavelength terms, calibration terms | global, per detector bank, per detector group | low | 5–30 | hybrid | Usually strongly shared; should not be duplicated unnecessarily |
| Resolution / instrumental broadening | U/V/W terms, TOF broadening terms, detector-dependent resolution coefficients | global, per detector group, per angular family | low to moderate | 3–20 | hybrid | Natural candidate for grouped rather than fully local scope |
| Background | polynomial coefficients, spline coefficients, basis weights, detector-specific background terms | per slice, per state, smooth model across slices or states | moderate to high | 5–500 | nuisance | One of the most likely sources of parameter explosion |
| Container / parasitic scattering | known component scales, broad scattering terms, structured nuisance signatures | global, per state, per slice | moderate | 1–50 | nuisance / hybrid | Particularly important for in situ and operando experiments |
| Amorphous / diffuse component | broad component amplitudes, widths, basis coefficients | per phase, per state, per slice | moderate | 1–30 | physical / hybrid | Important for non-ideal real-world samples |
| Sample displacement / positioning | displacement, height, offset-like terms | per state, per detector group | low to moderate | 1–10 | nuisance | May correlate strongly with other model components |
| Intensity correction terms | absorption, extinction, polarization-like corrections, empirical intensity factors | global, per state, per slice | low to moderate | 1–20 | hybrid | Scope depends strongly on experiment geometry |
| Noise / uncertainty model | variance scale factors, robust-loss parameters, overdispersion terms | global, per state, per slice | low to moderate | 1–20 | nuisance | May be essential for stable fitting on difficult data |
| Outlier / masking model | excluded regions, binary masks, outlier thresholds, contamination flags | local, per slice, per state | moderate | 1–20 | nuisance | Important for robust treatment of high-throughput real data |
| State-evolution terms | coefficients describing change with time, temperature, stress, composition, dose | across states, global, per phase | moderate | 1–50 | physical | Useful when fitting whole series rather than independent states |
| Cross-state regularization terms | smoothness penalties, sparsity penalties, basis weights across measurement index | across states | moderate | 1–100 | nuisance / structural | Not directly physical, but may be crucial for tractable inference |
| Learned latent variables | low-rank coefficients, embedding coordinates, surrogate-model parameters | global, per state, per phase | moderate to high | 2–100 | structural | Relevant if machine-learning-assisted approaches are considered |
| Metadata-driven terms | functions of temperature, pressure, load, chemistry, processing variables | across states, per phase, global | moderate | 1–50 | physical / structural | Useful for high-throughput datasets with strong external metadata |

---

## Scope definitions

For consistency across prototype designs, the following scope levels are recommended.

| Scope level | Meaning |
|---|---|
| Global | Shared across the entire campaign or analysis run |
| Experiment | Shared across a defined experiment, sample family, or acquisition batch |
| State | Shared across all angular slices belonging to one measurement state |
| Slice group | Shared across a subset of slices, such as detector bank, azimuthal family, or angular cluster |
| Slice | Specific to one retained 1D pattern |
| Local point-range | Specific to a narrow region of one pattern, such as a masked or contaminated interval |

These scope levels should be treated as first-class concepts in the prototype design, not merely as conventions.

---

## Key observations from the matrix

### 1. Most physically meaningful parameters should not scale linearly with slice count

Crystal structure, many instrumental terms, and many microstructural parameters are expected to be shared across multiple slices. A design that duplicates all such parameters for every pattern will become both computationally inefficient and scientifically fragile.

### 2. Background and nuisance models are likely to dominate parameter count

The most obvious source of parameter explosion is not necessarily the core crystallographic model, but the nuisance structure required to model:

- backgrounds
- parasitic scattering
- diffuse or amorphous contributions
- detector-specific artefacts
- masking and outlier handling

This implies that compact and structured representations of nuisance terms are likely to be central design requirements.

### 3. Texture and anisotropy are likely to be key beneficiaries of retained dimensionality

If the prototype succeeds scientifically, it is likely to do so first in parameter classes that are explicitly directional, such as:

- preferred orientation
- anisotropic strain
- stress-related peak shifts or distortions
- angle-dependent intensity redistribution

These should therefore be high-priority use cases in benchmarking.

### 4. Cross-state structure may be as important as within-state structure

In high-throughput campaigns, many parameters may evolve smoothly or sparsely across time, temperature, load, composition, or processing conditions. Treating every state as independent may waste information and create instability.

This suggests the prototype should at least leave room for models that couple states as well as slices.

---

## Architectural implications

The matrix suggests several design implications that should be used to screen prototype architectures.

### Parameter scoping must be explicit

The prototype should support explicit declaration of scope. It should be possible to state clearly whether a parameter is:

- global
- shared within a state
- shared within a slice group
- local to a slice
- governed by a structured model across slices or states

A design in which scope is only implicit in naming conventions will not scale well.

### Compact parameterizations will be essential

Wherever many local parameters can be replaced by a structured lower-dimensional representation, the prototype should support that naturally.

Examples include:

- background basis expansions
- smooth angular variation models
- low-rank state evolution models
- grouped or hierarchical nuisance terms

### Physical and nuisance parameters should be distinguishable

The system should make it easy to separate:

- parameters whose scientific interpretation is central
- parameters that primarily stabilize the fit
- parameters that absorb instrument or environment artefacts

This distinction will matter for both optimisation strategy and scientific reporting.

### The prototype should support structured coupling, not just parameter sharing

Some parameters will not simply be identical across many objects; instead they may be coupled by:

- functional dependence
- smoothness assumptions
- regularization
- low-dimensional latent models
- metadata-driven relationships

Prototype designs that only support exact equality constraints may be too limited.

---

## Suggested additions when using this matrix in practice

For each real use case, it will be useful to extend this table with additional columns such as:

- **applies to use case**
- **likely dominant cost**
- **likely dominant correlation / identifiability issue**
- **candidate compact representation**
- **estimated count for this use case**

Example:

| Parameter class | Applies to use case | Estimated count | Candidate compact representation | Main risk |
|---|---|---:|---|---|
| Background | in situ furnace, battery cell, pressure cell | 200–2000 | spline basis or learned low-rank basis | parameter explosion |
| Texture | additively manufactured alloy | 10–40 | spherical harmonics | identifiability versus scale |
| Instrument geometry | all | 10–20 | grouped shared parameters | over-duplication |

These additions will make the matrix more directly useful for choosing between prototype architectures.

---

## Recommended next step

The most useful next step is to instantiate this matrix for several representative scientific use cases and attach rough parameter counts to each class.

That exercise should make clear:

- where the main scaling pressure arises
- which parameter classes require special treatment
- which kinds of architectural ideas are plausible
- which kinds are unlikely to remain tractable