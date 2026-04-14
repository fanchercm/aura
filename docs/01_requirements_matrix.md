# Requirements Matrix for a Prototype Framework for Partially Integrated, High-Throughput Diffraction Analysis

## Purpose

This document defines a first-pass requirements matrix for a prototype framework intended to analyse partially integrated diffraction datasets at scales beyond the assumptions of conventional Rietveld software.

The prototype target is a workflow in which:

- each measurement state is represented not by a single fully integrated 1D pattern, but by approximately 50–100 linked 1D patterns
- each pattern retains directional or angularly resolved information
- high-throughput campaigns may contain thousands of measurement states
- the total analysis object may therefore contain on the order of 10^5 linked 1D patterns

The matrix is intended to:

1. define what the prototype must do
2. define what scale it must reach
3. identify what classes of design it rules out
4. define how success should be evaluated

---

## Scientific and analysis requirements

| ID | Requirement | Why it matters | Priority | Candidate quantitative target | Filters out |
|---|---|---|---|---|---|
| SCI-1 | The prototype must operate on partially integrated diffraction data, not only fully integrated 1D patterns | Preserves directional information lost in full azimuthal integration | Must | 50–100 1D patterns per measurement state | Any design centered on a single-pattern refinement object |
| SCI-2 | The prototype must support simultaneous coupled fitting of all patterns belonging to one measurement state | Physical parameters are shared across angular slices and should be inferred jointly | Must | One refinement may span 50–100 linked histograms | Independent-per-pattern fitting workflows |
| SCI-3 | The prototype must preserve and exploit angular dependence of diffraction intensity and/or profile shape | This is the core scientific motivation | Must | Angular metadata retained for every pattern | Designs that treat angle only as a label for plotting |
| SCI-4 | The prototype must support comparison against conventional fully integrated analysis | Needed to quantify benefit of the new approach | Must | Same dataset analysable in collapsed 1-pattern form and multi-pattern form | Designs that cannot reduce to standard workflows |
| SCI-5 | The prototype must support multiple scientific use cases with differing parameter-sharing structures | Prevents overfitting the software design to one narrow case | Must | At least 3 representative use cases | Highly hard-coded, case-specific implementations |
| SCI-6 | The prototype should support texture, anisotropic strain/stress, multiphase mixtures, and structured backgrounds | These are key classes of information likely preserved by higher-dimensional treatment | Should | At least one demonstrator for each class, or explicit rationale for exclusions | Designs unable to express angle-dependent physics |

---

## Data model requirements

| ID | Requirement | Why it matters | Priority | Candidate quantitative target | Filters out |
|---|---|---|---|---|---|
| DATA-1 | The internal data model must be hierarchical | The natural structure is campaign → sample state → angular slice → data points | Must | At least 3 explicit hierarchy levels | Flat list-of-patterns models with no grouping semantics |
| DATA-2 | The model must represent metadata associated with each pattern | Angle, detector, time, temperature, stress, sample ID, etc. affect interpretation | Must | Pattern object includes measurement and geometry metadata | Data containers that only hold intensity vectors |
| DATA-3 | The model must support parameter scoping across hierarchy levels | Many parameters are shared across subsets, not globally or locally only | Must | Global / experiment / state / slice / local scopes | Systems with only global and local parameters |
| DATA-4 | The model must support sparse linkage between patterns and parameters | Full dense coupling is computationally and conceptually wasteful | Must | Parameter-to-pattern mapping stored explicitly or via rules | Architectures assuming every parameter touches every pattern |
| DATA-5 | The model should support missing or irregular data | Real campaigns may have missing slices, failed measurements, masked ranges | Should | Robust handling of absent patterns and masked channels | Designs assuming perfectly regular blocks |

---

## Parameterization and modelling requirements

| ID | Requirement | Why it matters | Priority | Candidate quantitative target | Filters out |
|---|---|---|---|---|---|
| MOD-1 | The prototype must distinguish shared physical parameters from local nuisance parameters | Otherwise parameter count grows uncontrollably | Must | Explicit parameter classes or scopes | Giant undifferentiated parameter vector designs |
| MOD-2 | The prototype must support constraints and ties across patterns | Essential for stability and physical meaning | Must | Arbitrary equality or functional ties across scopes | Fully independent local fits |
| MOD-3 | The prototype must support low-dimensional parameterizations of patterned variation | Needed to replace many local parameters with compact models | Must | Example: background or texture described by basis coefficients rather than one set per slice | Designs that only scale by duplicating parameters |
| MOD-4 | The prototype should support regularization or prior structure | Prevents overfitting in very large coupled problems | Should | At least L2, smoothness, group penalties, or equivalent | Pure unconstrained least-squares-only approaches |
| MOD-5 | The prototype should support both explicit physical models and empirical nuisance models | Real data will need both | Should | Physical peak model plus flexible background or noise model | Rigid pipelines that cannot absorb experiment-specific artefacts |
| MOD-6 | The prototype must allow block structure in the model | Enables staged fitting and efficient optimisation | Must | Parameters partitionable into logical blocks | Monolithic models with no decomposition |

---

## Scale and performance requirements

| ID | Requirement | Why it matters | Priority | Candidate quantitative target | Filters out |
|---|---|---|---|---|---|
| PERF-1 | The prototype must support high-throughput campaigns | This is one of the core departures from conventional Rietveld workflows | Must | 10^3–10^4 measurement states | Designs only practical for single refinements |
| PERF-2 | The prototype must support total problem sizes on the order of 10^5 1D patterns | Sets the scale of the challenge | Must | Approximately 100,000 patterns overall | Architectures requiring all patterns to be manually curated or separately configured |
| PERF-3 | The prototype must avoid dense all-at-once memory assumptions | Jacobians and residuals may become too large | Must | Chunked evaluation over subsets of patterns | Dense in-memory normal-equation-style designs only |
| PERF-4 | The prototype should support parallel forward evaluation | Forward model cost is likely dominant | Should | Parallelism across patterns, states, or blocks | Strictly serial implementations |
| PERF-5 | The prototype should support staged or incremental optimisation | Needed for tractability and robustness | Should | Fit-by-block, fit-by-state, or progressive refinement modes | Single giant solve as only execution mode |
| PERF-6 | The prototype must expose runtime and memory profiling hooks | Essential for learning from the prototype | Must | Wall time, memory, and model-evaluation timing recorded per run | Black-box pipelines with poor observability |

---

## Workflow and usability requirements

| ID | Requirement | Why it matters | Priority | Candidate quantitative target | Filters out |
|---|---|---|---|---|---|
| UX-1 | The prototype must be scriptable and reproducible | Early-stage development needs automation and repeatability | Must | Full analysis configured from code or declarative config | GUI-only workflows |
| UX-2 | The prototype must support concise declaration of shared structure | Manually wiring 10^5 patterns is impossible | Must | Linkage generated from templates or rules, not per-pattern editing | Systems requiring manual histogram-level setup |
| UX-3 | The prototype must generate diagnostics at multiple aggregation levels | Users cannot inspect all fits individually | Must | Global, state, slice, and parameter-block summaries | Designs that only emit per-pattern residuals |
| UX-4 | The prototype should support drill-down from summary diagnostics to raw residuals | Needed to debug failures and understand scientific edge cases | Should | Outlier identification and pattern-level inspection | Aggregated-only reporting |
| UX-5 | The prototype should support checkpointing and restart | Large analyses should survive interruptions | Should | Save and load analysis state or fit state | Stateless batch-only systems |

---

## Validation and success criteria

| ID | Requirement | Why it matters | Priority | Candidate quantitative target | Filters out |
|---|---|---|---|---|---|
| VAL-1 | The prototype must be testable on synthetic data with known ground truth | Needed to determine whether coupled fitting recovers true parameters | Must | At least one synthetic benchmark per use case | Designs too entangled with beamline-specific data to validate cleanly |
| VAL-2 | The prototype must be testable on real experimental datasets | Synthetic success alone is insufficient | Must | At least one real dataset per target use case | Purely theoretical architectures |
| VAL-3 | The prototype must allow direct comparison of alternative architectures | This is the purpose of the requirements exercise | Must | Common benchmark suite and metrics | One-off prototypes that cannot be fairly compared |
| VAL-4 | The prototype should quantify whether retained dimensionality improves inference | Must show benefit over conventional collapse | Should | Improvement in identifiability, uncertainty, fit stability, or scientific interpretability | Designs with no baseline comparison path |
| VAL-5 | The prototype must define failure modes explicitly | Important for choosing future development directions | Must | Memory failure, convergence failure, model mismatch, and overparameterization tracked separately | Ambiguous "it didn’t work" outcomes |

---

## Recommended additional columns

For project use, the matrix will become more valuable if three further columns are added:

- **Example parameter counts**
- **Applies to use cases**
- **Notes / open questions**

Example:

| ID | Requirement | Example parameter counts | Applies to use cases | Notes |
|---|---|---:|---|---|
| MOD-3 | Low-dimensional angular variation model | 6–20 coefficients instead of 100 local parameters | texture, anisotropic strain, background variation | Candidate basis functions not yet chosen |

These extra columns will be particularly useful once representative use cases and parameter counts have been filled in.

---

## Design-screening questions derived from the matrix

Any candidate architecture should be assessed against the following questions:

1. Can it represent shared structure without manually wiring tens of thousands of links?
2. Can it avoid parameter explosion?
3. Can it recover known truth on synthetic data?
4. Can it exploit angular structure rather than merely store it?
5. Can it process a reduced realistic dataset in practical time?
6. Can a scientist understand why it failed?
7. Can it degrade gracefully to a conventional 1D workflow for benchmarking?

Architectures that fail several of these early are unlikely to be worthwhile prototype candidates.

---

## Notes

This matrix is deliberately phrased to support comparison of multiple prototype strategies rather than commit prematurely to one implementation pattern.

The next recommended step is to attach concrete parameter-count estimates for several representative scientific use cases, and then refine the quantitative targets accordingly.