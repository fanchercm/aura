# Annotated Bibliography
### Toward a next-generation, parametric-first Rietveld engine for dynamic experiments

Each entry states the work, then why it matters for the four project goals — **parametric/sequential
correctness**, **performance/stability**, **AI integration**, and **agentic development discipline**.
Entries are grouped thematically. URLs are given for retrievable sources; classic references give the
canonical citation.

---

## 1. Refinement methodology: the parametric foundation

**Stinton, G. W. & Evans, J. S. O. (2007). "Parametric Rietveld refinement." *J. Appl. Cryst.* 40, 87–95.**
`doi:10.1107/S0021889806043275` · https://journals.iucr.org/paper?S0021889806043275=
The conceptual cornerstone of this project. Fits an entire ensemble of patterns collected versus an
external variable (T, t, P, field) to a *single evolving structural model* rather than refining each
pattern in isolation. Documents the concrete payoffs we are targeting: higher parameter precision,
the ability to impose physically realistic constraints (a thermal-expansion law, a kinetic rate
equation), direct refinement of *non-crystallographic* quantities such as temperature or rate
constants, and reduced susceptibility to false minima. We treat this surface method as the engine's
default mode, with independent and sequential refinement as degenerate special cases.

**Dinnebier, R. E., Leineweber, A. & Evans, J. S. O. (2018). *Rietveld Refinement: Practical Powder
Diffraction Pattern Analysis using TOPAS.* De Gruyter.**
The standard modern practitioner text; Chapter 12 develops surface/parametric refinement in depth
(the WO₃ and ZrP₂O₇ worked examples). Source of canonical, expert-accepted results that become entries
in our ground-truth regression corpus, and of the conventions (parameter taxonomy, profile
conventions) the executable spec encodes.

**Evans, J. S. O. Durham parametric-refinement tutorials.**
https://topas.webspace.durham.ac.uk/tutorial_surface_new/ · https://johnevans.webspace.durham.ac.uk/tutorial_surface_zrp2o7/
Step-by-step surface refinements of variable-temperature WO₃ (P2₁/n → P-1 → Pc on cooling) and a
three-phase ZrP₂O₇ + Si + Al₂O₃ standard refinement fitting 37–51 datasets simultaneously, including
refinement of a temperature-calibration polynomial directly from the data. These are ideal
*end-to-end acceptance fixtures*: the WO₃ case specifically encodes the false-minimum trap (a
lower-symmetry model fits better even where it is not the true phase), which our invariant suite
tests for explicitly.

**Madsen, I. C., Scarlett, N. V. Y. *et al.* — QPA accuracy round robin & "effect of data quality and
model parameters on quantitative phase analysis."** https://arxiv.org/pdf/2008.11046
Establishes how data range, ADP/scale-factor correlation, and refinement strategy bias quantitative
phase analysis, and notes that *in situ* analyses mitigate the ADP–scale correlation by refining in a
parametric manner. Directly motivates two invariants: the phase-fraction sum constraint and the
scale/ADP correlation-conditioning check.

---

## 2. Incumbent open-source engines: baselines and documented failure modes

**Toby, B. H. & Von Dreele, R. B. — GSAS-II.** https://gsas-ii.readthedocs.io ·
https://github.com/AdvancedPhotonSource/GSAS-II
The most relevant open baseline: Python/NumPy with a small Fortran core, covering CW, pink-beam and
TOF data from lab, synchrotron, spallation and reactor sources. Its sequential-fitting model — a
*separate* refinement per histogram with a single shared phase-parameter set and copy-forward /
use-previous seeding — is exactly the architecture whose limitations we are responding to. Studying
it defines both our compatibility targets and our anti-patterns.

**O'Donnell, J. H., Von Dreele, R. B., Chan, M. K. Y. & Toby, B. H. (2018). "A scripting interface for
GSAS-II." *J. Appl. Cryst.*** https://www.osti.gov/servlets/purl/1465500
Documents `GSASIIscriptable` and the `G2Project` object model (`add_powder_histogram`, `add_phase`,
`do_refinements`) plus MPI parallelization of scripted refinements across cores/nodes. The cleanest
existing API contract for programmatic refinement and the natural reference for our engine's
scripting surface and for an MPI/GPU performance baseline.

**GSAS-II sequential-refinement user reports (mailing list / ResearchGate, 2016–2017).**
https://www.mail-archive.com/gsas-ii@mailman.aps.anl.gov/msg00163.html
Primary evidence of the instability we are fixing: the recurring `invalid metric tensor / cell-Dij
refinement not advised` exception, phase fractions not summing to unity despite wildcard constraints,
and results that compute but fail to propagate to plots. These reports are converted directly into
regression tests (metric-tensor positive-definiteness guard; phase-fraction conservation).

**GSAS-II sequential tutorial & "what is new" notes.**
https://advancedphotonsource.github.io/GSAS-II-tutorials/SeqRefine/SequentialTutorial.htm
Reveals the triple lattice abstraction (user sees a,b,c; engine refines reciprocal A-tensor terms;
sequential mode refines hydrostatic/elastic Dij offsets). Identifying this leaky abstraction as a
root cause of the metric-tensor errors is what motivates our single, direct, autodiff-friendly cell
parameterization.

**Rodríguez-Carvajal, J. — FullProf Suite.** https://www.ill.eu/sites/fullprof/
The reference Fortran engine for neutron (CW and TOF, nuclear and magnetic) and X-ray powder data;
maximally flexible and reliable but with a steep, error-prone control-file interface. Benchmark for
correctness of neutron/magnetic scattering and TOF handling.

**Lutterotti, L. — MAUD (Materials Analysis Using Diffraction).** https://luttero.github.io/maud/ ·
https://www.iucr.org/resources/other-directories/software/maud
The strongest open tool for the dynamic-experiment observables we care about beyond structure:
crystallographic texture (E-WIMV), residual stress/strain, and size-strain microstructure with
anisotropy, across X-ray/synchrotron/neutron/TOF/electron data and multi-bank geometries. Defines the
feature targets for our texture/strain/size-strain domain modules and is the reference for
multi-detector TOF (HIPPO-style) refinement.

**Maino, S. A. *et al.* (NIST) — "MAUD Rietveld refinement for crystallographic texture."**
https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=932239
A careful instructional account of texture refinement of Ti-6Al-4V from HIPPO neutron data and EBSD,
explicitly cataloguing the "hidden challenges" and UX pitfalls of the current workflow. A rich source
of UX failure cases to design against and of validated texture results for the corpus.

**EasyScience — easyDiffraction.** https://easydiffraction.github.io ·
https://github.com/EasyScience/EasyDiffractionApp
The most architecturally modern open tool: a QML GUI wrapping external calculation engines
(CrysPy/CrysFML) with pluggable minimizers (lmfit, bumps, DFO-LS) for CW and TOF neutron data, with
STAR/CIF-based human-readable I/O. Important both as a model for engine/GUI decoupling and as a
cautionary data point — its own documentation labels structure refinement "yet unstable," underscoring
that a clean GUI does not by itself deliver refinement stability.

---

## 3. Reusable scattering & crystallography libraries (build-on, don't reinvent)

**Grosse-Kunstleve, R. W., Sauter, N. K., Moriarty, N. W. & Adams, P. D. (2002). "The Computational
Crystallography Toolbox." *J. Appl. Cryst.* 35, 126–136.** https://github.com/cctbx/cctbx_project ·
https://cctbx.github.io/
The reference reusable toolkit: ISO C++ with Python bindings, organized as small modules covering unit
cells, space groups, scatterers, ADPs, X-ray scattering, reflection data, and refinement building
blocks, plus the PHIL hierarchical configuration system. Its explicit rationale — that ad-hoc,
re-implemented symmetry tables are a recurring bug source — is exactly why we adopt it for symmetry,
scatterers, and CIF I/O instead of rewriting them. PHIL also informs our typed, hierarchical project
config.

**CrysPy / CrysFML (used by easyDiffraction).**
The calculation engines behind easyDiffraction; relevant as concrete, modern examples of a scattering
library cleanly separated from a GUI, and as candidate components or correctness references for the
CW/TOF neutron forward model.

---

## 4. AI / ML for diffraction: triage, hypothesis search, and hard landscapes

**Lee, J. *et al.* — CPICANN: convolutional self-attention crystallographic phase identifier (IUCr,
2024).** https://journals.iucr.org/m/issues/2024/04/00/fc5077/index.html
A neural phase identifier trained on tens of thousands of simulated single-phase patterns (COD-derived)
that recommends candidate structures for *subsequent* Rietveld/Le Bail refinement. The canonical model
for the "AI proposes, the engine disposes" layering: it supplies seeds and bounds, never final numbers.
Defines the interface contract for our phase-ID triage service.

**Dara: "Automated multiple-hypothesis phase identification and refinement from powder XRD" (2026).**
https://pmc.ncbi.nlm.nih.gov/articles/PMC12895389/
An agentic framework performing an exhaustive tree search over plausible phase combinations within a
chemical space, validating *each* hypothesis with a real Rietveld engine (BGMN), deployed behind a web
UI and API in autonomous and standard labs. The architectural template for our multi-hypothesis
orchestrator and the strongest argument for keeping a deterministic, auditable engine at the core of
any agentic loop.

**Benrabah *et al.* (2026). "Deep learning for real-time phase quantification from XRD." *Adv. Eng.
Mater.*** https://advanced.onlinelibrary.wiley.com/doi/10.1002/adem.202503172
A CNN trained on 40,000+ real synchrotron patterns predicts two-phase steel fractions at ~2% MAE and
millisecond inference — roughly two orders of magnitude faster than Rietveld. Establishes the role of
fast ML inference for *live* monitoring during dynamic experiments and a quantitative accuracy/latency
target for our triage layer; also a reminder that ML estimates need engine validation before they are
trusted as results.

**"Neural networks for rapid phase quantification of cultural-heritage XRD" (IUCr, 2024).**
https://journals.iucr.org/j/issues/2024/03/00/yr5124/index.html
Dense-NN phase mapping of XRD-CT, seeded by an initial Rietveld refinement that supplies peak widths
and integrated intensities to make simulated training data realistic. A concrete pattern for closing
the loop between physics-based refinement and ML training-data generation — relevant to how we
synthesize our own training/regression corpus.

**Li, Q. *et al.* (2024/2025). "Powder diffraction crystal structure determination using generative
models" (PXRDGen). *Nat. Commun.* 16, 7428.** https://arxiv.org/pdf/2409.04727
Generative structure solution followed by automated Rietveld refinement. Its key empirical finding
disciplines our AI layer: refinement reliably converges only when the generated model is already
close (RMSE below ~0.05) and largely fails past ~0.4 — i.e., the value of AI is in producing
*good seeds*, and the engine must report when a seed is too poor to refine.

**Physically-constrained autoencoder-assisted Bayesian optimization for high-dimensional,
defect-sensitive refinement (arXiv 2026).** https://arxiv.org/pdf/2601.00855
Benchmarks traditional BO, SAAS-BO, and pc-VAE BO for refining correlated, defect-sensitive structure
parameters (Ho₂Ti₂O₇) on top of a conventional GSAS-II Rietveld fit. The reference for our optional
BO module that attacks high-dimensional, ill-conditioned parameter spaces where local least squares
stalls — and a model for embedding domain priors as constraints.

---

## 5. Differentiable & GPU-accelerated least squares (the engine substrate)

**Hofer, L. R., Krstajić, M. & Smith, R. P. (2022). "JAXFit: trust-region nonlinear least-squares
curve fitting on the GPU." arXiv:2208.12187.** https://arxiv.org/pdf/2208.12187
Demonstrates a GPU trust-region least-squares solver where the Jacobian is obtained by automatic
differentiation rather than finite differences, with fit functions written in plain Python. The direct
architectural model for our refinement engine: AD Jacobians remove a major source of the numerical
fragility seen in legacy finite-difference Rietveld codes, and the GPU substrate addresses throughput.

**Campagne, J.-E. *et al.* (2023). "JAX-COSMO: end-to-end differentiable, GPU-accelerated cosmology."
arXiv:2302.05163.** https://arxiv.org/pdf/2302.05163
A mature exemplar of an entire physical forward model made differentiable in JAX. Its argument is
decisive for parametric refinement: finite differences are hard to stabilize and need ≥2N+1 model
evaluations for N parameters, which is impractical inside an outer iterative loop — precisely the
shared-parameter outer loop a parametric refinement is. Motivates building the whole
`pattern = f(params)` map as a differentiable program.

**Mortensen et al.; general differentiable-programming-for-optimization survey (arXiv 2026).**
https://arxiv.org/html/2601.16510v1
Background on embedding optimization within autodiff frameworks (PyTorch/TF/JAX). Informs design
choices such as differentiating *through* the inner solve when we later want gradients of refined
outputs with respect to experimental controls.

---

## 6. Agentic & spec-driven software engineering (the development discipline)

**"From vibe coding to spec-driven development" (Towards Data Science, 2026).**
https://towardsdatascience.com/from-vibe-coding-to-spec-driven-development/
Articulates the prevailing professional practice: drive agents against detailed specifications with
human oversight, implementing in reviewed task groups rather than one-shot, with the developer steering
architecture and review. The organizing principle of our Phase 0–6 pipeline and of the
requirements/plan/validation document triad.

**Mathews, N. S. & Nagappan, M. (2024). "Test-driven development and LLM-based code generation." ASE
2024.**
Empirical evidence that supplying tests up front improves the correctness of LLM-generated code. The
foundation for making our physics-derived invariant suite the *primary* artifact agents implement
against, not an afterthought.

**"FeatureBench: benchmarking agentic coding for complex feature development" (arXiv 2026).**
https://arxiv.org/html/2602.10975v1
Shows that test-driven task formulation with explicit interfaces and expected-behavior descriptions
enables reliable execution-based evaluation of coding agents. Validates our approach of pairing each
spec interface with executable acceptance tests so that "passes the suite" ≈ "correct implementation."

**"A survey on code generation with LLM-based agents" (arXiv 2508.00083, 2025);
"Vibe coding vs. agentic coding" (arXiv 2505.19443, 2025);
agentic-issue-resolution and reproducible-evaluation surveys (arXiv 2512.22256; 2604.01437).**
Collectively document the known weaknesses of coding agents — logical defects, performance pitfalls,
and the need to feed agents non-public conventions and API contracts — and emerging mitigations
(PRM-based course correction, reproducible/explainable evaluation). These justify the hard guardrails
in our pipeline: human-designed layer boundaries, a project-context/conventions document, and
engine-touching merges gated on full-corpus runs.

**HPC/SciML agentic-engineering reports — Fortran→Kokkos migration (arXiv 2509.12443); "SciML agents:
write the solver, not the solution" (arXiv 2509.09936); agentic evaluation in PETSc (arXiv 2603.15976);
SWE-Perf (arXiv 2507.12415).**
Evidence that agents can perform numerically sensitive migration and performance work *when* bounded by
strong tests and benchmarks. Directly relevant to our Phase 4 (performance as a measured, gated step)
and to any decision to port rather than rewrite a legacy kernel.

---

## 7. Reference framing

**Rietveld, H. M. (1969). "A profile refinement method for nuclear and magnetic structures." *J. Appl.
Cryst.* 2, 65–71.**
The original method. Defines the least-squares objective minimizing the weighted sum of squared
observed–calculated profile differences — the quantity every invariant in our oracle is ultimately
defined against.

**Coelho, A. A. — TOPAS / TOPAS-Academic.** http://www.topas-academic.net/
The commercial benchmark for both parametric refinement and raw performance/stability. Not open
source, but its scripting model and surface-refinement capabilities define the practical bar our open
engine aims to meet or exceed for parametric experiments.
