# tests/physics — physics/core tier (every PR)

The agentic oracle (`test_invariants.py`) and engine-driven physics invariants:
forward-model purity, Jacobian-vs-finite-difference, refinement fixpoint /
synthetic recovery / GoF calibration, parametric-engine equivalence, conservation
guards, seed discipline. Runs against the `engine` fixture (reference today,
production as it lands). An invariant becomes CI-blocking the moment a production
engine *can* fail it.
