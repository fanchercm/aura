# tests/unit — fast tier (every commit)

Sub-second, engine-independent tests: container model, parameter handling, I/O
readers, the bridge, provenance, and the pure-math crystallographic identities
that hold for any correct implementation. CI-blocking on every push/PR.

Houses the migrated `test_models.py`, `test_parameters_spec.py`, `test_cli.py`,
and the I/O reader tests (Phase 1+).
