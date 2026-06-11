# tests/performance — performance tier (nightly, non-blocking)

Scaling and memory invariants: O(points × nearby_peaks) via a deterministic
peak-contribution counter (not just timing), sequential memory stability (<5%
RSS growth over a 10k-pattern campaign), and backend (numpy/JAX) consistency.
Marked `@pytest.mark.slow`; nightly only.

(Empty until the chunked engine lands — Phase 8.)
