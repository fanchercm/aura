# tests/ai — AI-layer guardrails

"AI proposes, engine disposes": the `PhaseIdentifier` propose-only contract,
candidate engine round-trip, provenance of AI recommendations, and hallucination
containment. The acceptance signal must always be an engine output (Rwp/GoF),
never an AI confidence. Runs every PR with the physics tier.
