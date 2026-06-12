"""
AI services — propose-only triage that never returns final scientific results.

The contract (CPICANN / Dara / PXRDGen lesson): **AI proposes, the engine
disposes.** An AI service may emit candidate phases, seeds, and confidences; they
round-trip through the deterministic engine, and acceptance is gated by an engine
output (Rwp/GoF), never by an AI confidence. AI recommendations carry provenance
and are subject to hallucination containment (impossible inputs are rejected or
flagged for repair, never fabricated into a result).
"""

from __future__ import annotations

from aura.ai.identify import AIProposal, PhaseIdentifier

__all__ = ["AIProposal", "PhaseIdentifier"]
