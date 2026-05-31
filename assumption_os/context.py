"""Formatting helpers for injecting activated assumptions into existing solvers."""

from __future__ import annotations

from .schema import ActivatedSubgraph, AssumptionType


def format_assumption_context(subgraph: ActivatedSubgraph, *, max_nodes: int = 6) -> str:
    lines = [
        "## Activated Assumption Subgraph",
        "Use these as falsifiable operating assumptions, not decorative advice.",
    ]
    for i, node in enumerate(subgraph.nodes[:max_nodes], start=1):
        score = subgraph.scores.get(node.id, 0.0)
        lines.append(f"\n[{i}] {node.id} ({node.type.value}, score={score:.3f})")
        lines.append(f"Claim: {node.claim}")
        if node.context_conditions:
            lines.append("Trigger/context: " + "; ".join(node.context_conditions[:3]))
        if node.predicted_effects:
            lines.append("Expected effect: " + "; ".join(node.predicted_effects[:2]))
        if node.risk_predictions:
            lines.append("Risks: " + "; ".join(node.risk_predictions[:2]))
        if node.verifiers:
            lines.append("Verifier: " + "; ".join(node.verifiers[:3]))

    residuals = [n for n in subgraph.nodes if n.type == AssumptionType.RESIDUAL]
    if residuals:
        lines.append("\n## Similar Historical Residuals")
        for r in residuals[:3]:
            lines.append(f"- {r.claim}")

    cases = [n for n in subgraph.nodes if n.type == AssumptionType.CASE]
    if cases:
        lines.append("\n## Anchoring Cases")
        for c in cases[:3]:
            lines.append(f"- {c.claim}")

    return "\n".join(lines).strip()
