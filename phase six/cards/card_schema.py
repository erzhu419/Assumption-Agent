"""Card schema for Loop v2 ('Triggered Cognitive Patches').

A card is a structured, slice-targeted micro-procedure with a trigger,
a procedural core, a verification step, and an optional worked example.
This is the unit of evolution in Loop v2 (replacing v1's free-form
aphorisms).

The procedural core is what we test for content effect (vs an ablation
that keeps trigger + style but removes the core).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Card:
    name: str                       # short, e.g. "denominator_lock"
    slice: str                      # which slice it primarily targets
    trigger: str                    # human-readable trigger description
    failure_prevented: str          # the specific cognitive bias / error pattern
    patch: List[str]                # ordered procedural steps (the core)
    verification: str               # self-check step
    example: Optional[str] = None   # optional worked mini-example
    notes: str = ""                 # provenance / design notes

    def render_full(self) -> str:
        """Full prompt-injectable form (used in WITH-CARD condition)."""
        lines = [
            f"## METHODOLOGICAL PATCH: {self.name}",
            f"Trigger: {self.trigger}",
            f"Failure prevented: {self.failure_prevented}",
            "Procedure:",
        ]
        for i, step in enumerate(self.patch, 1):
            lines.append(f"  {i}. {step}")
        lines.append(f"Verification: {self.verification}")
        if self.example:
            lines.append(f"Worked example: {self.example}")
        return "\n".join(lines)

    def render_ablated(self) -> str:
        """Ablated form: keep trigger + verification, remove procedural core.

        The ablation tests whether the gate's preference is driven by the
        actual procedure or only by the surrounding rhetoric / structure.
        If wr stays high under ablation, the card is a 'style' card
        (rejected). If wr drops, the procedure is the driver (accepted)."""
        lines = [
            f"## METHODOLOGICAL HINT: {self.name}",
            f"Trigger: {self.trigger}",
            f"Failure to avoid: {self.failure_prevented}",
            f"After answering, briefly verify: {self.verification}",
        ]
        return "\n".join(lines)


def load_cards(path) -> List[Card]:
    raw = json.loads(open(path, encoding="utf-8").read())
    return [Card(**c) for c in raw]


def save_cards(cards: List[Card], path) -> None:
    open(path, "w", encoding="utf-8").write(
        json.dumps([asdict(c) for c in cards], ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    # Smoke-test the schema
    c = Card(
        name="denominator_lock",
        slice="bayesian",
        trigger="Problem gives percentages, base rates, or conditional "
                  "probabilities across different groups.",
        failure_prevented="Base-rate neglect; likelihood/posterior inversion.",
        patch=[
            "Choose one denominator (preferably 1000 or 10000 cases).",
            "Convert every rate into integer counts on that denominator.",
            "Compute the requested probability using counts only.",
        ],
        verification="Did you confuse P(evidence|hypothesis) with "
                      "P(hypothesis|evidence)?",
        example=(
            "Disease 1% prevalence, test 99% sens, 99% spec.\n"
            "Of 10000: 100 sick (99 test+); 9900 healthy (99 test+).\n"
            "P(sick|test+) = 99 / (99+99) = 0.5."
        ),
    )
    print(c.render_full())
    print()
    print("=== ABLATED ===")
    print(c.render_ablated())
