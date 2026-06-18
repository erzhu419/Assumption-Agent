from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.hle_selector_backprop import (
    build_hle_selector_backprop_payload,
    format_hle_selector_backprop_markdown,
)


class TestHleSelectorBackprop(unittest.TestCase):
    def test_backprop_finds_hipporag_preserve_policy_gain_without_raw_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "run"
            source.mkdir()
            rows = [
                _row("p1", "raw", False, "A"),
                _row("p1", "hipporag_baseline", True, "B"),
                _row("p1", "assumption_agent_recursive_verify", False, "A", gate="abstained"),
                _row("p2", "raw", True, "A"),
                _row("p2", "hipporag_baseline", True, "A"),
                _row("p2", "assumption_agent_recursive_verify", True, "A", gate="allowed"),
            ]
            (source / "unit_shard_000.json").write_text(
                json.dumps({"rows": rows, "metrics": {"raw_content_persisted": False}}),
                encoding="utf-8",
            )

            payload = build_hle_selector_backprop_payload(root=root, sources=[source])
            markdown = format_hle_selector_backprop_markdown(payload)

        self.assertTrue(payload["pass"])
        self.assertEqual(payload["metrics"]["complete_triad_count"], 2)
        self.assertEqual(payload["metrics"]["policy_simulation"]["agent_current"]["correct"], 1)
        self.assertEqual(payload["metrics"]["policy_simulation"]["verified_else_hipporag"]["correct"], 2)
        self.assertEqual(
            payload["metrics"]["recommended_adjustments"],
            {"prefer_hipporag_preserve_selector_for_unverified_mc": 1},
        )
        self.assertFalse(payload["raw_content_persisted"])
        self.assertNotIn("_question", json.dumps(payload))
        self.assertIn("verified_else_hipporag", markdown)


def _row(
    problem_id: str,
    variant: str,
    correct: bool,
    prediction_hash: str,
    *,
    gate: str = "unknown",
) -> dict[str, object]:
    row: dict[str, object] = {
        "problem_id_hash": problem_id,
        "question_hash": f"q-{problem_id}",
        "answer_hash": f"a-{problem_id}",
        "model": "gpt-5.4-mini",
        "variant": variant,
        "category": "Science",
        "raw_subject": "Physics",
        "answer_type": "multipleChoice",
        "correct": correct,
        "prediction_hash": prediction_hash,
        "prediction_text_persisted": False,
        "raw_question_persisted": False,
        "gold_answer_persisted": False,
        "module_trace": [],
        "call_metadata": {},
        "error": None,
    }
    if variant.startswith("assumption_agent"):
        row["component_efficacy"] = {
            "flags": {
                "recursive_child_validation_activated": True,
                "verified_or_abstain_abstained": gate == "abstained",
                "verified_or_abstain_allowed": gate == "allowed",
            },
            "selection": {
                "selection_method": (
                    "verified_or_abstain_direct_fallback"
                    if gate == "abstained"
                    else "candidate_claim_verifier_priority"
                ),
                "verified_or_abstain_gate": {"status": gate},
            },
        }
    else:
        row["component_efficacy"] = {}
    return row


if __name__ == "__main__":
    unittest.main()
