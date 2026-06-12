import tempfile
import unittest
from pathlib import Path

from assumption_os.simulator_transition_schema import (
    build_simulator_transition_schema_payload,
    make_transition_row,
    validate_transition_row,
    validate_transition_rows,
)


class SimulatorTransitionSchemaTest(unittest.TestCase):
    def test_schema_payload_validates_current_345_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(__file__).resolve().parents[1]
            out_dir = Path(td)
            payload = build_simulator_transition_schema_payload(
                root=root,
                eval_id="unit_simulator_transition_schema",
                schema_out=out_dir / "schema.json",
                dataset_out=out_dir / "dataset.jsonl",
                quarantine_out=out_dir / "quarantine.jsonl",
                write_artifacts=True,
            )
            metrics = payload["metrics"]

            self.assertTrue(payload["pass"], payload["failed_gates"])
            self.assertEqual(metrics["raw_row_count"], 345)
            self.assertEqual(metrics["valid_row_count"], 345)
            self.assertEqual(metrics["invalid_row_count"], 0)
            self.assertEqual(metrics["redacted_row_count"], 345)
            self.assertTrue(metrics["provenance_hash_unique"])
            self.assertEqual(set(metrics["split_counts"]), {"train", "validation", "test"})
            self.assertTrue((out_dir / "schema.json").exists())
            self.assertEqual(len((out_dir / "dataset.jsonl").read_text().splitlines()), 345)
            self.assertEqual((out_dir / "quarantine.jsonl").read_text(), "")

    def test_redaction_violation_goes_to_quarantine_report(self):
        row = _valid_row("redaction_case")
        row["state"]["prompt"] = "raw prompt should not be present"
        report = validate_transition_rows([row])

        self.assertEqual(report.valid_row_count, 0)
        self.assertEqual(report.quarantine_row_count, 1)
        self.assertIn("redaction_forbidden_payload", report.issue_counts)

    def test_provenance_hash_tamper_detected(self):
        row = _valid_row("tamper_case")
        row["outcome"]["utility_vs_baseline"] = 0.0
        issues = validate_transition_row(row)
        issue_names = {issue.issue for issue in issues}

        self.assertIn("provenance_hash_mismatch", issue_names)


def _valid_row(row_id: str) -> dict:
    return make_transition_row(
        row_id=row_id,
        state={
            "domain": "unit",
            "pattern": "unit_pattern",
            "active_assumptions": ["unit_assumption"],
            "residual_cluster": "unit_cluster",
            "formal_gate_state": "not_applicable",
            "preflight_state": "ready",
            "world_model_features": ["domain:unit", "pattern:unit_pattern"],
        },
        action={"type": "run_ablation", "arm": "candidate_vs_baseline"},
        prediction={"p_accept": 0.7, "p_regress": 0.1, "expected_utility": 0.8, "uncertainty": 0.2},
        outcome={"accepted": True, "utility_vs_baseline": 1.0, "control_harm": False, "regression": False, "cost": 1.0},
        provenance={"artifact_id": "unit_artifact", "source_row_id": row_id, "split": "train"},
    )


if __name__ == "__main__":
    unittest.main()
