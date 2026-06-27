import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_os.hle_operator_cohort_preflight import (
    _choose_underfilled_family,
    _cohort_metrics,
    _cohort_row,
    _operator_applicability_probe,
    _operator_family_tags_from_stage,
    _parse_family_targets,
    _programmatic_domain_rule_candidate_tags,
    _programmatic_domain_rule_cohort_row,
    _programmatic_domain_rule_family,
    build_hle_operator_cohort_preflight_payload,
    format_markdown,
)


class HleOperatorCohortPreflightTest(unittest.TestCase):
    def test_cohort_row_is_metadata_only(self) -> None:
        problem = {
            "scanned_index": 49,
            "id_hash": "pid",
            "question_hash": "qid",
            "category": "Math",
            "raw_subject": "Mathematics",
            "answer_type": "multipleChoice",
            "_question": "raw HLE text must not persist",
            "_answer": "B",
        }
        stage = {
            "status": "activated",
            "reason": "operator_specs_compiled_for_internal_reasoning",
            "operator_source_ids": ["strategy_S01"],
            "operator_source_types": ["method"],
            "required_slot_count": 2,
            "verifier_check_count": 1,
            "operator_specs": [
                {
                    "source_id": "strategy_S01",
                    "required_output_slots": ["control_or_baseline", "decision_rule"],
                }
            ],
            "fallback_retrieval": {
                "top_node_ids": ["strategy_S01"],
                "top_scores": [0.42],
                "relevance_gate": {
                    "kept_node_ids": ["strategy_S01"],
                    "kept_answer_time_family_hits": {"strategy_S01": ["controlled_variable"]},
                    "rejected_reasons": {"strategy_S11": "generic_numbered_strategy_disabled"},
                },
            },
        }

        row = _cohort_row(problem=problem, domain="math", stage=stage)
        payload = {
            "eval_id": "probe",
            "metrics": _cohort_metrics([row], {"scanned": 50, "selected": 1}),
            "rows": [row],
            "claim_boundary": "metadata-only",
        }
        markdown = format_markdown(payload)

        self.assertEqual(row["seed_offset"], 48)
        self.assertEqual(row["operator_source_ids"], ["strategy_S01"])
        self.assertEqual(row["operator_family_tags"], ["controlled_variable"])
        self.assertEqual(row["cohort_family"], "controlled_variable")
        self.assertEqual(row["required_slots"], ["control_or_baseline", "decision_rule"])
        self.assertNotIn("_question", row)
        self.assertNotIn("_answer", row)
        self.assertNotIn("raw HLE text", markdown)
        self.assertEqual(payload["metrics"]["operator_family_counts"], {"controlled_variable": 1})

    def test_family_targets_are_canonicalized_and_balanced(self) -> None:
        self.assertEqual(
            _parse_family_targets("causal=2,migration=1,structural_transfer=3,unknown=9"),
            {
                "controlled_variable": 2,
                "incremental_replacement": 1,
                "structural_transfer": 3,
            },
        )
        choice = _choose_underfilled_family(
            family_tags=["controlled_variable", "structural_transfer"],
            family_counts={"controlled_variable": 2, "structural_transfer": 0},
            family_targets={"controlled_variable": 2, "structural_transfer": 1},
        )
        self.assertEqual(choice, "structural_transfer")

    def test_operator_family_tags_include_builtin_and_fallback_hits(self) -> None:
        stage = {
            "operator_source_ids": ["framework_incremental_replacement_migration"],
            "operator_admissibility": {
                "kept_answer_time_family_hits": {
                    "strategy_S02": ["structural_transfer"],
                },
            },
            "fallback_retrieval": {
                "relevance_gate": {
                    "kept_answer_time_family_hits": {
                        "strategy_S03": ["controlled_variable"],
                    },
                },
            },
        }
        self.assertEqual(
            _operator_family_tags_from_stage(stage),
            ["controlled_variable", "incremental_replacement", "structural_transfer"],
        )

    def test_operator_applicability_probe_keeps_metadata_only(self) -> None:
        problem = {
            "scanned_index": 49,
            "id_hash": "pid",
            "question_hash": "qid",
            "category": "Science",
            "raw_subject": "Biology",
            "answer_type": "multipleChoice",
            "_question": "raw HLE text must not persist",
            "_answer": "B",
        }
        stage = {
            "status": "activated",
            "operator_source_ids": ["strategy_S01"],
            "operator_specs": [
                {
                    "source_id": "strategy_S01",
                    "source_claim": "Use a controlled contrast.",
                    "required_output_slots": ["control_or_baseline", "decision_rule"],
                }
            ],
        }

        def fake_call_model(**kwargs):
            self.assertIn("raw HLE text must not persist", kwargs["prompt"])
            return json.dumps({
                "answer": "B",
                "operator_audit": {
                    "used_operator_ids": ["strategy_S01"],
                    "required_slots_filled": ["control_or_baseline", "decision_rule"],
                    "operator_changed_candidate": True,
                    "decorative_use": False,
                },
            })

        probe = _operator_applicability_probe(
            problem=problem,
            stage=stage,
            model="gpt-5.4-mini",
            max_tokens=128,
            min_slot_rate=0.75,
            call_model=fake_call_model,
        )
        row = _cohort_row(
            problem=problem,
            domain="science",
            stage={
                **stage,
                "reason": "operator_specs_compiled_for_internal_reasoning",
                "operator_source_types": ["method"],
                "required_slot_count": 2,
                "verifier_check_count": 1,
            },
            applicability_probe=probe,
        )
        metrics = _cohort_metrics([row], {"selected": 1})

        self.assertEqual(probe["status"], "passed")
        self.assertEqual(probe["slot_completion_rate"], 1.0)
        self.assertFalse(probe["decorative_use"])
        self.assertEqual(row["applicability_probe"]["used_operator_ids"], ["strategy_S01"])
        self.assertNotIn("raw HLE text", json.dumps(row))
        self.assertEqual(metrics["applicability_probe_status_counts"], {"passed": 1})

    def test_programmatic_domain_rule_row_is_metadata_only(self) -> None:
        problem = {
            "scanned_index": 831,
            "id_hash": "pid",
            "question_hash": "qid",
            "category": "Biology/Medicine",
            "raw_subject": "Biochemistry",
            "answer_type": "multipleChoice",
            "_question": "SEC-MALS raw HLE text must not persist. Answer Choices: F. correct J. lure",
            "_answer": "F",
        }
        decision = {
            "label": "F",
            "rule_id": "sec_mals_mass_balance_affinity_monomer",
            "confidence": "verified",
            "evidence_required": False,
        }
        candidate_tags = _programmatic_domain_rule_candidate_tags(
            problem=problem,
            stem=str(problem["_question"]),
            options={"F": "Protein B can have higher affinity", "J": "None of the above"},
        )
        family = _programmatic_domain_rule_family(decision["rule_id"])

        row = _programmatic_domain_rule_cohort_row(
            problem=problem,
            domain="science",
            decision=decision,
            family=family,
            candidate_tags=candidate_tags,
        )
        payload = {
            "eval_id": "probe",
            "metrics": _cohort_metrics([row], {"selected": 1}),
            "rows": [row],
            "claim_boundary": "metadata-only",
        }
        markdown = format_markdown(payload)

        self.assertEqual(row["row_kind"], "programmatic_domain_rule")
        self.assertEqual(row["seed_offset"], 830)
        self.assertEqual(row["cohort_family"], "sec_mals_mass_balance")
        self.assertEqual(row["programmatic_domain_rule"]["rule_id"], decision["rule_id"])
        self.assertEqual(payload["metrics"]["programmatic_domain_rule_family_counts"], {"sec_mals_mass_balance": 1})
        self.assertIn("sec_mals_mass_balance", row["programmatic_domain_rule_family_tags"])
        self.assertNotIn("_question", row)
        self.assertNotIn("_answer", row)
        self.assertNotIn("raw HLE text", json.dumps(row))
        self.assertNotIn("raw HLE text", markdown)

    def test_preflight_diagnostic_log_is_metadata_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_out = root / "preflight.jsonl"
            payload = build_hle_operator_cohort_preflight_payload(
                root=root,
                eval_id="log-probe",
                target_size=0,
                max_scan=10,
                graph_dir=root / "graph",
                log_out=log_out,
                diagnostic_log_interval=1,
            )
            events = [
                json.loads(line)
                for line in log_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(payload["diagnostic_log_out"], str(log_out))
        self.assertTrue(events)
        self.assertEqual(events[0]["event"], "hle_operator_cohort_preflight_started")
        self.assertEqual(events[-1]["event"], "hle_operator_cohort_preflight_completed")
        self.assertTrue(all(event.get("raw_content_persisted") is False for event in events))
        self.assertNotIn("raw HLE text", json.dumps(events, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
