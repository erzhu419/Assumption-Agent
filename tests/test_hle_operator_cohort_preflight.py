import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_programmatic_domain_rule_families_cover_answer_time_rules(self) -> None:
        self.assertEqual(
            _programmatic_domain_rule_family("bioinformatics_reference_imputation_pi_only_bias"),
            "reference_imputation_diversity_bias",
        )
        self.assertEqual(
            _programmatic_domain_rule_family("quant_genetics_heritability_pgs_necessity_none_true"),
            "quant_genetics_necessity",
        )
        self.assertEqual(
            _programmatic_domain_rule_family("ecology_voc_latitude_alpha_beta_direction_matrix"),
            "ecology_effect_direction",
        )
        self.assertEqual(
            _programmatic_domain_rule_family("enclosed_signal_not_available_for_between_host_navigation"),
            "enclosed_signal_navigation",
        )
        self.assertEqual(
            _programmatic_domain_rule_family("ontario_former_client_confidential_screen"),
            "legal_confidential_screen",
        )

        tagged = _programmatic_domain_rule_candidate_tags(
            problem={
                "category": "Biology/Medicine",
                "raw_subject": "Bioinformatics",
            },
            stem=(
                "Reference genome imputed low quality variants; Watterson theta and nucleotide diversity pi "
                "are compared. A GWAS polygenic score is evaluated against heritability. Plant VOC latitude "
                "effects list alpha and beta diversity directions. A syconium signal is solely within the "
                "enclosed structure and asks about long distance host tree navigation. An Ontario law firm "
                "has confidential former client information."
            ),
            options={},
        )

        self.assertIn("reference_imputation_diversity_bias", tagged)
        self.assertIn("quant_genetics_necessity", tagged)
        self.assertIn("ecology_effect_direction", tagged)
        self.assertIn("enclosed_signal_navigation", tagged)
        self.assertIn("legal_confidential_screen", tagged)

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

    def test_preflight_can_exclude_existing_hle_hashes(self) -> None:
        fake_rows = [
            {"question": "raw seen text", "answer": "A", "answer_type": "multipleChoice"},
            {"question": "raw new text", "answer": "B", "answer_type": "multipleChoice"},
        ]
        fake_stage = {
            "status": "activated",
            "reason": "operator_specs_compiled_for_internal_reasoning",
            "operator_source_ids": ["framework_answer_bearing_relation"],
            "operator_source_types": ["framework"],
            "operator_specs": [
                {
                    "source_id": "framework_answer_bearing_relation",
                    "required_output_slots": ["question_target_relation"],
                }
            ],
            "fallback_retrieval": {},
            "verifier_check_count": 1,
        }

        def fake_problem_from_row(row, *, scanned, skipped_before):
            return {
                "scanned_index": scanned,
                "id_hash": "seen" if scanned == 1 else "new",
                "question_hash": f"q{scanned}",
                "category": "Biology/Medicine",
                "raw_subject": "Biology",
                "answer_type": "multipleChoice",
                "_question": row["question"],
                "_answer": row["answer"],
            }

        with TemporaryDirectory() as tmp, patch(
            "assumption_os.hle_operator_cohort_preflight._load_hle_test_dataset",
            return_value=fake_rows,
        ), patch(
            "assumption_os.hle_operator_cohort_preflight._collect_existing_hle_problem_hashes",
            return_value={"seen"},
        ), patch(
            "assumption_os.hle_operator_cohort_preflight._problem_from_row",
            side_effect=fake_problem_from_row,
        ), patch(
            "assumption_os.hle_operator_cohort_preflight._compile_hle_operator_stage",
            return_value=fake_stage,
        ), patch(
            "assumption_os.hle_operator_cohort_preflight._classify_hle_domain",
            return_value="science",
        ):
            payload = build_hle_operator_cohort_preflight_payload(
                root=Path(tmp),
                eval_id="exclude-existing",
                target_size=1,
                max_scan=2,
                graph_dir=Path(tmp) / "graph",
                exclude_existing_hle_artifacts=True,
            )

        self.assertEqual(payload["sampling"]["excluded_existing_problem_count"], 1)
        self.assertEqual(payload["metrics"]["scan_summary"]["skipped_existing_problem_hash"], 1)
        self.assertEqual([row["problem_id_hash"] for row in payload["rows"]], ["new"])
        self.assertNotIn("raw seen text", json.dumps(payload))
        self.assertNotIn("raw new text", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
