import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_os.hle_three_part_eval import build_three_part_eval_payload, format_markdown


def _row(problem_hash: str, variant: str, correct: bool, method: str = "") -> dict:
    selection = {"selection_method": method} if method else {}
    if method == "verified_or_abstain_direct_fallback":
        selection["verified_or_abstain_gate"] = {"reason": "unverified_selection_method"}
    return {
        "problem_id_hash": problem_hash,
        "question_hash": f"q-{problem_hash}",
        "variant": variant,
        "model": "gpt-5.4-mini",
        "answer_type": "multipleChoice",
        "correct": correct,
        "error": None,
        "raw_question_persisted": False,
        "prediction_text_persisted": False,
        "gold_answer_persisted": False,
        "component_efficacy": {"selection": selection},
    }


class HleThreePartEvalTest(unittest.TestCase):
    def test_three_part_eval_uses_metadata_only_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate = {
                "eval_id": "current",
                "raw_content_persisted": False,
                "failed_gates": ["operator_application_fidelity_if_verified"],
                "paper_clean_failed_gates": ["operator_application_fidelity_if_verified"],
                "pollution_pass": True,
                "runtime_policy": {"process_timeout_policy": "watch_only"},
                "failure_diagnostics": {
                    "agent_gain_loss": {"agent_only_correct": 1, "all_three_wrong": 0}
                },
                "metrics": {
                    "operator_activation_summary": {"passed": True, "activated_row_count": 2},
                    "operator_application_summary": {"passed": False, "decorative_use_rate": 1.0},
                },
            }
            baseline = {
                "eval_id": "baseline",
                "raw_content_persisted": False,
                "metrics": {
                    "operator_activation_summary": {"passed": True},
                    "operator_application_summary": {"passed": False},
                },
            }
            current_rows = [
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "assumption_agent_recursive_verify", True, "orthogonal_structural_elimination_choice"),
                _row("p2", "raw", True),
                _row("p2", "hipporag_baseline", True),
                _row("p2", "assumption_agent_recursive_verify", True, "verified_or_abstain_direct_fallback"),
            ]
            baseline_rows = [
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "assumption_agent_recursive_verify", False, "operator_application_fidelity_choice"),
                _row("p2", "raw", True),
                _row("p2", "hipporag_baseline", True),
                _row("p2", "assumption_agent_recursive_verify", True, "verified_or_abstain_direct_fallback"),
            ]
            preflight = {
                "rows": [
                    {"problem_id_hash": "p1", "cohort_family": "structural_transfer"},
                    {"problem_id_hash": "p2", "cohort_family": "controlled_variable"},
                ]
            }
            (run_dir / "current.json").write_text(json.dumps(aggregate), encoding="utf-8")
            (run_dir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
            (run_dir / "current_shard_000.json").write_text(json.dumps({"rows": current_rows}), encoding="utf-8")
            (run_dir / "baseline_shard_000.json").write_text(json.dumps({"rows": baseline_rows}), encoding="utf-8")
            (run_dir / "current_shard_000.jsonl").write_text(
                json.dumps({
                    "event": "operator_application_selection_deferred",
                    "reason": "operator_source_adjudicator_not_run",
                    "source_defer": {"reason": "operator_source_adjudicator_not_run"},
                })
                + "\n",
                encoding="utf-8",
            )
            preflight_path = run_dir / "preflight.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

            payload = build_three_part_eval_payload(
                run_dir=run_dir,
                eval_id="current",
                baseline_eval_id="baseline",
                preflight_path=preflight_path,
            )
            markdown = format_markdown(payload)

        answer = payload["panels"]["answer_quality"]
        fidelity = payload["panels"]["application_fidelity"]
        residual = payload["panels"]["residual_family"]
        safety = payload["panels"]["metadata_safety"]

        self.assertTrue(answer["agent_above_raw"])
        self.assertEqual(answer["agent_minus_raw_accuracy"], 0.5)
        self.assertFalse(fidelity["passed"])
        self.assertEqual(fidelity["direct_operator_selection_count"], 0)
        self.assertEqual(
            fidelity["operator_source_defer_reason_counts"],
            {"operator_source_adjudicator_not_run": 1},
        )
        self.assertEqual(residual["selector_before_after_comparison"]["agent_improved_count"], 1)
        self.assertIn("structural_transfer", residual["by_family"])
        self.assertIn("structural_transfer", residual["baseline_by_family"])
        self.assertTrue(residual["single_round_family_delta_measured"])
        self.assertEqual(
            residual["agent_error_rate_delta_by_family"]["structural_transfer"]["current_minus_baseline_error_rate"],
            -1.0,
        )
        self.assertFalse(residual["residual_family_learning_measured"])
        self.assertTrue(safety["raw_content_not_persisted"])
        self.assertNotIn("raw HLE", markdown)

    def test_three_part_eval_does_not_count_vacuous_fidelity_as_application_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate = {
                "eval_id": "current",
                "raw_content_persisted": False,
                "paper_clean_pass": False,
                "metrics": {
                    "operator_activation_summary": {"passed": True, "activated_row_count": 2},
                    "operator_application_summary": {
                        "passed": True,
                        "applied_row_count": 0,
                        "application_coverage_rate": 0.0,
                        "deferred_not_applied_count": 2,
                        "decorative_use_rate": 0.0,
                    },
                },
            }
            rows = [
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "assumption_agent_recursive_verify", False, "verified_or_abstain_direct_fallback"),
                _row("p2", "raw", True),
                _row("p2", "hipporag_baseline", True),
                _row("p2", "assumption_agent_recursive_verify", True, "verified_or_abstain_direct_fallback"),
            ]
            (run_dir / "current.json").write_text(json.dumps(aggregate), encoding="utf-8")
            (run_dir / "current_shard_000.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            (run_dir / "current_shard_000.jsonl").write_text("", encoding="utf-8")
            log_out = run_dir / "three_part.jsonl"

            payload = build_three_part_eval_payload(run_dir=run_dir, eval_id="current")
            markdown = format_markdown(payload)
            logged_payload = build_three_part_eval_payload(
                run_dir=run_dir,
                eval_id="current",
                log_out=log_out,
            )
            events = [
                json.loads(line)
                for line in log_out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        fidelity = payload["panels"]["application_fidelity"]
        self.assertTrue(fidelity["applied_row_fidelity_passed"])
        self.assertFalse(fidelity["application_coverage_present"])
        self.assertFalse(fidelity["operator_application_evidence_passed"])
        self.assertFalse(fidelity["passed"])
        self.assertTrue(payload["pass_summary"]["applied_row_fidelity_passed"])
        self.assertFalse(payload["pass_summary"]["operator_application_fidelity_passed"])
        self.assertFalse(payload["pass_summary"]["operator_application_evidence_passed"])
        self.assertIn("application coverage present: `False`", markdown)
        self.assertEqual(logged_payload["diagnostic_log_out"], str(log_out))
        self.assertTrue(any(event["event"] == "hle_three_part_eval_panel_operator_fidelity" for event in events))
        self.assertTrue(all(event.get("raw_content_persisted") is False for event in events))

    def test_three_part_eval_separates_programmatic_domain_rule_fidelity(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate = {
                "eval_id": "current",
                "raw_content_persisted": False,
                "paper_clean_pass": True,
                "metrics": {
                    "operator_activation_summary": {"passed": False, "activated_row_count": 0},
                    "operator_application_summary": {
                        "passed": False,
                        "applied_row_count": 0,
                        "application_coverage_rate": 0.0,
                    },
                },
            }
            agent = _row("p1", "assumption_agent_recursive_verify", True, "domain_rule_verifier_priority")
            agent["component_efficacy"]["domain_rule_mc_verifier"] = {
                "status": "activated",
                "rule_id": "sec_mals_mass_balance_affinity_monomer",
                "confidence": "verified",
                "selected_domain_rule_candidate": True,
                "candidate_correct_for_eval": True,
                "short_circuited_child_generation": True,
            }
            agent["component_efficacy"]["recursive"] = {
                "execution_mode": "domain_rule_preverified",
                "early_stop_reason": "domain_rule_preverified",
            }
            rows = [
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                agent,
            ]
            (run_dir / "current.json").write_text(json.dumps(aggregate), encoding="utf-8")
            (run_dir / "current_shard_000.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            (run_dir / "current_shard_000.jsonl").write_text("", encoding="utf-8")

            payload = build_three_part_eval_payload(run_dir=run_dir, eval_id="current")
            markdown = format_markdown(payload)

        operator_fidelity = payload["panels"]["application_fidelity"]
        domain_rule = payload["panels"]["programmatic_domain_rule_fidelity"]
        self.assertFalse(operator_fidelity["passed"])
        self.assertTrue(domain_rule["passed"])
        self.assertEqual(domain_rule["activated_count"], 1)
        self.assertEqual(domain_rule["selected_count"], 1)
        self.assertEqual(domain_rule["short_circuit_count"], 1)
        self.assertTrue(payload["pass_summary"]["programmatic_domain_rule_fidelity_passed"])
        self.assertIn("Programmatic Domain-Rule Fidelity", markdown)

    def test_separate_baseline_preflight_supports_family_before_after(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate = {
                "eval_id": "current",
                "raw_content_persisted": False,
                "paper_clean_pass": True,
                "metrics": {
                    "operator_activation_summary": {"passed": True},
                    "operator_application_summary": {
                        "passed": True,
                        "applied_row_count": 1,
                        "application_coverage_rate": 1.0,
                    },
                },
            }
            baseline = {
                "eval_id": "baseline",
                "raw_content_persisted": False,
                "metrics": {},
            }
            current_rows = [
                _row("current-only", "raw", False),
                _row("current-only", "hipporag_baseline", False),
                _row("current-only", "assumption_agent_recursive_verify", True),
            ]
            baseline_rows = [
                _row("baseline-only", "raw", False),
                _row("baseline-only", "hipporag_baseline", False),
                _row("baseline-only", "assumption_agent_recursive_verify", False),
            ]
            current_preflight = {
                "rows": [{"problem_id_hash": "current-only", "cohort_family": "controlled_variable"}]
            }
            baseline_preflight = {
                "rows": [{"problem_id_hash": "baseline-only", "cohort_family": "controlled_variable"}]
            }
            (run_dir / "current.json").write_text(json.dumps(aggregate), encoding="utf-8")
            (run_dir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
            (run_dir / "current_shard_000.json").write_text(json.dumps({"rows": current_rows}), encoding="utf-8")
            (run_dir / "baseline_shard_000.json").write_text(json.dumps({"rows": baseline_rows}), encoding="utf-8")
            (run_dir / "current_shard_000.jsonl").write_text("", encoding="utf-8")
            current_preflight_path = run_dir / "current_preflight.json"
            baseline_preflight_path = run_dir / "baseline_preflight.json"
            current_preflight_path.write_text(json.dumps(current_preflight), encoding="utf-8")
            baseline_preflight_path.write_text(json.dumps(baseline_preflight), encoding="utf-8")

            payload = build_three_part_eval_payload(
                run_dir=run_dir,
                eval_id="current",
                baseline_eval_id="baseline",
                preflight_path=current_preflight_path,
                baseline_preflight_path=baseline_preflight_path,
                residual_family_protocol=True,
            )
            markdown = format_markdown(payload)

        residual = payload["panels"]["residual_family"]
        self.assertTrue(residual["residual_family_before_after_measured"])
        self.assertTrue(residual["residual_family_learning_measured"])
        self.assertEqual(
            residual["agent_error_rate_delta_by_family"]["controlled_variable"][
                "current_minus_baseline_error_rate"
            ],
            -1.0,
        )
        self.assertTrue(payload["pass_summary"]["residual_family_before_after_measured"])
        self.assertTrue(payload["pass_summary"]["residual_family_learning_measured"])
        self.assertIn("baseline family map provided: `True`", markdown)

    def test_source_prefetch_problem_tags_can_supply_family_map(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate = {
                "eval_id": "current",
                "raw_content_persisted": False,
                "paper_clean_pass": False,
                "metrics": {
                    "operator_activation_summary": {"passed": True},
                    "operator_application_summary": {"passed": True},
                },
            }
            rows = [
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "assumption_agent_recursive_verify", False),
            ]
            source_prefetch = {
                "problems": [
                    {
                        "problem_id_hash": "p1",
                        "operator_family_tags": ["structural_transfer", "incremental_replacement"],
                    }
                ]
            }
            (run_dir / "current.json").write_text(json.dumps(aggregate), encoding="utf-8")
            (run_dir / "current_shard_000.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            (run_dir / "current_shard_000.jsonl").write_text("", encoding="utf-8")
            preflight_path = run_dir / "source_prefetch.json"
            preflight_path.write_text(json.dumps(source_prefetch), encoding="utf-8")

            payload = build_three_part_eval_payload(
                run_dir=run_dir,
                eval_id="current",
                preflight_path=preflight_path,
            )

        residual = payload["panels"]["residual_family"]
        family = "incremental_replacement+structural_transfer"
        self.assertTrue(residual["family_map_provided"])
        self.assertIn(family, residual["by_family"])


if __name__ == "__main__":
    unittest.main()
