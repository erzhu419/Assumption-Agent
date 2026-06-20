import unittest
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from assumption_os.hle_module_activation_audit import build_hle_module_activation_audit_payload
from assumption_os.hle_smoke_eval import (
    _aggregate_rows,
    _apply_math_candidate_claim_verifier,
    _apply_verified_or_abstain_selection,
    _agent_child_model,
    _answer_format_repair_timeout,
    _build_agent_hipporag_child_context,
    _build_hle_evidence_bridge_context,
    _candidate_evidence_queries,
    _clean_evidence_text,
    _canonicalize_exact_answer_candidate,
    _canonicalize_multiple_choice_answer,
    _can_stop_recursive_children_early,
    _collect_existing_hle_problem_hashes,
    _component_efficacy_from_plan,
    _component_efficacy_summary,
    _control_comparison,
    _cost_aware_hipporag_preserve_trigger,
    _cost_aware_raw_budget_preserve_trigger,
    _cost_aware_raw_preserve_trigger,
    _counter_assumption_challenge_trigger,
    _default_call_timeout,
    _deterministic_math_tool_answer,
    _execute_math_tool_plan_text,
    _endpoint_error_pressure_abort_summary,
    _extract_multiple_choice_options,
    _expected_but_missing_modules,
    _filter_answer_bearing_evidence_results,
    _force_serial_child_execution_reason,
    _format_evidence_context,
    _has_two_vote_majority,
    _hipporag_style_rerank,
    _is_correct,
    _module_activation_summary,
    _model_router_subprocess_calls_enabled,
    _maybe_run_forced_alternative_challenge,
    _maybe_run_critic_synthesis_child,
    _maybe_add_route_arbitrator_candidate,
    _maybe_add_mc_option_sweep_candidates,
    _maybe_run_domain_rule_mc_verifier,
    _maybe_run_evidence_guided_option_challenge,
    _maybe_run_mc_option_evidence_scorer,
    _maybe_run_structural_option_audit_child,
    _maybe_run_option_elimination_challenge,
    _maybe_run_hipporag_preserve_selector_child,
    _maybe_run_raw_budget_preserve_selector_child,
    _maybe_run_raw_preserve_selector_child,
    _maybe_run_child_model_failover_child,
    _maybe_run_timeout_recovery_child,
    _math_tool_child_timeout,
    _maybe_add_answer_bearing_evidence_candidate,
    _maybe_mark_answer_bearing_evidence_attempt,
    _module_trace,
    _model_router_per_attempt_timeout,
    _model_router_extra_body,
    _needs_exact_answer_repair,
    _needs_evidence_grounded_child,
    _orthogonalize_child_prompt_specs,
    _parse_answer_json,
    _parse_verifier_choice,
    _prompt_for,
    _retrieval_summary_is_generic_harness_only,
    _recursive_child_prompt_specs,
    _recursive_timeout_recovery_trigger,
    _recursive_verifier_timeout,
    _child_model_failover_trigger,
    _call_recursive_verified_answer,
    _run_math_tool_attempt,
    _run_child_batch,
    _single_model_subprocess_call,
    _score_prediction,
    _select_recursive_child_answer,
    _should_run_math_tool_child,
    _should_run_candidate_claim_verifier,
    _should_use_agent_context,
    _should_prime_evidence_bridge,
    _same_run_cache_route_candidates,
    _route_arbitrator_should_lock,
    _route_arbitrator_lock_decision,
    _route_credit_table,
    _route_value_of_information_gate_summary,
    build_hle_text_smoke_eval_payload,
)


class HleSmokeEvalTest(unittest.TestCase):
    def test_parse_answer_json(self):
        self.assertEqual(_parse_answer_json('{"answer": "D"}'), "D")

    def test_extract_multiple_choice_options_supports_beyond_h(self):
        question = (
            "Which axiom is inconsistent?\n\n"
            "Answer Choices:\n"
            "A. Propositional extensionality\n"
            "H. Excluded middle\n"
            "I. Markov's principle"
        )
        options, first_start = _extract_multiple_choice_options(question)

        self.assertIsNotNone(first_start)
        self.assertEqual(options["A"], "Propositional extensionality")
        self.assertEqual(options["H"], "Excluded middle")
        self.assertEqual(options["I"], "Markov's principle")
        self.assertEqual(_parse_answer_json('```json\n{"answer": "Z+Z"}\n```'), "Z+Z")
        self.assertIsNone(_parse_answer_json("not json"))

    def test_multiple_choice_and_exact_scoring(self):
        self.assertTrue(_is_correct("D", "D", answer_type="multipleChoice"))
        self.assertTrue(_is_correct("The answer is D.", "D", answer_type="multipleChoice"))
        self.assertFalse(_is_correct("C", "D", answer_type="multipleChoice"))
        self.assertTrue(_is_correct(" Z + Z ", "Z + Z", answer_type="exactMatch"))
        self.assertFalse(_is_correct("Z+Z", "Z + Z", answer_type="exactMatch"))

    def test_score_prediction_does_not_persist_hle_content(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_hash": "aid",
            "category": "Math",
            "raw_subject": "Mathematics",
            "answer_type": "multipleChoice",
            "_question": "secret gated question",
            "_answer": "B",
        }
        row = _score_prediction(problem=problem, model="m", variant="raw", prediction='{"answer":"B"}')

        self.assertTrue(row["correct"])
        self.assertFalse(row["prediction_text_persisted"])
        self.assertFalse(row["raw_question_persisted"])
        self.assertFalse(row["gold_answer_persisted"])
        self.assertNotIn("_question", row)
        self.assertNotIn("_answer", row)

    def test_aggregate_rows(self):
        rows = [
            {"correct": True, "answer_type": "multipleChoice", "error": None},
            {"correct": False, "answer_type": "multipleChoice", "error": None},
            {"correct": True, "answer_type": "exactMatch", "error": None},
        ]
        summary = _aggregate_rows(rows)

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["accuracy"], 0.6667)
        self.assertEqual(summary["multiple_choice_accuracy"], 0.5)
        self.assertEqual(summary["exact_match_accuracy"], 1.0)

    def test_assumption_wrapper_trace_marks_true_agent_modules_skipped(self):
        problem = {
            "answer_type": "exactMatch",
            "category": "Chemistry",
            "raw_subject": "Chemistry",
        }
        trace = _module_trace(problem, variant="assumption_wrapper")
        by_module = {item["module"]: item for item in trace}

        self.assertEqual(by_module["prompt_scaffold"]["status"], "activated")
        self.assertEqual(by_module["assumption_graph_retrieval"]["status"], "skipped")
        self.assertEqual(by_module["structural_morphism_transfer"]["status"], "skipped")
        self.assertEqual(by_module["world_model_router"]["status"], "skipped")
        self.assertEqual(by_module["recursive_assumption_runner"]["status"], "skipped")

    def test_assumption_agent_trace_uses_real_stage_statuses(self):
        problem = {
            "answer_type": "exactMatch",
            "category": "Science",
            "raw_subject": "Physics",
        }
        plan = {
            "stages": {
                "domain_router": {"status": "activated"},
                "assumption_graph_retrieval": {"status": "activated"},
                "structural_morphism_transfer": {"status": "activated"},
                "world_model_router": {"status": "activated"},
                "recursive_assumption_runner": {"status": "activated"},
                "prompt_builder": {"status": "activated"},
            }
        }
        trace = _module_trace(problem, variant="assumption_agent", agent_plan=plan)
        by_module = {item["module"]: item for item in trace}

        self.assertEqual(by_module["assumption_graph_retrieval"]["status"], "activated")
        self.assertEqual(by_module["structural_morphism_transfer"]["status"], "activated")
        self.assertEqual(by_module["world_model_router"]["status"], "activated")
        self.assertEqual(by_module["recursive_assumption_runner"]["status"], "activated")
        self.assertEqual(by_module["multi_candidate_self_verifier"]["expected"], False)

    def test_recursive_verify_trace_uses_real_child_validation_stage(self):
        problem = {
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
        }
        plan = {
            "stages": {
                "domain_router": {"status": "activated"},
                "assumption_graph_retrieval": {"status": "activated"},
                "structural_morphism_transfer": {"status": "activated"},
                "world_model_router": {"status": "activated"},
                "recursive_assumption_runner": {"status": "activated"},
                "recursive_child_validation": {"status": "activated"},
                "multi_candidate_self_verifier": {"status": "activated"},
                "hle_evidence_bridge": {"status": "activated"},
                "hle_math_tool_solver": {"status": "activated"},
                "candidate_claim_verifier": {"status": "activated"},
                "prompt_builder": {"status": "activated"},
            }
        }
        trace = _module_trace(problem, variant="assumption_agent_recursive_verify", agent_plan=plan)
        by_module = {item["module"]: item for item in trace}

        self.assertTrue(by_module["recursive_child_validation"]["expected"])
        self.assertEqual(by_module["recursive_child_validation"]["status"], "activated")
        self.assertTrue(by_module["multi_candidate_self_verifier"]["expected"])
        self.assertEqual(by_module["multi_candidate_self_verifier"]["status"], "activated")
        self.assertEqual(by_module["hle_evidence_bridge"]["status"], "activated")
        self.assertFalse(by_module["hle_math_tool_solver"]["expected"])
        self.assertFalse(by_module["candidate_claim_verifier"]["expected"])

    def test_assumption_agent_prompt_abstain_matches_raw_prompt(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "secret gated question",
        }
        raw = _prompt_for(problem, variant="raw")
        agent = _prompt_for(problem, variant="assumption_agent", agent_plan={"prompt_context": ""})

        self.assertEqual(agent, raw)

    def test_recursive_verify_prompt_abstain_matches_raw_prompt(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "secret gated question",
        }
        raw = _prompt_for(problem, variant="raw")
        agent = _prompt_for(problem, variant="assumption_agent_recursive_verify", agent_plan={"prompt_context": ""})

        self.assertEqual(agent, raw)

    def test_generic_harness_retrieval_summary_is_detected(self):
        self.assertTrue(_retrieval_summary_is_generic_harness_only({"top_node_types": ["harness", "harness"]}))
        self.assertFalse(_retrieval_summary_is_generic_harness_only({"top_node_types": ["harness", "assumption"]}))
        self.assertFalse(_retrieval_summary_is_generic_harness_only({"top_node_types": []}))

    def test_recursive_child_prompt_specs_include_context_only_when_present(self):
        problem = {
            "answer_type": "exactMatch",
            "_question": "secret gated question",
        }

        plan = {}
        base_specs = _recursive_child_prompt_specs(problem, agent_plan=plan)
        self.assertEqual(len(base_specs), 3)
        self.assertEqual(len({spec["branch_axis"] for spec in base_specs}), len(base_specs))
        self.assertTrue(all(spec.get("orthogonal_branch_id") for spec in base_specs))
        self.assertTrue(all("recursive child branch" in spec["prompt"] for spec in base_specs))
        self.assertEqual(
            plan["stages"]["recursive_child_diversity_planner"]["unique_branch_axis_count"],
            3,
        )
        with patch.dict(os.environ, {"HLE_ENABLE_EXACT_TRAJECTORY_SEARCH": "1"}):
            trajectory_specs = _recursive_child_prompt_specs(problem, agent_plan={})
        self.assertEqual(
            [spec["prompt_kind"] for spec in trajectory_specs[3:6]],
            ["decomposition_answer", "adversarial_alternative_answer", "literal_constraint_answer"],
        )
        self.assertEqual(len({spec["branch_axis"] for spec in trajectory_specs}), len(trajectory_specs))
        specs = _recursive_child_prompt_specs(problem, agent_plan={"prompt_context": "graph context"})
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[-1]["prompt_kind"], "agent_context_answer")

        mc_specs = _recursive_child_prompt_specs(
            {"answer_type": "multipleChoice", "_question": "secret gated question"},
            agent_plan={"prompt_context": "graph context"},
        )
        self.assertEqual(mc_specs[1]["prompt_kind"], "agent_context_answer")
        mc_kinds = [spec["prompt_kind"] for spec in mc_specs]
        mc_axes = [spec["branch_axis"] for spec in mc_specs]
        self.assertIn("option_matrix_reasoner_answer", mc_kinds)
        self.assertIn("option_matrix_reasoning", mc_axes)

        evidence_specs = _recursive_child_prompt_specs(
            problem,
            agent_plan={"hle_evidence_context": "[Evidence 1] source=wikipedia; title=X; snippet=Y."},
        )
        self.assertEqual(evidence_specs[1]["prompt_kind"], "evidence_bridge_answer")

        mc_evidence_specs = _recursive_child_prompt_specs(
            {"answer_type": "multipleChoice", "_question": "secret gated question"},
            agent_plan={
                "prompt_context": "graph context",
                "hle_evidence_context": "[Evidence 1] source=wikipedia; title=X; snippet=Y.",
                "hipporag_prompt_context": "[Evidence 1] source=wikipedia; title=H; snippet=R.",
            },
        )
        self.assertEqual(
            [spec["prompt_kind"] for spec in mc_evidence_specs[:4]],
            ["direct_short_answer", "evidence_bridge_answer", "agent_context_answer", "hipporag_context_answer"],
        )

        exact_hipporag_specs = _recursive_child_prompt_specs(problem, agent_plan={})
        self.assertNotIn("hipporag_context_answer", [spec["prompt_kind"] for spec in exact_hipporag_specs])
        with patch.dict(os.environ, {"HLE_ENABLE_EXACT_AGENT_HIPPORAG_CHILD": "1"}):
            exact_hipporag_specs = _recursive_child_prompt_specs(
                problem,
                agent_plan={"hipporag_prompt_context": "[Evidence 1] source=wikipedia; title=H; snippet=R."},
            )
        self.assertIn("hipporag_context_answer", [spec["prompt_kind"] for spec in exact_hipporag_specs])

    def test_recursive_child_prompt_specs_include_code_semantics_for_compile_mc(self):
        problem = {
            "id_hash": "p",
            "question_hash": "q",
            "answer_type": "multipleChoice",
            "category": "Engineering",
            "raw_subject": "Computer Engineering",
            "_question": (
                "Will the following Rust code compile?\n"
                "```\nfn main() { println!(\"hi\"); }\n```\n"
                "A. It compiles\nB. It does not compile"
            ),
        }

        specs = _recursive_child_prompt_specs(problem, agent_plan={})
        kinds = [spec["prompt_kind"] for spec in specs]
        axes = [spec["branch_axis"] for spec in specs]

        self.assertIn("code_semantics_answer", kinds)
        self.assertIn("code_semantics", axes)
        self.assertLess(kinds.index("code_semantics_answer"), kinds.index("option_elimination_answer"))
        self.assertIn("option_matrix_reasoner_answer", kinds)

    def test_orthogonal_child_planner_deduplicates_same_axis_specs(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. one\nB. two",
        }
        plan = {}
        specs = _orthogonalize_child_prompt_specs(
            problem,
            [
                {"prompt_kind": "direct_short_answer", "prompt": "first"},
                {"prompt_kind": "direct_short_answer", "prompt": "duplicate"},
                {"prompt_kind": "constraint_checked_answer", "prompt": "constraint"},
            ],
            agent_plan=plan,
        )
        self.assertEqual([spec["prompt_kind"] for spec in specs], ["direct_short_answer", "constraint_checked_answer"])
        self.assertEqual([spec["branch_axis"] for spec in specs], ["closed_book_direct", "format_constraint"])
        summary = plan["stages"]["recursive_child_diversity_planner"]
        self.assertEqual(summary["duplicate_branch_axes_removed"], 1)
        self.assertEqual(summary["skipped_duplicate_branch_axes"][0]["branch_axis"], "closed_book_direct")

    def test_answer_bearing_evidence_filter_blocks_generic_context(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "Which scientist discovered the X process?\nA. Ada Lovelace\nB. Marie Curie",
        }
        selected, certificate = _filter_answer_bearing_evidence_results(
            problem=problem,
            results=[
                {"title": "Generic science", "snippet": "This article discusses laboratories and researchers.", "source": "wikipedia"},
                {"title": "Ada Lovelace", "snippet": "Ada Lovelace wrote notes on computing machines.", "source": "wikipedia"},
            ],
            candidate_answers=[],
            max_results=5,
        )

        self.assertEqual(selected, [])
        self.assertEqual(certificate["status"], "blocked_non_answer_bearing")
        self.assertEqual(certificate["selected_result_count"], 0)

    def test_answer_bearing_evidence_filter_accepts_option_or_candidate_overlap(self):
        mc_problem = {
            "answer_type": "multipleChoice",
            "_question": "Which scientist discovered the radium isolation process?\nA. Ada Lovelace\nB. Marie Curie",
        }
        selected, certificate = _filter_answer_bearing_evidence_results(
            problem=mc_problem,
            results=[
                {"title": "Marie Curie", "snippet": "Marie Curie discovered radium isolation process evidence.", "source": "wikipedia"},
            ],
            candidate_answers=[],
            max_results=5,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(certificate["status"], "answer_bearing")
        self.assertEqual(certificate["option_hit_labels"], ["B"])

        exact_problem = {
            "answer_type": "exactMatch",
            "_question": "Name the scientist associated with radium isolation process.",
        }
        exact_selected, exact_certificate = _filter_answer_bearing_evidence_results(
            problem=exact_problem,
            results=[
                {"title": "Marie Curie", "snippet": "Marie Curie is associated with the radium isolation process.", "source": "wikipedia"},
            ],
            candidate_answers=["Marie Curie"],
            max_results=5,
        )

        self.assertEqual(len(exact_selected), 1)
        self.assertEqual(exact_certificate["status"], "answer_bearing")
        self.assertEqual(exact_certificate["candidate_hit_count"], 1)
        self.assertEqual(len(exact_certificate["candidate_hit_answer_norm_hashes"]), 1)

    def test_candidate_specific_evidence_queries_and_verified_marking(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "raw_subject": "Computer Science",
            "category": "Computer Science/AI",
            "answer_type": "exactMatch",
            "_question": "Which person is named by the described compiler result?",
        }
        queries = _candidate_evidence_queries(problem, candidate_answers=["Ada Lovelace", "Grace Hopper"])

        self.assertIn("Ada Lovelace", queries)
        self.assertIn("Grace Hopper", queries)
        self.assertTrue(any("Ada Lovelace Computer Science" in query for query in queries))

        selected, certificate = _filter_answer_bearing_evidence_results(
            problem=problem,
            results=[
                {
                    "title": "Ada Lovelace",
                    "snippet": "Ada Lovelace is named by the described compiler result in computer science history.",
                    "source": "wikipedia",
                }
            ],
            candidate_answers=["Ada Lovelace", "Grace Hopper"],
            max_results=5,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(certificate["status"], "answer_bearing")
        self.assertEqual(certificate["candidate_hit_count"], 1)
        self.assertEqual(certificate["candidate_raw_hit_count"], 1)

        attempt = {
            "child_id": "e1",
            "child_index": 4,
            "prompt_kind": "evidence_grounded_answer",
            "parsed_answer": "Ada Lovelace",
            "status": "answered",
        }
        mark = _maybe_mark_answer_bearing_evidence_attempt(
            problem=problem,
            attempt=attempt,
            evidence_summary={"answer_bearing_certificate": certificate},
        )
        self.assertEqual(mark["status"], "marked_verified")
        self.assertEqual(attempt["candidate_verifier_state"], "verified")
        self.assertEqual(attempt["candidate_verifier_backend"], "answer_bearing_evidence_bridge")

        selection = _select_recursive_child_answer(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "Grace Hopper"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "Grace Hopper"},
                attempt,
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=None,
            max_tokens=32,
        )
        self.assertEqual(selection["selection_method"], "candidate_claim_verifier_priority")
        self.assertEqual(selection["selected_answer"], "Ada Lovelace")

        wrong_evidence_child = {
            "child_id": "e2",
            "child_index": 4,
            "prompt_kind": "evidence_grounded_answer",
            "parsed_answer": "Grace Hopper",
            "status": "answered",
        }
        rejected_mark = _maybe_mark_answer_bearing_evidence_attempt(
            problem=problem,
            attempt=wrong_evidence_child,
            evidence_summary={"answer_bearing_certificate": certificate},
        )
        self.assertEqual(rejected_mark["status"], "not_marked")
        supported_attempt, supported_summary = _maybe_add_answer_bearing_evidence_candidate(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "Ada Lovelace"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "Grace Hopper"},
                wrong_evidence_child,
            ],
            evidence_summary={"answer_bearing_certificate": certificate},
        )
        self.assertEqual(supported_summary["status"], "emitted")
        self.assertEqual(supported_attempt["prompt_kind"], "answer_bearing_evidence_candidate")
        self.assertEqual(supported_attempt["parsed_answer"], "Ada Lovelace")
        self.assertEqual(supported_attempt["candidate_verifier_state"], "verified")

        supported_selection = _select_recursive_child_answer(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "Ada Lovelace"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "Grace Hopper"},
                {"child_id": "c3", "child_index": 3, "prompt_kind": "decomposition_answer", "parsed_answer": "Grace Hopper"},
                supported_attempt,
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=None,
            max_tokens=32,
        )
        self.assertEqual(supported_selection["selection_method"], "candidate_claim_verifier_priority")
        self.assertEqual(supported_selection["selected_answer"], "Ada Lovelace")

    def test_candidate_specific_evidence_filter_relaxes_overlap_with_subject_anchor(self):
        problem = {
            "answer_type": "exactMatch",
            "raw_subject": "Computer Science",
            "category": "Computer Science/AI",
            "_question": "Which compiler pioneer is indicated by the described result?",
        }
        selected, certificate = _filter_answer_bearing_evidence_results(
            problem=problem,
            results=[
                {
                    "title": "Grace Hopper",
                    "snippet": "Grace Hopper was a computer scientist and programming language pioneer.",
                    "source": "wikipedia",
                }
            ],
            candidate_answers=["Grace Hopper"],
            max_results=5,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(certificate["status"], "answer_bearing")
        self.assertEqual(certificate["candidate_raw_hit_count"], 1)
        self.assertEqual(certificate["candidate_relaxed_overlap_count"], 1)
        self.assertEqual(certificate["candidate_hits_blocked_by_question_overlap"], 0)

    def test_answer_bearing_evidence_filter_blocks_non_discriminative_option_context(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "Which scientist discovered the radium isolation process?\nA. Ada Lovelace\nB. Marie Curie",
        }
        selected, certificate = _filter_answer_bearing_evidence_results(
            problem=problem,
            results=[
                {
                    "title": "Radium isolation history",
                    "snippet": "Ada Lovelace and Marie Curie are both mentioned while discussing discovered radium isolation process.",
                    "source": "wikipedia",
                },
            ],
            candidate_answers=[],
            max_results=5,
        )

        self.assertEqual(selected, [])
        self.assertEqual(certificate["status"], "blocked_non_discriminative_option_evidence")
        self.assertEqual(certificate["option_hit_labels"], ["A", "B"])

    def test_agent_hipporag_child_context_requires_answer_bearing_certificate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Science",
            "_question": "Which scientist discovered the radium isolation process?\nA. Ada Lovelace\nB. Marie Curie",
        }

        def fake_search(query, *, limit, timeout):
            return [
                {
                    "title": "Radium isolation history",
                    "snippet": "Ada Lovelace and Marie Curie are both mentioned while discussing discovered radium isolation process.",
                    "source": "wikipedia",
                }
            ]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            context, summary = _build_agent_hipporag_child_context(
                problem=problem,
                eval_id="e",
                call_id="c",
                model="m",
                logger=None,
                context_max_chars=1200,
            )

        self.assertEqual(context, "")
        self.assertEqual(summary["status"], "blocked_non_discriminative_option_evidence")
        self.assertEqual(summary["answer_bearing_certificate"]["option_hit_labels"], ["A", "B"])

    def test_parse_verifier_choice(self):
        self.assertEqual(_parse_verifier_choice('{"choice": 2}', max_index=3), 2)
        self.assertEqual(_parse_verifier_choice("choose 3", max_index=3), 3)
        self.assertIsNone(_parse_verifier_choice('{"choice": 4}', max_index=3))
        self.assertIsNone(_parse_verifier_choice("no choice", max_index=3))

    def test_two_vote_majority_detects_stable_child_answers(self):
        self.assertTrue(_has_two_vote_majority(
            [{"parsed_answer": "B"}, {"parsed_answer": "The answer is B."}],
            answer_type="multipleChoice",
        ))
        self.assertFalse(_has_two_vote_majority(
            [{"parsed_answer": "alpha"}, {"parsed_answer": "beta"}],
            answer_type="exactMatch",
        ))
        self.assertFalse(_has_two_vote_majority(
            [{"parsed_answer": "B"}, {"parsed_answer": "B"}],
            answer_type="exactMatch",
        ))

    def test_endpoint_error_pressure_abort_requires_error_storm_without_candidates(self):
        problem = {"answer_type": "multipleChoice", "_question": "Which option?\nA. one\nB. two"}
        errors = [
            {"prompt_kind": "direct_short_answer", "status": "error", "error_type": "RuntimeError", "parsed_answer": ""},
            {"prompt_kind": "constraint_checked_answer", "status": "error", "error_type": "RuntimeError", "parsed_answer": ""},
            {"prompt_kind": "recursive_assumption_answer", "status": "error", "error_type": "RuntimeError", "parsed_answer": ""},
        ]
        summary = _endpoint_error_pressure_abort_summary(problem=problem, attempts=errors)
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["valid_candidate_count"], 0)

        with_candidate = errors + [
            {"prompt_kind": "option_elimination_answer", "status": "answered", "parsed_answer": "A"},
            {"prompt_kind": "adversarial_alternative_answer", "status": "answered", "parsed_answer": "B"},
        ]
        summary = _endpoint_error_pressure_abort_summary(problem=problem, attempts=with_candidate)
        self.assertEqual(summary["status"], "not_required")

    def test_route_credit_table_separates_recoverable_selector_loss(self):
        rows = [
            {"model": "m", "variant": "raw", "problem_id_hash": "p1", "answer_type": "multipleChoice", "correct": False, "error": None},
            {"model": "m", "variant": "raw_budget_matched", "problem_id_hash": "p1", "answer_type": "multipleChoice", "correct": True, "error": None},
            {"model": "m", "variant": "hipporag_baseline", "problem_id_hash": "p1", "answer_type": "multipleChoice", "correct": False, "error": None},
            {"model": "m", "variant": "hipporag_budget_matched", "problem_id_hash": "p1", "answer_type": "multipleChoice", "correct": False, "error": None},
            {
                "model": "m",
                "variant": "assumption_agent_recursive_verify",
                "problem_id_hash": "p1",
                "answer_type": "multipleChoice",
                "correct": False,
                "error": None,
                "component_efficacy": {
                    "selection": {"selection_method": "route_value_verifier_choice"},
                    "route_arbitrator": {
                        "selected_route_type": "hipporag_preserve",
                        "selected_route_trust_reason": "answer_bearing_hipporag_route",
                        "value_of_information_gate": {
                            "status": "continue_exploration",
                            "recommended_action": "continue_exploration",
                        },
                    },
                },
            },
            {"model": "m", "variant": "raw", "problem_id_hash": "p2", "answer_type": "multipleChoice", "correct": False, "error": None},
            {"model": "m", "variant": "raw_budget_matched", "problem_id_hash": "p2", "answer_type": "multipleChoice", "correct": False, "error": None},
            {"model": "m", "variant": "hipporag_baseline", "problem_id_hash": "p2", "answer_type": "multipleChoice", "correct": False, "error": None},
            {"model": "m", "variant": "hipporag_budget_matched", "problem_id_hash": "p2", "answer_type": "multipleChoice", "correct": False, "error": None},
            {
                "model": "m",
                "variant": "assumption_agent_recursive_verify",
                "problem_id_hash": "p2",
                "answer_type": "multipleChoice",
                "correct": True,
                "error": None,
                "component_efficacy": {
                    "selection": {"selection_method": "candidate_claim_verifier_priority"},
                    "route_arbitrator": {"selected_route_type": "answer_bearing_evidence"},
                },
            },
        ]

        table = _route_credit_table(rows)
        model_table = table["by_model"]["m"]

        self.assertEqual(model_table["problem_count"], 2)
        self.assertEqual(model_table["recoverable_agent_error_count"], 1)
        self.assertEqual(model_table["unrecoverable_agent_error_count"], 0)
        self.assertEqual(model_table["agent_unique_correct_count"], 1)
        self.assertEqual(model_table["agent_loss_to_control_counts"], {"raw_budget_matched": 1})
        self.assertEqual(
            model_table["agent_selected_route_type_recoverable_loss_counts"],
            {"hipporag_preserve": 1},
        )

    def test_route_voi_gate_is_diagnostic_by_default_and_lockable_by_env(self):
        summary = {
            "status": "activated",
            "candidate_emitted": True,
            "selected_route_type": "raw_budget_consensus",
            "selected_route_child_id": "rawb",
            "selected_route_score": 12.0,
            "runner_up_score": 7.0,
            "selected_route_trusted": True,
            "selected_route_trust_reason": "strong_raw_budget_consensus",
            "raw_budget_strong_consensus": True,
            "raw_budget_top_vote_count": 4,
            "route_scores": [
                {
                    "route_type": "raw_budget_consensus",
                    "prompt_kind": "raw_budget_preserve_selector_answer",
                    "child_id": "rawb",
                    "score": 12.0,
                    "value_profile": {
                        "confidence": "high",
                        "reason_tags": ["raw_budget_strong_consensus"],
                        "risk_tags": [],
                    },
                    "baseline_cache_support_count": 2,
                    "answer_hash": "ha",
                    "normalized_answer_hash": "hna",
                },
                {
                    "route_type": "hipporag_preserve",
                    "prompt_kind": "hipporag_preserve_selector_answer",
                    "child_id": "hippo",
                    "score": 7.0,
                    "value_profile": {
                        "confidence": "medium",
                        "reason_tags": ["weak_option_linked_retrieval"],
                        "risk_tags": ["weak_option_linked_retrieval_without_budget"],
                    },
                    "context_answer_supported": False,
                    "baseline_cache_support_count": 1,
                    "answer_hash": "hb",
                    "normalized_answer_hash": "hnb",
                },
            ],
        }
        summary["value_of_information_gate"] = _route_value_of_information_gate_summary(summary)

        self.assertEqual(summary["value_of_information_gate"]["status"], "preserve_route")
        self.assertFalse(_route_arbitrator_lock_decision(summary))
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_VOI_HARD_GATE": "1"}, clear=False):
            summary["value_of_information_gate"] = _route_value_of_information_gate_summary(summary)
            self.assertTrue(_route_arbitrator_lock_decision(summary))
            self.assertTrue(summary["value_of_information_gate"]["hard_gate_applied"])

    def test_exact_early_stop_requires_independent_context_child(self):
        math_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Find the integer n.",
        }
        non_math_problem = {
            "answer_type": "exactMatch",
            "category": "History",
            "raw_subject": "History",
            "_question": "Who wrote the text?",
        }
        direct_constraint = [
            {"prompt_kind": "direct_short_answer", "parsed_answer": "42"},
            {"prompt_kind": "constraint_checked_answer", "parsed_answer": "42"},
        ]
        direct_evidence = direct_constraint + [
            {"prompt_kind": "evidence_bridge_answer", "parsed_answer": "42"},
        ]
        with_recursive = direct_constraint + [
            {"prompt_kind": "recursive_assumption_answer", "parsed_answer": "42"},
        ]
        under_diverse_two_vote = [
            {"prompt_kind": "direct_short_answer", "parsed_answer": "42"},
            {"prompt_kind": "recursive_assumption_answer", "parsed_answer": "42"},
        ]

        self.assertFalse(_can_stop_recursive_children_early(math_problem, direct_constraint))
        self.assertFalse(_can_stop_recursive_children_early(math_problem, under_diverse_two_vote))
        self.assertTrue(_can_stop_recursive_children_early(math_problem, direct_evidence))
        self.assertTrue(_can_stop_recursive_children_early(math_problem, with_recursive))
        self.assertFalse(_can_stop_recursive_children_early(non_math_problem, direct_constraint))
        self.assertTrue(_can_stop_recursive_children_early(non_math_problem, with_recursive))

        mc_problem = {
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?",
        }
        mc_direct_evidence = [
            {"prompt_kind": "direct_short_answer", "parsed_answer": "B"},
            {"prompt_kind": "evidence_bridge_answer", "parsed_answer": "B"},
        ]
        mc_with_context = mc_direct_evidence + [
            {"prompt_kind": "agent_context_answer", "parsed_answer": "B"},
        ]
        mc_core_axes = [
            {"prompt_kind": "direct_short_answer", "parsed_answer": "B"},
            {"prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
            {"prompt_kind": "recursive_assumption_answer", "parsed_answer": "B"},
            {"prompt_kind": "option_matrix_reasoner_answer", "parsed_answer": "B"},
            {"prompt_kind": "option_elimination_answer", "parsed_answer": "C"},
            {"prompt_kind": "adversarial_alternative_answer", "parsed_answer": "B"},
        ]
        self.assertFalse(_can_stop_recursive_children_early(mc_problem, mc_direct_evidence))
        self.assertFalse(_can_stop_recursive_children_early(mc_problem, mc_with_context))
        self.assertTrue(_can_stop_recursive_children_early(mc_problem, mc_core_axes))

    def test_parallel_child_batch_records_wall_clock_timeouts(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
        }
        specs = [
            {"prompt_kind": "slow_a", "prompt": "a"},
            {"prompt_kind": "slow_b", "prompt": "b"},
        ]

        def slow_child_attempt(**kwargs):
            time.sleep(0.2)
            return {
                "child_id": f"child-{kwargs['child_index']}",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "late",
                "parsed_answer_hash": "late",
                "prediction_hash": "late",
                "latency_sec": 0.2,
                "status": "answered",
            }

        started = time.monotonic()
        with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=slow_child_attempt):
            result = _run_child_batch(
                problem=problem,
                specs=specs,
                start_index=1,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=0.01,
                max_tokens=32,
                max_workers=2,
            )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual([row["status"] for row in result["attempts"]], ["timeout", "timeout"])
        self.assertEqual(result["underlying_model_calls"], 0)

    def test_parallel_child_execution_forces_serial_when_timeout_is_finite(self):
        self.assertEqual(
            _force_serial_child_execution_reason(mode="parallel_quorum", timeout=180),
            "finite_timeout_requires_main_thread_deadline",
        )
        self.assertEqual(_force_serial_child_execution_reason(mode="parallel_quorum", timeout=None), "")
        self.assertEqual(_force_serial_child_execution_reason(mode="serial", timeout=180), "")
        with patch.dict(os.environ, {"HLE_DISABLE_STRICT_SERIAL_CHILD_TIMEOUT": "1"}, clear=False):
            self.assertEqual(_force_serial_child_execution_reason(mode="parallel_quorum", timeout=180), "")

    def test_agent_child_model_env_override_is_optional(self):
        self.assertEqual(_agent_child_model("gpt-5.4-mini"), "gpt-5.4-mini")
        with patch.dict(os.environ, {"HLE_AGENT_CHILD_MODEL": "gpt-5.5"}, clear=False):
            self.assertEqual(_agent_child_model("gpt-5.4-mini"), "gpt-5.5")

    def test_recursive_answer_routes_child_generation_model(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute the value.",
        }
        plan = {"stages": {}}
        execute_result = {
            "attempts": [
                {
                    "child_id": "c1",
                    "child_index": 1,
                    "prompt_kind": "direct_short_answer",
                    "parsed_answer": "42",
                    "parsed_answer_hash": "h42",
                    "status": "answered",
                }
            ],
            "underlying_model_calls": 1,
            "early_stop_reason": None,
            "skipped_prompt_kinds": [],
            "execution_mode": "serial",
            "serial_forced_reason": None,
            "child_timeout_sec": 30,
            "child_max_workers": 1,
        }
        selection = {
            "status": "activated",
            "selection_method": "normalized_majority",
            "selected_child_id": "c1",
            "selected_answer": "42",
            "underlying_model_calls": 0,
        }
        with patch.dict(os.environ, {"HLE_AGENT_CHILD_MODEL": "gpt-5.5"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._execute_recursive_child_attempts", return_value=execute_result) as execute:
                with patch("assumption_os.hle_smoke_eval._select_recursive_child_answer", return_value=selection):
                    with patch("assumption_os.hle_smoke_eval._maybe_run_timeout_recovery_child", return_value=(None, None)):
                        with patch("assumption_os.hle_smoke_eval._should_run_math_tool_child", return_value=False):
                            with patch("assumption_os.hle_smoke_eval._should_run_candidate_claim_verifier", return_value=False):
                                with patch("assumption_os.hle_smoke_eval._maybe_run_mc_option_evidence_scorer", return_value=(None, None)):
                                    with patch("assumption_os.hle_smoke_eval._maybe_run_domain_rule_mc_verifier", return_value=(None, None)):
                                        with patch("assumption_os.hle_smoke_eval._maybe_run_counter_assumption_challenge", return_value=(None, None)):
                                            with patch("assumption_os.hle_smoke_eval._maybe_run_critic_synthesis_child", return_value=(None, None)):
                                                with patch("assumption_os.hle_smoke_eval._maybe_add_mc_option_sweep_candidates", return_value=([], None)):
                                                    result = _call_recursive_verified_answer(
                                                        problem=problem,
                                                        model="gpt-5.4-mini",
                                                        agent_plan=plan,
                                                        eval_id="e",
                                                        call_id="call",
                                                        logger=None,
                                                        timeout=60,
                                                        child_mode="serial",
                                                        child_timeout=30,
                                                        max_tokens=128,
                                                        evidence_bridge_enabled=False,
                                                    )

        self.assertEqual(json.loads(result["answer_text"])["answer"], "42")
        self.assertEqual(execute.call_args.kwargs["model"], "gpt-5.5")
        self.assertEqual(plan["stages"]["child_model_router"]["child_model"], "gpt-5.5")
        self.assertEqual(plan["stages"]["recursive_child_validation"]["child_model"], "gpt-5.5")

    def test_timeout_recovery_child_triggers_only_on_candidate_shortage(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute the value.",
        }
        timeout_attempts = [
            {"status": "timeout", "prompt_kind": "direct_short_answer", "parsed_answer": ""},
            {"status": "error", "prompt_kind": "constraint_checked_answer", "parsed_answer": ""},
        ]
        trigger = _recursive_timeout_recovery_trigger(problem=problem, attempts=timeout_attempts)
        self.assertEqual(trigger["status"], "activated")
        self.assertEqual(trigger["reason"], "timeout_or_error_with_candidate_shortage")

        enough_attempts = timeout_attempts + [
            {"status": "answered", "prompt_kind": "direct_short_answer", "parsed_answer": "42"},
            {"status": "answered", "prompt_kind": "constraint_checked_answer", "parsed_answer": "43"},
        ]
        abstained = _recursive_timeout_recovery_trigger(problem=problem, attempts=enough_attempts)
        self.assertEqual(abstained["status"], "abstained")
        self.assertEqual(abstained["reason"], "sufficient_candidate_diversity")

        verified_math = _recursive_timeout_recovery_trigger(
            problem=problem,
            attempts=timeout_attempts,
            math_tool_summary={"confidence": "verified_symbolic"},
        )
        self.assertEqual(verified_math["reason"], "math_tool_already_verified")

    def test_timeout_recovery_child_emits_logged_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Science",
            "raw_subject": "Chemistry",
            "_question": "Name the isolated active alkaloid.",
        }
        attempts = [
            {
                "child_id": "c1",
                "child_index": 1,
                "status": "timeout",
                "prompt_kind": "direct_short_answer",
                "parsed_answer": "",
            },
            {
                "child_id": "c2",
                "child_index": 2,
                "status": "error",
                "prompt_kind": "constraint_checked_answer",
                "parsed_answer": "",
            },
        ]
        with patch.dict(os.environ, {"HLE_TIMEOUT_RECOVERY_MODEL": "gpt-5.5"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"answer":"morphine"}') as call_model:
                attempt, summary = _maybe_run_timeout_recovery_child(
                    problem=problem,
                    attempts=attempts,
                    math_tool_summary=None,
                    model="gpt-5.4-mini",
                    eval_id="e",
                    call_id="call",
                    logger=None,
                    timeout=30,
                    max_tokens=512,
                )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["prompt_kind"], "timeout_recovery_answer")
        self.assertEqual(attempt["status"], "answered")
        self.assertEqual(attempt["parsed_answer"], "morphine")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["candidate_emitted"])
        self.assertEqual(summary["recovery_model"], "gpt-5.5")
        self.assertLessEqual(summary["recovery_max_tokens"], 160)
        self.assertEqual(call_model.call_args.kwargs["model"], "gpt-5.5")

    def test_child_model_failover_triggers_when_candidate_diversity_is_low(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Computer Science/AI",
            "raw_subject": "Computer Science",
            "_question": "Name the algorithm.",
        }
        failed_attempts = [
            {"status": "error", "prompt_kind": "direct_short_answer", "parsed_answer": ""},
            {"status": "timeout", "prompt_kind": "constraint_checked_answer", "parsed_answer": ""},
        ]
        trigger = _child_model_failover_trigger(
            problem=problem,
            attempts=failed_attempts,
            base_model="gpt-5.4-mini",
            child_model="gpt-5.5",
        )
        self.assertEqual(trigger["status"], "activated")
        self.assertEqual(trigger["reason"], "child_model_failed_without_valid_candidate")

        with_candidate = failed_attempts + [
            {"status": "answered", "prompt_kind": "direct_short_answer", "parsed_answer": "Dijkstra"},
        ]
        partial = _child_model_failover_trigger(
            problem=problem,
            attempts=with_candidate,
            base_model="gpt-5.4-mini",
            child_model="gpt-5.5",
        )
        self.assertEqual(partial["status"], "activated")
        self.assertEqual(partial["reason"], "child_model_failure_with_low_candidate_diversity")
        self.assertEqual(partial["unique_candidate_count"], 1)

        diverse_candidates = failed_attempts + [
            {"status": "answered", "prompt_kind": "direct_short_answer", "parsed_answer": "Dijkstra"},
            {"status": "answered", "prompt_kind": "constraint_checked_answer", "parsed_answer": "Bellman-Ford"},
            {"status": "answered", "prompt_kind": "recursive_assumption_answer", "parsed_answer": "A*"},
            {"status": "answered", "prompt_kind": "decomposition_answer", "parsed_answer": "Floyd-Warshall"},
        ]
        abstained = _child_model_failover_trigger(
            problem=problem,
            attempts=diverse_candidates,
            base_model="gpt-5.4-mini",
            child_model="gpt-5.5",
        )
        self.assertEqual(abstained["status"], "abstained")
        self.assertEqual(abstained["reason"], "valid_candidate_diversity_already_available")

    def test_child_model_failover_uses_base_model_for_one_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Computer Science/AI",
            "raw_subject": "Computer Science",
            "_question": "Name the algorithm.",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "status": "error", "prompt_kind": "direct_short_answer", "parsed_answer": ""},
            {"child_id": "c2", "child_index": 2, "status": "error", "prompt_kind": "constraint_checked_answer", "parsed_answer": ""},
        ]
        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"answer":"Dijkstra"}') as call_model:
            attempt, summary = _maybe_run_child_model_failover_child(
                problem=problem,
                attempts=attempts,
                base_model="gpt-5.4-mini",
                child_model="gpt-5.5",
                eval_id="e",
                call_id="call",
                logger=None,
                timeout=240,
                max_tokens=512,
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["prompt_kind"], "child_model_failover_answer")
        self.assertEqual(attempt["parsed_answer"], "Dijkstra")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["candidate_emitted"])
        self.assertEqual(summary["base_model"], "gpt-5.4-mini")
        self.assertEqual(summary["failed_child_model"], "gpt-5.5")
        self.assertEqual(summary["unique_candidate_count_before"], 0)
        self.assertEqual(call_model.call_args.kwargs["model"], "gpt-5.4-mini")

    def test_disable_recursive_runner_disables_child_verifier_path(self):
        sample = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_hash": "ahid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Chemistry",
            "_question": "Which option is correct?",
            "_answer": "A",
            "choices": ["A. alpha", "B. beta"],
            "scanned_index": 0,
        }
        plan = {
            "prompt_context": "bounded graph context",
            "stages": {
                "assumption_graph_retrieval": {"status": "activated"},
                "structural_morphism_transfer": {"status": "activated"},
                "world_model_router": {"status": "activated", "decision": "inject"},
            },
        }
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HLE_DISABLE_RECURSIVE_ASSUMPTION_RUNNER": "1"}, clear=False):
                with patch(
                    "assumption_os.hle_smoke_eval._access_preflight",
                    return_value={"dataset_accessible": True},
                ):
                    with patch("assumption_os.hle_smoke_eval._load_text_only_sample", return_value=[sample]):
                        with patch("assumption_os.hle_smoke_eval._build_assumption_agent_plan", return_value=plan):
                            with patch(
                                "assumption_os.hle_smoke_eval._call_recursive_verified_answer",
                                side_effect=AssertionError("recursive verifier should be disabled"),
                            ):
                                with patch(
                                    "assumption_os.hle_smoke_eval._call_model",
                                    return_value='{"answer":"A"}',
                                ) as call_model:
                                    payload = build_hle_text_smoke_eval_payload(
                                        root=Path(tmp),
                                        eval_id="unit_no_recursive",
                                        sample_size=1,
                                        execute_live=True,
                                        models=["gpt-5.4-mini"],
                                        variants=["assumption_agent_recursive_verify"],
                                    )

        self.assertEqual(call_model.call_count, 1)
        self.assertEqual(payload["api_summary"]["underlying_model_calls_executed"], 1)
        row = payload["rows"][0]
        self.assertTrue(row["correct"])
        by_module = {item["module"]: item for item in row["module_trace"]}
        self.assertEqual(by_module["recursive_child_validation"]["status"], "disabled")
        self.assertEqual(by_module["multi_candidate_self_verifier"]["status"], "disabled")
        efficacy = row["component_efficacy"]
        self.assertFalse(efficacy["flags"]["recursive_child_validation_activated"])
        self.assertEqual(efficacy["recursive"]["status"], "disabled")

    def test_raw_budget_matched_control_uses_multiple_calls_without_agent_modules(self):
        sample = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_hash": "ahid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. alpha\nB. beta\nC. gamma",
            "_answer": "A",
            "choices": ["A. alpha", "B. beta", "C. gamma"],
            "scanned_index": 0,
        }
        with TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "HLE_BUDGET_MATCHED_CANDIDATE_COUNT": "3",
                    "HLE_BUDGET_MATCHED_MAX_WORKERS": "2",
                },
                clear=False,
            ):
                with patch("assumption_os.hle_smoke_eval._access_preflight", return_value={"dataset_accessible": True}):
                    with patch("assumption_os.hle_smoke_eval._load_text_only_sample", return_value=[sample]):
                        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"answer":"A"}') as call_model:
                            payload = build_hle_text_smoke_eval_payload(
                                root=Path(tmp),
                                eval_id="unit_raw_budget_matched",
                                sample_size=1,
                                execute_live=True,
                                models=["gpt-5.5"],
                                variants=["raw_budget_matched"],
                            )

        self.assertEqual(call_model.call_count, 3)
        self.assertEqual(payload["api_summary"]["underlying_model_calls_executed"], 3)
        row = payload["rows"][0]
        self.assertTrue(row["correct"])
        self.assertEqual(row["variant"], "raw_budget_matched")
        by_module = {item["module"]: item for item in row["module_trace"]}
        self.assertEqual(by_module["budget_matched_self_consistency"]["status"], "activated")
        self.assertEqual(by_module["assumption_graph_retrieval"]["status"], "not_applicable")
        self.assertEqual(by_module["structural_morphism_transfer"]["status"], "not_applicable")
        self.assertEqual(by_module["world_model_router"]["status"], "not_applicable")
        self.assertEqual(by_module["recursive_assumption_runner"]["status"], "not_applicable")
        efficacy = row["component_efficacy"]
        self.assertEqual(efficacy["kind"], "budget_matched_control")
        self.assertEqual(efficacy["budget_matched_control"]["base_variant"], "raw")
        self.assertEqual(efficacy["budget_matched_control"]["candidate_count"], 3)
        self.assertTrue(efficacy["flags"]["budget_matched_control_activated"])

    def test_hipporag_budget_matched_control_keeps_retrieval_but_not_agent_modules(self):
        sample = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_hash": "ahid",
            "answer_type": "multipleChoice",
            "category": "Humanities/Social Science",
            "raw_subject": "History",
            "_question": "Which option is correct?\nA. alpha\nB. beta\nC. gamma",
            "_answer": "B",
            "choices": ["A. alpha", "B. beta", "C. gamma"],
            "scanned_index": 0,
        }
        hippo_plan = {
            "prompt_context": "[HippoRAG evidence] beta is explicitly supported.",
            "stages": {"prompt_builder": {"status": "activated", "context_injected": True}},
        }
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HLE_BUDGET_MATCHED_CANDIDATE_COUNT": "2"}, clear=False):
                with patch("assumption_os.hle_smoke_eval._access_preflight", return_value={"dataset_accessible": True}):
                    with patch("assumption_os.hle_smoke_eval._load_text_only_sample", return_value=[sample]):
                        with patch("assumption_os.hle_smoke_eval._build_hipporag_baseline_plan", return_value=hippo_plan):
                            with patch(
                                "assumption_os.hle_smoke_eval._call_model",
                                return_value='{"answer":"B"}',
                            ) as call_model:
                                payload = build_hle_text_smoke_eval_payload(
                                    root=Path(tmp),
                                    eval_id="unit_hippo_budget_matched",
                                    sample_size=1,
                                    execute_live=True,
                                    models=["gpt-5.5"],
                                    variants=["hipporag_budget_matched"],
                                )

        self.assertEqual(call_model.call_count, 2)
        row = payload["rows"][0]
        self.assertTrue(row["correct"])
        by_module = {item["module"]: item for item in row["module_trace"]}
        self.assertEqual(by_module["baseline_prompt_builder"]["status"], "activated")
        self.assertEqual(by_module["budget_matched_self_consistency"]["status"], "activated")
        self.assertEqual(by_module["assumption_graph_retrieval"]["status"], "not_applicable")
        self.assertEqual(by_module["recursive_assumption_runner"]["status"], "not_applicable")
        self.assertEqual(row["component_efficacy"]["budget_matched_control"]["base_variant"], "hipporag_baseline")

    def test_recursive_verifier_timeout_is_capped_separately_from_call_timeout(self):
        self.assertIsNone(_recursive_verifier_timeout(None))
        self.assertEqual(_recursive_verifier_timeout(7200), 7200.0)
        self.assertEqual(_recursive_verifier_timeout(45), 45.0)
        with patch.dict(os.environ, {"HLE_RECURSIVE_VERIFIER_TIMEOUT_SEC": "37"}, clear=False):
            self.assertEqual(_recursive_verifier_timeout(7200), 37.0)
        with patch.dict(os.environ, {"HLE_RECURSIVE_VERIFIER_TIMEOUT_SEC": "0"}, clear=False):
            self.assertIsNone(_recursive_verifier_timeout(7200))

    def test_model_router_timeout_defaults_are_unbounded(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_default_call_timeout())
            self.assertIsNone(_model_router_per_attempt_timeout())
            self.assertIsNone(_math_tool_child_timeout(None))
            self.assertIsNone(_answer_format_repair_timeout(None))
        with patch.dict(os.environ, {"MODEL_ROUTER_TIMEOUT": "240", "MODEL_ROUTER_PER_ATTEMPT_TIMEOUT": "30"}, clear=True):
            self.assertEqual(_default_call_timeout(), 240.0)
            self.assertEqual(_model_router_per_attempt_timeout(), 30.0)
        with patch.dict(os.environ, {"MODEL_ROUTER_TIMEOUT": "none", "MODEL_ROUTER_PER_ATTEMPT_TIMEOUT": "0"}, clear=True):
            self.assertIsNone(_default_call_timeout())
            self.assertIsNone(_model_router_per_attempt_timeout())
        with patch.dict(os.environ, {"HLE_ANSWER_FORMAT_REPAIR_TIMEOUT_SEC": "0"}, clear=True):
            self.assertIsNone(_answer_format_repair_timeout(120))
        with patch.dict(os.environ, {"HLE_ANSWER_FORMAT_REPAIR_TIMEOUT_SEC": "45"}, clear=True):
            self.assertEqual(_answer_format_repair_timeout(None), 45.0)

    def test_model_router_extra_body_is_env_only_and_protected(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_model_router_extra_body(), {})
        with patch.dict(
            os.environ,
            {
                "MODEL_ROUTER_REASONING_EFFORT": "low",
                "MODEL_ROUTER_EXTRA_BODY_JSON": '{"top_p":0.9,"model":"bad","messages":[]}',
            },
            clear=True,
        ):
            body = _model_router_extra_body()
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["top_p"], 0.9)
        self.assertNotIn("model", body)
        self.assertNotIn("messages", body)

    def test_model_subprocess_call_does_not_put_api_key_in_command_args(self):
        with patch.dict(os.environ, {"MODEL_ROUTER_SUBPROCESS_CALLS": "1"}, clear=False):
            self.assertTrue(_model_router_subprocess_calls_enabled())

        completed = type("Completed", (), {"returncode": 0, "stdout": "ok\n", "stderr": ""})()
        with patch("assumption_os.hle_smoke_eval.subprocess.run", return_value=completed) as run:
            text = _single_model_subprocess_call(
                env={"base_url": "https://example.test", "api_key": "secret-key", "model": "m"},
                payload={"model": "m", "messages": []},
                request_timeout=3,
            )

        self.assertEqual(text, "ok")
        args = run.call_args.args[0]
        self.assertNotIn("secret-key", " ".join(args))
        self.assertIn("secret-key", run.call_args.kwargs["input"])
        self.assertEqual(run.call_args.kwargs["timeout"], 8.0)

        with patch("assumption_os.hle_smoke_eval.subprocess.run", return_value=completed) as run_unbounded:
            text = _single_model_subprocess_call(
                env={"base_url": "https://example.test", "api_key": "secret-key", "model": "m"},
                payload={"model": "m", "messages": []},
                request_timeout=None,
            )
        self.assertEqual(text, "ok")
        self.assertIsNone(run_unbounded.call_args.kwargs["timeout"])
        self.assertIn('"request_timeout": null', run_unbounded.call_args.kwargs["input"])

    def test_exact_answer_repair_gate(self):
        exact_problem = {"answer_type": "exactMatch"}
        mc_problem = {"answer_type": "multipleChoice"}
        math_exact_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute the exact value.",
        }

        self.assertTrue(_needs_exact_answer_repair(exact_problem, "B"))
        self.assertTrue(_needs_exact_answer_repair(exact_problem, ""))
        self.assertFalse(_needs_exact_answer_repair(exact_problem, "Ada Lovelace"))
        self.assertFalse(_needs_exact_answer_repair(mc_problem, "B"))
        self.assertTrue(_needs_evidence_grounded_child(
            exact_problem,
            [{"parsed_answer": "B"}, {"parsed_answer": "B"}],
        ))
        self.assertFalse(_needs_evidence_grounded_child(
            exact_problem,
            [{"parsed_answer": "Ada Lovelace"}, {"parsed_answer": "B"}],
        ))
        self.assertFalse(_needs_evidence_grounded_child(
            math_exact_problem,
            [{"parsed_answer": "B"}, {"parsed_answer": "B"}],
        ))
        with patch.dict(os.environ, {"HLE_ENABLE_EXACT_DIVERSE_EVIDENCE_BRIDGE": "1"}):
            self.assertTrue(_needs_evidence_grounded_child(
                exact_problem,
                [{"parsed_answer": "Ada Lovelace"}, {"parsed_answer": "Marie Curie"}],
            ))
            self.assertFalse(_needs_evidence_grounded_child(
                math_exact_problem,
                [{"parsed_answer": "41"}, {"parsed_answer": "42"}],
            ))
        self.assertTrue(_should_prime_evidence_bridge(
            exact_problem,
            {"world_model_router": {"decision": "abstain_to_raw_prompt"}, "prompt_context": ""},
        ))
        self.assertTrue(_should_prime_evidence_bridge(
            mc_problem,
            {"world_model_router": {"decision": "abstain_to_raw_prompt"}, "prompt_context": ""},
        ))
        with patch.dict(os.environ, {"HLE_DISABLE_MC_EVIDENCE_BRIDGE": "1"}):
            self.assertFalse(_should_prime_evidence_bridge(
                mc_problem,
                {"world_model_router": {"decision": "abstain_to_raw_prompt"}, "prompt_context": ""},
            ))
        self.assertFalse(_should_prime_evidence_bridge(
            math_exact_problem,
            {"world_model_router": {"decision": "abstain_to_raw_prompt"}, "prompt_context": ""},
        ))

    def test_exact_selection_prefers_non_suspicious_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "_question": "Return the exact answer.",
        }
        selection = _select_recursive_child_answer(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct", "parsed_answer": "B"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "checked", "parsed_answer": "B"},
                {"child_id": "c3", "child_index": 3, "prompt_kind": "evidence", "parsed_answer": "Ada Lovelace"},
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(selection["selected_answer"], "Ada Lovelace")

        with patch.dict(os.environ, {"HLE_ENABLE_EXACT_EVIDENCE_OVERRIDE": "1"}):
            evidence_selection = _select_recursive_child_answer(
                problem=problem,
                attempts=[
                    {"child_id": "c1", "child_index": 1, "prompt_kind": "direct", "parsed_answer": "wrong theorem"},
                    {"child_id": "c2", "child_index": 2, "prompt_kind": "checked", "parsed_answer": "wrong theorem"},
                    {"child_id": "c3", "child_index": 3, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "right theorem"},
                ],
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(evidence_selection["selection_method"], "evidence_bridge_priority_over_closed_book_majority")
        self.assertEqual(evidence_selection["selected_answer"], "right theorem")

        direct_selection = _select_recursive_child_answer(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "direct answer"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "evidence answer"},
                {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "third answer"},
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )
        self.assertEqual(direct_selection["selection_method"], "exact_direct_fallback")
        self.assertEqual(direct_selection["selected_answer"], "direct answer")

        with patch.dict(os.environ, {"HLE_ENABLE_EXACT_DIVERSE_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice": 2}') as call_model:
                verifier_selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=[
                        {
                            "child_id": "c1",
                            "child_index": 1,
                            "prompt_kind": "direct_short_answer",
                            "parsed_answer": "direct answer",
                        },
                        {
                            "child_id": "c2",
                            "child_index": 2,
                            "prompt_kind": "constraint_checked_answer",
                            "parsed_answer": "checked answer",
                        },
                        {
                            "child_id": "c3",
                            "child_index": 3,
                            "prompt_kind": "recursive_assumption_answer",
                            "parsed_answer": "third answer",
                        },
                    ],
                    model="critic",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=None,
                    max_tokens=32,
                )

        self.assertEqual(verifier_selection["selection_method"], "verifier_choice")
        self.assertEqual(verifier_selection["selected_child_id"], "c2")
        self.assertEqual(verifier_selection["selected_answer"], "checked answer")
        self.assertTrue(verifier_selection["verifier_model_call"])
        call_model.assert_called_once()

    def test_verified_or_abstain_falls_back_to_direct_for_unverified_selection(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw\nB. context",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "context", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "B"},
        ]
        selection = {
            "selection_method": "normalized_majority",
            "selected_child_id": "context",
            "selected_answer": "B",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }

        gated = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)

        self.assertEqual(gated["selection_method"], "verified_or_abstain_direct_fallback")
        self.assertEqual(gated["selected_child_id"], "direct")
        self.assertEqual(gated["selected_answer"], "A")
        self.assertEqual(gated["verified_or_abstain_gate"]["status"], "abstained")
        self.assertEqual(gated["verified_or_abstain_gate"]["original_selection_method"], "normalized_majority")

        verifier_selection = {
            "selection_method": "verifier_choice",
            "selected_child_id": "context",
            "selected_answer": "B",
            "underlying_model_calls": 1,
            "verifier_model_call": True,
        }
        verifier_gated = _apply_verified_or_abstain_selection(
            problem=problem,
            attempts=attempts,
            selection=verifier_selection,
        )

        self.assertEqual(verifier_gated["selection_method"], "verifier_choice")
        self.assertEqual(verifier_gated["selected_child_id"], "context")
        self.assertEqual(verifier_gated["verified_or_abstain_gate"]["status"], "allowed")

    def test_raw_preserve_selector_candidate_and_fallback_priority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw\nB. context",
        }
        with patch.dict(os.environ, {"HLE_ENABLE_RAW_PRESERVE_SELECTOR": "1"}), patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"answer":"A"}',
        ) as call_model:
            attempt, summary = _maybe_run_raw_preserve_selector_child(
                problem=problem,
                attempts=[
                    {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "B"}
                ],
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNotNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(attempt["prompt_kind"], "raw_preserve_selector_answer")
        self.assertEqual(attempt["parsed_answer"], "A")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["underlying_model_calls"], 1)
        call_model.assert_called_once()

        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "B"},
            {"child_id": "rawp", "child_index": 2, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
        ]
        selection = {
            "selection_method": "normalized_majority",
            "selected_child_id": "direct",
            "selected_answer": "B",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
        with patch.dict(os.environ, {"HLE_ENABLE_RAW_PRESERVE_SELECTOR": "1"}):
            gated = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)

        self.assertEqual(gated["selection_method"], "verified_or_abstain_direct_fallback")
        self.assertEqual(gated["selected_child_id"], "rawp")
        self.assertEqual(gated["selected_answer"], "A")
        self.assertEqual(gated["verified_or_abstain_gate"]["fallback_prompt_kind"], "raw_preserve_selector_answer")

    def test_hipporag_preserve_selector_candidate_and_fallback_priority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Biology/Medicine",
            "raw_subject": "Ecology",
            "_question": "Which option is correct?\nA. direct\nB. retrieved",
        }
        baseline_plan = {
            "stages": {
                "hipporag_context_retrieval": {
                    "status": "activated",
                    "query_count": 2,
                    "candidate_doc_count": 3,
                },
                "hipporag_associative_rerank": {
                    "status": "activated",
                    "selected_doc_count": 2,
                },
                "prompt_builder": {
                    "status": "activated",
                    "context_char_count": 120,
                },
            },
            "prompt_context": "supporting context",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "recursive", "child_index": 2, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR": "1"}, clear=False), patch(
            "assumption_os.hle_smoke_eval._build_hipporag_baseline_plan",
            return_value=baseline_plan,
        ) as build_plan, patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"answer":"B"}',
        ) as call_model:
            attempt, summary = _maybe_run_hipporag_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan={"stages": {"world_model_router": {"generic_graph_context_only": True}}},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNotNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(attempt["prompt_kind"], "hipporag_preserve_selector_answer")
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["retrieval_status"], "activated")
        self.assertEqual(summary["candidate_doc_count"], 3)
        self.assertEqual(summary["underlying_model_calls"], 1)
        build_plan.assert_called_once()
        call_model.assert_called_once()

        selection = {
            "selection_method": "normalized_majority",
            "selected_child_id": "direct",
            "selected_answer": "A",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
        gated = _apply_verified_or_abstain_selection(
            problem=problem,
            attempts=[
                {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
                {"child_id": "rawp", "child_index": 2, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "C"},
                {
                    "child_id": "hipp",
                    "child_index": 3,
                    "prompt_kind": "hipporag_preserve_selector_answer",
                    "parsed_answer": "B",
                    "preserve_context_char_count": 120,
                    "preserve_selected_doc_count": 2,
                    "preserve_candidate_doc_count": 3,
                },
            ],
            selection=selection,
        )

        self.assertEqual(gated["selection_method"], "verified_or_abstain_direct_fallback")
        self.assertEqual(gated["selected_child_id"], "rawp")
        self.assertEqual(gated["selected_answer"], "C")
        self.assertEqual(
            gated["verified_or_abstain_gate"]["fallback_prompt_kind"],
            "raw_preserve_selector_answer",
        )

    def test_hipporag_preserve_selector_blocks_contextless_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. direct\nB. retrieved",
        }
        baseline_plan = {
            "stages": {
                "hipporag_context_retrieval": {
                    "status": "activated",
                    "query_count": 2,
                    "candidate_doc_count": 0,
                },
                "prompt_builder": {
                    "status": "activated",
                    "context_char_count": 0,
                },
            },
            "prompt_context": "",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "recursive", "child_index": 2, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR": "1"}, clear=False), patch(
            "assumption_os.hle_smoke_eval._build_hipporag_baseline_plan",
            return_value=baseline_plan,
        ), patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"answer":"B"}',
        ) as call_model:
            attempt, summary = _maybe_run_hipporag_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan={},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "blocked_non_answer_bearing")
        self.assertEqual(summary["block_reason"], "hipporag_preserve_requires_retrieved_context")
        self.assertEqual(summary["underlying_model_calls"], 0)
        call_model.assert_not_called()

        gated = _apply_verified_or_abstain_selection(
            problem=problem,
            attempts=[
                {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
                {"child_id": "rawp", "child_index": 2, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "C"},
                {
                    "child_id": "hipp",
                    "child_index": 3,
                    "prompt_kind": "hipporag_preserve_selector_answer",
                    "parsed_answer": "B",
                    "preserve_context_char_count": 0,
                    "preserve_selected_doc_count": 0,
                    "preserve_candidate_doc_count": 0,
                },
            ],
            selection={
                "selection_method": "normalized_majority",
                "selected_child_id": "direct",
                "selected_answer": "A",
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            },
        )

        self.assertEqual(gated["selected_child_id"], "rawp")
        self.assertEqual(gated["verified_or_abstain_gate"]["fallback_prompt_kind"], "raw_preserve_selector_answer")

    def test_preserve_selectors_use_same_run_baseline_cache_without_extra_calls(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. wrong\nB. raw\nC. hippo",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {
                    "variant": "raw",
                    "answer": "B",
                    "answer_hash": "raw-hash",
                },
                "raw_budget_matched": {
                    "variant": "raw_budget_matched",
                    "answer": "B",
                    "answer_hash": "raw-budget-hash",
                    "budget_candidate_count": 5,
                    "budget_answered_candidate_count": 5,
                    "budget_top_candidate_vote_count": 4,
                    "budget_strong_consensus": True,
                    "selection_method": "same_run_control",
                },
                "hipporag_budget_matched": {
                    "variant": "hipporag_budget_matched",
                    "answer": "C",
                    "answer_hash": "hippo-budget-hash",
                    "context_char_count": 800,
                    "candidate_doc_count": 4,
                    "selected_doc_count": 2,
                    "budget_candidate_count": 5,
                    "budget_answered_candidate_count": 5,
                    "selection_method": "same_run_control",
                },
            }
        }
        with patch.dict(
            os.environ,
            {
                "HLE_ENABLE_RAW_PRESERVE_SELECTOR": "1",
                "HLE_ENABLE_RAW_BUDGET_PRESERVE_SELECTOR": "1",
                "HLE_ENABLE_HIPPORAG_PRESERVE_SELECTOR": "1",
            },
            clear=False,
        ), patch("assumption_os.hle_smoke_eval._call_model") as call_model, patch(
            "assumption_os.hle_smoke_eval._build_hipporag_baseline_plan"
        ) as build_hippo:
            raw_attempt, raw_summary = _maybe_run_raw_preserve_selector_child(
                problem=problem,
                attempts=[],
                agent_plan=agent_plan,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )
            raw_budget_attempt, raw_budget_summary = _maybe_run_raw_budget_preserve_selector_child(
                problem=problem,
                attempts=[raw_attempt],
                agent_plan=agent_plan,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )
            hippo_attempt, hippo_summary = _maybe_run_hipporag_preserve_selector_child(
                problem=problem,
                attempts=[raw_attempt, raw_budget_attempt],
                agent_plan=agent_plan,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        call_model.assert_not_called()
        build_hippo.assert_not_called()
        self.assertEqual(raw_attempt["parsed_answer"], "B")
        self.assertEqual(raw_summary["policy"], "same_run_raw_baseline_cache_candidate")
        self.assertEqual(raw_summary["underlying_model_calls"], 0)
        self.assertEqual(raw_budget_attempt["parsed_answer"], "B")
        self.assertEqual(raw_budget_summary["policy"], "same_run_raw_budget_matched_cache_candidate")
        self.assertTrue(raw_budget_summary["strong_consensus"])
        self.assertEqual(raw_budget_summary["top_candidate_vote_count"], 4)
        self.assertEqual(hippo_attempt["parsed_answer"], "C")
        self.assertEqual(hippo_summary["policy"], "same_run_hipporag_cache_candidate")
        self.assertEqual(hippo_summary["context_char_count"], 800)
        self.assertEqual(hippo_summary["underlying_model_calls"], 0)

    def test_hipporag_preserve_selector_emits_contextless_same_run_cache_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw\nB. retrieved",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "hipporag_baseline": {
                    "variant": "hipporag_baseline",
                    "answer": "B",
                    "answer_hash": "hippo-hash",
                    "context_char_count": 0,
                    "candidate_doc_count": 0,
                    "selected_doc_count": 0,
                },
            }
        }
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_HIPPORAG_PRESERVE_SELECTOR": "1"},
            clear=False,
        ), patch("assumption_os.hle_smoke_eval._call_model") as call_model, patch(
            "assumption_os.hle_smoke_eval._build_hipporag_baseline_plan"
        ) as build_hippo:
            attempt, summary = _maybe_run_hipporag_preserve_selector_child(
                problem=problem,
                attempts=[],
                agent_plan=agent_plan,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        call_model.assert_not_called()
        build_hippo.assert_not_called()
        self.assertIsNotNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertFalse(attempt["same_run_cache_has_usable_context"])
        self.assertEqual(summary["policy"], "same_run_hipporag_cache_candidate_no_context")
        self.assertEqual(summary["retrieval_status"], "same_run_cache_no_context")
        self.assertEqual(summary["context_char_count"], 0)
        self.assertEqual(summary["underlying_model_calls"], 0)

    def test_hipporag_preserve_selector_can_use_budget_matched_candidates(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Biology/Medicine",
            "raw_subject": "Ecology",
            "_question": "Which option is correct?\nA. direct\nB. retrieved",
        }
        baseline_plan = {
            "stages": {
                "hipporag_context_retrieval": {
                    "status": "activated",
                    "query_count": 2,
                    "candidate_doc_count": 3,
                },
                "hipporag_associative_rerank": {
                    "status": "activated",
                    "selected_doc_count": 2,
                },
                "prompt_builder": {
                    "status": "activated",
                    "context_char_count": 120,
                },
            },
            "prompt_context": "supporting context",
        }
        child_attempts = [
            {
                "child_id": "hippo-budget-a",
                "child_index": 10,
                "prompt_kind": "direct_short_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hash-b",
                "status": "answered",
            },
            {
                "child_id": "hippo-budget-b",
                "child_index": 11,
                "prompt_kind": "constraint_checked_answer",
                "parsed_answer": "C",
                "parsed_answer_hash": "hash-c",
                "status": "answered",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_HIPPORAG_BUDGET_PRESERVE_SELECTOR": "1"}, clear=False), patch(
            "assumption_os.hle_smoke_eval._build_hipporag_baseline_plan",
            return_value=baseline_plan,
        ), patch(
            "assumption_os.hle_smoke_eval._run_child_batch",
            return_value={"attempts": child_attempts, "underlying_model_calls": 2, "max_workers": 2},
        ) as run_batch, patch(
            "assumption_os.hle_smoke_eval._select_recursive_child_answer",
            return_value={
                "selection_method": "verified_or_abstain_direct_fallback",
                "selected_child_id": "hippo-budget-a",
                "selected_answer": "B",
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            },
        ):
            attempt, summary = _maybe_run_hipporag_preserve_selector_child(
                problem=problem,
                attempts=[
                    {
                        "child_id": "direct",
                        "child_index": 1,
                        "prompt_kind": "direct_short_answer",
                        "parsed_answer": "A",
                    }
                ],
                agent_plan={},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["prompt_kind"], "hipporag_preserve_selector_answer")
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(attempt["branch_axis"], "hipporag_preserve_baseline")
        self.assertTrue(summary["budget_matched"])
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["answered_candidate_count"], 2)
        self.assertEqual(summary["underlying_model_calls"], 2)
        run_batch.assert_called_once()

    def test_cost_aware_hipporag_preserve_trigger_abstains_on_verified_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. a\nB. b",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR": "1"}, clear=False):
            trigger = _cost_aware_hipporag_preserve_trigger(problem=problem, attempts=attempts, agent_plan={})

        self.assertEqual(trigger["status"], "activated")
        self.assertEqual(trigger["reason"], "unverified_multiple_choice_baseline_preserve")

        verified = attempts + [{
            "child_id": "v",
            "child_index": 3,
            "prompt_kind": "candidate_claim_verifier_answer",
            "parsed_answer": "B",
            "candidate_verifier_state": "verified",
        }]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR": "1"}, clear=False):
            verified_trigger = _cost_aware_hipporag_preserve_trigger(problem=problem, attempts=verified, agent_plan={})

        self.assertEqual(verified_trigger["status"], "abstained")
        self.assertEqual(verified_trigger["reason"], "trusted_verified_candidate_available")

    def test_cost_aware_raw_preserve_trigger_is_narrow_and_unverified(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Humanities/Social Science",
            "raw_subject": "History",
            "_question": "Which interpretation is best?\nA. a\nB. b\nC. c\nD. d",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
            {"child_id": "c4", "child_index": 4, "prompt_kind": "critic_synthesis_answer", "parsed_answer": "D"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_RAW_PRESERVE_SELECTOR": "1"}, clear=False):
            trigger = _cost_aware_raw_preserve_trigger(problem=problem, attempts=attempts, agent_plan={})

        self.assertEqual(trigger["status"], "activated")
        self.assertEqual(trigger["reason"], "high_regression_domain_with_unverified_divergent_candidates")

        verified_attempts = attempts + [{
            "child_id": "v",
            "child_index": 5,
            "prompt_kind": "candidate_claim_verifier_answer",
            "parsed_answer": "B",
            "candidate_verifier_state": "verified",
        }]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_RAW_PRESERVE_SELECTOR": "1"}, clear=False):
            verified_trigger = _cost_aware_raw_preserve_trigger(
                problem=problem,
                attempts=verified_attempts,
                agent_plan={},
            )

        self.assertEqual(verified_trigger["status"], "abstained")
        self.assertEqual(verified_trigger["reason"], "trusted_verified_candidate_available")

    def test_cost_aware_raw_preserve_selector_runs_only_when_triggered(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Humanities/Social Science",
            "raw_subject": "History",
            "_question": "Which interpretation is best?\nA. a\nB. b\nC. c\nD. d",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
            {"child_id": "c4", "child_index": 4, "prompt_kind": "critic_synthesis_answer", "parsed_answer": "D"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_RAW_PRESERVE_SELECTOR": "1"}, clear=False), patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"answer":"A"}',
        ) as call_model:
            attempt, summary = _maybe_run_raw_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan={},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(summary["policy"], "cost_aware_regression_guard_no_context_candidate")
        self.assertEqual(summary["trigger"]["status"], "activated")
        call_model.assert_called_once()

        science_problem = dict(problem, category="Science", raw_subject="Physics")
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_RAW_PRESERVE_SELECTOR": "1"}, clear=False), patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"answer":"A"}',
        ) as no_call_model:
            skipped_attempt, skipped_summary = _maybe_run_raw_preserve_selector_child(
                problem=science_problem,
                attempts=attempts[:2],
                agent_plan={},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNone(skipped_attempt)
        self.assertIsNone(skipped_summary)
        no_call_model.assert_not_called()

    def test_cost_aware_raw_budget_preserve_trigger_catches_uncertain_unverified_routes(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which mechanism best explains this?\nA. a\nB. b\nC. c\nD. d",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
        ]
        agent_plan = {
            "stages": {
                "world_model_router": {"generic_graph_context_only": True},
                "structural_morphism_transfer": {
                    "structural_morphism_hits": [{"decision": "repair_level"}],
                    "formal_mapping_hits": [],
                },
            }
        }
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_RAW_BUDGET_PRESERVE_SELECTOR": "1"}, clear=False):
            trigger = _cost_aware_raw_budget_preserve_trigger(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
            )

        self.assertEqual(trigger["status"], "activated")
        self.assertEqual(trigger["reason"], "unverified_mc_route_uncertain_use_raw_budget_preserve")
        self.assertTrue(trigger["generic_graph_context_only"])
        self.assertTrue(trigger["weak_morphism_only"])

        verified_attempts = attempts + [{
            "child_id": "v",
            "child_index": 4,
            "prompt_kind": "candidate_claim_verifier_answer",
            "parsed_answer": "B",
            "candidate_verifier_state": "verified",
        }]
        with patch.dict(os.environ, {"HLE_ENABLE_COST_AWARE_RAW_BUDGET_PRESERVE_SELECTOR": "1"}, clear=False):
            verified_trigger = _cost_aware_raw_budget_preserve_trigger(
                problem=problem,
                attempts=verified_attempts,
                agent_plan=agent_plan,
            )

        self.assertEqual(verified_trigger["status"], "abstained")
        self.assertEqual(verified_trigger["reason"], "trusted_verified_candidate_available")

    def test_raw_budget_preserve_selector_runs_budget_matched_candidates(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which mechanism best explains this?\nA. a\nB. b\nC. c\nD. d",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
        ]
        child_attempts = [
            {
                "child_id": "rb1",
                "child_index": 4,
                "prompt_kind": "direct_short_answer",
                "parsed_answer": "C",
                "parsed_answer_hash": "hC",
                "status": "answered",
            },
            {
                "child_id": "rb2",
                "child_index": 5,
                "prompt_kind": "constraint_checked_answer",
                "parsed_answer": "C",
                "parsed_answer_hash": "hC",
                "status": "answered",
            },
            {
                "child_id": "rb3",
                "child_index": 6,
                "prompt_kind": "skeptical_recheck_answer",
                "parsed_answer": "C",
                "parsed_answer_hash": "hC",
                "status": "answered",
            },
            {
                "child_id": "rb4",
                "child_index": 7,
                "prompt_kind": "literal_constraint_answer",
                "parsed_answer": "D",
                "parsed_answer_hash": "hD",
                "status": "answered",
            },
            {
                "child_id": "rb5",
                "child_index": 8,
                "prompt_kind": "option_elimination_baseline_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hB",
                "status": "answered",
            },
        ]
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_COST_AWARE_RAW_BUDGET_PRESERVE_SELECTOR": "1"},
            clear=False,
        ), patch(
            "assumption_os.hle_smoke_eval._run_child_batch",
            return_value={"attempts": child_attempts, "underlying_model_calls": 5, "max_workers": 5},
        ) as run_batch, patch(
            "assumption_os.hle_smoke_eval._select_recursive_child_answer",
            return_value={
                "selection_method": "normalized_majority",
                "selected_child_id": "rb4",
                "selected_answer": "D",
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            },
        ) as select_answer:
            attempt, summary = _maybe_run_raw_budget_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan={"stages": {"world_model_router": {"generic_graph_context_only": True}}},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNotNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(attempt["prompt_kind"], "raw_budget_preserve_selector_answer")
        self.assertEqual(attempt["parsed_answer"], "C")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["candidate_count"], 5)
        self.assertEqual(summary["answered_candidate_count"], 5)
        self.assertEqual(summary["top_candidate_vote_count"], 3)
        self.assertEqual(summary["selection_method"], "raw_budget_top_vote_consensus")
        self.assertEqual(summary["selected_child_id"], "rb1")
        self.assertEqual(summary["underlying_model_calls"], 5)
        run_batch.assert_called_once()
        select_answer.assert_called_once()

    def test_raw_budget_preserve_selector_blocks_weak_consensus_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which mechanism best explains this?\nA. a\nB. b\nC. c\nD. d",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "B"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "C"},
        ]
        child_attempts = [
            {
                "child_id": f"rb{idx}",
                "child_index": idx + 3,
                "prompt_kind": "direct_short_answer",
                "parsed_answer": answer,
                "parsed_answer_hash": f"h{answer}",
                "status": "answered",
            }
            for idx, answer in enumerate(["A", "B", "C", "D", "E"], start=1)
        ]
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_COST_AWARE_RAW_BUDGET_PRESERVE_SELECTOR": "1"},
            clear=False,
        ), patch(
            "assumption_os.hle_smoke_eval._run_child_batch",
            return_value={"attempts": child_attempts, "underlying_model_calls": 5, "max_workers": 5},
        ), patch(
            "assumption_os.hle_smoke_eval._select_recursive_child_answer",
            return_value={
                "selection_method": "normalized_majority",
                "selected_child_id": "rb1",
                "selected_answer": "A",
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            },
        ):
            attempt, summary = _maybe_run_raw_budget_preserve_selector_child(
                problem=problem,
                attempts=attempts,
                agent_plan={"stages": {"world_model_router": {"generic_graph_context_only": True}}},
                model="base-model",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=32,
            )

        self.assertIsNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["status"], "blocked_weak_consensus")
        self.assertEqual(summary["top_candidate_vote_count"], 1)
        self.assertFalse(summary["candidate_emitted"])
        self.assertEqual(summary["underlying_model_calls"], 5)

    def test_raw_budget_preserve_selector_has_fallback_priority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw\nB. hippo\nC. budget",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "rawp", "child_index": 2, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 120,
                "preserve_selected_doc_count": 1,
                "preserve_candidate_doc_count": 1,
            },
            {
                "child_id": "rawbp",
                "child_index": 4,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
        ]
        selection = {
            "selection_method": "normalized_majority",
            "selected_child_id": "direct",
            "selected_answer": "A",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
        gated = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)

        self.assertEqual(gated["selection_method"], "verified_or_abstain_direct_fallback")
        self.assertEqual(gated["selected_child_id"], "rawbp")
        self.assertEqual(gated["selected_answer"], "C")
        self.assertEqual(
            gated["verified_or_abstain_gate"]["fallback_prompt_kind"],
            "raw_budget_preserve_selector_answer",
        )

    def test_route_arbitrator_prefers_contextual_hipporag_when_supported_and_raw_budget_is_weak(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. direct\nB. retrieved\nC. raw budget",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "B"},
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 900,
                "preserve_selected_doc_count": 2,
                "preserve_candidate_doc_count": 4,
                "context_answer_supported": True,
                "context_answer_overlap_count": 2,
                "context_question_overlap_count": 2,
                "same_route_agreement_count": 2,
            },
        ]
        raw_budget_summary = {
            "status": "activated",
            "strong_consensus": False,
            "top_candidate_vote_count": 1,
        }
        hipporag_summary = {
            "status": "activated",
            "context_char_count": 900,
            "selected_doc_count": 2,
            "candidate_doc_count": 4,
        }
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary=raw_budget_summary,
                hipporag_preserve_summary=hipporag_summary,
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertIsNotNone(summary)
        self.assertEqual(attempt["prompt_kind"], "route_arbitrator_answer")
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertEqual(summary["route_count"], 3)
        self.assertEqual(summary["unique_answer_count"], 2)

        gated = _apply_verified_or_abstain_selection(
            problem=problem,
            attempts=attempts + [attempt],
            selection={
                "selection_method": "normalized_majority",
                "selected_child_id": "direct",
                "selected_answer": "A",
                "underlying_model_calls": 0,
                "verifier_model_call": False,
            },
        )
        self.assertEqual(gated["selected_child_id"], attempt["child_id"])
        self.assertEqual(gated["selected_answer"], "B")
        self.assertEqual(gated["verified_or_abstain_gate"]["fallback_prompt_kind"], "route_arbitrator_answer")

    def test_route_arbitrator_keeps_strong_raw_budget_consensus(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. direct\nB. retrieved\nC. raw budget",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 500,
                "preserve_selected_doc_count": 1,
                "preserve_candidate_doc_count": 2,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={
                    "status": "activated",
                    "context_char_count": 500,
                    "selected_doc_count": 1,
                    "candidate_doc_count": 2,
                },
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "C")
        self.assertEqual(summary["selected_route_type"], "raw_budget_consensus")
        self.assertTrue(summary["raw_budget_strong_consensus"])
        self.assertFalse(_route_arbitrator_should_lock(summary))
        with patch.dict(os.environ, {"HLE_ENABLE_RAW_BUDGET_CACHE_FIRST_LOCK": "1"}, clear=False):
            self.assertTrue(_route_arbitrator_should_lock(summary))

    def test_route_arbitrator_penalizes_unsupported_hipporag_context(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. direct\nB. retrieved\nC. raw budget",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 2200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 8,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": False,
                    "top_candidate_vote_count": 1,
                },
                hipporag_preserve_summary={
                    "status": "activated",
                    "context_char_count": 2200,
                    "selected_doc_count": 5,
                    "candidate_doc_count": 8,
                },
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertNotEqual(summary["selected_route_type"], "hipporag_preserve")

    def test_route_arbitrator_retains_contextless_hipporag_cache_as_low_trust_route(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw budget\nB. retrieved",
        }
        attempts = [
            {
                "child_id": "rawbp",
                "child_index": 1,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
            },
            {
                "child_id": "hipp",
                "child_index": 2,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "preserve_context_char_count": 0,
                "preserve_selected_doc_count": 0,
                "preserve_candidate_doc_count": 0,
            },
        ]
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "B"},
            }
        }
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={
                    "status": "activated",
                    "policy": "same_run_hipporag_cache_candidate_no_context",
                    "context_char_count": 0,
                    "selected_doc_count": 0,
                    "candidate_doc_count": 0,
                },
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(summary["hipporag_context_route_count"], 1)
        hippo_rows = [
            row for row in summary["route_scores"]
            if row["prompt_kind"] == "hipporag_preserve_selector_answer"
        ]
        self.assertEqual(len(hippo_rows), 1)
        self.assertIn("retrieval_not_answer_bearing", hippo_rows[0]["value_profile"]["risk_tags"])
        if summary["selected_route_type"] == "hipporag_preserve":
            self.assertFalse(summary["selected_route_trusted"])

    def test_route_arbitrator_keeps_strong_raw_budget_over_unsupported_hipporag_family(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw family\nB. unsupported hippo\nC. raw budget",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "C"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "B"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 4,
                "preserve_candidate_doc_count": 8,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 2,
                "same_route_agreeing_variants": ["hipporag_baseline", "hipporag_budget_matched"],
                "context_answer_supported": False,
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 4,
                "preserve_candidate_doc_count": 8,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 2,
                "same_route_agreeing_variants": ["hipporag_baseline", "hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_answer_supported": False,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1200},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "C")
        self.assertEqual(summary["selected_route_type"], "raw_budget_consensus")

    def test_route_arbitrator_blocks_raw_budget_when_raw_and_hipporag_base_pair_disagree(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. base pair\nB. budget",
        }
        attempts = [
            {
                "child_id": "rawp",
                "child_index": 1,
                "prompt_kind": "raw_preserve_selector_answer",
                "parsed_answer": "A",
            },
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "B",
            },
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 2,
                "preserve_candidate_doc_count": 4,
                "context_answer_supported": True,
                "context_answer_overlap_count": 1,
                "context_question_overlap_count": 4,
            },
        ]
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "B"},
            }
        }
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={
                    "status": "activated",
                    "context_char_count": 1200,
                    "selected_doc_count": 2,
                    "candidate_doc_count": 4,
                    "context_answer_supported": True,
                    "context_answer_overlap_count": 1,
                    "context_question_overlap_count": 4,
                },
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        raw_budget_rows = [
            row for row in summary["route_scores"]
            if row["route_type"] == "raw_budget_consensus"
        ]
        self.assertEqual(len(raw_budget_rows), 1)
        self.assertIn(
            "conflicts_with_raw_hipporag_base_pair",
            raw_budget_rows[0]["value_profile"]["risk_tags"],
        )
        if summary["selected_route_type"] == "raw_budget_consensus":
            self.assertFalse(summary["selected_route_trusted"])
            self.assertEqual(
                summary["selected_route_trust_reason"],
                "raw_budget_conflicts_with_raw_hipporag_base_pair",
            )

    def test_route_arbitrator_uses_hipporag_same_route_consensus(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. direct\nB. retrieved\nC. raw budget",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 7,
                "same_route_agreement_count": 2,
                "context_answer_supported": True,
                "context_answer_overlap_count": 3,
                "context_question_overlap_count": 2,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={
                    "status": "activated",
                    "context_char_count": 1200,
                    "selected_doc_count": 5,
                    "candidate_doc_count": 7,
                },
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")

    def test_route_arbitrator_uses_cache_multi_arm_support_for_hipporag_budget(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. direct\nB. retrieved budget\nC. raw budget\nD. hippo baseline",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "C"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "D"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "rawp", "child_index": 2, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawbp",
                "child_index": 3,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "C",
            },
            {
                "child_id": "hipp",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1600,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 11,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "budget_matched": True,
                "context_answer_supported": True,
                "context_answer_overlap_count": 3,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1600},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertEqual(summary["baseline_cache_unique_answer_count"], 4)

    def test_cache_first_route_candidates_keep_distinct_hipporag_arms(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": (
                "Which option is best?\n"
                "A. raw choice\n"
                "B. raw budget choice\n"
                "C. hippo base choice\n"
                "D. answer bearing hippo budget choice"
            ),
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A", "answer_hash": "ha"},
                "raw_budget_matched": {
                    "variant": "raw_budget_matched",
                    "answer": "B",
                    "answer_hash": "hb",
                    "budget_candidate_count": 5,
                    "budget_answered_candidate_count": 5,
                    "budget_top_candidate_vote_count": 2,
                    "budget_strong_consensus": False,
                },
                "hipporag_baseline": {
                    "variant": "hipporag_baseline",
                    "answer": "C",
                    "answer_hash": "hc",
                    "context_char_count": 900,
                    "candidate_doc_count": 4,
                    "selected_doc_count": 2,
                    "context_answer_supported": False,
                },
                "hipporag_budget_matched": {
                    "variant": "hipporag_budget_matched",
                    "answer": "D",
                    "answer_hash": "hd",
                    "context_char_count": 1200,
                    "candidate_doc_count": 5,
                    "selected_doc_count": 3,
                    "context_answer_supported": True,
                    "context_answer_overlap_count": 4,
                    "context_question_overlap_count": 3,
                    "budget_candidate_count": 5,
                    "budget_answered_candidate_count": 5,
                    "budget_top_candidate_vote_count": 4,
                    "budget_strong_consensus": True,
                },
            }
        }
        attempts, raw_summary, raw_budget_summary, hippo_summary = _same_run_cache_route_candidates(
            problem=problem,
            agent_plan=agent_plan,
            call_id="c",
            start_index=100,
        )
        self.assertEqual(len(attempts), 4)
        self.assertIsNotNone(raw_summary)
        self.assertIsNotNone(raw_budget_summary)
        self.assertIsNotNone(hippo_summary)
        hippo_variants = {
            attempt.get("same_run_baseline_cache_variant")
            for attempt in attempts
            if attempt.get("prompt_kind") == "hipporag_preserve_selector_answer"
        }
        self.assertEqual(hippo_variants, {"hipporag_baseline", "hipporag_budget_matched"})
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            route_attempt, route_summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary=raw_budget_summary,
                hipporag_preserve_summary=hippo_summary,
                call_id="c",
            )
        self.assertIsNotNone(route_attempt)
        self.assertEqual(route_attempt["parsed_answer"], "D")
        self.assertEqual(route_summary["selected_route_type"], "hipporag_preserve")
        self.assertTrue(route_summary["selected_route_trusted"])
        self.assertEqual(route_attempt["candidate_verifier_state"], "verified")
        self.assertEqual(route_summary["route_count"], 4)
        self.assertTrue(_route_arbitrator_should_lock(route_summary))

    def test_cache_first_route_keeps_hipporag_budget_arm_without_context_stats(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw\nB. hippo budget\nC. raw budget",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A", "answer_hash": "ha"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "C", "answer_hash": "hc"},
                "hipporag_baseline": {
                    "variant": "hipporag_baseline",
                    "answer": "A",
                    "answer_hash": "ha",
                    "context_char_count": 900,
                    "candidate_doc_count": 4,
                    "selected_doc_count": 2,
                },
                "hipporag_budget_matched": {
                    "variant": "hipporag_budget_matched",
                    "answer": "B",
                    "answer_hash": "hb",
                    "budget_candidate_count": 5,
                    "budget_answered_candidate_count": 5,
                    "budget_top_candidate_vote_count": 3,
                    "budget_strong_consensus": True,
                },
            }
        }
        attempts, _, _, hippo_summary = _same_run_cache_route_candidates(
            problem=problem,
            agent_plan=agent_plan,
            call_id="c",
            start_index=100,
        )
        hippo_variants = {
            attempt.get("same_run_baseline_cache_variant")
            for attempt in attempts
            if attempt.get("prompt_kind") == "hipporag_preserve_selector_answer"
        }
        self.assertIn("hipporag_budget_matched", hippo_variants)
        self.assertIn("hipporag_budget_matched", hippo_summary["candidate_variants"])

    def test_route_arbitrator_keeps_raw_budget_when_cache_supports_raw_family(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw family\nB. retrieved\nC. hippo baseline",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "C"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
            },
            {
                "child_id": "hipp",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1600,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 11,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "budget_matched": True,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1600},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "A")
        self.assertEqual(summary["selected_route_type"], "raw_budget_consensus")

    def test_route_arbitrator_abstains_without_route_conflict(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. same\nB. other",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {
                "child_id": "rawbp",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary={"status": "activated", "strong_consensus": True},
                call_id="c",
            )

        self.assertIsNone(attempt)
        self.assertEqual(summary["status"], "abstained")
        self.assertEqual(summary["reason"], "routes_agree")

    def test_route_value_verifier_promotes_strong_budgeted_retrieval_counter(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": (
                "Which option is best?\n"
                "A. raw family\nB. budgeted retrieved counter\nC. other"
            ),
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
            },
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 9,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "ha",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1400,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_question_overlap_count": 8,
                "context_answer_supported": True,
                "context_answer_overlap_count": 1,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2600},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertTrue(summary["selected_route_trusted"])
        self.assertEqual(summary["selected_route_trust_reason"], "high_value_budgeted_retrieval_route")
        self.assertGreater(summary["retrieval_budget_counter_norm_count"], 0)
        self.assertEqual(attempt["route_value_confidence"], "high")

    def test_route_value_promotes_option_linked_budgeted_retrieval_counter(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw family\nB. budget retrieval counter\nC. other",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 9,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "ha",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1500,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2700},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertTrue(summary["selected_route_trusted"])
        self.assertEqual(summary["selected_route_trust_reason"], "retrieval_budget_counter_to_raw_budget")
        self.assertIn(
            "retrieval_budget_counter_signal",
            summary["selected_route_value_profile"]["components"],
        )

    def test_route_voi_keeps_budgeted_retrieval_disagreement_as_counter_not_promotion(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw family\nB. budgeted retrieval counter",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreeing_variants": ["hipporag_baseline"],
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1100,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 11,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 3,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1100},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(summary["selected_route_type"], "raw_budget_consensus")
        self.assertEqual(summary["retrieval_budget_counter_norm_count"], 0)
        self.assertEqual(summary["value_of_information_gate"]["status"], "continue_exploration")
        self.assertEqual(
            summary["value_of_information_gate"]["credible_counter_routes"][0]["credible_counter_kind"],
            "budgeted_retrieval_disagreement",
        )

    def test_route_value_verifier_can_be_disabled_for_ablation(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw family\nB. retrieved counter",
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo-budget",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1400,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_ROUTE_ARBITRATOR": "1", "HLE_DISABLE_ROUTE_VALUE_VERIFIER": "1"},
            clear=False,
        ):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1400},
                call_id="c",
            )

        self.assertIsNone(attempt)
        self.assertEqual(summary["reason"], "route_value_verifier_disabled")
        self.assertFalse(summary["route_value_verifier_enabled"])

    def test_route_value_trusts_answer_bearing_independent_hipporag_counter(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option follows from the retrieved physics evidence?\nA. raw majority\nB. retrieved counter",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {
                    "variant": "raw_budget_matched",
                    "answer": "A",
                    "budget_top_candidate_vote_count": 5,
                    "budget_strong_consensus": True,
                },
                "hipporag_baseline": {
                    "variant": "hipporag_baseline",
                    "answer": "B",
                    "context_char_count": 1080,
                    "selected_doc_count": 5,
                    "candidate_doc_count": 10,
                    "context_answer_supported": True,
                    "context_answer_overlap_count": 3,
                    "context_question_overlap_count": 5,
                    "context_answer_option_hash": "hb",
                },
                "hipporag_budget_matched": {
                    "variant": "hipporag_budget_matched",
                    "answer": "A",
                    "budget_top_candidate_vote_count": 2,
                    "budget_strong_consensus": False,
                    "context_char_count": 1059,
                    "selected_doc_count": 5,
                    "candidate_doc_count": 5,
                    "context_answer_supported": True,
                    "context_answer_overlap_count": 3,
                    "context_question_overlap_count": 3,
                    "context_answer_option_hash": "ha",
                },
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 5,
            },
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1080,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_answer_supported": True,
                "context_answer_overlap_count": 3,
                "context_question_overlap_count": 5,
                "context_answer_option_hash": "hb",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1059,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 5,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": False,
                "budget_top_candidate_vote_count": 2,
                "context_answer_supported": True,
                "context_answer_overlap_count": 3,
                "context_question_overlap_count": 3,
                "context_answer_option_hash": "ha",
            },
            {"child_id": "direct", "child_index": 5, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2139},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertTrue(summary["selected_route_trusted"])
        self.assertEqual(
            summary["selected_route_trust_reason"],
            "answer_bearing_hipporag_counter_to_budget_echo",
        )
        self.assertEqual(summary["independent_hippo_counter_norm_count"], 1)
        self.assertEqual(summary["value_of_information_gate"]["recommended_action"], "preserve_route")

    def test_route_value_does_not_trust_fragmented_weak_budgeted_retrieval(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw\nB. hippo base\nC. budgeted retrieval",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {
                    "variant": "raw_budget_matched",
                    "answer": "A",
                    "budget_top_candidate_vote_count": 5,
                    "budget_strong_consensus": True,
                },
                "hipporag_baseline": {
                    "variant": "hipporag_baseline",
                    "answer": "B",
                    "context_char_count": 1099,
                    "selected_doc_count": 5,
                    "candidate_doc_count": 9,
                    "context_answer_supported": False,
                    "context_answer_overlap_count": 0,
                    "context_question_overlap_count": 4,
                    "context_answer_option_hash": "hb",
                },
                "hipporag_budget_matched": {
                    "variant": "hipporag_budget_matched",
                    "answer": "C",
                    "budget_top_candidate_vote_count": 4,
                    "budget_strong_consensus": True,
                    "context_char_count": 1099,
                    "selected_doc_count": 5,
                    "candidate_doc_count": 9,
                    "context_answer_supported": False,
                    "context_answer_overlap_count": 1,
                    "context_question_overlap_count": 4,
                    "context_answer_option_hash": "hc",
                },
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 5,
            },
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1099,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 9,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_answer_supported": False,
                "context_answer_overlap_count": 0,
                "context_question_overlap_count": 4,
                "context_answer_option_hash": "hb",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "C",
                "preserve_context_char_count": 1099,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 9,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_answer_supported": False,
                "context_answer_overlap_count": 1,
                "context_question_overlap_count": 4,
                "context_answer_option_hash": "hc",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2198},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertFalse(summary["selected_route_trusted"])
        self.assertEqual(
            summary["selected_route_trust_reason"],
            "budgeted_retrieval_fragmented_low_support",
        )

    def test_route_consensus_guard_keeps_unverified_consensus_diagnostic(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. consensus\nB. challenge",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "A"},
            }
        }
        base_attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hippo", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 900, "preserve_selected_doc_count": 3},
            {"child_id": "direct", "child_index": 4, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            route_attempt, route_summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=base_attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": False,
                    "top_candidate_vote_count": 2,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 900},
                call_id="c",
            )

        self.assertIsNotNone(route_attempt)
        self.assertTrue(route_summary["route_consensus"])
        self.assertFalse(route_summary["selected_route_trusted"])
        self.assertEqual(route_summary["selected_route_trust_reason"], "raw_budget_support_below_trust_threshold")
        self.assertEqual(route_attempt.get("candidate_verifier_state"), None)
        self.assertFalse(_route_arbitrator_lock_decision(route_summary))
        self.assertIn(
            "unverified_route_family_consensus",
            route_summary["selected_route_value_profile"]["risk_tags"],
        )
        attempts = base_attempts + [
            route_attempt,
            {"child_id": "challenge", "child_index": 20, "prompt_kind": "option_elimination_challenge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

    def test_route_consensus_guard_still_trusts_answer_bearing_consensus(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. consensus\nB. challenge",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "A"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "hippo",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "preserve_context_char_count": 900,
                "preserve_selected_doc_count": 3,
                "context_answer_supported": True,
                "context_answer_overlap_count": 2,
                "context_question_overlap_count": 3,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            route_attempt, route_summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 900},
                call_id="c",
            )

        self.assertIsNotNone(route_attempt)
        self.assertTrue(route_summary["route_consensus"])
        self.assertTrue(route_summary["selected_route_trusted"])
        self.assertEqual(route_summary["selected_route_trust_reason"], "route_family_consensus")
        self.assertTrue(_route_arbitrator_lock_decision(route_summary))
        continue_summary = dict(route_summary)
        continue_summary["value_of_information_gate"] = {
            "status": "continue_exploration",
            "recommended_action": "continue_exploration",
        }
        self.assertFalse(_route_arbitrator_lock_decision(continue_summary))

    def test_route_consensus_guard_can_be_disabled_for_ablation(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. consensus\nB. challenge",
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 900,
                "preserve_selected_doc_count": 3,
            },
        ]
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_ROUTE_ARBITRATOR": "1", "HLE_DISABLE_ROUTE_CONSENSUS_GUARD": "1"},
            clear=False,
        ):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 900},
                call_id="c",
            )

        self.assertIsNone(attempt)
        self.assertEqual(summary["reason"], "routes_agree")
        self.assertFalse(summary["route_consensus_guard_enabled"])

    def test_route_value_verifier_prefers_independent_hipporag_over_budget_echo(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw echo\nB. independent retrieval\nC. other",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "B"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "A"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_answer_overlap_count": 1,
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "hb",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_answer_overlap_count": 1,
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "ha",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2400},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["selected_route_type"], "hipporag_preserve")
        self.assertEqual(summary["independent_hippo_counter_norm_count"], 1)
        self.assertIn(
            "independent_hipporag_counter_to_budget_echo",
            summary["selected_route_value_profile"]["reason_tags"],
        )

    def test_route_value_trusts_three_control_answer_bearing_hipporag_family(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw only\nB. family supported\nC. other",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "B"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "B"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "B",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
            },
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 8,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 2,
                "same_route_agreeing_variants": ["hipporag_baseline", "hipporag_budget_matched"],
                "context_answer_supported": True,
                "context_question_overlap_count": 1,
                "context_answer_overlap_count": 2,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1200},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertTrue(summary["selected_route_trusted"])
        self.assertEqual(
            summary["selected_route_trust_reason"],
            "baseline_supported_answer_bearing_hipporag_family_route",
        )

    def test_route_value_blocks_hipporag_family_without_answer_bearing_certificate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. consensus\nB. alternative",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "A"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
            },
            {
                "child_id": "hippo",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 8,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 2,
                "same_route_agreeing_variants": ["hipporag_baseline", "hipporag_budget_matched"],
                "context_question_overlap_count": 5,
                "context_answer_overlap_count": 2,
                "context_answer_option_hash": "ha",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            _, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1200},
                call_id="c",
            )

        self.assertFalse(summary["selected_route_trusted"])
        self.assertEqual(
            summary["selected_route_trust_reason"],
            "hipporag_family_consensus_without_answer_bearing_certificate",
        )
        self.assertEqual(
            summary["value_of_information_gate"]["recommended_action"],
            "continue_exploration",
        )

    def test_route_value_trusts_raw_budget_before_unverified_family_risk(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. budget consensus\nB. alternative",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "A"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "A",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 5,
            },
            {
                "child_id": "hippo",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 900,
                "preserve_selected_doc_count": 3,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            _, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 5,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 900},
                call_id="c",
            )

        self.assertEqual(summary["selected_route_type"], "raw_budget_consensus")
        self.assertTrue(summary["selected_route_trusted"])
        self.assertEqual(summary["selected_route_trust_reason"], "strong_raw_budget_consensus")

    def test_route_value_prefers_budget_pair_over_weak_option_linked_base_pair(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. base pair\nB. budget pair\nC. other",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "B"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "A"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "B"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "rawb",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "B",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 3,
            },
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1400,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 8,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_question_overlap_count": 6,
                "context_answer_overlap_count": 0,
                "context_answer_option_hash": "ha",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 8,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_question_overlap_count": 5,
                "context_answer_overlap_count": 0,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 3,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2600},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertIn(summary["selected_route_type"], {"raw_budget_consensus", "hipporag_preserve"})
        self.assertTrue(summary["selected_route_trusted"])
        self.assertIn(
            "raw_and_hipporag_budget_pair_consensus",
            summary["selected_route_value_profile"]["reason_tags"],
        )

    def test_budget_echo_guard_can_be_disabled_for_ablation(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is best?\nA. raw echo\nB. independent retrieval\nC. other",
        }
        agent_plan = {
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "B"},
                "hipporag_budget_matched": {"variant": "hipporag_budget_matched", "answer": "A"},
            }
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {"child_id": "rawb", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo-base",
                "child_index": 3,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_baseline",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_baseline"],
                "context_answer_overlap_count": 1,
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "hb",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "A",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 10,
                "same_run_baseline_cache_variant": "hipporag_budget_matched",
                "same_route_agreement_count": 1,
                "same_route_agreeing_variants": ["hipporag_budget_matched"],
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 4,
                "context_answer_overlap_count": 1,
                "context_question_overlap_count": 8,
                "context_answer_option_hash": "ha",
            },
        ]
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_ROUTE_ARBITRATOR": "1", "HLE_DISABLE_BUDGET_ECHO_GUARD": "1"},
            clear=False,
        ):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                agent_plan=agent_plan,
                raw_budget_preserve_summary={
                    "status": "activated",
                    "strong_consensus": True,
                    "top_candidate_vote_count": 4,
                },
                hipporag_preserve_summary={"status": "activated", "context_char_count": 2400},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(summary["independent_hippo_counter_norm_count"], 0)
        self.assertFalse(summary["budget_echo_guard_enabled"])
        self.assertNotIn(
            "independent_hipporag_counter_to_budget_echo",
            summary["selected_route_value_profile"]["reason_tags"],
        )

    def test_llm_route_arbitrator_can_override_heuristic_route(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. heuristic\nB. llm selected",
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo",
                "child_index": 2,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 900,
                "preserve_selected_doc_count": 2,
                "preserve_candidate_doc_count": 4,
                "context_answer_supported": True,
                "context_answer_overlap_count": 2,
                "context_question_overlap_count": 2,
            },
        ]
        with patch.dict(
            os.environ,
            {"HLE_ENABLE_ROUTE_ARBITRATOR": "1", "HLE_ENABLE_LLM_ROUTE_ARBITRATOR": "1"},
            clear=False,
        ), patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"route_id":1,"confidence":"high","reason_tag":"answer_bearing"}',
        ) as call_model:
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                model="m",
                timeout=None,
                max_tokens=64,
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["llm_route_arbitrator"]["status"], "activated")
        self.assertEqual(summary["underlying_model_calls"], 1)
        call_model.assert_called_once()

    def test_route_arbitrator_does_not_trust_unsupported_hipporag_route(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "_question": "Which option is correct?\nA. raw\nB. retrieved",
        }
        attempts = [
            {"child_id": "raw", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A"},
            {
                "child_id": "hippo",
                "child_index": 2,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "preserve_context_char_count": 1200,
                "preserve_selected_doc_count": 5,
                "preserve_candidate_doc_count": 7,
                "same_route_agreement_count": 2,
                "context_answer_supported": False,
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_ROUTE_ARBITRATOR": "1"}, clear=False):
            attempt, summary = _maybe_add_route_arbitrator_candidate(
                problem=problem,
                attempts=attempts,
                hipporag_preserve_summary={"status": "activated", "context_char_count": 1200},
                call_id="c",
            )

        self.assertIsNotNone(attempt)
        self.assertFalse(summary["selected_route_trusted"])
        self.assertNotEqual(attempt.get("candidate_verifier_state"), "verified")

    def test_untrusted_route_arbitrator_candidate_does_not_lock_deep_challenges(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. direct\nB. retrieved",
            "choices": ["A. direct", "B. retrieved"],
        }
        plan = {
            "stages": {},
            "hle_same_run_baseline_cache": {
                "raw": {"variant": "raw", "answer": "A"},
                "raw_budget_matched": {"variant": "raw_budget_matched", "answer": "A"},
                "hipporag_baseline": {"variant": "hipporag_baseline", "answer": "B"},
            },
        }
        execute_result = {
            "attempts": [
                {
                    "child_id": "direct",
                    "child_index": 1,
                    "prompt_kind": "direct_short_answer",
                    "parsed_answer": "A",
                    "parsed_answer_hash": "hA",
                    "status": "answered",
                    "branch_axis": "closed_book_direct",
                },
                {
                    "child_id": "format",
                    "child_index": 2,
                    "prompt_kind": "literal_constraint_answer",
                    "parsed_answer": "A",
                    "parsed_answer_hash": "hA",
                    "status": "answered",
                    "branch_axis": "format_constraint",
                },
            ],
            "underlying_model_calls": 2,
            "early_stop_reason": None,
            "skipped_prompt_kinds": [],
            "skipped_branch_axes": [],
            "execution_mode": "parallel_quorum",
            "serial_forced_reason": None,
            "child_timeout_sec": 30,
            "child_max_workers": 2,
        }
        route_attempt = {
            "child_id": "route",
            "child_index": 3,
            "prompt_kind": "route_arbitrator_answer",
            "parsed_answer": "B",
            "parsed_answer_hash": "hB",
            "status": "answered",
            "route_arbitrator_trusted": False,
        }
        route_summary = {
            "status": "activated",
            "candidate_emitted": True,
            "child_id": "route",
            "selected_route_type": "hipporag_preserve",
            "selected_route_trusted": False,
            "selected_route_trust_reason": "hipporag_context_not_answer_bearing",
            "selected_route_score": 3.0,
            "runner_up_score": 1.0,
            "underlying_model_calls": 0,
        }
        selection = {
            "status": "activated",
            "selection_method": "normalized_majority",
            "selected_child_id": "direct",
            "selected_answer": "A",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
        with patch.dict(
            os.environ,
            {
                "HLE_ENABLE_ROUTE_ARBITRATOR": "1",
                "HLE_DISABLE_CACHE_FIRST_ROUTE_ARBITRATOR": "1",
            },
            clear=False,
        ):
            with patch("assumption_os.hle_smoke_eval._execute_recursive_child_attempts", return_value=execute_result):
                with patch("assumption_os.hle_smoke_eval._should_run_math_tool_child", return_value=False):
                    with patch("assumption_os.hle_smoke_eval._maybe_run_timeout_recovery_child", return_value=(None, None)):
                        with patch("assumption_os.hle_smoke_eval._maybe_run_child_model_failover_child", return_value=(None, None)):
                            with patch("assumption_os.hle_smoke_eval._should_run_candidate_claim_verifier", return_value=False):
                                with patch("assumption_os.hle_smoke_eval._maybe_run_mc_option_evidence_scorer", return_value=(None, None)):
                                    with patch("assumption_os.hle_smoke_eval._maybe_run_domain_rule_mc_verifier", return_value=(None, None)):
                                        with patch("assumption_os.hle_smoke_eval._maybe_run_raw_preserve_selector_child", return_value=(None, None)):
                                            with patch("assumption_os.hle_smoke_eval._maybe_run_raw_budget_preserve_selector_child", return_value=(None, None)):
                                                with patch("assumption_os.hle_smoke_eval._maybe_run_hipporag_preserve_selector_child", return_value=(None, None)):
                                                    with patch("assumption_os.hle_smoke_eval._maybe_add_route_arbitrator_candidate", return_value=(route_attempt, route_summary)):
                                                        with patch("assumption_os.hle_smoke_eval._maybe_run_counter_assumption_challenge", return_value=(None, None)) as counter:
                                                            with patch("assumption_os.hle_smoke_eval._maybe_run_critic_synthesis_child", return_value=(None, None)):
                                                                with patch("assumption_os.hle_smoke_eval._maybe_add_mc_option_sweep_candidates", return_value=([], None)):
                                                                    with patch("assumption_os.hle_smoke_eval._select_recursive_child_answer", return_value=selection):
                                                                        result = _call_recursive_verified_answer(
                                                                            problem=problem,
                                                                            model="gpt-5.4-mini",
                                                                            agent_plan=plan,
                                                                            eval_id="e",
                                                                            call_id="call",
                                                                            logger=None,
                                                                            timeout=60,
                                                                            child_mode="parallel_quorum",
                                                                            child_timeout=30,
                                                                            max_tokens=128,
                                                                            evidence_bridge_enabled=False,
                                                                        )

        self.assertEqual(json.loads(result["answer_text"])["answer"], "A")
        counter.assert_called_once()
        self.assertFalse(plan["stages"]["route_arbitrator"]["route_locked"])
        self.assertFalse(plan["stages"]["route_arbitrator"]["selected_route_trusted"])

    def test_verified_or_abstain_allows_verified_selection(self):
        problem = {
            "answer_type": "exactMatch",
            "_question": "Compute the answer.",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41"},
            {"child_id": "math", "child_index": 2, "prompt_kind": "math_tool_answer", "parsed_answer": "42"},
        ]
        selection = {
            "selection_method": "verified_math_tool_priority",
            "selected_child_id": "math",
            "selected_answer": "42",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }

        gated = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)

        self.assertEqual(gated["selection_method"], "verified_math_tool_priority")
        self.assertEqual(gated["selected_child_id"], "math")
        self.assertEqual(gated["verified_or_abstain_gate"]["status"], "allowed")

        math_selection = _select_recursive_child_answer(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct", "parsed_answer": "wrong"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "checked", "parsed_answer": "wrong"},
                {
                    "child_id": "c3",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "42",
                    "tool_confidence": "verified_symbolic",
                    "tool_source": "deterministic_parser",
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(math_selection["selection_method"], "verified_math_tool_priority")
        self.assertEqual(math_selection["selected_answer"], "42")

        verifier_problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute $2^5 + 10$.",
        }
        verifier_attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "41"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "constraint_checked_answer", "parsed_answer": "The answer is $42$."},
        ]
        verifier_summary = _apply_math_candidate_claim_verifier(verifier_problem, verifier_attempts)
        verifier_selection = _select_recursive_child_answer(
            problem=verifier_problem,
            attempts=verifier_attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(verifier_summary["verified_count"], 1)
        self.assertEqual(verifier_summary["refuted_count"], 2)
        self.assertEqual(verifier_attempts[2]["candidate_verifier_state"], "verified")
        self.assertEqual(verifier_selection["selection_method"], "candidate_claim_verifier_priority")
        self.assertEqual(verifier_selection["selected_answer"], "42")

        planner_problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Use the described process to determine the final numeric value.",
        }
        planner_attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41", "status": "answered"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "42", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"operation":"evaluate","expression":"6*7","equation":"","variable":"x","modulus":""}',
        ):
            planner_summary = _apply_math_candidate_claim_verifier(
                planner_problem,
                planner_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(planner_summary["backend"], "sympy_candidate_reference_planner")
        self.assertEqual(planner_summary["verified_count"], 1)
        self.assertEqual(planner_summary["refuted_count"], 1)
        self.assertEqual(planner_summary["underlying_model_calls"], 1)
        self.assertEqual(planner_attempts[1]["candidate_verifier_state"], "verified")

        weak_attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "42", "status": "answered"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "42", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"operation":"evaluate","expression":"6*7","equation":"","variable":"x","modulus":""}',
        ):
            weak_summary = _apply_math_candidate_claim_verifier(
                planner_problem,
                weak_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(weak_summary["status"], "weak_single_candidate_confirmation")
        self.assertNotIn("candidate_verifier_state", weak_attempts[0])

        unsupported_planner_selection = _select_recursive_child_answer(
            problem=problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct", "parsed_answer": "closed book"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "checked", "parsed_answer": "closed book"},
                {
                    "child_id": "c3",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "planner-only",
                    "tool_confidence": "verified_symbolic",
                    "tool_source": "llm_planner",
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(unsupported_planner_selection["selection_method"], "normalized_majority")
        self.assertEqual(unsupported_planner_selection["selected_answer"], "closed book")

        math_majority = _select_recursive_child_answer(
            problem={"id_hash": "pid", "question_hash": "qid", "answer_type": "exactMatch", "category": "Math"},
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "direct"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "evidence"},
                {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "evidence"},
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(math_majority["selection_method"], "math_exact_normalized_majority")
        self.assertEqual(math_majority["selected_answer"], "evidence")

        math_direct_fallback = _select_recursive_child_answer(
            problem={"id_hash": "pid", "question_hash": "qid", "answer_type": "exactMatch", "category": "Math"},
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "direct"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "evidence"},
                {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "recursive"},
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(math_direct_fallback["selection_method"], "math_exact_direct_fallback")
        self.assertEqual(math_direct_fallback["selected_answer"], "direct")

    def test_exact_math_candidate_claim_verifier_uses_executable_equivalence(self):
        solve_problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Use the described symbolic process to solve for x.",
        }
        solve_attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "x = -2 or x = 2", "status": "answered"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "2, 3", "status": "answered"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "constraint_checked_answer", "parsed_answer": "±2", "status": "answered"},
            {"child_id": "c4", "child_index": 4, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "The roots are x = -2 and x = 2.", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"operation":"solve","expression":"","equation":"x**2 - 4 = 0","variable":"x","modulus":""}',
        ):
            solve_summary = _apply_math_candidate_claim_verifier(
                solve_problem,
                solve_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(solve_summary["backend"], "sympy_candidate_reference_planner")
        self.assertEqual(solve_summary["verified_count"], 3)
        self.assertEqual(solve_summary["refuted_count"], 1)
        self.assertEqual(solve_attempts[0]["candidate_verifier_state"], "verified")
        self.assertEqual(solve_attempts[0]["candidate_verifier_match_method"], "unordered_collection_equivalence")
        self.assertEqual(solve_attempts[1]["candidate_verifier_state"], "refuted")
        self.assertEqual(solve_attempts[2]["candidate_verifier_state"], "verified")
        self.assertEqual(solve_attempts[2]["candidate_verifier_match_method"], "unordered_collection_equivalence")
        self.assertEqual(solve_attempts[3]["candidate_verifier_state"], "verified")
        self.assertEqual(solve_attempts[3]["candidate_verifier_match_method"], "unordered_collection_equivalence")

        solve_selection = _select_recursive_child_answer(
            problem=solve_problem,
            attempts=solve_attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )
        self.assertEqual(solve_selection["selection_method"], "math_exact_normalized_majority")
        self.assertEqual(solve_selection["selected_answer"], "x = -2 or x = 2")
        self.assertEqual(solve_attempts[0]["candidate_verifier_trust"], "weak_llm_reference_planner")

        equivalent_majority = _select_recursive_child_answer(
            problem=solve_problem,
            attempts=[
                {"child_id": "m1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "x = -2 or x = 2", "status": "answered"},
                {"child_id": "m2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "±2", "status": "answered"},
                {"child_id": "m3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "2, 3", "status": "answered"},
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )
        self.assertEqual(equivalent_majority["selection_method"], "math_exact_normalized_majority")
        self.assertEqual(equivalent_majority["selected_answer"], "x = -2 or x = 2")

        with patch.dict(os.environ, {"HLE_DISABLE_CANDIDATE_CLAIM_VERIFIER": "1"}):
            self.assertFalse(_should_run_candidate_claim_verifier(solve_problem))

        numeric_problem = {
            "id_hash": "pid2",
            "question_hash": "qid2",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Use the described arithmetic process to determine the final value.",
        }
        numeric_attempts = [
            {"child_id": "n1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "0.5", "status": "answered"},
            {"child_id": "n2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "0.25", "status": "answered"},
            {"child_id": "n3", "child_index": 3, "prompt_kind": "constraint_checked_answer", "parsed_answer": "The final answer is \\boxed{1/2}.", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"operation":"evaluate","expression":"1/2","equation":"","variable":"x","modulus":""}',
        ):
            numeric_summary = _apply_math_candidate_claim_verifier(
                numeric_problem,
                numeric_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(numeric_summary["verified_count"], 2)
        self.assertEqual(numeric_summary["refuted_count"], 1)
        self.assertEqual(numeric_attempts[0]["candidate_verifier_state"], "verified")
        self.assertIn(
            numeric_attempts[0]["candidate_verifier_match_method"],
            {"numeric_equivalence", "numeric_tolerance"},
        )
        self.assertEqual(numeric_attempts[2]["candidate_verifier_state"], "verified")

        routed_attempts = [
            {"child_id": "rt1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "0.5", "status": "answered"},
            {"child_id": "rt2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "0.25", "status": "answered"},
        ]
        with patch.dict(os.environ, {"HLE_CANDIDATE_CLAIM_PLANNER_MODEL": "gpt-5.5"}):
            with patch(
                "assumption_os.hle_smoke_eval._call_model",
                return_value='{"operation":"evaluate","expression":"1/2","equation":"","variable":"x","modulus":""}',
            ) as call_model:
                routed_summary = _apply_math_candidate_claim_verifier(
                    numeric_problem,
                    routed_attempts,
                    model="gpt-5.4-mini",
                    timeout=1,
                    max_tokens=64,
                )
        self.assertEqual(routed_summary["verified_count"], 1)
        self.assertEqual(call_model.call_args.kwargs["model"], "gpt-5.5")

        multiplan_attempts = [
            {"child_id": "mp1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "0.5", "status": "answered"},
            {"child_id": "mp2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "0.25", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value=(
                '{"plans":['
                '{"operation":"none","expression":"","equation":"","variable":"x"},'
                '{"operation":"evaluate","expression":"1/2","equation":"","variable":"x"}'
                ']}'
            ),
        ):
            multiplan_summary = _apply_math_candidate_claim_verifier(
                numeric_problem,
                multiplan_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )
        self.assertEqual(multiplan_summary["verified_count"], 1)
        self.assertEqual(multiplan_summary["refuted_count"], 1)
        self.assertEqual(multiplan_attempts[0]["candidate_verifier_state"], "verified")

        leak_skip_attempts = [
            {"child_id": "ls1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "0.5", "status": "answered"},
            {"child_id": "ls2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "0.25", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value=(
                '{"plans":['
                '{"operation":"evaluate","expression":"0.5","equation":"","variable":"x"},'
                '{"operation":"evaluate","expression":"1/2","equation":"","variable":"x"}'
                ']}'
            ),
        ):
            leak_skip_summary = _apply_math_candidate_claim_verifier(
                numeric_problem,
                leak_skip_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )
        self.assertEqual(leak_skip_summary["verified_count"], 1)
        self.assertEqual(leak_skip_summary["refuted_count"], 1)
        self.assertEqual(leak_skip_attempts[0]["candidate_verifier_state"], "verified")

        repair_attempts = [
            {"child_id": "rp1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "0.5", "status": "answered"},
            {"child_id": "rp2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "0.25", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            side_effect=[
                '{"plans":[{"operation":"none","expression":"","equation":"","variable":"x"}]}',
                '{"plans":[{"operation":"evaluate","expression":"1/2","equation":"","variable":"x"}]}',
            ],
        ):
            repair_summary = _apply_math_candidate_claim_verifier(
                numeric_problem,
                repair_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )
        self.assertEqual(repair_summary["verified_count"], 1)
        self.assertEqual(repair_summary["refuted_count"], 1)
        self.assertEqual(repair_summary["underlying_model_calls"], 2)
        self.assertEqual(repair_attempts[0]["candidate_verifier_state"], "verified")

        python_problem = {
            "id_hash": "pid5",
            "question_hash": "qid5",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Combinatorics",
            "_question": "Use the described combinatorial process to determine the final count.",
        }
        python_attempts = [
            {"child_id": "py1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "120", "status": "answered"},
            {"child_id": "py2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "100", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"plans":[{"operation":"python","expression":"comb(10, 3)","variable":"x"}]}',
        ):
            python_summary = _apply_math_candidate_claim_verifier(
                python_problem,
                python_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )
        self.assertEqual(python_summary["verified_count"], 1)
        self.assertEqual(python_summary["refuted_count"], 1)
        self.assertEqual(python_attempts[0]["candidate_verifier_state"], "verified")

        weak_planner_attempts = [
            {"child_id": "wp1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "100", "status": "answered"},
            {"child_id": "wp2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "120", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"plans":[{"operation":"python","expression":"comb(10, 3)","variable":"x"}]}',
        ):
            weak_planner_summary = _apply_math_candidate_claim_verifier(
                python_problem,
                weak_planner_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )
        self.assertEqual(weak_planner_summary["status"], "weak_planner_single_verified")
        self.assertEqual(weak_planner_attempts[1]["candidate_verifier_state"], "verified")
        self.assertEqual(weak_planner_attempts[1]["candidate_verifier_trust"], "weak_single_planner")
        weak_selection = _select_recursive_child_answer(
            problem=python_problem,
            attempts=weak_planner_attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )
        self.assertNotEqual(weak_selection["selection_method"], "candidate_claim_verifier_priority")

        error_attempts = [
            {"child_id": "e1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "42", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            side_effect=RuntimeError("endpoint disconnected"),
        ):
            error_summary = _apply_math_candidate_claim_verifier(
                numeric_problem,
                error_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(error_summary["status"], "no_executable_claim")
        self.assertEqual(error_summary["reference_reason"], "planner_error")
        self.assertEqual(error_summary["reference_error_type"], "RuntimeError")
        self.assertEqual(error_summary["underlying_model_calls"], 1)

        deterministic_solve_problem = {
            "id_hash": "pid3",
            "question_hash": "qid3",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Solve $x^2 - 4 = 0$.",
        }
        deterministic_attempts = [
            {"child_id": "d1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "2, 3", "status": "answered"},
            {"child_id": "d2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "±2", "status": "answered"},
        ]
        with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
            deterministic_summary = _apply_math_candidate_claim_verifier(
                deterministic_solve_problem,
                deterministic_attempts,
                model=None,
                timeout=1,
                max_tokens=64,
            )
        call_model.assert_not_called()
        self.assertEqual(deterministic_summary["backend"], "sympy_deterministic")
        self.assertEqual(deterministic_summary["reference_operation"], "solve")
        self.assertEqual(deterministic_summary["verified_count"], 1)
        self.assertEqual(deterministic_summary["refuted_count"], 1)
        self.assertEqual(deterministic_summary["underlying_model_calls"], 0)
        self.assertEqual(deterministic_attempts[1]["candidate_verifier_state"], "verified")

        deterministic_simplify_problem = {
            "id_hash": "pid4",
            "question_hash": "qid4",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Simplify $x + x$.",
        }
        simplify_attempts = [
            {"child_id": "s1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "x", "status": "answered"},
            {"child_id": "s2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "2*x", "status": "answered"},
        ]
        with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
            simplify_summary = _apply_math_candidate_claim_verifier(
                deterministic_simplify_problem,
                simplify_attempts,
                model=None,
                timeout=1,
                max_tokens=64,
            )
        call_model.assert_not_called()
        self.assertEqual(simplify_summary["backend"], "sympy_deterministic")
        self.assertEqual(simplify_summary["reference_operation"], "simplify")
        self.assertEqual(simplify_summary["verified_count"], 1)
        self.assertEqual(simplify_summary["refuted_count"], 1)
        self.assertEqual(simplify_attempts[1]["candidate_verifier_state"], "verified")

        all_wrong_simplify_attempts = [
            {"child_id": "aw1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "x", "status": "answered"},
            {"child_id": "aw2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "3*x", "status": "answered"},
        ]
        all_wrong_summary = _apply_math_candidate_claim_verifier(
            deterministic_simplify_problem,
            all_wrong_simplify_attempts,
            model=None,
            timeout=1,
            max_tokens=64,
        )
        self.assertEqual(all_wrong_summary["verified_count"], 1)
        self.assertEqual(all_wrong_summary["refuted_count"], 2)
        self.assertEqual(all_wrong_simplify_attempts[-1]["prompt_kind"], "candidate_claim_verifier_answer")
        self.assertEqual(all_wrong_simplify_attempts[-1]["parsed_answer"], "2*x")
        synthetic_selection = _select_recursive_child_answer(
            problem=deterministic_simplify_problem,
            attempts=all_wrong_simplify_attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )
        self.assertEqual(synthetic_selection["selection_method"], "candidate_claim_verifier_priority")
        self.assertEqual(synthetic_selection["selected_answer"], "2*x")

    def test_mc_candidate_claim_verifier_can_override_wrong_majority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute $6*7$.\nA. 41\nB. 42\nC. 43\nD. 44",
        }
        options, first_start = _extract_multiple_choice_options(problem["_question"])

        self.assertEqual(first_start, len("Compute $6*7$."))
        self.assertEqual(options["B"], "42")
        self.assertTrue(_should_run_candidate_claim_verifier(problem))
        with patch.dict(os.environ, {"HLE_DISABLE_MC_CANDIDATE_CLAIM_VERIFIER": "1"}, clear=False):
            self.assertFalse(_should_run_candidate_claim_verifier(problem))

        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "status": "answered"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "status": "answered"},
        ]
        summary = _apply_math_candidate_claim_verifier(problem, attempts)
        selection = _select_recursive_child_answer(
            problem=problem,
            attempts=attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(summary["backend"], "sympy_mc_option_deterministic")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["verified_count"], 1)
        self.assertEqual(summary["refuted_count"], 2)
        self.assertEqual(attempts[-1]["prompt_kind"], "candidate_claim_verifier_answer")
        self.assertEqual(selection["selection_method"], "candidate_claim_verifier_priority")
        self.assertEqual(selection["selected_answer"], "B")

    def test_source_grounded_mc_verifier_can_override_closed_book_majority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option matches the evidence?\nA. wrong\nB. right\nC. distractor",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
        ]
        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
                evidence_context="[Evidence 1] source=wikipedia; title=X; snippet=right.",
            )

        self.assertEqual(selection["selection_method"], "source_grounded_verifier_choice")
        self.assertEqual(selection["selected_answer"], "B")
        self.assertEqual(selection["underlying_model_calls"], 1)

    def test_counter_assumption_challenge_triggers_only_for_unverified_majority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. wrong\nB. right",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "B"},
        ]
        trigger = _counter_assumption_challenge_trigger(problem, attempts)

        self.assertEqual(trigger["status"], "activated")
        self.assertEqual(trigger["reason"], "majority_without_independent_verification")
        self.assertEqual(trigger["top_candidate_count"], 2)
        self.assertEqual(trigger["unique_candidate_count"], 2)

        verified_trigger = _counter_assumption_challenge_trigger(
            problem,
            attempts,
            candidate_verifier_summary={"verified_count": 1},
        )
        self.assertEqual(verified_trigger["status"], "abstained")
        self.assertEqual(verified_trigger["reason"], "candidate_claim_already_verified")

    def test_counter_assumption_challenge_forces_verifier_when_it_disagrees(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. wrong\nB. right",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "counter_assumption_challenge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(
            os.environ,
            {
                "HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1",
                "HLE_ENABLE_OPTION_SWEEP_VOI_COUNTER": "1",
                "HLE_ALLOW_STRUCTURAL_COUNTER_OVER_BASELINE_CONSENSUS": "1",
            },
        ):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "c3")
        self.assertEqual(selection["selected_answer"], "B")
        self.assertEqual(selection["underlying_model_calls"], 1)

    def test_option_elimination_runs_after_counter_challenge_confirms_majority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. wrong\nB. right",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "counter_assumption_challenge_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
        ]

        def fake_child_attempt(**kwargs):
            return {
                "child_id": "c4",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "prediction_hash": "pb",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=fake_child_attempt):
            attempt, summary = _maybe_run_option_elimination_challenge(
                problem=problem,
                attempts=attempts,
                counter_challenge_summary={"status": "activated", "challenge_disagreed_with_majority": False},
                evidence_context="[Evidence 1] source=wikipedia; title=X; snippet=right.",
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(attempt["prompt_kind"], "option_elimination_challenge_answer")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["challenge_disagreed_with_majority"])

    def test_option_elimination_still_runs_after_counter_challenge_disagrees(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. wrong\nB. distractor\nC. right",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "counter_assumption_challenge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]

        def fake_child_attempt(**kwargs):
            self.assertIn("Do not anchor", kwargs["spec"]["prompt"])
            return {
                "child_id": "c4",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "C",
                "parsed_answer_hash": "hc",
                "prediction_hash": "pc",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=fake_child_attempt):
            attempt, summary = _maybe_run_option_elimination_challenge(
                problem=problem,
                attempts=attempts,
                counter_challenge_summary={"status": "activated", "challenge_disagreed_with_majority": True},
                evidence_context="",
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(attempt["prompt_kind"], "option_elimination_challenge_answer")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["reason"], "counter_challenge_disagreed_run_full_option_elimination")
        self.assertTrue(summary["challenge_disagreed_with_majority"])

    def test_mc_option_evidence_scorer_emits_only_clear_margin_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option names the transformer architecture?\nA. ResNet\nB. Attention Is All You Need\nC. AlexNet",
            "_answer": "B",
        }

        def fake_search(query, *, limit, timeout):
            if "Attention" in query:
                return [
                    {
                        "title": "Attention Is All You Need",
                        "snippet": "Attention Is All You Need introduced the Transformer architecture.",
                    },
                    {
                        "title": "Transformer (deep learning architecture)",
                        "snippet": "The Transformer architecture was introduced in the paper Attention Is All You Need.",
                    },
                    {
                        "title": "Transformer paper",
                        "snippet": "Attention Is All You Need is the paper that introduced the Transformer architecture.",
                    },
                ]
            return [{"title": "Unrelated", "snippet": "No matching architecture evidence."}]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                attempt, summary = _maybe_run_mc_option_evidence_scorer(
                    problem=problem,
                    attempts=[],
                    eval_id="e",
                    call_id="c",
                    model="m",
                    logger=None,
                )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["prompt_kind"], "mc_option_evidence_scorer_answer")
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(attempt["candidate_verifier_state"], "verified")
        self.assertEqual(attempt["tool_confidence"], "verified_option_evidence_margin")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["candidate_emitted"])
        self.assertEqual(summary["candidate_verifier_state"], "verified")
        self.assertGreaterEqual(summary["top_support_doc_count"], 3)
        self.assertEqual(summary["top_ambiguous_doc_count"], 0)
        self.assertTrue(summary["candidate_correct_for_eval"])

    def test_mc_option_evidence_scorer_blocks_ambiguous_multisupport_docs(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which named method is the intended answer?\nA. Alpha Method\nB. Beta Method\nC. Gamma Method",
            "_answer": "B",
        }

        def fake_search(query, *, limit, timeout):
            return [
                {
                    "title": "Alpha Method and Beta Method comparison",
                    "snippet": "The source discusses both Alpha Method and Beta Method in the same context.",
                }
            ]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                attempt, summary = _maybe_run_mc_option_evidence_scorer(
                    problem=problem,
                    attempts=[],
                    eval_id="e",
                    call_id="c",
                    model="m",
                    logger=None,
                )

        self.assertIsNone(attempt)
        self.assertEqual(summary["status"], "blocked_ambiguous_option_evidence")
        self.assertFalse(summary["candidate_emitted"])
        self.assertEqual(summary["candidate_verifier_state"], "not_verified")
        self.assertGreaterEqual(summary["any_ambiguous_doc_count"], 1)

    def test_mc_option_evidence_scorer_blocks_low_absolute_support_score(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": (
                "Which option names the relevant mechanism in coupled oscillators?\n"
                "A. Alpha coupling\nB. Beta lock\nC. Gamma drift"
            ),
            "_answer": "B",
        }

        def fake_search(query, *, limit, timeout):
            if "Beta lock" in query:
                return [
                    {"title": "Beta", "snippet": "Relevant mechanism for coupled oscillators."},
                    {"title": "Beta overview", "snippet": "Mechanism note for coupled oscillators."},
                ]
            return [{"title": "Unrelated", "snippet": "No support."}]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                attempt, summary = _maybe_run_mc_option_evidence_scorer(
                    problem=problem,
                    attempts=[],
                    eval_id="e",
                    call_id="c",
                    model="m",
                    logger=None,
                )

        self.assertIsNone(attempt)
        self.assertFalse(summary["candidate_emitted"])
        self.assertEqual(summary["candidate_verifier_state"], "not_verified")
        self.assertGreaterEqual(summary["top_support_doc_count"], 2)
        self.assertLess(summary["top_score"], summary["min_verified_score"])

    def test_mc_option_evidence_ranks_stable_support_over_single_high_overlap_doc(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": (
                "Which option names the principle introduced for stable transformer sequence modeling?\n"
                "A. Alpha Principle\nB. Beta Principle\nC. Gamma Principle"
            ),
            "_answer": "B",
        }

        def fake_search(query, *, limit, timeout):
            if "Alpha Principle" in query:
                return [
                    {
                        "title": "Alpha Principle stable transformer sequence modeling",
                        "snippet": (
                            "Alpha Principle Alpha Principle Alpha Principle is discussed with stable "
                            "transformer sequence modeling terminology."
                        ),
                    }
                ]
            if "Beta Principle" in query:
                return [
                    {
                        "title": "Beta Principle",
                        "snippet": "Beta Principle was introduced for stable transformer sequence modeling.",
                    },
                    {
                        "title": "Stable sequence modeling",
                        "snippet": "The method known as Beta Principle is used for stable transformer sequence modeling.",
                    },
                    {
                        "title": "Beta Principle sequence modeling",
                        "snippet": "Stable transformer sequence modeling uses the Beta Principle.",
                    },
                ]
            return [{"title": "Gamma note", "snippet": "Gamma Principle is unrelated."}]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                attempt, summary = _maybe_run_mc_option_evidence_scorer(
                    problem=problem,
                    attempts=[],
                    eval_id="e",
                    call_id="c",
                    model="m",
                    logger=None,
                )

        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["top_support_doc_count"], 3)
        self.assertGreater(summary["rank_margin"], 0)
        self.assertTrue(summary["candidate_correct_for_eval"])

    def test_mc_option_evidence_requires_question_stem_support(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Engineering",
            "raw_subject": "Computer Engineering",
            "_question": (
                "Which method reduces timing hazards in asynchronous circuit arbitration?\n"
                "A. Alpha Synchronizer\nB. Beta Protocol\nC. Gamma Filter"
            ),
            "_answer": "C",
        }

        def fake_search(query, *, limit, timeout):
            if "Alpha Synchronizer" in query:
                return [
                    {"title": "Alpha Synchronizer", "snippet": "Alpha Synchronizer is a named method."},
                    {"title": "Alpha Synchronizer overview", "snippet": "The Alpha Synchronizer method is widely cited."},
                ]
            return [{"title": "Unrelated", "snippet": "No support."}]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                attempt, summary = _maybe_run_mc_option_evidence_scorer(
                    problem=problem,
                    attempts=[],
                    eval_id="e",
                    call_id="c",
                    model="m",
                    logger=None,
                )

        self.assertIsNone(attempt)
        self.assertFalse(summary["candidate_emitted"])
        self.assertEqual(summary["candidate_verifier_state"], "not_verified")
        self.assertEqual(summary["top_support_doc_count"], 0)

    def test_evidence_guided_option_challenge_emits_unverified_variation(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": (
                "Which principle predicts restoring opposition to an imposed perturbation?\n"
                "A. Unrelated Rule\nB. Lenz law\nC. Random drift"
            ),
            "_answer": "B",
        }

        def fake_search(query, *, limit, timeout):
            if "Lenz" in query:
                return [
                    {
                        "title": "Lenz's law",
                        "snippet": (
                            "Lenz law is a principle predicting restoring opposition to an imposed "
                            "perturbation: an induced current opposes the change that produced it."
                        ),
                    }
                ]
            return [{"title": "Unrelated", "snippet": "No direct support."}]

        def fake_child(**kwargs):
            return {
                "child_id": "eg1",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "prediction_hash": "pb",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=fake_child):
                    attempt, summary, context = _maybe_run_evidence_guided_option_challenge(
                        problem=problem,
                        attempts=[
                            {
                                "child_id": "c1",
                                "child_index": 1,
                                "prompt_kind": "direct_short_answer",
                                "parsed_answer": "A",
                                "parsed_answer_hash": "ha",
                            }
                        ],
                        option_evidence_summary={"status": "weak_margin"},
                        model="m",
                        eval_id="e",
                        call_id="c",
                        logger=None,
                        timeout=1,
                        max_tokens=64,
                    )

        self.assertIsNotNone(attempt)
        self.assertIn("Option B", context)
        self.assertEqual(attempt["prompt_kind"], "evidence_guided_option_challenge_answer")
        self.assertEqual(attempt["candidate_verifier_state"], "not_verified")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["candidate_emitted"])
        self.assertTrue(summary["candidate_correct_for_eval"])

    def test_evidence_guided_option_challenge_blocks_context_without_support_docs(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Which option is correct?\nA. alpha\nB. beta\nC. gamma",
        }

        def fake_search(query, *, limit, timeout):
            return [{"title": "Generic", "snippet": "This snippet does not connect any option to the question."}]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                with patch("assumption_os.hle_smoke_eval._run_child_attempt") as child:
                    attempt, summary, context = _maybe_run_evidence_guided_option_challenge(
                        problem=problem,
                        attempts=[],
                        option_evidence_summary={"status": "weak_margin"},
                        model="m",
                        eval_id="e",
                        call_id="c",
                        logger=None,
                        timeout=1,
                        max_tokens=64,
                    )

        child.assert_not_called()
        self.assertIsNone(attempt)
        self.assertEqual(context, "")
        self.assertEqual(summary["status"], "abstained")
        self.assertEqual(summary["reason"], "no_discriminative_support_docs")
        self.assertEqual(summary["underlying_model_calls"], 0)

    def test_structural_option_audit_triggers_on_collapsed_unverified_mc_candidates(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Which option is correct?\nA. consensus\nB. least-assumption alternative\nC. distractor",
            "_answer": "B",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "A"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "option_matrix_reasoner_answer", "parsed_answer": "A"},
        ]

        def fake_child(**kwargs):
            return {
                "child_id": "audit",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "prediction_hash": "pb",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=fake_child) as child:
            attempt, summary = _maybe_run_structural_option_audit_child(
                problem=problem,
                attempts=attempts,
                option_evidence_summary={"status": "weak_margin"},
                evidence_guided_option_summary={"status": "abstained"},
                evidence_context="",
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=64,
            )

        child.assert_called_once()
        self.assertIsNotNone(attempt)
        self.assertEqual(attempt["prompt_kind"], "structural_option_audit_answer")
        self.assertEqual(attempt["candidate_verifier_state"], "not_verified")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["candidate_disagreed_with_majority"])
        self.assertTrue(summary["candidate_correct_for_eval"])

    def test_evidence_guided_option_challenge_ignores_route_verified_gate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": (
                "Which principle predicts restoring opposition to an imposed perturbation?\n"
                "A. Unrelated Rule\nB. Lenz law"
            ),
        }

        def fake_search(query, *, limit, timeout):
            return [{
                "title": "Lenz's law",
                "snippet": (
                    "Lenz law is the principle predicting restoring opposition to an imposed perturbation; "
                    "it opposes the change that produced it."
                ),
            }]

        def fake_child(**kwargs):
            return {
                "child_id": "eg2",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "prediction_hash": "pb",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            with patch("assumption_os.hle_smoke_eval._domain_evidence_search", return_value=[]):
                with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=fake_child):
                    attempt, summary, _ = _maybe_run_evidence_guided_option_challenge(
                        problem=problem,
                        attempts=[
                            {
                                "child_id": "route",
                                "child_index": 1,
                                "prompt_kind": "route_arbitrator_answer",
                                "parsed_answer": "A",
                                "parsed_answer_hash": "ha",
                                "candidate_verifier_state": "verified",
                                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                            }
                        ],
                        option_evidence_summary={"status": "weak_margin"},
                        model="m",
                        eval_id="e",
                        call_id="c",
                        logger=None,
                        timeout=1,
                        max_tokens=64,
                    )

        self.assertIsNotNone(attempt)
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["reason"], "unverified_option_specific_evidence_variation")

    def test_unverified_evidence_guided_option_challenge_does_not_bypass_gate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. direct\nB. evidence-guided",
        }
        attempts = [
            {
                "child_id": "c1",
                "child_index": 1,
                "prompt_kind": "direct_short_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
            },
            {
                "child_id": "c2",
                "child_index": 2,
                "prompt_kind": "evidence_guided_option_challenge_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "candidate_verifier_state": "not_verified",
            },
        ]

        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
                evidence_context="Option B: weak retrieved support.",
            )
        gated = _apply_verified_or_abstain_selection(problem=problem, attempts=attempts, selection=selection)

        self.assertEqual(selection["selection_method"], "verifier_choice")
        self.assertEqual(selection["selected_answer"], "B")
        self.assertEqual(gated["selection_method"], "verifier_choice")
        self.assertEqual(gated["verified_or_abstain_gate"]["status"], "allowed")
        self.assertEqual(gated["verified_or_abstain_gate"]["reason"], "verified_selection_method")
        bypass_selection = {
            "selection_method": "normalized_majority",
            "selected_child_id": "c2",
            "selected_answer": "B",
            "underlying_model_calls": 0,
            "verifier_model_call": False,
        }
        bypass_gated = _apply_verified_or_abstain_selection(
            problem=problem,
            attempts=attempts,
            selection=bypass_selection,
        )
        self.assertEqual(bypass_gated["selection_method"], "verified_or_abstain_direct_fallback")
        self.assertEqual(bypass_gated["selected_answer"], "A")

    def test_route_voi_continue_exploration_defers_to_evidence_challenge_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. route\nB. evidence challenge",
        }
        attempts = [
            {
                "child_id": "route",
                "child_index": 1,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_value_confidence": "high",
                "route_value_score": 10.0,
                "route_value_of_information_gate_status": "continue_exploration",
            },
            {
                "child_id": "eg",
                "child_index": 2,
                "prompt_kind": "evidence_guided_option_challenge_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "candidate_verifier_state": "not_verified",
            },
        ]

        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
                evidence_context="Option B has retrieved evidence.",
            )

        self.assertEqual(selection["selection_method"], "verifier_choice")
        self.assertEqual(selection["selected_child_id"], "eg")
        self.assertEqual(selection["selected_answer"], "B")

    def test_verified_option_evidence_priority_can_override_unverified_majority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. evidence-backed",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "c3",
                "child_index": 3,
                "prompt_kind": "mc_option_evidence_scorer_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "private_option_evidence_context": "Option B: directly supported by evidence.",
                "candidate_verifier_state": "verified",
                "tool_confidence": "verified_option_evidence_margin",
            },
        ]
        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":1}'):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(selection["selection_method"], "verified_option_evidence_priority")
        self.assertEqual(selection["selected_child_id"], "c3")
        self.assertEqual(selection["selected_answer"], "B")

    def test_domain_rule_mc_verifier_handles_cross_resistance_minimality(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Biology/Medicine",
            "raw_subject": "Biology",
            "_question": (
                "Two bacteria are grown in lab. The first has common lateral transfer; the second has a stable "
                "genome with no lateral transfer. Yet the second acquired drug resistance at an equal pace.\n"
                "A. Rare mutations occurred.\n"
                "B. Compensatory mutations increased fitness and also led to cross-resistance.\n"
                "C. There was contamination by plasmids.\n"
                "D. Mutations did not have compensatory mutations and also led to cross-resistance.\n"
                "E. Compensatory mutations followed rare resistance mutations."
            ),
            "_answer": "D",
        }

        attempt, summary = _maybe_run_domain_rule_mc_verifier(
            problem=problem,
            attempts=[],
            evidence_context="",
            eval_id="e",
            call_id="c",
            model="m",
            logger=None,
        )

        self.assertEqual(attempt["parsed_answer"], "D")
        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["rule_id"], "bacterial_cross_resistance_minimal_extra_assumption")
        self.assertTrue(summary["candidate_correct_for_eval"])

    def test_domain_rule_mc_verifier_requires_lso_evidence_for_ontario_screen(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Humanities/Social Science",
            "raw_subject": "Law",
            "_question": (
                "A Toronto law firm has former-client confidential information but internal measures prevent "
                "sharing between groups. Which is correct?\n"
                "A. They can continue because matters are unrelated.\n"
                "B. They can continue because appropriate measures ensure confidential information is not shared.\n"
                "C. They are not allowed although measures exist.\n"
                "D. They are not allowed unless the former client consents."
            ),
            "_answer": "B",
        }

        no_evidence_attempt, no_evidence_summary = _maybe_run_domain_rule_mc_verifier(
            problem=problem,
            attempts=[],
            evidence_context="",
            eval_id="e",
            call_id="c",
            model="m",
            logger=None,
        )
        self.assertIsNone(no_evidence_attempt)
        self.assertEqual(no_evidence_summary["status"], "not_required")

        attempt, summary = _maybe_run_domain_rule_mc_verifier(
            problem=problem,
            attempts=[],
            evidence_context=(
                "[Evidence 1] source=lso_rules; title=Law Society of Ontario Rules; "
                "snippet=the law firm establishes that it has taken adequate measures on a timely basis "
                "to ensure that there will be no risk of disclosure of the former client's confidential information"
            ),
            eval_id="e",
            call_id="c",
            model="m",
            logger=None,
        )
        self.assertEqual(attempt["parsed_answer"], "B")
        self.assertEqual(summary["rule_id"], "ontario_former_client_confidential_screen")
        self.assertTrue(summary["candidate_correct_for_eval"])

    def test_option_elimination_disagreement_forces_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. wrong\nB. right",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "option_elimination_challenge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "c3")
        self.assertEqual(selection["selected_answer"], "B")

    def test_baseline_family_consensus_blocks_counter_verifier_override(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. consensus\nB. challenge",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hipp", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1},
            {"child_id": "direct", "child_index": 4, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "elim", "child_index": 5, "prompt_kind": "option_elimination_challenge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

    def test_option_matrix_disagreement_can_trigger_verifier_when_route_voi_explores(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. consensus\nB. matrix",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hipp", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1},
            {"child_id": "direct", "child_index": 4, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 5,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "route_arbitrator_trusted": False,
                "candidate_verifier_state": "not_verified",
                "route_value_of_information_gate_status": "continue_exploration",
                "route_value_of_information_recommended_action": "continue_exploration",
            },
            {"child_id": "matrix", "child_index": 6, "prompt_kind": "option_matrix_reasoner_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(
            os.environ,
            {
                "HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1",
                "HLE_ENABLE_OPTION_SWEEP_VOI_COUNTER": "1",
                "HLE_ALLOW_STRUCTURAL_COUNTER_OVER_BASELINE_CONSENSUS": "1",
            },
        ):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_called_once()
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "matrix")
        self.assertEqual(selection["selected_answer"], "B")

    def test_structural_option_audit_can_trigger_verifier_when_route_voi_explores(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. consensus\nB. structural audit",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hipp", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1},
            {"child_id": "direct", "child_index": 4, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 5,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "route_arbitrator_trusted": False,
                "candidate_verifier_state": "not_verified",
                "route_value_of_information_gate_status": "continue_exploration",
                "route_value_of_information_recommended_action": "continue_exploration",
            },
            {"child_id": "audit", "child_index": 6, "prompt_kind": "structural_option_audit_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_called_once()
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "audit")
        self.assertEqual(selection["selected_answer"], "B")

    def test_trusted_preserve_route_blocks_counter_verifier_override(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. trusted route\nB. noisy counter",
        }
        attempts = [
            {"child_id": "direct", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
            {"child_id": "matrix", "child_index": 2, "prompt_kind": "option_matrix_reasoner_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
            {
                "child_id": "route",
                "child_index": 3,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "route_arbitrator_trusted": True,
                "route_value_score": 9.0,
                "route_value_of_information_gate_status": "preserve_route",
                "route_value_of_information_recommended_action": "preserve_route",
            },
            {"child_id": "audit", "child_index": 4, "prompt_kind": "structural_option_audit_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "route_value_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "route")
        self.assertEqual(selection["selected_answer"], "A")

    def test_low_margin_raw_budget_route_can_defer_to_structural_counter_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. baseline route\nB. structural counter",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hipp", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1},
            {"child_id": "direct", "child_index": 4, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 5,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_arbitrator_selected_route": "raw_budget_consensus",
                "route_arbitrator_trust_reason": "strong_raw_budget_consensus",
                "route_value_confidence": "high",
                "route_value_score": 6.0,
                "route_value_score_margin": 0.8,
                "route_value_of_information_gate_status": "preserve_route",
                "route_value_of_information_recommended_action": "preserve_route",
            },
            {"child_id": "audit", "child_index": 6, "prompt_kind": "structural_option_audit_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_called_once()
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "audit")
        self.assertEqual(selection["selected_answer"], "B")

    def test_raw_budget_route_can_defer_to_budgeted_retrieval_counter_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Medicine",
            "_question": "Which option is correct?\nA. raw budget route\nB. budgeted retrieval counter",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 3,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_arbitrator_selected_route": "raw_budget_consensus",
                "route_arbitrator_trust_reason": "strong_raw_budget_consensus",
                "route_value_confidence": "high",
                "route_value_score": 12.8,
                "route_value_score_margin": 3.4,
                "route_value_of_information_gate_status": "preserve_route",
                "route_value_of_information_recommended_action": "preserve_route",
            },
            {
                "child_id": "hippo-budget",
                "child_index": 4,
                "prompt_kind": "hipporag_preserve_selector_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "budget_matched": True,
                "budget_strong_consensus": True,
                "budget_top_candidate_vote_count": 3,
                "context_answer_option_hash": "hb",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_called_once()
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "hippo-budget")
        self.assertEqual(selection["selected_answer"], "B")

    def test_high_margin_raw_budget_route_does_not_defer_to_structural_counter(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. baseline route\nB. structural counter",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hipp", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1},
            {
                "child_id": "route",
                "child_index": 4,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_arbitrator_selected_route": "raw_budget_consensus",
                "route_arbitrator_trust_reason": "strong_raw_budget_consensus",
                "route_value_confidence": "high",
                "route_value_score": 6.0,
                "route_value_score_margin": 2.5,
                "route_value_of_information_gate_status": "preserve_route",
                "route_value_of_information_recommended_action": "preserve_route",
            },
            {"child_id": "audit", "child_index": 5, "prompt_kind": "structural_option_audit_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "route_value_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "route")
        self.assertEqual(selection["selected_answer"], "A")

    def test_baseline_family_majority_blocks_structural_counter_by_default(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. consensus\nB. isolated counter",
        }
        attempts = [
            {"child_id": "rawp", "child_index": 1, "prompt_kind": "raw_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "rawbp", "child_index": 2, "prompt_kind": "raw_budget_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "hipp", "child_index": 3, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "A", "parsed_answer_hash": "ha", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1},
            {"child_id": "hippb", "child_index": 4, "prompt_kind": "hipporag_preserve_selector_answer", "parsed_answer": "B", "parsed_answer_hash": "hb", "preserve_context_char_count": 200, "preserve_selected_doc_count": 1, "budget_matched": True},
            {"child_id": "matrix", "child_index": 5, "prompt_kind": "option_matrix_reasoner_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}, clear=False):
            with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

    def test_route_value_continue_exploration_does_not_priority_select_route(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. route",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 3,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "candidate_verifier_state": "verified",
                "route_arbitrator_trusted": True,
                "route_value_confidence": "high",
                "route_value_of_information_gate_status": "continue_exploration",
                "route_value_of_information_recommended_action": "continue_exploration",
            },
        ]

        selection = _select_recursive_child_answer(
            problem=problem,
            attempts=attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

    def test_option_evidence_conflict_is_diagnostic_by_default(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. evidence-backed",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "c3",
                "child_index": 3,
                "prompt_kind": "mc_option_evidence_scorer_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "private_option_evidence_context": "Option B: directly supported by evidence.",
            },
        ]
        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_child_id"], "c1")
        self.assertEqual(selection["selected_answer"], "A")

    def test_option_evidence_conflict_can_use_evidence_aware_arbitrator_when_enabled(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. evidence-backed",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "c3",
                "child_index": 3,
                "prompt_kind": "mc_option_evidence_scorer_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "private_option_evidence_context": "Option B: directly supported by evidence.",
            },
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_OPTION_EVIDENCE_ARBITRATOR": "1", "HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        self.assertEqual(selection["selection_method"], "option_evidence_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "c3")
        self.assertEqual(selection["selected_answer"], "B")

    def test_option_sweep_candidates_do_not_trigger_counter_verifier_by_default(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. synthetic\nC. synthetic",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "mc_option_sweep_candidate", "parsed_answer": "B", "parsed_answer_hash": "hb"},
            {"child_id": "c4", "child_index": 4, "prompt_kind": "mc_option_sweep_candidate", "parsed_answer": "C", "parsed_answer_hash": "hc"},
        ]
        with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_child_id"], "c1")
        self.assertEqual(selection["selected_answer"], "A")

    def test_hipporag_context_child_requires_agreement_by_default(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. retrieval",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "hipporag_context_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        selection = _select_recursive_child_answer(
            problem=problem,
            attempts=attempts,
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )

        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_child_id"], "c1")

        with patch.dict(os.environ, {"HLE_ENABLE_BROAD_AGENT_HIPPORAG_PRIORITY": "1"}):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )
        self.assertEqual(selection["selection_method"], "hipporag_context_priority")
        self.assertEqual(selection["selected_child_id"], "c3")
        self.assertEqual(selection["selected_answer"], "B")

        with patch.dict(os.environ, {"HLE_ENABLE_BROAD_AGENT_HIPPORAG_PRIORITY": "1", "HLE_DISABLE_AGENT_HIPPORAG_PRIORITY": "1"}):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_child_id"], "c1")

    def test_source_grounded_verifier_can_precede_counter_challenge_when_broad_mode_enabled(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. evidence-backed",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "counter_assumption_challenge_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c4", "child_index": 4, "prompt_kind": "evidence_bridge_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
            {"child_id": "c5", "child_index": 5, "prompt_kind": "mc_option_sweep_candidate", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_BROAD_SOURCE_GROUNDED_MC": "1"}), patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"choice":2}',
        ):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
                evidence_context="[Evidence 1] source=wikipedia; title=X; snippet=Option B is directly supported.",
            )

        self.assertEqual(selection["selection_method"], "source_grounded_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "c4")
        self.assertEqual(selection["selected_answer"], "B")

    def test_critic_synthesis_child_adds_distinct_critic_candidate(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. critic",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "recursive_assumption_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
        ]

        def fake_child_attempt(**kwargs):
            return {
                "child_id": "c4",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "prediction_hash": "pb",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch("assumption_os.hle_smoke_eval._run_child_attempt", side_effect=fake_child_attempt):
            attempt, summary = _maybe_run_critic_synthesis_child(
                problem=problem,
                attempts=attempts,
                evidence_context="",
                base_model="gpt-5.4-mini",
                critic_model="gpt-5.5",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(attempt["prompt_kind"], "critic_synthesis_answer")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["critic_disagreed_with_majority"])

        attempts.append(attempt)
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="gpt-5.5",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_child_id"], "c4")

    def test_mc_option_sweep_adds_missing_finite_labels_for_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. missing correct\nC. distractor",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
        ]

        added, summary = _maybe_add_mc_option_sweep_candidates(problem=problem, attempts=attempts)

        self.assertEqual(summary["status"], "activated")
        self.assertEqual(summary["added_candidate_count"], 2)
        self.assertEqual([row["parsed_answer"] for row in added], ["B", "C"])

        attempts.extend(added)
        with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
            selection = _select_recursive_child_answer(
                problem=problem,
                attempts=attempts,
                model="gpt-5.5",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

        with patch.dict(os.environ, {"HLE_ENABLE_OPTION_SWEEP_COUNTER_TRIGGER": "1", "HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}'):
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="gpt-5.5",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

    def test_route_voi_continue_allows_full_option_space_counter_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. hidden correct\nC. distractor",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 3,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_value_confidence": "high",
                "route_value_of_information_gate_status": "continue_exploration",
                "route_value_of_information_recommended_action": "continue_exploration",
            },
        ]
        added, summary = _maybe_add_mc_option_sweep_candidates(problem=problem, attempts=attempts)
        self.assertEqual(summary["status"], "activated")
        attempts.extend(added)

        with patch.dict(
            os.environ,
            {
                "HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1",
                "HLE_ENABLE_OPTION_SWEEP_VOI_COUNTER": "1",
            },
        ):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="gpt-5.4-mini",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        self.assertEqual(call_model.call_count, 1)
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_answer"], "B")

    def test_route_voi_continue_option_sweep_counter_is_opt_in(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. hidden correct\nC. distractor",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 3,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_value_confidence": "high",
                "route_value_of_information_gate_status": "continue_exploration",
                "route_value_of_information_recommended_action": "continue_exploration",
            },
        ]
        added, _ = _maybe_add_mc_option_sweep_candidates(problem=problem, attempts=attempts)
        attempts.extend(added)

        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="gpt-5.4-mini",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        self.assertEqual(call_model.call_count, 0)
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_answer"], "A")

    def test_route_voi_preserve_blocks_full_option_space_counter_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. hidden correct\nC. distractor",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {
                "child_id": "route",
                "child_index": 3,
                "prompt_kind": "route_arbitrator_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
                "candidate_verifier_state": "verified",
                "candidate_verifier_trust": "route_arbitrator_evidence_gate",
                "route_arbitrator_trusted": True,
                "route_value_confidence": "high",
                "route_value_of_information_gate_status": "preserve_route",
                "route_value_of_information_recommended_action": "preserve_route",
            },
        ]
        added, _ = _maybe_add_mc_option_sweep_candidates(problem=problem, attempts=attempts)
        attempts.extend(added)

        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model", return_value='{"choice":2}') as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="gpt-5.4-mini",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        self.assertEqual(call_model.call_count, 0)
        self.assertEqual(selection["selection_method"], "route_value_verifier_choice")
        self.assertEqual(selection["selected_answer"], "A")

    def test_verified_or_abstain_prefers_raw_budget_over_single_raw_fallback(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. raw\nB. budget",
        }
        attempts = [
            {
                "child_id": "raw",
                "child_index": 1,
                "prompt_kind": "raw_preserve_selector_answer",
                "parsed_answer": "A",
                "parsed_answer_hash": "ha",
            },
            {
                "child_id": "budget",
                "child_index": 2,
                "prompt_kind": "raw_budget_preserve_selector_answer",
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
            },
        ]

        selection = _apply_verified_or_abstain_selection(
            problem=problem,
            attempts=attempts,
            selection={
                "selection_method": "normalized_majority",
                "selected_child_id": "raw",
                "selected_answer": "A",
            },
        )

        self.assertEqual(selection["selection_method"], "verified_or_abstain_direct_fallback")
        self.assertEqual(selection["selected_child_id"], "budget")
        self.assertEqual(selection["selected_answer"], "B")
        self.assertEqual(selection["verified_or_abstain_gate"]["fallback_prompt_kind"], "raw_budget_preserve_selector_answer")

    def test_forced_alternative_alone_does_not_trigger_counter_verifier(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. majority\nB. forced alternative",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "forced_alternative_answer", "parsed_answer": "B", "parsed_answer_hash": "hb"},
        ]
        with patch.dict(os.environ, {"HLE_ENABLE_COUNTER_ASSUMPTION_VERIFIER": "1"}):
            with patch("assumption_os.hle_smoke_eval._call_model") as call_model:
                selection = _select_recursive_child_answer(
                    problem=problem,
                    attempts=attempts,
                    model="m",
                    eval_id="e",
                    call_id="c",
                    logger=None,
                    timeout=1,
                    max_tokens=32,
                )

        call_model.assert_not_called()
        self.assertEqual(selection["selection_method"], "normalized_majority")
        self.assertEqual(selection["selected_child_id"], "c1")
        self.assertEqual(selection["selected_answer"], "A")

    def test_forced_alternative_runs_after_option_elimination_confirms_majority(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "Which option is correct?\nA. wrong\nB. right\nC. distractor",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
            {"child_id": "c3", "child_index": 3, "prompt_kind": "option_elimination_challenge_answer", "parsed_answer": "A", "parsed_answer_hash": "ha"},
        ]

        def fake_child_attempt(**kwargs):
            return {
                "child_id": "c4",
                "child_index": kwargs["child_index"],
                "prompt_kind": kwargs["spec"]["prompt_kind"],
                "parsed_answer": "B",
                "parsed_answer_hash": "hb",
                "prediction_hash": "pb",
                "latency_sec": 0.1,
                "status": "answered",
            }

        with patch.dict("os.environ", {"HLE_ENABLE_FORCED_ALTERNATIVE": "1"}), patch(
            "assumption_os.hle_smoke_eval._run_child_attempt",
            side_effect=fake_child_attempt,
        ):
            attempt, summary = _maybe_run_forced_alternative_challenge(
                problem=problem,
                attempts=attempts,
                option_elimination_summary={"status": "activated", "challenge_disagreed_with_majority": False},
                evidence_context="",
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=1,
                max_tokens=32,
            )

        self.assertEqual(attempt["prompt_kind"], "forced_alternative_answer")
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["challenge_disagreed_with_majority"])
        self.assertTrue(summary["answer_is_allowed_alternative"])

    def test_mc_candidate_claim_verifier_planner_and_ambiguous_guard(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_type": "multipleChoice",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Use the described arithmetic process to determine the final value.\nA. 41\nB. 42\nC. 43\nD. 44",
        }
        attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "A", "status": "answered"},
            {"child_id": "c2", "child_index": 2, "prompt_kind": "agent_context_answer", "parsed_answer": "A", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"operation":"evaluate","expression":"6*7","equation":"","variable":"x","modulus":""}',
        ):
            summary = _apply_math_candidate_claim_verifier(
                problem,
                attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(summary["backend"], "sympy_mc_option_planner")
        self.assertEqual(summary["verified_count"], 1)
        self.assertEqual(summary["refuted_count"], 2)
        self.assertEqual(summary["underlying_model_calls"], 1)
        self.assertEqual(attempts[-1]["parsed_answer"], "B")

        ambiguous = {
            **problem,
            "_question": "Use the described arithmetic process to determine the final value.\nA. 42\nB. 42\nC. 43\nD. 44",
        }
        ambiguous_attempts = [
            {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "C", "status": "answered"},
        ]
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            return_value='{"operation":"evaluate","expression":"6*7","equation":"","variable":"x","modulus":""}',
        ):
            ambiguous_summary = _apply_math_candidate_claim_verifier(
                ambiguous,
                ambiguous_attempts,
                model="m",
                timeout=1,
                max_tokens=64,
            )

        self.assertEqual(ambiguous_summary["status"], "ambiguous_option_match")
        self.assertEqual(len(ambiguous_attempts), 1)
        self.assertNotIn("candidate_verifier_state", ambiguous_attempts[0])

    def test_agent_exact_answer_canonicalizer_is_narrow(self):
        problem = {"answer_type": "exactMatch"}

        answer, summary = _canonicalize_exact_answer_candidate(problem, "The answer is $42$.")
        self.assertEqual(answer, "42")
        self.assertTrue(summary["changed"])

        title, title_summary = _canonicalize_exact_answer_candidate(problem, "Attention Is All You Need")
        self.assertEqual(title, "Attention Is All You Need")
        self.assertFalse(title_summary["changed"])

        mc_answer, mc_summary = _canonicalize_exact_answer_candidate({"answer_type": "multipleChoice"}, "The answer is B.")
        self.assertEqual(mc_answer, "The answer is B.")
        self.assertFalse(mc_summary["changed"])

    def test_multiple_choice_canonicalizer_maps_option_text_without_first_letter_fallback(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "Which paper introduced the Transformer?\nA. ResNet\nB. Attention Is All You Need\nC. AlexNet",
        }

        answer, summary = _canonicalize_multiple_choice_answer(problem, "Attention Is All You Need")
        self.assertEqual(answer, "B")
        self.assertTrue(summary["changed"])

        label, label_summary = _canonicalize_multiple_choice_answer(problem, "The answer is B.")
        self.assertEqual(label, "B")
        self.assertTrue(label_summary["changed"])

        ambiguous, ambiguous_summary = _canonicalize_multiple_choice_answer(problem, "Attention")
        self.assertEqual(ambiguous, "Attention")
        self.assertFalse(ambiguous_summary["changed"])

        scored_problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_hash": "ah",
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "AI",
            "_question": problem["_question"],
            "_answer": "B",
        }
        row = _score_prediction(
            problem=scored_problem,
            model="m",
            variant="raw",
            prediction='{"answer":"Attention Is All You Need"}',
        )
        self.assertTrue(row["correct"])

        scored_problem["_answer"] = "Attention Is All You Need"
        row = _score_prediction(
            problem=scored_problem,
            model="m",
            variant="raw",
            prediction='{"answer":"B"}',
        )
        self.assertTrue(row["correct"])

    def test_math_tool_deterministic_and_plan_execution(self):
        math_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute $2^5 + 10$.",
        }
        non_math_problem = {
            "answer_type": "exactMatch",
            "category": "History",
            "raw_subject": "History",
            "_question": "Who wrote the book?",
        }

        self.assertTrue(_should_run_math_tool_child(math_problem))
        self.assertFalse(_should_run_math_tool_child(non_math_problem))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_math_tool_child_timeout(7200), 7200)
            self.assertIsNone(_math_tool_child_timeout(None))
        with patch.dict(os.environ, {"HLE_MATH_TOOL_CHILD_TIMEOUT_SEC": "0"}):
            self.assertIsNone(_math_tool_child_timeout(7200))
        with patch.dict(os.environ, {"HLE_MATH_TOOL_CHILD_TIMEOUT_SEC": "180"}):
            self.assertEqual(_math_tool_child_timeout(7200), 180)
        result = _deterministic_math_tool_answer(math_problem)
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["confidence"], "verified_symbolic")

        solve_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Solve $x^2 - 4 = 0$.",
        }
        solve_result = _deterministic_math_tool_answer(solve_problem)
        self.assertEqual(solve_result["answer"], "-2, 2")
        self.assertEqual(solve_result["source"], "deterministic_equation_solver")
        self.assertEqual(solve_result["operation"], "solve")

        root_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Find the roots of $x^2 - 4$.",
        }
        root_result = _deterministic_math_tool_answer(root_problem)
        self.assertEqual(root_result["answer"], "-2, 2")
        self.assertEqual(root_result["source"], "deterministic_root_solver")

        simplify_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Simplify $x + x$.",
        }
        simplify_result = _deterministic_math_tool_answer(simplify_problem)
        self.assertEqual(simplify_result["answer"], "2*x")
        self.assertEqual(simplify_result["source"], "deterministic_transform_solver")
        self.assertEqual(simplify_result["operation"], "simplify")

        derivative_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Find the derivative of $x^3$ with respect to x.",
        }
        derivative_result = _deterministic_math_tool_answer(derivative_problem)
        self.assertEqual(derivative_result["answer"], "3*x**2")
        self.assertEqual(derivative_result["operation"], "differentiate")

        limit_problem = {
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Evaluate the limit of $sin(x)/x$ as x approaches 0.",
        }
        limit_result = _deterministic_math_tool_answer(limit_problem)
        self.assertEqual(limit_result["answer"], "1")
        self.assertEqual(limit_result["operation"], "limit")

        factor_result = _deterministic_math_tool_answer({
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Factor $x^2 - 4$.",
        })
        self.assertEqual(factor_result["answer"], "(x - 2)*(x + 2)")
        self.assertEqual(factor_result["operation"], "factor")

        plan_result = _execute_math_tool_plan_text(
            '{"operation":"solve","equation":"x**2 - 4 = 0","variable":"x","modulus":""}'
        )
        self.assertEqual(plan_result["answer"], "-2, 2")

        factor_plan_result = _execute_math_tool_plan_text(
            '{"operation":"factor","expression":"x**2 - 4","variable":"x","modulus":""}'
        )
        self.assertEqual(factor_plan_result["answer"], "(x - 2)*(x + 2)")

        selected_math_tool = _select_recursive_child_answer(
            problem={
                "id_hash": "pid",
                "question_hash": "qid",
                "answer_type": "exactMatch",
                "category": "Math",
                "raw_subject": "Mathematics",
                "_question": "Factor $x^2 - 4$.",
            },
            attempts=[
                {"child_id": "d", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "x**2 - 4", "status": "answered"},
                {
                    "child_id": "m",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "(x - 2)*(x + 2)",
                    "status": "answered",
                    "tool_confidence": "verified_symbolic",
                    "tool_source": "deterministic_transform_solver",
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=1,
            max_tokens=32,
        )
        self.assertEqual(selected_math_tool["selection_method"], "verified_math_tool_priority")
        self.assertEqual(selected_math_tool["selected_answer"], "(x - 2)*(x + 2)")

        symbolic_evaluate = _execute_math_tool_plan_text(
            '{"operation":"evaluate","expression":"x + x","variable":"x","modulus":""}'
        )
        self.assertEqual(symbolic_evaluate["operation"], "simplify")
        self.assertEqual(symbolic_evaluate["answer"], "2*x")
        self.assertEqual(symbolic_evaluate["coerced_from_operation"], "evaluate")

        multiplan_result = _execute_math_tool_plan_text(
            '{"plans":[{"operation":"none"},{"operation":"factor","expression":"x**2 - 4","variable":"x"}]}'
        )
        self.assertEqual(multiplan_result["operation"], "factor")
        self.assertEqual(multiplan_result["answer"], "(x - 2)*(x + 2)")
        self.assertEqual(multiplan_result["plan_index"], 1)
        self.assertEqual(multiplan_result["plan_count"], 2)

        consensus_result = _execute_math_tool_plan_text(
            '{"plans":['
            '{"operation":"evaluate","expression":"1/2","variable":"x"},'
            '{"operation":"python","expression":"Fraction(1, 2)","variable":"x"},'
            '{"operation":"evaluate","expression":"1/3","variable":"x"}'
            ']}'
        )
        self.assertEqual(consensus_result["confidence"], "verified_symbolic_consensus")
        self.assertEqual(consensus_result["answer"], "1/2")
        self.assertEqual(consensus_result["plan_agreement_count"], 2)

        consensus_selection = _select_recursive_child_answer(
            problem=math_problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "41"},
                {
                    "child_id": "c3",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "42",
                    "status": "answered",
                    "tool_confidence": "verified_symbolic_consensus",
                    "tool_source": "llm_planner",
                    "tool_summary": {"plan_agreement_count": 2, "plan_success_count": 2},
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=None,
            max_tokens=32,
        )
        self.assertEqual(consensus_selection["selection_method"], "verified_math_tool_priority")
        self.assertEqual(consensus_selection["selected_answer"], "42")

        single_plan_llm_selection = _select_recursive_child_answer(
            problem=math_problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "41"},
                {
                    "child_id": "c3",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "42",
                    "status": "answered",
                    "tool_confidence": "verified_symbolic",
                    "tool_source": "llm_planner",
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=None,
            max_tokens=32,
        )
        self.assertNotEqual(single_plan_llm_selection["selection_method"], "verified_math_tool_priority")

        refuted_consensus_selection = _select_recursive_child_answer(
            problem=math_problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "41"},
                {
                    "child_id": "c3",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "42",
                    "status": "answered",
                    "tool_confidence": "verified_symbolic_consensus",
                    "tool_source": "llm_planner",
                    "tool_summary": {"plan_agreement_count": 2, "plan_success_count": 2},
                    "candidate_verifier_state": "refuted",
                    "candidate_verifier_backend": "sympy_candidate_reference_planner",
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=None,
            max_tokens=32,
        )
        self.assertNotEqual(refuted_consensus_selection["selection_method"], "verified_math_tool_priority")

        weak_reference_consensus_selection = _select_recursive_child_answer(
            problem=math_problem,
            attempts=[
                {"child_id": "c1", "child_index": 1, "prompt_kind": "direct_short_answer", "parsed_answer": "41"},
                {"child_id": "c2", "child_index": 2, "prompt_kind": "constraint_checked_answer", "parsed_answer": "41"},
                {
                    "child_id": "c3",
                    "child_index": 9001,
                    "prompt_kind": "math_tool_answer",
                    "parsed_answer": "42",
                    "status": "answered",
                    "tool_confidence": "verified_symbolic_consensus",
                    "tool_source": "llm_planner",
                    "tool_summary": {"plan_agreement_count": 2, "plan_success_count": 2},
                    "candidate_verifier_trust": "weak_llm_reference_planner",
                },
            ],
            model="m",
            eval_id="e",
            call_id="c",
            logger=None,
            timeout=None,
            max_tokens=32,
        )
        self.assertNotEqual(weak_reference_consensus_selection["selection_method"], "verified_math_tool_priority")

        all_failed_result = _execute_math_tool_plan_text(
            '{"plans":[{"operation":"none"},{"operation":"evaluate","expression":"not a safe expression","variable":"x"}]}'
        )
        self.assertEqual(all_failed_result["confidence"], "abstain")
        self.assertIn(all_failed_result["reason"], {"all_candidate_plans_failed", "planner_abstained", "equation_solve_failed"})
        self.assertEqual(all_failed_result["plan_count"], 2)

        python_comb = _execute_math_tool_plan_text(
            '{"operation":"python","expression":"comb(10, 3)","variable":"x"}'
        )
        self.assertEqual(python_comb["answer"], "120")
        self.assertEqual(python_comb["operation"], "python")

        python_sum = _execute_math_tool_plan_text(
            '{"operation":"python","expression":"sum(i*i for i in range(4))","variable":"x"}'
        )
        self.assertEqual(python_sum["answer"], "14")

        repair_problem = {
            "id_hash": "repair_pid",
            "question_hash": "repair_qid",
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
            "_question": "Compute the finite count described by the combinatorial process.",
        }
        with patch(
            "assumption_os.hle_smoke_eval._call_model",
            side_effect=[
                '{"plans":[{"operation":"none","expression":"","equation":"","variable":"x"}]}',
                '{"plans":[{"operation":"python","expression":"comb(10, 3)","variable":"x"}]}',
            ],
        ) as call_model:
            repair_attempt = _run_math_tool_attempt(
                problem=repair_problem,
                model="m",
                eval_id="e",
                call_id="c",
                logger=None,
                timeout=None,
                max_tokens=64,
            )
        self.assertEqual(call_model.call_count, 2)
        self.assertEqual(repair_attempt["status"], "answered")
        self.assertEqual(repair_attempt["parsed_answer"], "120")
        self.assertEqual(repair_attempt["tool_confidence"], "verified_symbolic")
        self.assertEqual(repair_attempt["underlying_model_calls"], 2)
        self.assertEqual(repair_attempt["tool_summary"]["source"], "llm_planner_repair")
        self.assertEqual(repair_attempt["tool_summary"]["reason"], None)

        python_fraction = _execute_math_tool_plan_text(
            '{"operation":"python","expression":"Fraction(1, 3) + Fraction(1, 6)","variable":"x"}'
        )
        self.assertEqual(python_fraction["answer"], "1/2")

        unsafe_python = _execute_math_tool_plan_text(
            '{"operation":"python","expression":"__import__(\\\"os\\\").system(\\\"echo bad\\\")","variable":"x"}'
        )
        self.assertEqual(unsafe_python["confidence"], "abstain")
        self.assertEqual(unsafe_python["reason"], "unsafe_python_expression")

        repaired_symbolic_evaluate = _execute_math_tool_plan_text(
            '{"operation":"evaluate","expression":"f(x) = x + x where x is real","variable":"x","modulus":""}'
        )
        self.assertEqual(repaired_symbolic_evaluate["operation"], "simplify")
        self.assertEqual(repaired_symbolic_evaluate["answer"], "2*x")

        derivative = _execute_math_tool_plan_text(
            '{"operation":"differentiate","expression":"x**3","variable":"x","order":"1"}'
        )
        self.assertEqual(derivative["answer"], "3*x**2")

        integral = _execute_math_tool_plan_text(
            '{"operation":"integrate","expression":"x","variable":"x","lower":"0","upper":"2"}'
        )
        self.assertEqual(integral["answer"], "2")

        limit = _execute_math_tool_plan_text(
            '{"operation":"limit","expression":"sin(x)/x","variable":"x","point":"0"}'
        )
        self.assertEqual(limit["answer"], "1")

    def test_evidence_bridge_query_and_context_helpers(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "raw_subject": "Computer Science",
            "category": "Computer Science/AI",
            "answer_type": "exactMatch",
            "_question": 'Which paper introduced "Attention Is All You Need" in neural machine translation?',
        }
        queries = _candidate_evidence_queries(problem)

        self.assertTrue(any("Attention Is All You Need" in query for query in queries))
        self.assertEqual(_clean_evidence_text("<span>Transformer</span>&nbsp;model"), "Transformer model")
        context = _format_evidence_context(
            [{"title": "Attention Is All You Need", "snippet": "Transformer architecture."}],
            max_chars=200,
        )
        self.assertIn("source=wikipedia", context)
        self.assertIn("Transformer architecture", context)

        docs_by_query = {
            "Computer Science": [
                {"title": "Distractor", "snippet": "Unrelated note."},
            ],
            "Computer Science/AI": [
                {"title": "Machine learning", "snippet": "Generic AI overview."},
            ],
            "Attention Is All You Need": [
                {"title": "Attention Is All You Need", "snippet": "Transformer neural machine translation architecture."},
            ],
            "Which Applied Mathematics": [
                {"title": "Wrong", "snippet": "Wrong."},
            ],
        }

        def fake_search(query, *, limit, timeout):
            return docs_by_query.get(query, [])

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
            bridge_context, summary = _build_hle_evidence_bridge_context(
                problem=problem,
                eval_id="e",
                call_id="c",
                model="m",
                logger=None,
                candidate_answers=["Attention Is All You Need"],
            )

        self.assertEqual(summary["selection_policy"], "answer_bearing_hipporag_style_associative_rerank")
        self.assertEqual(summary["answer_bearing_certificate"]["status"], "answer_bearing")
        self.assertIn("Attention Is All You Need", bridge_context)

    def test_evidence_bridge_uses_domain_fallback_for_specialized_subjects(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "raw_subject": "Biology/Medicine",
            "category": "Biology/Medicine",
            "answer_type": "exactMatch",
            "_question": 'Which option best matches "adenosine receptor antagonist" in sleep pharmacology?',
        }

        def fake_wikipedia_search(query, *, limit, timeout):
            return []

        def fake_domain_search(query, *, problem, limit, timeout):
            return [
                {
                    "title": "Caffeine and adenosine receptors",
                    "snippet": "Caffeine is an antagonist at adenosine receptors in sleep pharmacology and promotes wakefulness.",
                    "source": "pubmed",
                }
            ]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_wikipedia_search), patch(
            "assumption_os.hle_smoke_eval._domain_evidence_search",
            side_effect=fake_domain_search,
        ):
            bridge_context, summary = _build_hle_evidence_bridge_context(
                problem=problem,
                eval_id="e",
                call_id="c",
                model="m",
                logger=None,
                candidate_answers=["Caffeine"],
            )

        self.assertEqual(summary["source"], "wikipedia_plus_domain_search")
        self.assertEqual(summary["source_counts"]["pubmed"], 1)
        self.assertEqual(summary["answer_bearing_certificate"]["status"], "answer_bearing")
        self.assertIn("source=pubmed", bridge_context)
        self.assertIn("adenosine receptors", bridge_context)

    def test_hipporag_baseline_trace_and_prompt_are_control_only(self):
        problem = {
            "answer_type": "multipleChoice",
            "category": "Science",
            "raw_subject": "Physics",
            "_question": "secret gated question",
        }
        plan = {
            "stages": {
                "hipporag_context_retrieval": {"status": "activated"},
                "hipporag_associative_rerank": {"status": "activated"},
                "prompt_builder": {"status": "activated"},
            },
            "prompt_context": "[Evidence 1] source=wikipedia; title=Lens; snippet=Optics.",
        }
        trace = _module_trace(problem, variant="hipporag_baseline", agent_plan=plan)
        by_module = {item["module"]: item for item in trace}
        prompt = _prompt_for(problem, variant="hipporag_baseline", agent_plan=plan)

        self.assertEqual(by_module["hipporag_context_retrieval"]["status"], "activated")
        self.assertEqual(by_module["hipporag_associative_rerank"]["status"], "activated")
        self.assertEqual(by_module["assumption_graph_retrieval"]["status"], "not_applicable")
        self.assertEqual(by_module["world_model_router"]["status"], "not_applicable")
        self.assertIn("does not include the gold answer or Assumption Agent graph", prompt)

    def test_hipporag_style_rerank_prefers_query_overlap(self):
        problem = {
            "raw_subject": "Computer Science",
            "category": "Computer Science",
            "_question": "Which architecture introduced attention for sequence transduction?",
        }
        docs = [
            {"title": "Unrelated", "snippet": "A historical article."},
            {"title": "Attention Is All You Need", "snippet": "Transformer architecture for sequence transduction."},
        ]
        ranked = _hipporag_style_rerank(problem, docs)

        self.assertEqual(ranked[0]["doc"]["title"], "Attention Is All You Need")

    def test_collect_existing_hle_problem_hashes_reads_json_and_jsonl(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hle_text_smoke_a.json").write_text(
                __import__("json").dumps({"rows": [{"problem_id_hash": "p1"}]}),
                encoding="utf-8",
            )
            (root / "hle_text_smoke_b.jsonl").write_text(
                __import__("json").dumps({"problem_id_hash": "p2"}) + "\n",
                encoding="utf-8",
            )

            hashes = _collect_existing_hle_problem_hashes(root=root, artifact_glob="hle_text_smoke*.json*")

        self.assertEqual(hashes, {"p1", "p2"})

    def test_collect_existing_hle_problem_hashes_reads_parallel_run_default_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "phase four" / "assumption_graph" / "paper_readiness_20260604" / "hle_parallel_runs"
            run_dir.mkdir(parents=True)
            (run_dir / "hle_old.json").write_text(
                __import__("json").dumps({"rows": [{"problem_id_hash": "parallel-old"}]}),
                encoding="utf-8",
            )

            hashes = _collect_existing_hle_problem_hashes(
                root=root,
                artifact_glob="phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/hle*.json*",
            )

        self.assertEqual(hashes, {"parallel-old"})

    def test_collect_existing_hle_problem_hashes_can_use_stale_frozen_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hle_text_smoke_a.json").write_text(
                __import__("json").dumps({"rows": [{"problem_id_hash": "fresh"}]}),
                encoding="utf-8",
            )
            cache = root / "frozen_hash_cache.json"
            cache.write_text(
                __import__("json").dumps(
                    {
                        "artifact_glob": "hle_text_smoke*.json*",
                        "manifest": [{"path_hash": "stale"}],
                        "problem_id_hashes": ["frozen"],
                        "raw_content_persisted": False,
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HLE_EXISTING_HASH_CACHE_PATH": str(cache)}, clear=False):
                default_hashes = _collect_existing_hle_problem_hashes(
                    root=root,
                    artifact_glob="hle_text_smoke*.json*",
                )
            cache.write_text(
                __import__("json").dumps(
                    {
                        "artifact_glob": "hle_text_smoke*.json*",
                        "manifest": [{"path_hash": "stale"}],
                        "problem_id_hashes": ["frozen"],
                        "raw_content_persisted": False,
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "HLE_EXISTING_HASH_CACHE_PATH": str(cache),
                    "HLE_EXISTING_HASH_CACHE_ALLOW_STALE": "1",
                },
                clear=False,
            ):
                frozen_hashes = _collect_existing_hle_problem_hashes(
                    root=root,
                    artifact_glob="hle_text_smoke*.json*",
                )

        self.assertEqual(default_hashes, {"fresh"})
        self.assertEqual(frozen_hashes, {"frozen"})

    def test_control_comparison_reports_agent_vs_controls(self):
        rows = [
            {"model": "m", "variant": "raw", "problem_id_hash": "p1", "correct": False, "answer_type": "multipleChoice"},
            {"model": "m", "variant": "raw", "problem_id_hash": "p2", "correct": True, "answer_type": "multipleChoice"},
            {"model": "m", "variant": "hipporag_baseline", "problem_id_hash": "p1", "correct": False, "answer_type": "multipleChoice"},
            {"model": "m", "variant": "hipporag_baseline", "problem_id_hash": "p2", "correct": False, "answer_type": "multipleChoice"},
            {"model": "m", "variant": "assumption_agent_recursive_verify", "problem_id_hash": "p1", "correct": True, "answer_type": "multipleChoice"},
            {"model": "m", "variant": "assumption_agent_recursive_verify", "problem_id_hash": "p2", "correct": False, "answer_type": "multipleChoice"},
        ]
        comparison = _control_comparison(rows)

        self.assertEqual(comparison["m"]["agent_vs_raw"]["agent_minus_control_accuracy"], 0.0)
        self.assertEqual(comparison["m"]["agent_vs_raw"]["agent_unique_correct_count"], 1)
        self.assertEqual(comparison["m"]["agent_vs_raw"]["control_unique_correct_count"], 1)
        self.assertEqual(comparison["m"]["agent_vs_hipporag_baseline"]["agent_minus_control_accuracy"], 0.5)

    def test_hle_agent_context_gate_blocks_weak_exact_match_injection(self):
        self.assertFalse(_should_use_agent_context(
            answer_type="exactMatch",
            top_score=0.32,
            formal_hit_count=0,
            structural_hit_count=1,
            strong_structural_hit_count=0,
            expected_utility=0.55,
        ))
        self.assertTrue(_should_use_agent_context(
            answer_type="exactMatch",
            top_score=0.12,
            formal_hit_count=1,
            structural_hit_count=0,
            strong_structural_hit_count=0,
            expected_utility=0.20,
        ))

    def test_module_activation_summary_reports_missing_expected_modules(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "answer_hash": "aid",
            "category": "Chemistry",
            "raw_subject": "Chemistry",
            "answer_type": "multipleChoice",
            "_question": "secret gated question",
            "_answer": "B",
        }
        row = _score_prediction(problem=problem, model="m", variant="assumption_wrapper", prediction='{"answer":"B"}')
        summary = _module_activation_summary([row])
        missing = _expected_but_missing_modules([row])

        self.assertEqual(summary["m::assumption_wrapper"]["prompt_scaffold"]["activated"], 1)
        self.assertIn("assumption_graph_retrieval", missing["m::assumption_wrapper"])
        self.assertIn("recursive_assumption_runner", missing["m::assumption_wrapper"])

        abstain_row = dict(row)
        abstain_row["variant"] = "assumption_agent_recursive_verify"
        abstain_row["module_trace"] = [
            {"module": "hle_math_tool_solver", "expected": True, "status": "abstained"},
            {"module": "candidate_claim_verifier", "expected": True, "status": "no_executable_claim"},
        ]
        self.assertEqual(_expected_but_missing_modules([abstain_row]), {})

    def test_component_efficacy_audit_marks_functional_roles(self):
        problem = {
            "answer_type": "multipleChoice",
            "category": "Math",
            "raw_subject": "Mathematics",
        }
        plan = {
            "stages": {
                "assumption_graph_retrieval": {
                    "status": "activated",
                    "node_count": 5,
                    "edge_count": 1,
                    "top_scores": [0.4, 0.2],
                    "top_node_types": ["harness", "hypothesis"],
                },
                "structural_morphism_transfer": {
                    "status": "activated",
                    "formal_mapping_hits": [{"mapping_id": "m1"}],
                    "structural_morphism_hits": [{"decision": "repair_under_specified"}],
                },
                "world_model_router": {
                    "status": "activated",
                    "decision": "use_context",
                    "expected_utility": 0.4,
                    "predicted_regression_risk": "low",
                },
                "prompt_builder": {"status": "activated", "context_injected": True},
                "hle_evidence_bridge": {
                    "status": "activated",
                    "selection_policy": "hipporag_style_associative_rerank",
                    "query_count": 2,
                    "selected_result_count": 5,
                    "evidence_char_count": 1000,
                },
                "recursive_child_validation": {
                    "status": "activated",
                    "execution_mode": "serial",
                    "planned_child_count": 5,
                    "child_count": 3,
                    "answered_child_count": 3,
                    "error_child_count": 0,
                    "early_stopped": True,
                    "early_stop_reason": "two_vote_majority",
                    "prompt_kinds": ["direct_short_answer", "evidence_bridge_answer", "agent_context_answer"],
                    "skipped_prompt_kinds": ["constraint_checked_answer", "recursive_assumption_answer"],
                    "candidate_answer_hashes": ["h1", "h1", "h1"],
                },
                "recursive_timeout_recovery_child": {
                    "status": "activated",
                    "reason": "timeout_or_error_with_candidate_shortage",
                    "valid_candidate_count_before": 1,
                    "unique_candidate_count_before": 1,
                    "timeout_child_count_before": 1,
                    "error_child_count_before": 1,
                    "candidate_emitted": True,
                    "selected_timeout_recovery_candidate": False,
                    "recovery_model": "gpt-5.5",
                },
                "candidate_claim_verifier": {
                    "status": "no_executable_claim",
                    "backend": "sympy_mc_option_planner",
                    "verified_count": 0,
                    "refuted_count": 0,
                    "inconclusive_count": 3,
                    "reference_operation": "none",
                },
                "counter_assumption_challenge": {
                    "status": "activated",
                    "reason": "majority_without_independent_verification",
                    "top_candidate_count": 3,
                    "unique_candidate_count": 1,
                    "challenge_disagreed_with_majority": False,
                    "selected_counter_challenge": False,
                    "option_elimination_challenge": {
                        "status": "activated",
                        "challenge_disagreed_with_majority": True,
                        "selected_option_elimination_challenge": False,
                    },
                    "forced_alternative_challenge": {
                        "status": "activated",
                        "challenge_disagreed_with_majority": True,
                        "selected_forced_alternative": False,
                    },
                },
                "multi_candidate_self_verifier": {
                    "status": "activated",
                    "selection_method": "normalized_majority",
                    "verifier_model_call": False,
                },
            }
        }
        efficacy = _component_efficacy_from_plan(
            problem=problem,
            variant="assumption_agent_recursive_verify",
            plan=plan,
            correct=False,
            error=None,
        )

        self.assertTrue(efficacy["flags"]["graph_context_injected"])
        self.assertTrue(efficacy["flags"]["evidence_bridge_activated"])
        self.assertTrue(efficacy["flags"]["evidence_child_executed"])
        self.assertTrue(efficacy["flags"]["recursive_collapsed_consensus"])
        self.assertTrue(efficacy["flags"]["recursive_timeout_recovery_activated"])
        self.assertTrue(efficacy["flags"]["recursive_timeout_recovery_emitted_candidate"])
        self.assertFalse(efficacy["flags"]["recursive_timeout_recovery_selected"])
        self.assertEqual(efficacy["recursive_timeout_recovery"]["recovery_model"], "gpt-5.5")
        self.assertTrue(efficacy["flags"]["claim_verifier_no_executable_claim"])
        self.assertTrue(efficacy["flags"]["counter_assumption_challenge_activated"])
        self.assertTrue(efficacy["flags"]["option_elimination_challenge_activated"])
        self.assertTrue(efficacy["flags"]["option_elimination_challenge_disagreed"])
        self.assertFalse(efficacy["flags"]["option_elimination_challenge_selected"])
        self.assertTrue(efficacy["flags"]["forced_alternative_activated"])
        self.assertTrue(efficacy["flags"]["forced_alternative_disagreed"])
        self.assertFalse(efficacy["flags"]["forced_alternative_selected"])
        self.assertTrue(efficacy["flags"]["majority_only_selection"])
        self.assertFalse(efficacy["flags"]["recursive_diverse_candidates"])

        summary = _component_efficacy_summary([
            {
                "model": "m",
                "variant": "assumption_agent_recursive_verify",
                "answer_type": "multipleChoice",
                "correct": False,
                "component_efficacy": efficacy,
            },
            {
                "model": "m",
                "variant": "assumption_agent_recursive_verify",
                "answer_type": "multipleChoice",
                "correct": True,
                "component_efficacy": {
                    "selection": {"selection_method": "candidate_claim_verifier_priority"},
                    "claim_verifier": {"status": "activated"},
                    "recursive": {"unique_candidate_count": 2},
                    "flags": {
                        "final_correct": True,
                        "candidate_claim_override": True,
                        "claim_verifier_verified_candidate": True,
                        "recursive_diverse_candidates": True,
                    },
                },
            },
        ])

        row = summary["m::assumption_agent_recursive_verify"]
        self.assertEqual(row["flag_counts"]["majority_only_selection"], 1)
        self.assertEqual(row["flag_accuracy"]["majority_only_selection"], 0.0)
        self.assertEqual(row["flag_counts"]["candidate_claim_override"], 1)
        self.assertEqual(row["flag_accuracy"]["candidate_claim_override"], 1.0)
        self.assertEqual(row["selection_method_counts"]["normalized_majority"], 1)
        self.assertEqual(row["selection_method_counts"]["candidate_claim_verifier_priority"], 1)

    def test_module_activation_audit_backfills_old_rows(self):
        artifact = {
            "eval_id": "old_hle",
            "metrics": {
                "sample_count": 1,
                "planned_live_model_calls": 2,
                "resolved_live_model_calls": 2,
                "live_model_call_error_count": 0,
                "by_model_variant": {},
            },
            "rows": [
                {
                    "problem_id_hash": "p",
                    "question_hash": "q",
                    "answer_hash": "a",
                    "model": "m",
                    "variant": "raw",
                    "category": "x",
                    "raw_subject": "x",
                    "answer_type": "multipleChoice",
                    "correct": True,
                    "error": None,
                },
                {
                    "problem_id_hash": "p",
                    "question_hash": "q",
                    "answer_hash": "a",
                    "model": "m",
                    "variant": "assumption_wrapper",
                    "category": "x",
                    "raw_subject": "x",
                    "answer_type": "multipleChoice",
                    "correct": True,
                    "error": None,
                },
            ],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
            audit = build_hle_module_activation_audit_payload([path])

        self.assertTrue(audit["pass"])
        self.assertFalse(audit["diagnosis"]["old_artifacts_had_module_level_telemetry"])
        self.assertIn("assumption_graph_retrieval", audit["expected_but_missing_modules"]["m::assumption_wrapper"])


if __name__ == "__main__":
    unittest.main()
