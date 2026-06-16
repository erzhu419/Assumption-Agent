import unittest
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from assumption_os.hle_module_activation_audit import build_hle_module_activation_audit_payload
from assumption_os.hle_smoke_eval import (
    _aggregate_rows,
    _apply_math_candidate_claim_verifier,
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
    _counter_assumption_challenge_trigger,
    _deterministic_math_tool_answer,
    _execute_math_tool_plan_text,
    _extract_multiple_choice_options,
    _expected_but_missing_modules,
    _format_evidence_context,
    _has_two_vote_majority,
    _hipporag_style_rerank,
    _is_correct,
    _module_activation_summary,
    _maybe_run_forced_alternative_challenge,
    _maybe_run_critic_synthesis_child,
    _maybe_add_mc_option_sweep_candidates,
    _maybe_run_domain_rule_mc_verifier,
    _maybe_run_mc_option_evidence_scorer,
    _maybe_run_option_elimination_challenge,
    _math_tool_child_timeout,
    _module_trace,
    _needs_exact_answer_repair,
    _needs_evidence_grounded_child,
    _parse_answer_json,
    _parse_verifier_choice,
    _prompt_for,
    _retrieval_summary_is_generic_harness_only,
    _recursive_child_prompt_specs,
    _run_child_batch,
    _score_prediction,
    _select_recursive_child_answer,
    _should_run_math_tool_child,
    _should_run_candidate_claim_verifier,
    _should_use_agent_context,
    _should_prime_evidence_bridge,
)


class HleSmokeEvalTest(unittest.TestCase):
    def test_parse_answer_json(self):
        self.assertEqual(_parse_answer_json('{"answer": "D"}'), "D")
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

        self.assertEqual(len(_recursive_child_prompt_specs(problem, agent_plan={})), 3)
        specs = _recursive_child_prompt_specs(problem, agent_plan={"prompt_context": "graph context"})
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[-1]["prompt_kind"], "agent_context_answer")

        mc_specs = _recursive_child_prompt_specs(
            {"answer_type": "multipleChoice", "_question": "secret gated question"},
            agent_plan={"prompt_context": "graph context"},
        )
        self.assertEqual(mc_specs[1]["prompt_kind"], "agent_context_answer")

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

    def test_math_exact_early_stop_requires_reflective_child(self):
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
        direct_evidence = [
            {"prompt_kind": "direct_short_answer", "parsed_answer": "42"},
            {"prompt_kind": "evidence_bridge_answer", "parsed_answer": "42"},
        ]
        with_reflective = direct_evidence + [
            {"prompt_kind": "constraint_checked_answer", "parsed_answer": "42"},
        ]

        self.assertFalse(_can_stop_recursive_children_early(math_problem, direct_evidence))
        self.assertTrue(_can_stop_recursive_children_early(math_problem, with_reflective))
        self.assertTrue(_can_stop_recursive_children_early(non_math_problem, direct_evidence))

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
        self.assertFalse(_can_stop_recursive_children_early(mc_problem, mc_direct_evidence))
        self.assertTrue(_can_stop_recursive_children_early(mc_problem, mc_with_context))

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
        self.assertTrue(_should_prime_evidence_bridge(
            exact_problem,
            {"world_model_router": {"decision": "abstain_to_raw_prompt"}, "prompt_context": ""},
        ))
        self.assertTrue(_should_prime_evidence_bridge(
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
                    }
                ]
            return [{"title": "Unrelated", "snippet": "No matching architecture evidence."}]

        with patch("assumption_os.hle_smoke_eval._wikipedia_search", side_effect=fake_search):
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
        self.assertEqual(summary["status"], "activated")
        self.assertTrue(summary["candidate_emitted"])
        self.assertTrue(summary["candidate_correct_for_eval"])

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
        self.assertEqual(selection["selection_method"], "counter_assumption_verifier_choice")
        self.assertEqual(selection["selected_answer"], "B")

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
            self.assertEqual(_math_tool_child_timeout(7200), 180)
        with patch.dict(os.environ, {"HLE_MATH_TOOL_CHILD_TIMEOUT_SEC": "0"}):
            self.assertEqual(_math_tool_child_timeout(7200), 7200)
        result = _deterministic_math_tool_answer(math_problem)
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["confidence"], "verified_symbolic")

        plan_result = _execute_math_tool_plan_text(
            '{"operation":"solve","equation":"x**2 - 4 = 0","variable":"x","modulus":""}'
        )
        self.assertEqual(plan_result["answer"], "-2, 2")

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
            )

        self.assertEqual(summary["selection_policy"], "hipporag_style_associative_rerank")
        self.assertIn("Attention Is All You Need", bridge_context)

    def test_evidence_bridge_uses_domain_fallback_for_specialized_subjects(self):
        problem = {
            "id_hash": "pid",
            "question_hash": "qid",
            "raw_subject": "Biology/Medicine",
            "category": "Biology/Medicine",
            "_question": 'Which option best matches "adenosine receptor antagonist" in sleep pharmacology?',
        }

        def fake_wikipedia_search(query, *, limit, timeout):
            return []

        def fake_domain_search(query, *, problem, limit, timeout):
            return [
                {
                    "title": "Caffeine and adenosine receptors",
                    "snippet": "Caffeine is an antagonist at adenosine receptors and promotes wakefulness.",
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
            )

        self.assertEqual(summary["source"], "wikipedia_plus_domain_search")
        self.assertEqual(summary["source_counts"]["pubmed"], 1)
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
