import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_os.hle_module_activation_audit import build_hle_module_activation_audit_payload
from assumption_os.hle_smoke_eval import (
    _aggregate_rows,
    _candidate_evidence_queries,
    _clean_evidence_text,
    _expected_but_missing_modules,
    _format_evidence_context,
    _has_two_vote_majority,
    _is_correct,
    _module_activation_summary,
    _module_trace,
    _needs_exact_answer_repair,
    _needs_evidence_grounded_child,
    _parse_answer_json,
    _parse_verifier_choice,
    _prompt_for,
    _recursive_child_prompt_specs,
    _score_prediction,
    _select_recursive_child_answer,
    _should_use_agent_context,
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

    def test_recursive_child_prompt_specs_include_context_only_when_present(self):
        problem = {
            "answer_type": "exactMatch",
            "_question": "secret gated question",
        }

        self.assertEqual(len(_recursive_child_prompt_specs(problem, agent_plan={})), 3)
        specs = _recursive_child_prompt_specs(problem, agent_plan={"prompt_context": "graph context"})
        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[-1]["prompt_kind"], "agent_context_answer")

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

    def test_exact_answer_repair_gate(self):
        exact_problem = {"answer_type": "exactMatch"}
        mc_problem = {"answer_type": "multipleChoice"}

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

    def test_evidence_bridge_query_and_context_helpers(self):
        problem = {
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
