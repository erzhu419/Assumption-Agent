import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from assumption_os.hle_module_activation_audit import build_hle_module_activation_audit_payload
from assumption_os.hle_smoke_eval import (
    _aggregate_rows,
    _expected_but_missing_modules,
    _is_correct,
    _module_activation_summary,
    _module_trace,
    _parse_answer_json,
    _prompt_for,
    _score_prediction,
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

    def test_assumption_agent_prompt_abstain_matches_raw_prompt(self):
        problem = {
            "answer_type": "multipleChoice",
            "_question": "secret gated question",
        }
        raw = _prompt_for(problem, variant="raw")
        agent = _prompt_for(problem, variant="assumption_agent", agent_plan={"prompt_context": ""})

        self.assertEqual(agent, raw)

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
