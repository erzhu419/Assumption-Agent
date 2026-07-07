import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.hle_transition_dataset import (
    TRANSITION_DATASET_VERSION,
    build_transition_dataset,
    build_transition_dataset_from_paths,
    load_hle_result_rows_from_path,
    transition_record_from_hle_row,
)


class HleTransitionDatasetTests(unittest.TestCase):
    def test_transition_record_hashes_question_and_labels(self):
        record = transition_record_from_hle_row({
            "seed": 1298,
            "question": "Which option is the probe?",
            "selected_label": "B",
            "gold_label": "C",
            "correct": False,
            "category": "Chemistry",
            "selection_method": "source_pair_binding_lane",
            "failure_bucket": "candidate_generation_missed_gold",
            "elapsed_seconds": 12.5,
            "unique_model_calls": 7,
            "route": {
                "router_payload_hash": "routehash",
                "fast_policy_memory": {
                    "selected_policy_ids": ["source_binding_v1"],
                    "fast_policy_payload_hash": "policyhash",
                },
            },
            "option_matrix": {
                "option_rows": [
                    {"label": "B", "option_hash": "option-b"},
                    {"label": "C", "option_hash": "option-c"},
                ],
            },
        })

        payload = record.to_dict()
        self.assertEqual(payload["dataset_version"], TRANSITION_DATASET_VERSION)
        self.assertEqual(payload["question_id"], "1298")
        self.assertEqual(payload["action"], "source_pair_binding_lane")
        self.assertEqual(payload["failure_bucket"], "candidate_generation_missed_gold")
        self.assertEqual(payload["fast_policy_ids"], ["source_binding_v1"])
        self.assertEqual(payload["path_hashes"]["router_payload_hash"], "routehash")
        self.assertEqual(payload["path_hashes"]["fast_policy_payload_hash"], "policyhash")
        self.assertEqual(payload["option_feature_hashes"]["C"], "option-c")
        self.assertFalse(payload["raw_content_persisted"])
        self.assertNotIn("Which option", str(payload))

    def test_transition_dataset_summary_counts_failures_and_no_fallback(self):
        dataset = build_transition_dataset([
            {
                "id": "a",
                "selected_label": "A",
                "gold_label": "A",
                "correct": True,
                "selection_method": "raw_fallback",
                "elapsed_seconds": 1.5,
            },
            {
                "id": "b",
                "selected_label": "B",
                "gold_label": "C",
                "correct": False,
                "selection_method": "source_lane",
                "failure_bucket": "verified_or_abstain no_fallback",
                "elapsed_seconds": 2.5,
            },
        ])

        summary = dataset["summary"]
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["known_correct_count"], 2)
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["no_fallback_count"], 1)
        self.assertEqual(summary["failure_buckets"]["verified_or_abstain no_fallback"], 1)
        self.assertEqual(summary["action_counts"]["raw_fallback"], 1)
        self.assertFalse(dataset["raw_content_persisted"])

    def test_transition_record_uses_hash_only_hle_fields(self):
        record = transition_record_from_hle_row({
            "problem_id_hash": "problem-hash",
            "question_hash": "question-hash",
            "prediction_hash": "prediction-hash",
            "answer_hash": "answer-hash",
            "correct": False,
            "category": "Chemistry",
            "raw_subject": "Chemistry",
            "variant": "assumption_agent_recursive_verify",
            "call_metadata": {
                "agent_plan_hash": "plan-hash",
                "call_id": "call-hash",
                "latency_sec": 42.25,
                "variant_watchdog": {
                    "model_call_count": 9,
                    "model_router_attempt_count": 10,
                    "status": "completed",
                },
            },
            "component_efficacy": {
                "selection": {
                    "selection_method": "verified_or_abstain_direct_fallback",
                    "verified_or_abstain_gate": {
                        "status": "no_fallback",
                        "reason": "no_direct_candidate",
                    },
                },
                "flags": {
                    "verified_or_abstain_no_fallback": True,
                },
            },
        })

        payload = record.to_dict()
        self.assertEqual(payload["question_id"], "problem-hash")
        self.assertEqual(payload["question_hash"], "question-hash")
        self.assertEqual(payload["selected_label_hash"], "prediction-hash")
        self.assertEqual(payload["gold_after_run_label_hash"], "answer-hash")
        self.assertEqual(payload["action"], "verified_or_abstain_direct_fallback")
        self.assertEqual(payload["failure_bucket"], "verified_or_abstain no_fallback")
        self.assertEqual(payload["latency_seconds"], 42.25)
        self.assertEqual(payload["cost"], 9.0)
        self.assertEqual(payload["path_hashes"]["agent_plan_hash"], "plan-hash")
        self.assertEqual(payload["path_hashes"]["variant_watchdog_model_call_count"], 9)
        self.assertEqual(payload["path_hashes"]["verified_or_abstain_gate_status"], "no_fallback")
        self.assertIn("verified_or_abstain_gate_reason_hash", payload["path_hashes"])
        self.assertFalse(payload["raw_content_persisted"])

    def test_transition_record_tracks_preserve_original_no_direct_fallback(self):
        dataset = build_transition_dataset([
            {
                "problem_id_hash": "problem-hash",
                "question_hash": "question-hash",
                "prediction_hash": "prediction-hash",
                "answer_hash": "answer-hash",
                "correct": False,
                "component_efficacy": {
                    "selection": {
                        "selection_method": "verified_or_abstain_direct_fallback",
                        "verified_or_abstain_gate": {
                            "status": "abstained",
                            "reason": "no_direct_candidate_preserve_original_selection",
                            "fallback_policy": (
                                "preserve_original_selection_no_direct_fallback"
                            ),
                        },
                    },
                    "flags": {
                        "verified_or_abstain_abstained": True,
                        "verified_or_abstain_preserve_original_no_direct_fallback": True,
                    },
                },
            },
        ])

        record = dataset["records"][0]
        self.assertEqual(
            record["failure_bucket"],
            "verified_or_abstain preserve_original_no_direct_fallback",
        )
        self.assertEqual(dataset["summary"]["no_fallback_count"], 0)
        self.assertEqual(
            dataset["summary"]["verified_or_abstain_gate_status_counts"]["abstained"],
            1,
        )
        self.assertEqual(
            record["path_hashes"]["verified_or_abstain_gate_fallback_policy"],
            "preserve_original_selection_no_direct_fallback",
        )
        self.assertFalse(dataset["raw_content_persisted"])

    def test_failure_bucket_prefers_candidate_generation_gap_flag(self):
        record = transition_record_from_hle_row({
            "problem_id_hash": "problem-hash",
            "question_hash": "question-hash",
            "prediction_hash": "prediction-hash",
            "answer_hash": "answer-hash",
            "correct": False,
            "component_efficacy": {
                "operator_failure_taxonomy": {
                    "category": "SourceEvidenceMissing",
                    "reason": "source_verifier_attempted_without_direct_accepted_evidence",
                },
                "selection": {
                    "verified_or_abstain_gate": {
                        "status": "abstained",
                        "reason": "source_verifier_generic_blocks_model_only_verified_selection",
                    },
                },
                "flags": {
                    "candidate_generation_missed_gold": True,
                    "verified_or_abstain_abstained": True,
                },
            },
        })

        self.assertEqual(record.failure_bucket, "candidate_generation_missed_gold")

    def test_summary_counts_no_fallback_gate_even_when_primary_bucket_differs(self):
        dataset = build_transition_dataset([
            {
                "problem_id_hash": "problem-hash",
                "question_hash": "question-hash",
                "prediction_hash": "prediction-hash",
                "answer_hash": "answer-hash",
                "correct": False,
                "component_efficacy": {
                    "selection": {
                        "verified_or_abstain_gate": {
                            "status": "no_fallback",
                            "reason": "no_direct_candidate",
                        },
                    },
                    "flags": {
                        "candidate_generation_missed_gold": True,
                    },
                },
            },
        ])

        self.assertEqual(dataset["summary"]["failure_buckets"]["candidate_generation_missed_gold"], 1)
        self.assertEqual(dataset["summary"]["no_fallback_count"], 1)
        self.assertEqual(dataset["summary"]["verified_or_abstain_gate_status_counts"]["no_fallback"], 1)

    def test_loader_expands_aggregate_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_path = root / "run_shard_000.json"
            shard_path.write_text(json.dumps({
                "eval_id": "shard-eval",
                "rows": [
                    {
                        "problem_id_hash": "p1",
                        "question_hash": "q1",
                        "prediction_hash": "pred1",
                        "answer_hash": "gold1",
                        "correct": True,
                        "component_efficacy": {
                            "selection": {"selection_method": "direct"},
                            "flags": {"final_correct": True},
                        },
                    }
                ],
            }), encoding="utf-8")
            aggregate_path = root / "aggregate.json"
            aggregate_path.write_text(json.dumps({
                "eval_id": "aggregate-eval",
                "eval_kind": "hle_parallel",
                "shards": [
                    {
                        "out": shard_path.name,
                        "status": "completed",
                        "shard_index": 0,
                        "elapsed_sec": 3.0,
                    }
                ],
            }), encoding="utf-8")

            rows = load_hle_result_rows_from_path(aggregate_path)
            self.assertEqual(len(rows), 1)
            dataset = build_transition_dataset_from_paths([aggregate_path])

        record = dataset["records"][0]
        self.assertEqual(record["question_id"], "p1")
        self.assertEqual(record["correct"], True)
        self.assertEqual(record["failure_bucket"], "none")
        self.assertEqual(record["action"], "direct")
        self.assertEqual(record["path_hashes"]["shard_index"], 0)
        self.assertIn("source_artifact_hash", record["path_hashes"])
        self.assertEqual(dataset["summary"]["record_count"], 1)


if __name__ == "__main__":
    unittest.main()
