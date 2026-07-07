import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from assumption_os.hle_smoke_eval import GoldAccessDuringDecisionError, _no_gold_decision_phase
from assumption_os.local_mc_transfer_eval import (
    _local_mc_problem_from_row,
    build_local_mc_transfer_eval_payload,
    format_markdown,
)


class LocalMcTransferEvalTest(unittest.TestCase):
    def test_dry_run_loads_local_mc_jsonl_without_persisting_raw_content(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "transfer.jsonl"
            rows = [
                {
                    "id": "unit-1",
                    "question": "Hidden transfer question alpha?",
                    "choices": ["red herring", "SecretCorrectOptionText", "other"],
                    "answer": "B",
                    "category": "unit_transfer",
                },
                {
                    "id": "unit-2",
                    "question": "Hidden transfer question beta?",
                    "choices": {"A": "first", "B": "second", "C": "third"},
                    "answer": "C",
                    "subject": "unit_subject",
                },
            ]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            payload = build_local_mc_transfer_eval_payload(
                root=root,
                input_jsonl=data,
                sample_size=2,
                execute_live=False,
            )
            markdown = format_markdown(payload)

        serialized = json.dumps(payload, ensure_ascii=False) + markdown
        self.assertTrue(payload["pass"])
        self.assertEqual(payload["eval_kind"], "local_mc_transfer_eval")
        self.assertEqual(payload["metrics"]["sample_count"], 2)
        self.assertEqual(payload["metrics"]["planned_live_model_calls"], 0)
        self.assertFalse(payload["raw_content_persisted"])
        self.assertNotIn("Hidden transfer question", serialized)
        self.assertNotIn("SecretCorrectOptionText", serialized)
        self.assertNotIn('"answer": "B"', serialized)

    def test_local_problem_uses_hle_no_gold_decision_guard(self) -> None:
        problem = _local_mc_problem_from_row(
            {
                "id": "guarded",
                "question": "Guarded transfer question?",
                "choices": ["one", "two"],
                "answer": "A",
            },
            scanned_index=1,
        )
        self.assertIsNotNone(problem)
        with _no_gold_decision_phase("unit_decision"):
            with self.assertRaises(GoldAccessDuringDecisionError):
                _ = problem["_answer"]

    def test_live_payload_defaults_to_transfer_watchdog_cap(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "empty.jsonl"
            data.write_text("", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "HLE_VARIANT_TOTAL_TIMEOUT_SEC": "",
                    "HLE_VARIANT_TOTAL_MODEL_CALL_BUDGET": "",
                    "HLE_VARIANT_RECURSIVE_SELECTION_RESERVED_MODEL_CALL_BUDGET": "",
                    "HLE_VARIANT_SELECTION_RESERVED_MODEL_CALL_BUDGET": "",
                    "HLE_VARIANT_RESERVED_SELECTION_MODEL_CALLS": "",
                    "HLE_WEAK_SOURCE_FALLBACK_CASCADE_SKIP_STRUCTURAL_AUDIT": "",
                    "LOCAL_MC_TRANSFER_VARIANT_TOTAL_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_VARIANT_TOTAL_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_SELECTION_RESERVED_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_SKIP_STRUCTURAL_AUDIT_ON_WEAK_SOURCE": "",
                    "MODEL_ROUTER_SUBPROCESS_CALLS": "",
                    "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC": "",
                    "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT": "",
                    "MODEL_ROUTER_NO_BYTE_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_MODEL_SUBPROCESS_NO_BYTE_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_MODEL_NO_BYTE_TIMEOUT_SEC": "",
                    "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC": "",
                    "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC": "",
                    "LOCAL_MC_TRANSFER_CHILD_BATCH_MAX_WAIT_SEC": "",
                    "HLE_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET": "",
                    "HLE_RECURSIVE_CHILD_TOTAL_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_LATE_CHILD_MODEL_CALL_BUDGET": "",
                    "HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET": "",
                    "HLE_RECURSIVE_SELECTION_ADJUDICATOR_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_SELECTION_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_SELECTION_MODEL_CALL_BUDGET": "",
                    "HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC": "",
                    "HLE_RECURSIVE_SELECTION_TOTAL_WALLCLOCK_SEC": "",
                    "HLE_RECURSIVE_SELECTION_TOTAL_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC": "",
                    "LOCAL_MC_TRANSFER_SELECTION_WALLCLOCK_BUDGET_SEC": "",
                    "HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS": "",
                    "LOCAL_MC_TRANSFER_AGENT_PARALLEL_CHILD_MAX_WORKERS": "",
                    "LOCAL_MC_TRANSFER_CHILD_MAX_WORKERS": "",
                    "HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_TIMEOUT_RECOVERY_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_TIMEOUT_RECOVERY_CALL_TIMEOUT_SEC": "",
                    "HLE_TIMEOUT_RECOVERY_MAX_TOKENS": "",
                    "LOCAL_MC_TRANSFER_TIMEOUT_RECOVERY_MAX_TOKENS": "",
                    "LOCAL_MC_TRANSFER_AGENT_MODEL_CALL_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_AGENT_CALL_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_AGENT_CHILD_TIMEOUT_SEC": "",
                    "LOCAL_MC_TRANSFER_CHILD_TIMEOUT_SEC": "",
                },
                clear=False,
            ):
                payload = build_local_mc_transfer_eval_payload(
                    root=root,
                    input_jsonl=data,
                    sample_size=0,
                    execute_live=True,
                )

        watchdog = payload["runtime_policy"]["variant_watchdog"]
        self.assertTrue(watchdog["enabled"])
        self.assertEqual(watchdog["total_timeout_sec"], 900.0)
        self.assertEqual(watchdog["total_model_call_budget"], 12)
        self.assertEqual(watchdog["recursive_selection_reserved_model_call_budget"], 1)
        defaults = payload["runtime_policy"]["local_transfer_runtime_defaults"]
        self.assertEqual(defaults["status"], "activated")
        self.assertTrue(defaults["variant_recursive_selection_reserve_applied"])
        self.assertTrue(defaults["weak_source_structural_audit_skip_applied"])
        self.assertTrue(defaults["subprocess_model_calls_applied"])
        self.assertTrue(defaults["subprocess_no_byte_timeout_applied"])
        self.assertEqual(defaults["subprocess_no_byte_timeout_sec"], 180.0)
        self.assertTrue(defaults["recursive_child_batch_max_wait_applied"])
        self.assertEqual(defaults["recursive_child_batch_max_wait_sec"], 120.0)
        self.assertTrue(defaults["recursive_late_child_model_call_budget_applied"])
        self.assertEqual(defaults["recursive_late_child_model_call_budget"], 7)
        self.assertTrue(defaults["recursive_selection_model_call_budget_applied"])
        self.assertEqual(defaults["recursive_selection_model_call_budget"], 1)
        self.assertTrue(defaults["recursive_selection_wallclock_budget_applied"])
        self.assertEqual(defaults["recursive_selection_wallclock_budget_sec"], 120.0)
        self.assertTrue(defaults["agent_parallel_child_max_workers_applied"])
        self.assertEqual(defaults["agent_parallel_child_max_workers"], 2)
        self.assertTrue(defaults["timeout_recovery_timeout_applied"])
        self.assertEqual(defaults["timeout_recovery_timeout_sec"], 60.0)
        self.assertTrue(defaults["timeout_recovery_max_tokens_applied"])
        self.assertEqual(defaults["timeout_recovery_max_tokens"], 64)
        self.assertTrue(defaults["agent_model_call_timeout_default_applied"])
        self.assertEqual(defaults["agent_model_call_timeout_sec"], 120.0)
        self.assertTrue(defaults["agent_child_timeout_default_applied"])
        self.assertEqual(defaults["agent_child_timeout_sec"], 120.0)
        self.assertTrue(payload["runtime_policy"]["subprocess_model_calls_enabled"])
        self.assertEqual(payload["runtime_policy"]["subprocess_no_byte_timeout_sec"], 180.0)
        self.assertEqual(payload["runtime_policy"]["recursive_child_batch_max_wait_sec"], 120.0)
        self.assertEqual(payload["runtime_policy"]["recursive_late_child_model_call_budget"], 7)
        self.assertEqual(payload["runtime_policy"]["recursive_selection_model_call_budget"], 1)
        self.assertEqual(payload["runtime_policy"]["recursive_selection_wallclock_budget_sec"], 120.0)
        self.assertEqual(payload["runtime_policy"]["agent_parallel_child_max_workers"], 2)
        self.assertEqual(payload["runtime_policy"]["timeout_recovery_timeout_sec"], 60.0)
        self.assertEqual(payload["runtime_policy"]["timeout_recovery_max_tokens"], 64)
        self.assertTrue(payload["api_summary"]["subprocess_model_calls_enabled"])
        self.assertEqual(payload["api_summary"]["subprocess_no_byte_timeout_sec"], 180.0)
        self.assertEqual(payload["api_summary"]["recursive_child_batch_max_wait_sec"], 120.0)
        self.assertEqual(payload["api_summary"]["recursive_late_child_model_call_budget"], 7)
        self.assertEqual(payload["api_summary"]["recursive_selection_model_call_budget"], 1)
        self.assertEqual(payload["api_summary"]["recursive_selection_wallclock_budget_sec"], 120.0)
        self.assertEqual(payload["api_summary"]["agent_parallel_child_max_workers"], 2)
        self.assertEqual(payload["api_summary"]["timeout_recovery_timeout_sec"], 60.0)
        self.assertEqual(payload["api_summary"]["timeout_recovery_max_tokens"], 64)
        self.assertEqual(payload["api_summary"]["agent_model_call_timeout_sec"], 120.0)
        self.assertEqual(payload["api_summary"]["agent_child_timeout_sec"], 120.0)

    def test_live_runtime_defaults_preserve_explicit_batch_and_late_budget_env(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "empty.jsonl"
            data.write_text("", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC": "90",
                    "HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC": "",
                    "HLE_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET": "3",
                    "HLE_RECURSIVE_CHILD_TOTAL_MODEL_CALL_BUDGET": "",
                    "LOCAL_MC_TRANSFER_RECURSIVE_LATE_CHILD_MODEL_CALL_BUDGET": "",
                    "HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET": "2",
                    "HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC": "45",
                    "HLE_AGENT_PARALLEL_CHILD_MAX_WORKERS": "4",
                    "HLE_TIMEOUT_RECOVERY_TIMEOUT_SEC": "30",
                    "HLE_TIMEOUT_RECOVERY_MAX_TOKENS": "32",
                },
                clear=False,
            ):
                payload = build_local_mc_transfer_eval_payload(
                    root=root,
                    input_jsonl=data,
                    sample_size=0,
                    execute_live=True,
                )

        defaults = payload["runtime_policy"]["local_transfer_runtime_defaults"]
        self.assertFalse(defaults["recursive_child_batch_max_wait_applied"])
        self.assertEqual(defaults["recursive_child_batch_max_wait_sec"], 90.0)
        self.assertFalse(defaults["recursive_late_child_model_call_budget_applied"])
        self.assertEqual(defaults["recursive_late_child_model_call_budget"], 3)
        self.assertFalse(defaults["recursive_selection_model_call_budget_applied"])
        self.assertEqual(defaults["recursive_selection_model_call_budget"], 2)
        self.assertFalse(defaults["recursive_selection_wallclock_budget_applied"])
        self.assertEqual(defaults["recursive_selection_wallclock_budget_sec"], 45.0)
        self.assertFalse(defaults["agent_parallel_child_max_workers_applied"])
        self.assertEqual(defaults["agent_parallel_child_max_workers"], 4)
        self.assertFalse(defaults["timeout_recovery_timeout_applied"])
        self.assertEqual(defaults["timeout_recovery_timeout_sec"], 30.0)
        self.assertFalse(defaults["timeout_recovery_max_tokens_applied"])
        self.assertEqual(defaults["timeout_recovery_max_tokens"], 32)
        self.assertEqual(payload["runtime_policy"]["recursive_child_batch_max_wait_sec"], 90.0)
        self.assertEqual(payload["runtime_policy"]["recursive_late_child_model_call_budget"], 3)
        self.assertEqual(payload["runtime_policy"]["recursive_selection_model_call_budget"], 2)
        self.assertEqual(payload["runtime_policy"]["recursive_selection_wallclock_budget_sec"], 45.0)
        self.assertEqual(payload["runtime_policy"]["agent_parallel_child_max_workers"], 4)
        self.assertEqual(payload["runtime_policy"]["timeout_recovery_timeout_sec"], 30.0)
        self.assertEqual(payload["runtime_policy"]["timeout_recovery_max_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
