from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from assumption_os.hle_parallel_shard_runner import (
    ShardRunState,
    aggregate_parallel_payload,
    build_error_stratification,
    build_heartbeat,
    build_shard_command,
    build_shard_specs,
    format_parallel_markdown,
)


class TestHleParallelShardRunner(unittest.TestCase):
    def test_build_shard_specs_splits_total_and_advances_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs"
            md_dir = Path(tmp) / "md"
            specs = build_shard_specs(
                eval_id="fresh30",
                total_sample_size=7,
                shard_size=3,
                seed_offset=100,
                seed_stride=11,
                run_dir=run_dir,
                md_dir=md_dir,
            )
        self.assertEqual([spec.sample_size for spec in specs], [3, 3, 1])
        self.assertEqual([spec.seed_offset for spec in specs], [100, 111, 122])
        self.assertEqual([spec.eval_id for spec in specs], ["fresh30_shard_000", "fresh30_shard_001", "fresh30_shard_002"])

    def test_build_shard_command_keeps_api_secrets_out_of_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="cmdtest",
                total_sample_size=2,
                shard_size=2,
                seed_offset=0,
                seed_stride=1,
                run_dir=root / "run",
                md_dir=root / "md",
            )[0]
            cmd = build_shard_command(
                spec,
                root=root,
                max_scan=50,
                models="gpt-5.4-mini",
                variants="raw,assumption_agent_recursive_verify",
                execute_live=True,
                call_timeout=900,
                max_tokens=384,
                graph_dir=root / "graph",
                agent_top_k=4,
                agent_context_max_chars=1600,
                agent_child_mode="parallel_quorum",
                agent_child_timeout=800,
                evidence_bridge_enabled=False,
                exclude_existing_hle_artifacts=True,
                exclude_artifact_glob="artifacts/*.json*",
                sample_answer_type="multipleChoice",
                sample_subject_contains="chem",
            )
        text = " ".join(cmd)
        self.assertIn("--hard-exit-after-write", cmd)
        self.assertIn("--execute-live", cmd)
        self.assertIn("--disable-evidence-bridge", cmd)
        self.assertIn("--exclude-existing-hle-artifacts", cmd)
        self.assertNotIn("sk-", text)
        self.assertNotIn("hf_", text)

    def test_build_shard_command_extends_max_scan_past_seed_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="scan",
                total_sample_size=1,
                shard_size=1,
                seed_offset=3600,
                seed_stride=1,
                run_dir=root / "run",
                md_dir=root / "md",
            )[0]
            cmd = build_shard_command(
                spec,
                root=root,
                max_scan=50000,
                models="gpt-5.4-mini",
                variants="raw",
                execute_live=False,
                call_timeout=None,
                max_tokens=384,
                graph_dir=root / "graph",
                agent_top_k=4,
                agent_context_max_chars=1600,
                agent_child_mode="parallel_quorum",
                agent_child_timeout=None,
                evidence_bridge_enabled=True,
                exclude_existing_hle_artifacts=False,
                exclude_artifact_glob="artifacts/*.json*",
                sample_answer_type="",
                sample_subject_contains="",
            )
        max_scan_index = cmd.index("--max-scan") + 1
        self.assertEqual(cmd[max_scan_index], "53600")

    def test_heartbeat_reports_latest_jsonl_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="hb",
                total_sample_size=1,
                shard_size=1,
                seed_offset=7,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )[0]
            spec.log_out.parent.mkdir(parents=True, exist_ok=True)
            spec.log_out.write_text(
                json.dumps({"event": "call_start", "model": "m", "variant": "raw", "problem_id_hash": "p1"}) + "\n"
                + json.dumps({"event": "call_error", "model": "m", "variant": "raw", "problem_id_hash": "p1", "error_type": "RuntimeError"}) + "\n",
                encoding="utf-8",
            )
            state = ShardRunState(spec=spec, command=["python", "-m", "x"], status="running")
            heartbeat = build_heartbeat([state])
        self.assertEqual(heartbeat["status_counts"], {"running": 1})
        self.assertEqual(heartbeat["shards"][0]["latest_event"]["event"], "call_error")
        self.assertEqual(heartbeat["shards"][0]["latest_event"]["error_type"], "RuntimeError")
        self.assertFalse(heartbeat["raw_content_persisted"])

    def test_error_stratification_counts_rows_jsonl_and_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="err",
                total_sample_size=2,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )
            specs[0].log_out.write_text(
                json.dumps({"event": "recursive_child_timeout", "variant": "assumption_agent_recursive_verify", "error_type": "TimeoutError"}) + "\n",
                encoding="utf-8",
            )
            rows = [
                _row("p1", "raw", False, error_type="RuntimeError"),
                _row("p1", "assumption_agent_recursive_verify", True),
            ]
            state0 = ShardRunState(spec=specs[0], command=[], status="soft_timed_out", soft_timeout_sent=True)
            state1 = ShardRunState(spec=specs[1], command=[], status="completed")
            errors = build_error_stratification(rows=rows, specs=specs, states=[state0, state1])
        self.assertEqual(errors["top_level_error_count"], 1)
        self.assertEqual(errors["top_level_errors_by_variant"], {"raw": 1})
        self.assertEqual(errors["jsonl_error_events_by_event"], {"recursive_child_timeout": 1})
        self.assertEqual(errors["process_timeout_count"], 1)

    def test_aggregate_parallel_payload_builds_clean_shared_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="agg",
                total_sample_size=2,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )
            payloads = [
                _payload([_row("p1", "raw", False), _row("p1", "assumption_agent_recursive_verify", True)]),
                _payload([_row("p2", "raw", True), _row("p2", "assumption_agent_recursive_verify", True)]),
            ]
            states = [ShardRunState(spec=spec, command=[], status="completed", returncode=0) for spec in specs]
            payload = aggregate_parallel_payload(
                eval_id="agg",
                specs=specs,
                states=states,
                shard_payloads=payloads,
                execute_live=True,
                models="gpt-5.4-mini",
                variants="raw,assumption_agent_recursive_verify",
                total_sample_size=2,
                shard_size=1,
                parallel_workers=2,
                soft_timeout_sec=900,
            )
            markdown = format_parallel_markdown(payload)
        self.assertTrue(payload["pass"])
        self.assertTrue(payload["paper_clean_pass"])
        self.assertEqual(payload["metrics"]["sample_count"], 2)
        self.assertEqual(payload["metrics"]["by_model_variant"]["gpt-5.4-mini::raw"]["accuracy"], 0.5)
        clean = payload["metrics"]["clean_shared_subset"]["gpt-5.4-mini"]["by_variant"]
        self.assertEqual(clean["assumption_agent_recursive_verify"]["accuracy"], 1.0)
        self.assertEqual(clean["raw"]["accuracy"], 0.5)
        self.assertIn("HLE Parallel Shard Evaluation", markdown)


def _row(problem_id: str, variant: str, correct: bool, *, error_type: str | None = None) -> dict[str, object]:
    return {
        "problem_id_hash": problem_id,
        "question_hash": f"q-{problem_id}",
        "answer_hash": f"a-{problem_id}",
        "model": "gpt-5.4-mini",
        "variant": variant,
        "category": "text",
        "raw_subject": "synthetic",
        "answer_type": "multipleChoice",
        "correct": correct,
        "prediction_hash": None if error_type else f"pred-{problem_id}-{variant}",
        "prediction_text_persisted": False,
        "raw_question_persisted": False,
        "gold_answer_persisted": False,
        "module_trace": [],
        "call_metadata": {},
        "component_efficacy": {},
        "error": {"type": error_type, "message": "synthetic"} if error_type else None,
    }


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "rows": rows,
        "metrics": {
            "sample_count": len({row["problem_id_hash"] for row in rows}),
            "planned_live_model_calls": len(rows),
            "live_model_calls_executed": len(rows),
            "underlying_model_calls_executed": len(rows),
            "resolved_live_model_calls": len(rows),
            "raw_content_persisted": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
