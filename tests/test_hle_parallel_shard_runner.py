from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from assumption_os.hle_module_ablation_runner import (
    build_ablation_plan,
    selected_ablation_profiles,
)
from assumption_os.hle_parallel_shard_runner import (
    ShardRunState,
    aggregate_parallel_payload,
    build_error_stratification,
    build_failure_diagnostics,
    build_heartbeat,
    build_model_budget_fairness_audit,
    build_pollution_audit,
    build_runner_env,
    build_shard_command,
    build_shard_specs,
    dedupe_shard_specs_by_sample_hash,
    format_parallel_markdown,
    mark_reusable_completed_shards,
    run_parallel_shards,
)


class TestHleParallelShardRunner(unittest.TestCase):
    def _ablation_args(self, tmp: str) -> Namespace:
        return Namespace(
            root=tmp,
            eval_id="ablate",
            profiles=(
                "full,no_graph,no_evidence,no_morphism,no_option_evidence,no_candidate_claim_verifier,"
                "no_world_model,no_recursive,raw_preserve_selector,verified_gate_off"
            ),
            profile_workers=1,
            dry_run=True,
            total_sample_size=3,
            shard_size=1,
            parallel_workers=1,
            max_scan=500,
            seed_offset=100,
            seed_stride=7,
            sample_answer_type="multipleChoice",
            sample_subject_contains="",
            models="gpt-5.4-mini",
            variants="raw,assumption_agent_recursive_verify,hipporag_baseline",
            execute_live=False,
            call_timeout=45,
            max_tokens=256,
            graph_dir="phase four/assumption_graph",
            agent_top_k=5,
            agent_context_max_chars=2800,
            agent_child_mode="parallel_quorum",
            agent_child_timeout=45,
            disable_evidence_bridge=False,
            exclude_existing_hle_artifacts=True,
            exclude_artifact_glob="phase four/assumption_graph/paper_readiness_20260604/hle*.json*",
            dedupe_shard_samples=True,
            dedupe_shard_max_attempts=11,
            run_dir=str(Path(tmp) / "runs"),
            md_dir=str(Path(tmp) / "md"),
            out="",
            md_out="",
            soft_timeout_sec=900,
            terminate_grace_sec=30,
            launch_stagger_sec=0.1,
            reuse_completed_shards=True,
            kill_on_soft_timeout=False,
            model_router_attempts=2,
            model_router_timeout=7200,
            model_router_per_attempt_timeout=90,
            model_router_backoff_base_sec=1.25,
            model_router_global_concurrency=1,
            model_router_global_concurrency_dir=str(Path(tmp) / "slots"),
            model_router_global_slot_ttl_sec=1800,
            model_router_global_slot_wait_sec=2400,
        )

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

    def test_module_ablation_plan_builds_real_toggles_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = build_ablation_plan(self._ablation_args(tmp))

        self.assertTrue(payload["gates"]["profiles_defined"])
        self.assertTrue(payload["gates"]["secrets_not_persisted"])
        self.assertFalse(payload["gates"]["raw_content_persisted"])
        by_profile = {row["profile"]: row for row in payload["profiles"]}
        self.assertEqual(by_profile["no_graph"]["agent_top_k"], 0)
        self.assertTrue(by_profile["no_evidence"]["disable_evidence_bridge"])
        self.assertEqual(
            by_profile["no_morphism"]["env_overrides"]["HLE_DISABLE_STRUCTURAL_MORPHISM_TRANSFER"],
            "1",
        )
        self.assertEqual(
            by_profile["no_option_evidence"]["env_overrides"]["HLE_DISABLE_MC_OPTION_EVIDENCE_SCORER"],
            "1",
        )
        self.assertEqual(
            by_profile["no_candidate_claim_verifier"]["env_overrides"]["HLE_DISABLE_CANDIDATE_CLAIM_VERIFIER"],
            "1",
        )
        self.assertEqual(
            by_profile["no_world_model"]["env_overrides"]["HLE_DISABLE_WORLD_MODEL_ROUTER"],
            "1",
        )
        self.assertEqual(
            by_profile["no_recursive"]["env_overrides"]["HLE_DISABLE_RECURSIVE_ASSUMPTION_RUNNER"],
            "1",
        )
        self.assertEqual(
            by_profile["raw_preserve_selector"]["env_overrides"]["HLE_ENABLE_RAW_PRESERVE_SELECTOR"],
            "1",
        )
        raw_preserve_command = by_profile["raw_preserve_selector"]["command"]
        self.assertIn("--dedupe-shard-samples", raw_preserve_command)
        self.assertEqual(
            raw_preserve_command[raw_preserve_command.index("--dedupe-shard-max-attempts") + 1],
            "11",
        )
        self.assertNotIn("--kill-on-soft-timeout", raw_preserve_command)
        self.assertIn("--reuse-completed-shards", raw_preserve_command)
        self.assertIn("--launch-stagger-sec", raw_preserve_command)
        self.assertEqual(
            raw_preserve_command[raw_preserve_command.index("--launch-stagger-sec") + 1],
            "0.1",
        )
        flattened = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("sk-", flattened)
        self.assertNotIn("hf_", flattened)

    def test_module_ablation_rejects_unknown_profile(self) -> None:
        with self.assertRaises(ValueError):
            selected_ablation_profiles("full,does_not_exist")

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

    def test_reuse_completed_shards_marks_valid_payloads_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="reuse",
                total_sample_size=1,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root / "run",
                md_dir=root / "md",
            )[0]
            spec.out.parent.mkdir(parents=True, exist_ok=True)
            spec.out.write_text(
                json.dumps({
                    "rows": [_row("p1", "raw", True)],
                    "metrics": {"raw_content_persisted": False},
                }),
                encoding="utf-8",
            )
            state = ShardRunState(spec=spec, command=["should-not-run"])

            summary = mark_reusable_completed_shards([state])

        self.assertEqual(summary["reused_shard_count"], 1)
        self.assertEqual(summary["pending_shard_count"], 0)
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.returncode, 0)
        self.assertTrue(state.reused_existing_payload)

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

    def test_dedupe_shard_specs_advances_colliding_seed_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="dedupe",
                total_sample_size=2,
                shard_size=1,
                seed_offset=0,
                seed_stride=37,
                run_dir=root,
                md_dir=root,
            )

            def fake_loader(**kwargs):
                seed = kwargs["seed_offset"]
                return [{"id_hash": "same"}] if seed < 100 else [{"id_hash": f"unique-{seed}"}]

            deduped, summary = dedupe_shard_specs_by_sample_hash(
                root=root,
                specs=specs,
                max_scan=1000,
                seed_stride=37,
                exclude_existing_hle_artifacts=False,
                exclude_artifact_glob="unused",
                sample_answer_type="multipleChoice",
                sample_subject_contains="",
                max_attempts=4,
                sample_loader=fake_loader,
            )

        self.assertEqual(deduped[0].seed_offset, 0)
        self.assertEqual(deduped[1].seed_offset, 111)
        self.assertEqual(summary["accepted_shard_count"], 2)
        self.assertEqual(summary["remaps"][1]["attempt_count"], 3)
        self.assertFalse(summary["raw_content_persisted"])

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
                json.dumps({
                    "event": "recursive_child_timeout",
                    "variant": "assumption_agent_recursive_verify",
                    "error_type": "TimeoutError",
                    "error": "model request failed: RemoteDisconnected",
                }) + "\n",
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
        self.assertEqual(errors["jsonl_error_events_by_label"], {"model request failed: RemoteDisconnected": 1})
        self.assertEqual(errors["top_level_errors_by_label"], {"synthetic": 1})
        self.assertEqual(errors["process_timeout_count"], 1)

    def test_soft_timeout_is_watch_only_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="watch",
                total_sample_size=1,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )[0]
            state = ShardRunState(
                spec=spec,
                command=[sys.executable, "-c", "import time; time.sleep(0.3)"],
            )
            states = run_parallel_shards(
                root=root,
                shard_states=[state],
                parallel_workers=1,
                heartbeat_path=root / "heartbeat.json",
                poll_interval_sec=0.01,
                heartbeat_interval_sec=0.01,
                soft_timeout_sec=0.02,
                terminate_grace_sec=0.01,
                kill_on_soft_timeout=False,
                launch_stagger_sec=0.0,
                env=os.environ.copy(),
            )
            heartbeat = json.loads((root / "heartbeat.json").read_text(encoding="utf-8"))

        self.assertEqual(states[0].status, "completed")
        self.assertEqual(states[0].returncode, 0)
        self.assertTrue(states[0].soft_timeout_observed)
        self.assertFalse(states[0].soft_timeout_sent)
        self.assertFalse(states[0].hard_kill_sent)
        self.assertEqual(states[0].process_timeout_policy, "watch_only")
        self.assertTrue(heartbeat["shards"][0]["soft_timeout_observed"])

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
                _payload([
                    _row("p1", "raw", False),
                    _row("p1", "hipporag_baseline", False),
                    _row("p1", "assumption_agent_recursive_verify", True, component_efficacy=_agent_single_call_ce()),
                ]),
                _payload([
                    _row("p2", "raw", True),
                    _row("p2", "hipporag_baseline", True),
                    _row("p2", "assumption_agent_recursive_verify", True, component_efficacy=_agent_single_call_ce()),
                ]),
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
                kill_on_soft_timeout=False,
            )
            markdown = format_parallel_markdown(payload)
        self.assertTrue(payload["pass"])
        self.assertTrue(payload["paper_clean_pass"])
        self.assertEqual(payload["runtime_policy"]["process_timeout_policy"], "watch_only")
        self.assertEqual(payload["metrics"]["sample_count"], 2)
        self.assertEqual(payload["metrics"]["by_model_variant"]["gpt-5.4-mini::raw"]["accuracy"], 0.5)
        clean = payload["metrics"]["clean_shared_subset"]["gpt-5.4-mini"]["by_variant"]
        self.assertEqual(clean["assumption_agent_recursive_verify"]["accuracy"], 1.0)
        self.assertEqual(clean["raw"]["accuracy"], 0.5)
        self.assertTrue(payload["pollution_pass"])
        self.assertEqual(
            payload["pollution_audit"]["claim_guard"]["recommended_hle_claim_scope"],
            "full_resolved_rows",
        )
        self.assertIn("HLE Parallel Shard Evaluation", markdown)
        self.assertIn("Pollution Audit", markdown)
        self.assertIn("Model Budget Fairness", markdown)

    def test_model_budget_fairness_blocks_unfair_strong_child_agent_claim(self) -> None:
        rows = [
            _row("p1", "raw", False, model="gpt-5.4-mini"),
            _row("p1", "hipporag_baseline", False, model="gpt-5.4-mini"),
            _row(
                "p1",
                "assumption_agent_recursive_verify",
                True,
                model="gpt-5.4-mini",
                component_efficacy=_agent_strong_child_ce(),
            ),
        ]

        audit = build_model_budget_fairness_audit(rows=rows)

        self.assertTrue(audit["gates"]["same_model_controls_present"])
        self.assertFalse(audit["gates"]["strong_baseline_controls_present_if_needed"])
        self.assertFalse(audit["gates"]["budget_matched_controls_present_if_needed"])
        self.assertFalse(audit["gates"]["model_budget_fairness_accounted"])
        self.assertEqual(audit["stronger_or_different_effective_models"], ["gpt-5.5"])
        self.assertEqual(audit["missing_strong_baseline_controls"][0]["model"], "gpt-5.5")
        self.assertEqual(
            audit["missing_budget_matched_controls"][0]["missing_variants"],
            ["raw_budget_matched", "hipporag_budget_matched"],
        )

    def test_model_budget_fairness_passes_with_same_strong_and_budget_controls(self) -> None:
        rows = [
            _row("p1", "raw", False, model="gpt-5.4-mini"),
            _row("p1", "hipporag_baseline", False, model="gpt-5.4-mini"),
            _row("p1", "raw", False, model="gpt-5.5"),
            _row("p1", "hipporag_baseline", False, model="gpt-5.5"),
            _row("p1", "raw_budget_matched", False, model="gpt-5.5"),
            _row("p1", "hipporag_budget_matched", False, model="gpt-5.5"),
            _row(
                "p1",
                "assumption_agent_recursive_verify",
                True,
                model="gpt-5.4-mini",
                component_efficacy=_agent_strong_child_ce(),
            ),
        ]

        audit = build_model_budget_fairness_audit(rows=rows)

        self.assertTrue(audit["gates"]["same_model_controls_present"])
        self.assertTrue(audit["gates"]["strong_baseline_controls_present_if_needed"])
        self.assertTrue(audit["gates"]["budget_matched_controls_present_if_needed"])
        self.assertTrue(audit["gates"]["model_budget_fairness_accounted"])
        self.assertEqual(audit["failed_gates"], [])

    def test_aggregate_paper_clean_fails_when_model_budget_fairness_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="unfair",
                total_sample_size=1,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )
            rows = [
                _row("p1", "raw", False, model="gpt-5.4-mini"),
                _row("p1", "hipporag_baseline", False, model="gpt-5.4-mini"),
                _row(
                    "p1",
                    "assumption_agent_recursive_verify",
                    True,
                    model="gpt-5.4-mini",
                    component_efficacy=_agent_strong_child_ce(),
                ),
            ]
            states = [ShardRunState(spec=specs[0], command=[], status="completed", returncode=0)]
            payload = aggregate_parallel_payload(
                eval_id="unfair",
                specs=specs,
                states=states,
                shard_payloads=[_payload(rows)],
                execute_live=True,
                models="gpt-5.4-mini",
                variants="raw,hipporag_baseline,assumption_agent_recursive_verify",
                total_sample_size=1,
                shard_size=1,
                parallel_workers=1,
                soft_timeout_sec=900,
                kill_on_soft_timeout=False,
            )

        self.assertTrue(payload["pass"])
        self.assertFalse(payload["paper_clean_pass"])
        self.assertIn("model_budget_fairness_accounted", payload["paper_clean_failed_gates"])

    def test_runner_env_sets_retry_and_global_concurrency_without_secrets(self) -> None:
        env = build_runner_env(
            model_router_attempts=7,
            model_router_timeout=7200,
            model_router_per_attempt_timeout=90,
            model_router_backoff_base_sec=1.25,
            model_router_global_concurrency=2,
            model_router_global_concurrency_dir="/tmp/hle-slots",
            model_router_global_slot_ttl_sec=1800,
            model_router_global_slot_wait_sec=2400,
        )
        self.assertEqual(env["MODEL_ROUTER_ATTEMPTS"], "7")
        self.assertEqual(env["MODEL_ROUTER_TIMEOUT"], "7200")
        self.assertEqual(env["MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"], "90")
        self.assertEqual(env["MODEL_ROUTER_BACKOFF_BASE_SEC"], "1.25")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_CONCURRENCY"], "2")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR"], "/tmp/hle-slots")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC"], "1800")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC"], "2400")
        configured_values = " ".join(
            env[key]
            for key in (
                "MODEL_ROUTER_ATTEMPTS",
                "MODEL_ROUTER_TIMEOUT",
                "MODEL_ROUTER_PER_ATTEMPT_TIMEOUT",
                "MODEL_ROUTER_BACKOFF_BASE_SEC",
                "MODEL_ROUTER_GLOBAL_CONCURRENCY",
                "MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR",
                "MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC",
                "MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC",
            )
        )
        self.assertNotIn("sk-", configured_values)
        self.assertNotIn("hf_", configured_values)

    def test_pollution_audit_tracks_generic_context_selection_and_endpoint_scope(self) -> None:
        rows = [
            _row(
                "p1",
                "assumption_agent_recursive_verify",
                True,
                component_efficacy={
                    "flags": {
                        "graph_context_injected": True,
                        "agent_hipporag_context_activated": True,
                        "morphism_hit": True,
                    },
                    "graph": {"status": "activated", "top_node_type_counts": {"harness": 3}},
                    "agent_hipporag": {"status": "activated"},
                    "morphism": {"structural_hit_count": 1},
                    "selection": {"method": "normalized_majority"},
                },
            ),
            _row("p1", "raw", False, error_type="RuntimeError"),
        ]
        metrics = {
            "raw_content_persisted": False,
            "duplicate_sample_problem_count": 0,
            "planned_live_model_calls": 2,
            "resolved_live_model_calls": 1,
            "live_model_calls_executed": 1,
            "underlying_model_calls_executed": 1,
            "clean_shared_subset": {
                "gpt-5.4-mini": {
                    "shared_clean_problem_count": 0,
                    "by_variant": {},
                }
            },
        }
        errors = {
            "top_level_error_count": 1,
            "process_timeout_count": 0,
            "top_level_errors_by_variant": {"raw": 1},
        }
        audit = build_pollution_audit(
            rows=rows,
            shard_payloads=[_payload(rows)],
            metrics=metrics,
            error_stratification=errors,
            execute_live=True,
        )
        summary = audit["context_pollution"]["summary"]
        self.assertEqual(summary["graph_generic_harness_context"], 1)
        self.assertEqual(summary["hipporag_context_correct"], 1)
        self.assertEqual(summary["morphism_correct"], 1)
        self.assertEqual(
            audit["module_credit_assignment"]["by_selection_method"]["normalized_majority"]["accuracy"],
            1.0,
        )
        self.assertEqual(
            audit["claim_guard"]["recommended_hle_claim_scope"],
            "clean_shared_subset_due_to_endpoint_noise",
        )
        self.assertIn("clean_shared_subset_available_if_endpoint_errors", audit["failed_gates"])

    def test_pollution_audit_separates_graph_retrieval_from_context_injection(self) -> None:
        rows = [
            _row(
                "p1",
                "assumption_agent_recursive_verify",
                False,
                component_efficacy={
                    "flags": {
                        "graph_retrieved_nodes": True,
                        "graph_context_injected": False,
                        "graph_context_discarded": True,
                        "generic_graph_context_only": True,
                    },
                    "graph": {"status": "activated", "top_node_type_counts": {"harness": 4}},
                    "world_model": {
                        "decision": "abstain_to_raw_prompt",
                        "context_abstain_reason": "generic_harness_graph_context_only",
                    },
                },
            )
        ]
        audit = build_pollution_audit(
            rows=rows,
            shard_payloads=[_payload(rows)],
            metrics={
                "raw_content_persisted": False,
                "duplicate_sample_problem_count": 0,
                "planned_live_model_calls": 1,
                "resolved_live_model_calls": 1,
                "live_model_calls_executed": 1,
                "underlying_model_calls_executed": 1,
                "clean_shared_subset": {},
            },
            error_stratification={
                "top_level_error_count": 0,
                "process_timeout_count": 0,
                "top_level_errors_by_variant": {},
            },
            execute_live=True,
        )
        summary = audit["context_pollution"]["summary"]
        self.assertEqual(summary["graph_retrieval_activated"], 1)
        self.assertEqual(summary["graph_generic_harness_retrieved"], 1)
        self.assertEqual(summary["graph_context_discarded"], 1)
        self.assertNotIn("graph_context_used", summary)

    def test_failure_diagnostics_bucket_agent_loss_sources(self) -> None:
        agent = _row(
            "p1",
            "assumption_agent_recursive_verify",
            False,
            component_efficacy={
                "flags": {
                    "verified_or_abstain_abstained": True,
                    "morphism_hit": True,
                    "strong_morphism_hit": False,
                    "morphism_context_injected": True,
                },
                "selection": {
                    "selection_method": "verified_or_abstain_direct_fallback",
                    "verified_or_abstain_gate": {"status": "abstained"},
                },
            },
        )
        agent.update({
            "answer_type": "exactMatch",
            "category": "Math",
            "raw_subject": "Mathematics",
        })
        rows = [
            agent,
            _row("p1", "raw", False),
            _row("p1", "hipporag_baseline", False),
        ]

        diagnostics = build_failure_diagnostics(rows=rows)

        self.assertEqual(diagnostics["agent_problem_count"], 1)
        self.assertEqual(diagnostics["agent_failure_buckets"]["math_exact_failed"], 1)
        self.assertEqual(diagnostics["agent_failure_buckets"]["weak_morphism_unhelpful"], 1)
        self.assertEqual(diagnostics["agent_failure_buckets"]["verified_or_abstain_fallback_wrong"], 1)
        self.assertEqual(diagnostics["agent_gain_loss"]["raw_also_wrong_agent_no_gain"], 1)
        self.assertEqual(diagnostics["agent_gain_loss"]["hipporag_also_wrong_agent_no_gain"], 1)
        self.assertEqual(diagnostics["agent_gain_loss"]["all_three_wrong"], 1)
        self.assertEqual(
            diagnostics["agent_selection_methods"]["verified_or_abstain_direct_fallback"],
            1,
        )
        self.assertEqual(diagnostics["verified_or_abstain_gate_status"]["abstained"], 1)


def _row(
    problem_id: str,
    variant: str,
    correct: bool,
    *,
    model: str = "gpt-5.4-mini",
    error_type: str | None = None,
    component_efficacy: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "problem_id_hash": problem_id,
        "question_hash": f"q-{problem_id}",
        "answer_hash": f"a-{problem_id}",
        "model": model,
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
        "component_efficacy": component_efficacy or {},
        "error": {"type": error_type, "message": "synthetic"} if error_type else None,
    }


def _agent_single_call_ce() -> dict[str, object]:
    return {
        "kind": "assumption_agent_recursive_verify",
        "flags": {},
        "recursive": {
            "status": "disabled",
            "planned_child_count": 0,
            "child_count": 0,
            "answered_child_count": 0,
        },
        "selection": {
            "selection_method": "single_call_fallback",
            "verifier_model_call": False,
        },
    }


def _agent_strong_child_ce() -> dict[str, object]:
    return {
        "kind": "assumption_agent_recursive_verify",
        "flags": {
            "recursive_child_validation_activated": True,
            "child_model_used": True,
            "critic_model_used": True,
        },
        "recursive": {
            "status": "activated",
            "base_model": "gpt-5.4-mini",
            "child_model": "gpt-5.5",
            "planned_child_count": 4,
            "child_count": 4,
            "answered_child_count": 4,
        },
        "child_model": {
            "status": "activated",
            "base_model": "gpt-5.4-mini",
            "child_model": "gpt-5.5",
        },
        "critic_model": {
            "status": "activated",
            "base_model": "gpt-5.4-mini",
            "critic_model": "gpt-5.5",
        },
        "selection": {
            "selection_method": "normalized_majority",
            "verifier_model_call": True,
        },
    }


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    sample_hashes = sorted({str(row["problem_id_hash"]) for row in rows})
    return {
        "rows": rows,
        "sampling": {
            "sample_problem_hashes": sample_hashes,
        },
        "metrics": {
            "sample_count": len(sample_hashes),
            "planned_live_model_calls": len(rows),
            "live_model_calls_executed": len(rows),
            "underlying_model_calls_executed": len(rows),
            "resolved_live_model_calls": len(rows),
            "raw_content_persisted": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
