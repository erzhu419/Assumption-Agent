from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from assumption_os.hle_module_ablation_runner import (
    build_ablation_plan,
    selected_ablation_profiles,
)
from assumption_os.hle_parallel_shard_runner import (
    ShardRunState,
    aggregate_parallel_payload,
    apply_generalization_holdout_defaults,
    apply_hle_offline_defaults,
    apply_live_network_defaults,
    build_error_stratification,
    build_endpoint_retry_manifest,
    build_failure_diagnostics,
    build_heartbeat,
    build_model_budget_fairness_audit,
    build_payload_without_execution,
    build_pollution_audit,
    build_runner_env,
    build_split_fair_controls_payload,
    build_shard_command,
    build_shard_specs,
    build_shard_specs_for_seed_offsets,
    dedupe_shard_specs_by_sample_hash,
    distinct_shard_sample_requirement_violation,
    format_parallel_markdown,
    mark_reusable_completed_shards,
    model_router_policy_from_env,
    model_router_primary_key_present,
    run_live_model_preflight,
    run_parallel_shards,
    runtime_feature_flags_from_args,
    source_policy_from_env,
)


class TestHleParallelShardRunner(unittest.TestCase):
    def _ablation_args(self, tmp: str) -> Namespace:
        return Namespace(
            root=tmp,
            eval_id="ablate",
            profiles=(
                "full,no_graph,no_evidence,no_morphism,no_option_evidence,no_candidate_claim_verifier,"
                "no_world_model,no_recursive,raw_preserve_selector,hipporag_preserve_selector,verified_gate_off"
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
            exclude_artifact_glob="phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/hle*.json*",
            dedupe_shard_samples=True,
            dedupe_shard_max_attempts=11,
            generalization_holdout=False,
            generalization_holdout_preserve_explicit_seed_offsets=False,
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
            model_router_subprocess_calls=True,
            model_router_no_byte_timeout_sec=120,
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

    def test_build_shard_specs_for_explicit_seed_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs"
            md_dir = Path(tmp) / "md"
            specs = build_shard_specs_for_seed_offsets(
                eval_id="explicit",
                seed_offsets=[498, 499, 527],
                run_dir=run_dir,
                md_dir=md_dir,
            )
        self.assertEqual([spec.sample_size for spec in specs], [1, 1, 1])
        self.assertEqual([spec.seed_offset for spec in specs], [498, 499, 527])
        self.assertEqual([spec.eval_id for spec in specs], ["explicit_shard_000", "explicit_shard_001", "explicit_shard_002"])

    def test_generalization_holdout_remaps_explicit_seed_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._ablation_args(tmp)
            args.eval_id = "generalization"
            args.total_sample_size = 2
            args.shard_size = 1
            args.seed_offsets = "10,20"
            args.generalization_holdout = True
            args.exclude_existing_hle_artifacts = False
            args.dedupe_shard_samples = False
            apply_generalization_holdout_defaults(args)

            def fake_dedupe(**kwargs):
                self.assertTrue(kwargs["exclude_existing_hle_artifacts"])
                self.assertEqual([spec.seed_offset for spec in kwargs["specs"]], [10, 20])
                return (
                    [replace(spec, seed_offset=spec.seed_offset + 1000) for spec in kwargs["specs"]],
                    {
                        "enabled": True,
                        "status": "ok",
                        "raw_content_persisted": False,
                        "distinct_problem_hash_count": 2,
                    },
                )

            with patch(
                "assumption_os.hle_parallel_shard_runner.dedupe_shard_specs_by_sample_hash",
                side_effect=fake_dedupe,
            ) as dedupe:
                specs, states = build_payload_without_execution(args)

        self.assertTrue(args.exclude_existing_hle_artifacts)
        self.assertTrue(args.dedupe_shard_samples)
        self.assertTrue(args._generalization_holdout_policy["explicit_seed_offsets_remapped"])
        dedupe.assert_called_once()
        self.assertEqual([spec.seed_offset for spec in specs], [1010, 1020])
        self.assertEqual(args._shard_sample_dedupe_summary["status"], "ok")
        self.assertIn("--exclude-existing-hle-artifacts", states[0].command)

    def test_generalization_holdout_can_preserve_preflighted_explicit_seed_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._ablation_args(tmp)
            args.eval_id = "generalization_preflighted"
            args.total_sample_size = 2
            args.shard_size = 1
            args.seed_offsets = "2022,2087"
            args.generalization_holdout = True
            args.generalization_holdout_preserve_explicit_seed_offsets = True
            args.exclude_existing_hle_artifacts = False
            args.dedupe_shard_samples = True
            apply_generalization_holdout_defaults(args)

            with patch(
                "assumption_os.hle_parallel_shard_runner.dedupe_shard_specs_by_sample_hash",
            ) as dedupe:
                specs, states = build_payload_without_execution(args)

        self.assertFalse(args.exclude_existing_hle_artifacts)
        self.assertFalse(args.dedupe_shard_samples)
        self.assertFalse(args._generalization_holdout_policy["explicit_seed_offsets_remapped"])
        self.assertTrue(args._generalization_holdout_policy["explicit_seed_offsets_preserved"])
        self.assertFalse(args._generalization_holdout_policy["exclude_existing_hle_artifacts"])
        dedupe.assert_not_called()
        self.assertEqual([spec.seed_offset for spec in specs], [2022, 2087])
        self.assertEqual(
            args._shard_sample_dedupe_summary["reason"],
            "preflighted_explicit_seed_offsets_preserved_for_generalization_holdout",
        )
        self.assertNotIn("--exclude-existing-hle-artifacts", states[0].command)

    def test_distinct_shard_sample_requirement_violation_flags_duplicate_fallback(self) -> None:
        violation = distinct_shard_sample_requirement_violation(
            dedupe_summary={
                "enabled": True,
                "accepted_shard_count": 5,
                "duplicate_fallback_count": 7,
                "distinct_problem_hash_count": 5,
                "raw_content_persisted": False,
            },
            shard_count=12,
        )

        self.assertIsNotNone(violation)
        self.assertEqual(violation["reason"], "distinct_shard_sample_requirement_not_met")
        self.assertEqual(violation["duplicate_fallback_count"], 7)
        self.assertFalse(violation["raw_content_persisted"])

    def test_distinct_shard_sample_requirement_violation_accepts_full_distinct_cohort(self) -> None:
        violation = distinct_shard_sample_requirement_violation(
            dedupe_summary={
                "enabled": True,
                "accepted_shard_count": 12,
                "duplicate_fallback_count": 0,
                "distinct_problem_hash_count": 12,
                "raw_content_persisted": False,
            },
            shard_count=12,
        )

        self.assertIsNone(violation)

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
        self.assertEqual(
            by_profile["hipporag_preserve_selector"]["env_overrides"]["HLE_ENABLE_COST_AWARE_HIPPORAG_PRESERVE_SELECTOR"],
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
        self.assertIn("--model-router-subprocess-calls", raw_preserve_command)
        self.assertEqual(
            raw_preserve_command[raw_preserve_command.index("--model-router-no-byte-timeout-sec") + 1],
            "120",
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
                variant_total_timeout_sec=1200,
                variant_total_model_call_budget=9,
                enable_assumption_operators=True,
                assumption_operator_domains="science,hle_general",
                assumption_operator_max_specs=3,
                enable_assumption_operator_retrieval_fallback=True,
                assumption_operator_fallback_min_score=0.2,
                enable_operator_application_verifier=True,
                enable_operator_policy_gate=True,
                disable_domain_rule_verifier=True,
                enable_option_claim_contrastive_adjudicator=True,
                enable_option_claim_span_directness_verifier=True,
                enable_option_claim_relation_span_comparator=True,
                enable_option_claim_relation_query_planner=True,
                enable_option_claim_source_cache_corpus_backfill=True,
                enable_option_claim_source_verifier_repair_context=True,
                enable_option_claim_source_verifier_acceptance_quality_gate=True,
                enable_option_claim_source_verifier_structured_context=True,
            )
        text = " ".join(cmd)
        self.assertIn("--hard-exit-after-write", cmd)
        self.assertIn("--execute-live", cmd)
        self.assertIn("--variant-total-timeout-sec 1200", text)
        self.assertIn("--variant-total-model-call-budget 9", text)
        self.assertIn("--disable-evidence-bridge", cmd)
        self.assertIn("--enable-assumption-operators", cmd)
        self.assertIn("--assumption-operator-domains science,hle_general", text)
        self.assertIn("--assumption-operator-max-specs 3", text)
        self.assertIn("--enable-assumption-operator-retrieval-fallback", cmd)
        self.assertIn("--assumption-operator-fallback-min-score 0.2", text)
        self.assertIn("--enable-operator-application-verifier", cmd)
        self.assertIn("--enable-operator-policy-gate", cmd)
        self.assertIn("--disable-domain-rule-verifier", cmd)
        self.assertIn("--enable-option-claim-contrastive-adjudicator", cmd)
        self.assertIn("--enable-option-claim-span-directness-verifier", cmd)
        self.assertIn("--enable-option-claim-relation-span-comparator", cmd)
        self.assertIn("--enable-option-claim-relation-query-planner", cmd)
        self.assertIn("--enable-option-claim-source-cache-corpus-backfill", cmd)
        self.assertIn("--enable-option-claim-source-verifier-repair-context", cmd)
        self.assertIn("--enable-option-claim-source-verifier-acceptance-quality-gate", cmd)
        self.assertIn("--enable-option-claim-source-verifier-structured-context", cmd)
        self.assertIn("--exclude-existing-hle-artifacts", cmd)
        self.assertNotIn("sk-", text)
        self.assertNotIn("hf_", text)

    def test_runtime_feature_flags_and_source_policy_record_toggles_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._ablation_args(tmp)
            args.enable_assumption_operators = True
            args.disable_assumption_operators = False
            args.assumption_operator_domains = "science,hle_general"
            args.assumption_operator_skip_domains = "business"
            args.assumption_operator_max_specs = 2
            args.allow_assumption_operators_without_context = False
            args.enable_assumption_operator_retrieval_fallback = True
            args.enable_operator_application_verifier = True
            args.enable_operator_policy_gate = True
            args.disable_domain_rule_verifier = True
            args.enable_option_claim_contrastive_adjudicator = True
            args.disable_option_claim_contrastive_adjudicator = False
            args.enable_option_claim_span_directness_verifier = True
            args.disable_option_claim_span_directness_verifier = False
            args.enable_option_claim_relation_span_comparator = True
            args.disable_option_claim_relation_span_comparator = False
            args.enable_option_claim_relation_query_planner = True
            args.disable_option_claim_relation_query_planner = False
            args.enable_option_claim_source_cache_corpus_backfill = True
            args.disable_option_claim_source_cache_corpus_backfill = False
            args.enable_option_claim_source_verifier_repair_context = True
            args.disable_option_claim_source_verifier_repair_context = False
            args.enable_option_claim_source_verifier_acceptance_quality_gate = True
            args.disable_option_claim_source_verifier_acceptance_quality_gate = False
            args.enable_option_claim_source_verifier_structured_context = True
            args.disable_option_claim_source_verifier_structured_context = False
            args.recursive_selection_model_call_budget = 2
            args.recursive_selection_wallclock_budget_sec = 180

            flags = runtime_feature_flags_from_args(args)

        source_policy = source_policy_from_env({
            "HLE_EVIDENCE_SOURCE_CACHE_ONLY": "1",
            "HLE_SOURCE_SEARCH_CACHE_ONLY": "1",
            "HLE_DISABLE_LIVE_SOURCE_SEARCH": "1",
            "HLE_ALLOW_LIVE_SOURCE_SEARCH": "0",
            "HLE_EVIDENCE_SOURCE_CORPUS_PATHS": "/tmp/local-cache.jsonl",
            "HLE_ENABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER": "1",
            "HLE_DISABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER": "",
            "HLE_ENABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR": "1",
            "HLE_DISABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR": "",
            "HLE_ENABLE_OPTION_CLAIM_EARLY_SOURCE_QUEUE_RELATION_SPAN_COMPARATOR": "1",
            "HLE_DISABLE_OPTION_CLAIM_EARLY_SOURCE_QUEUE_RELATION_SPAN_COMPARATOR": "",
            "HLE_ENABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL": "1",
            "HLE_DISABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL": "",
            "HLE_ENABLE_SOURCE_CACHE_ANSWER_BEARING_OPTION_CLAIM_RETRY": "1",
            "HLE_DISABLE_SOURCE_CACHE_ANSWER_BEARING_OPTION_CLAIM_RETRY": "",
            "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT": "1",
            "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT": "",
            "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE": "1",
            "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE": "",
            "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT": "1",
            "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT": "",
            "HLE_SOURCE_GROUNDED_OPTION_CLAIM_VERIFIER_CANDIDATE_LIMIT": "2",
            "HLE_SOURCE_GROUNDED_OPTION_CLAIM_VERIFIER_MODEL_CALL_LIMIT": "2",
            "HLE_OPTION_CLAIM_SOURCE_DIRECTNESS_MODEL_CALL_CAP": "4",
            "HLE_SOURCE_GROUNDED_OPTION_CLAIM_RETRY_TOP_K": "1",
            "HLE_SOURCE_GROUNDED_OPTION_CLAIM_MISSING_MODEL_RETRY_LIMIT": "1",
            "HLE_SOURCE_GROUNDED_OPTION_CLAIM_SOURCE_QUALITY_CHALLENGER_LIMIT": "1",
            "HLE_LOW_SUPPORT_OPTION_CLAIM_SOURCE_VERIFIER_LIMIT": "1",
            "HLE_ZERO_QUALITY_SWEEP_GAP_OPTION_CLAIM_SOURCE_VERIFIER_LIMIT": "1",
            "HLE_ENABLE_SOURCE_VERIFIER_SEMANTIC_GENERIC_BACKOFF_ATTEMPT_PRESSURE": "1",
            "HLE_SOURCE_VERIFIER_SEMANTIC_GENERIC_BACKOFF_ATTEMPT_PRESSURE_MIN_ATTEMPTS": "2",
            "HLE_SOURCE_VERIFIER_SEMANTIC_GENERIC_BACKOFF_MIN_ATTEMPTS": "2",
            "SEMANTIC_SCHOLAR_API_KEY": "test-semantic-key",
            "OPENALEX_API_KEY": "test-openalex-key",
        })
        self.assertTrue(flags["operator_policy_gate_enabled"])
        self.assertTrue(flags["option_claim_contrastive_adjudicator_enabled"])
        self.assertTrue(flags["option_claim_span_directness_verifier_enabled"])
        self.assertTrue(flags["option_claim_relation_span_comparator_enabled"])
        self.assertTrue(flags["option_claim_relation_query_planner_enabled"])
        self.assertTrue(flags["option_claim_source_cache_corpus_backfill_enabled"])
        self.assertTrue(flags["option_claim_source_verifier_repair_context_enabled"])
        self.assertTrue(flags["option_claim_source_verifier_acceptance_quality_gate_enabled"])
        self.assertTrue(flags["option_claim_source_verifier_structured_context_enabled"])
        self.assertEqual(flags["recursive_selection_model_call_budget"], 2)
        self.assertEqual(flags["recursive_selection_wallclock_budget_sec"], 180)
        self.assertEqual(flags["assumption_operator_domains"], "science,hle_general")
        self.assertEqual(source_policy["source_search_cache_only"], "1")
        self.assertEqual(source_policy["option_claim_relation_query_planner_env"], "1")
        self.assertEqual(source_policy["option_claim_relation_span_comparator_env"], "1")
        self.assertEqual(
            source_policy[
                "option_claim_early_source_queue_relation_span_comparator_env"
            ],
            "1",
        )
        self.assertEqual(source_policy["option_claim_source_cache_corpus_backfill_env"], "1")
        self.assertEqual(source_policy["option_claim_source_verifier_repair_context_env"], "1")
        self.assertEqual(
            source_policy["option_claim_source_verifier_acceptance_quality_gate_env"],
            "1",
        )
        self.assertEqual(
            source_policy["option_claim_source_verifier_structured_context_env"],
            "1",
        )
        self.assertEqual(source_policy["source_grounded_option_claim_verifier_candidate_limit"], "2")
        self.assertEqual(source_policy["source_grounded_option_claim_verifier_model_call_limit"], "2")
        self.assertEqual(source_policy["option_claim_source_directness_model_call_cap"], "4")
        self.assertEqual(source_policy["source_grounded_option_claim_retry_top_k"], "1")
        self.assertEqual(source_policy["source_grounded_option_claim_missing_model_retry_limit"], "1")
        self.assertEqual(source_policy["source_grounded_option_claim_source_quality_challenger_limit"], "1")
        self.assertEqual(source_policy["low_support_option_claim_source_verifier_limit"], "1")
        self.assertEqual(source_policy["zero_quality_sweep_gap_option_claim_source_verifier_limit"], "1")
        self.assertEqual(
            source_policy["source_verifier_semantic_generic_backoff_attempt_pressure_env"],
            "1",
        )
        self.assertEqual(
            source_policy[
                "source_verifier_semantic_generic_backoff_attempt_pressure_min_attempts"
            ],
            "2",
        )
        self.assertEqual(
            source_policy["source_verifier_semantic_generic_backoff_min_attempts"],
            "2",
        )
        self.assertTrue(source_policy["semantic_scholar_api_key_present"])
        self.assertTrue(source_policy["openalex_api_key_present"])
        flattened = json.dumps({"feature_flags": flags, "source_policy": source_policy}, sort_keys=True)
        self.assertNotIn("test-semantic-key", flattened)
        self.assertNotIn("test-openalex-key", flattened)
        self.assertNotIn("sk-", flattened)
        self.assertNotIn("hf_", flattened)

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

    def test_dedupe_single_row_shards_uses_fast_scan_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="dedupe-fast",
                total_sample_size=3,
                shard_size=1,
                seed_offset=498,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )
            candidates = [
                {"scanned_index": 499, "id_hash": "a"},
                {"scanned_index": 500, "id_hash": "b"},
                {"scanned_index": 527, "id_hash": "c"},
            ]
            with patch(
                "assumption_os.hle_parallel_shard_runner._load_text_only_candidate_index",
                return_value=candidates,
            ) as load_index:
                deduped, summary = dedupe_shard_specs_by_sample_hash(
                    root=root,
                    specs=specs,
                    max_scan=1000,
                    seed_stride=1,
                    exclude_existing_hle_artifacts=False,
                    exclude_artifact_glob="unused",
                    sample_answer_type="multipleChoice",
                    sample_subject_contains="",
                    max_attempts=4,
                )

        self.assertTrue(summary["fast_single_pass"])
        self.assertEqual(summary["distinct_problem_hash_count"], 3)
        self.assertEqual([spec.seed_offset for spec in deduped], [498, 499, 526])
        self.assertEqual(
            [row["selected_problem_hashes"] for row in summary["remaps"]],
            [["a"], ["b"], ["c"]],
        )
        load_index.assert_called_once()

    def test_dedupe_single_row_shards_wraps_seed_offsets_past_candidate_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="dedupe-wrap",
                total_sample_size=2,
                shard_size=1,
                seed_offset=2600,
                seed_stride=43,
                run_dir=root,
                md_dir=root,
            )
            candidates = [
                {"scanned_index": 17, "id_hash": "early-a"},
                {"scanned_index": 271, "id_hash": "early-b"},
                {"scanned_index": 2471, "id_hash": "tail-c"},
            ]
            with patch(
                "assumption_os.hle_parallel_shard_runner._load_text_only_candidate_index",
                return_value=candidates,
            ):
                deduped, summary = dedupe_shard_specs_by_sample_hash(
                    root=root,
                    specs=specs,
                    max_scan=8000,
                    seed_stride=43,
                    exclude_existing_hle_artifacts=False,
                    exclude_artifact_glob="unused",
                    sample_answer_type="multipleChoice",
                    sample_subject_contains="",
                    max_attempts=4,
                )

        self.assertEqual(summary["accepted_shard_count"], 2)
        self.assertEqual(summary["distinct_problem_hash_count"], 2)
        self.assertTrue(all(row["status"] == "accepted_wrapped" for row in summary["remaps"]))
        self.assertEqual(len({spec.seed_offset for spec in deduped}), 2)
        self.assertTrue(all(spec.seed_offset in {16, 270, 2470} for spec in deduped))
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
        self.assertEqual(heartbeat["shards"][0]["jsonl_line_count"], 2)
        self.assertIsInstance(heartbeat["shards"][0]["jsonl_age_sec"], float)
        self.assertFalse(heartbeat["raw_content_persisted"])

    def test_heartbeat_includes_running_shard_memory_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="memhb",
                total_sample_size=1,
                shard_size=1,
                seed_offset=7,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )[0]
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; buf='x'*1000000; time.sleep(2)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                state = ShardRunState(
                    spec=spec,
                    command=[sys.executable, "-m", "x"],
                    status="running",
                    process=process,
                )
                heartbeat = build_heartbeat([state])
            finally:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

        shard = heartbeat["shards"][0]
        self.assertEqual(shard["process_pid"], process.pid)
        self.assertGreater(shard["process_memory"]["rss_kb"], 0)
        self.assertEqual(shard["process_peak_rss_kb"], state.peak_rss_kb)

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

    def test_endpoint_retry_manifest_collects_retryable_top_level_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_shard_specs(
                eval_id="retry",
                total_sample_size=1,
                shard_size=1,
                seed_offset=2225,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )[0]
            row = _row("p1", "raw", False, error_type="RuntimeError")
            row["error"]["message"] = (
                "model request failed: RemoteDisconnected: "
                "Remote end closed connection without response"
            )
            payload = _payload([row])
            payload["eval_id"] = spec.eval_id
            payload["sampling"]["seed_offset"] = 2225
            state = ShardRunState(spec=spec, command=[], status="completed", returncode=0)

            manifest = build_endpoint_retry_manifest(
                rows=[row],
                shard_payloads=[payload],
                states=[state],
            )

        self.assertEqual(manifest["retryable_endpoint_error_count"], 1)
        self.assertEqual(manifest["by_variant"], {"raw": 1})
        self.assertEqual(manifest["retry_items"][0]["seed_offset"], 2225)
        self.assertEqual(manifest["retry_items"][0]["retry_key"], "gpt-5.4-mini::raw::p1")
        self.assertFalse(manifest["raw_content_persisted"])

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
                diagnostic_log_out=root / "agg.diagnostic.jsonl",
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
        self.assertEqual(payload["diagnostic_log_out"], str(root / "agg.diagnostic.jsonl"))
        self.assertEqual(payload["logging_policy"]["event_stream"], "jsonl")
        self.assertEqual(
            payload["pollution_audit"]["claim_guard"]["recommended_hle_claim_scope"],
            "full_resolved_rows",
        )
        self.assertIn("HLE Parallel Shard Evaluation", markdown)
        self.assertIn("Pollution Audit", markdown)
        self.assertIn("Model Budget Fairness", markdown)

    def test_aggregate_fails_when_requested_sample_rows_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="partial",
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
                ])
            ]
            states = [ShardRunState(spec=spec, command=[], status="completed", returncode=0) for spec in specs]
            payload = aggregate_parallel_payload(
                eval_id="partial",
                specs=specs,
                states=states,
                shard_payloads=payloads,
                execute_live=True,
                models="gpt-5.4-mini",
                variants="raw,hipporag_baseline,assumption_agent_recursive_verify",
                total_sample_size=2,
                shard_size=1,
                parallel_workers=2,
                soft_timeout_sec=900,
                kill_on_soft_timeout=False,
            )

        self.assertFalse(payload["pass"])
        self.assertFalse(payload["paper_clean_pass"])
        self.assertIn("requested_sample_rows_loaded", payload["failed_gates"])
        self.assertEqual(payload["metrics"]["sample_count"], 1)

    def test_split_fair_controls_combines_variant_batches_without_double_counting_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls = _payload([
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "raw_budget_matched", False),
                _row("p1", "hipporag_budget_matched", False),
                _row("p2", "raw", False),
                _row("p2", "hipporag_baseline", False),
                _row("p2", "raw_budget_matched", False),
                _row("p2", "hipporag_budget_matched", False),
            ])
            controls.update({
                "eval_id": "controls",
                "eval_kind": "hle_parallel_shard_runner",
                "pass": True,
                "paper_clean_pass": True,
                "runtime_policy": {"execute_live": True, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            agent = _payload([
                _row(
                    "p1",
                    "assumption_agent_recursive_verify",
                    True,
                    component_efficacy=_agent_multi_call_same_model_ce(),
                ),
                _row(
                    "p2",
                    "assumption_agent_recursive_verify",
                    False,
                    component_efficacy=_agent_multi_call_same_model_ce(),
                ),
            ])
            agent.update({
                "eval_id": "agent",
                "eval_kind": "hle_parallel_shard_runner",
                "pass": True,
                "paper_clean_pass": False,
                "runtime_policy": {"execute_live": True, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            controls_path = root / "controls.json"
            agent_path = root / "agent.json"
            controls_path.write_text(json.dumps(controls), encoding="utf-8")
            agent_path.write_text(json.dumps(agent), encoding="utf-8")

            payload = build_split_fair_controls_payload(
                eval_id="combined",
                input_paths=[controls_path, agent_path],
                diagnostic_log_out=root / "combined.jsonl",
            )
            markdown = format_parallel_markdown(payload)

        self.assertTrue(payload["pass"])
        self.assertTrue(payload["paper_clean_pass"])
        self.assertEqual(payload["eval_kind"], "hle_split_fair_controls_combined")
        self.assertEqual(payload["metrics"]["sample_count"], 2)
        self.assertEqual(payload["metrics"]["distinct_sample_problem_count"], 2)
        self.assertEqual(payload["metrics"]["scored_row_count"], 10)
        self.assertEqual(payload["split_run_audit"]["failed_gates"], [])
        self.assertEqual(payload["model_budget_fairness_audit"]["failed_gates"], [])
        self.assertEqual(
            payload["metrics"]["by_model_variant"]["gpt-5.4-mini::assumption_agent_recursive_verify"]["accuracy"],
            0.5,
        )
        self.assertEqual(payload["metrics"]["by_model_variant"]["gpt-5.4-mini::raw"]["accuracy"], 0.0)
        self.assertIn("HLE Split Fair Controls Combined Evaluation", markdown)
        self.assertIn("Split Inputs", markdown)

    def test_split_fair_controls_fails_when_inputs_cover_different_problem_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls = _payload([
                _row("p1", "raw", False),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "raw_budget_matched", False),
                _row("p1", "hipporag_budget_matched", False),
                _row("p2", "raw", False),
                _row("p2", "hipporag_baseline", False),
                _row("p2", "raw_budget_matched", False),
                _row("p2", "hipporag_budget_matched", False),
            ])
            controls.update({
                "eval_id": "controls",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {"execute_live": False, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            agent = _payload([
                _row(
                    "p1",
                    "assumption_agent_recursive_verify",
                    True,
                    component_efficacy=_agent_multi_call_same_model_ce(),
                )
            ])
            agent.update({
                "eval_id": "agent",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {"execute_live": False, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            controls_path = root / "controls.json"
            agent_path = root / "agent.json"
            controls_path.write_text(json.dumps(controls), encoding="utf-8")
            agent_path.write_text(json.dumps(agent), encoding="utf-8")

            payload = build_split_fair_controls_payload(
                eval_id="combined-mismatch",
                input_paths=[controls_path, agent_path],
            )

        self.assertFalse(payload["pass"])
        self.assertFalse(payload["paper_clean_pass"])
        self.assertIn("split_inputs_cover_same_problem_set", payload["failed_gates"])
        self.assertEqual(
            payload["split_run_audit"]["problem_set_mismatches"][0]["missing_from_reference"],
            ["p2"],
        )

    def test_split_retry_clean_replacement_can_clear_endpoint_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls = _payload([
                _row("p1", "raw", False, error_type="RuntimeError"),
                _row("p1", "hipporag_baseline", False),
                _row("p1", "raw_budget_matched", False),
                _row("p1", "hipporag_budget_matched", False),
            ])
            controls.update({
                "eval_id": "controls",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {"execute_live": True, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            agent = _payload([
                _row(
                    "p1",
                    "assumption_agent_recursive_verify",
                    True,
                    component_efficacy=_agent_multi_call_same_model_ce(),
                )
            ])
            agent.update({
                "eval_id": "agent",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {"execute_live": True, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            retry = _payload([
                _row("p1", "raw", False),
            ])
            retry.update({
                "eval_id": "raw-retry",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {"execute_live": True, "raw_content_persisted": False},
                "raw_content_persisted": False,
            })
            controls_path = root / "controls.json"
            agent_path = root / "agent.json"
            retry_path = root / "retry.json"
            controls_path.write_text(json.dumps(controls), encoding="utf-8")
            agent_path.write_text(json.dumps(agent), encoding="utf-8")
            retry_path.write_text(json.dumps(retry), encoding="utf-8")

            payload = build_split_fair_controls_payload(
                eval_id="combined-retry",
                input_paths=[controls_path, agent_path],
                retry_input_paths=[retry_path],
                allow_clean_retry_replacements=True,
            )

        self.assertTrue(payload["pass"])
        self.assertTrue(payload["paper_clean_pass"])
        self.assertEqual(payload["metrics"]["sample_count"], 1)
        self.assertEqual(payload["split_run_audit"]["retry_input_count"], 1)
        duplicate_rows = payload["split_run_audit"]["duplicate_rows"]
        self.assertEqual(duplicate_rows["allowed_clean_retry_replacement_count"], 1)
        self.assertEqual(duplicate_rows["retry_new_variant_problem_row_count"], 0)
        self.assertEqual(payload["error_stratification"]["top_level_error_count"], 0)
        self.assertEqual(payload["split_run_audit"]["failed_gates"], [])

    def test_split_retry_rejects_changed_inference_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls = _payload([_row("p1", "raw", False, error_type="RuntimeError")])
            controls.update({
                "eval_id": "controls",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {
                    "execute_live": True,
                    "model_router": {"max_tokens": 512, "reasoning_effort": ""},
                    "raw_content_persisted": False,
                },
                "raw_content_persisted": False,
            })
            retry = _payload([_row("p1", "raw", False)])
            retry.update({
                "eval_id": "raw-retry",
                "eval_kind": "hle_parallel_shard_runner",
                "runtime_policy": {
                    "execute_live": True,
                    "model_router": {"max_tokens": 128, "reasoning_effort": "low"},
                    "raw_content_persisted": False,
                },
                "raw_content_persisted": False,
            })
            controls_path = root / "controls.json"
            retry_path = root / "retry.json"
            controls_path.write_text(json.dumps(controls), encoding="utf-8")
            retry_path.write_text(json.dumps(retry), encoding="utf-8")

            payload = build_split_fair_controls_payload(
                eval_id="combined-policy-mismatch",
                input_paths=[controls_path],
                retry_input_paths=[retry_path],
                allow_clean_retry_replacements=True,
            )

        self.assertFalse(payload["pass"])
        self.assertFalse(payload["paper_clean_pass"])
        self.assertIn(
            "split_inputs_share_inference_policy",
            payload["split_run_audit"]["failed_gates"],
        )
        self.assertEqual(
            payload["split_run_audit"]["inference_policies"],
            [
                {"max_tokens": 512, "reasoning_effort": ""},
                {"max_tokens": 128, "reasoning_effort": "low"},
            ],
        )

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

    def test_aggregate_fails_when_selected_operator_never_activates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="operator-noop",
                total_sample_size=1,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )
            operator_noop_ce = {
                "kind": "assumption_agent_recursive_verify",
                "flags": {
                    "operator_specs_requested": True,
                    "operator_specs_activated": False,
                    "operator_context_injected": False,
                    "operator_specs_blocked": True,
                },
                "operator_specs": {
                    "status": "skipped",
                    "reason": "generic_harness_graph_context_only",
                    "operator_count": 0,
                    "operator_source_types": [],
                },
                "selection": {"selection_method": "normalized_majority"},
            }
            rows = [
                _row("p1", "raw", False, model="gpt-5.4-mini"),
                _row("p1", "hipporag_baseline", False, model="gpt-5.4-mini"),
                _row("p1", "raw_budget_matched", False, model="gpt-5.4-mini"),
                _row("p1", "hipporag_budget_matched", False, model="gpt-5.4-mini"),
                _row(
                    "p1",
                    "assumption_agent_recursive_verify",
                    True,
                    model="gpt-5.4-mini",
                    component_efficacy=operator_noop_ce,
                ),
            ]
            states = [ShardRunState(spec=specs[0], command=[], status="completed", returncode=0)]
            payload = aggregate_parallel_payload(
                eval_id="operator-noop",
                specs=specs,
                states=states,
                shard_payloads=[_payload(rows)],
                execute_live=True,
                models="gpt-5.4-mini",
                variants=(
                    "raw,hipporag_baseline,raw_budget_matched,hipporag_budget_matched,"
                    "assumption_agent_recursive_verify"
                ),
                total_sample_size=1,
                shard_size=1,
                parallel_workers=1,
                soft_timeout_sec=900,
                kill_on_soft_timeout=False,
            )

        self.assertFalse(payload["pass"])
        self.assertIn("assumption_operator_activated_if_selected", payload["failed_gates"])
        summary = payload["metrics"]["operator_activation_summary"]
        self.assertEqual(summary["selected_row_count"], 1)
        self.assertEqual(summary["activated_row_count"], 0)
        self.assertEqual(summary["reason_counts"]["generic_harness_graph_context_only"], 1)

    def test_aggregate_allows_context_gate_abstain_without_operator_activation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_shard_specs(
                eval_id="operator-context-abstain",
                total_sample_size=1,
                shard_size=1,
                seed_offset=0,
                seed_stride=1,
                run_dir=root,
                md_dir=root,
            )
            operator_context_abstain_ce = {
                "kind": "assumption_agent_recursive_verify",
                "flags": {
                    "operator_specs_requested": True,
                    "operator_specs_activated": False,
                    "operator_context_injected": False,
                    "operator_specs_blocked": True,
                },
                "operator_specs": {
                    "status": "skipped",
                    "reason": "context_gate_abstained",
                    "operator_count": 0,
                    "operator_source_types": [],
                },
                "selection": {"selection_method": "verified_or_abstain_direct_fallback"},
            }
            rows = [
                _row("p1", "raw", False, model="gpt-5.4-mini"),
                _row("p1", "hipporag_baseline", False, model="gpt-5.4-mini"),
                _row("p1", "raw_budget_matched", False, model="gpt-5.4-mini"),
                _row("p1", "hipporag_budget_matched", False, model="gpt-5.4-mini"),
                _row(
                    "p1",
                    "assumption_agent_recursive_verify",
                    True,
                    model="gpt-5.4-mini",
                    component_efficacy=operator_context_abstain_ce,
                ),
            ]
            states = [ShardRunState(spec=specs[0], command=[], status="completed", returncode=0)]
            payload = aggregate_parallel_payload(
                eval_id="operator-context-abstain",
                specs=specs,
                states=states,
                shard_payloads=[_payload(rows)],
                execute_live=True,
                models="gpt-5.4-mini",
                variants=(
                    "raw,hipporag_baseline,raw_budget_matched,hipporag_budget_matched,"
                    "assumption_agent_recursive_verify"
                ),
                total_sample_size=1,
                shard_size=1,
                parallel_workers=1,
                soft_timeout_sec=900,
                kill_on_soft_timeout=False,
                feature_flags={
                    "option_claim_contrastive_adjudicator_enabled": True,
                    "option_claim_span_directness_verifier_enabled": True,
                    "raw_content_persisted": False,
                },
                source_policy={
                    "evidence_source_cache_only": "1",
                    "source_search_cache_only": "1",
                    "live_source_search_disabled": "1",
                    "live_source_search_allowed": "0",
                    "raw_content_persisted": False,
                },
            )

        self.assertTrue(payload["pass"])
        self.assertNotIn("assumption_operator_activated_if_selected", payload["failed_gates"])
        self.assertTrue(
            payload["runtime_policy"]["feature_flags"]["option_claim_contrastive_adjudicator_enabled"]
        )
        self.assertEqual(payload["runtime_policy"]["source_policy"]["source_search_cache_only"], "1")
        summary = payload["metrics"]["operator_activation_summary"]
        self.assertEqual(summary["requested_row_count"], 1)
        self.assertEqual(summary["selected_row_count"], 0)
        self.assertEqual(summary["context_abstained_row_count"], 1)
        self.assertEqual(summary["blocked_row_count"], 1)
        self.assertEqual(summary["reason_counts"]["context_gate_abstained"], 1)

    def test_runner_env_sets_retry_and_global_concurrency_without_secrets(self) -> None:
        env = build_runner_env(
            model_router_attempts=7,
            model_router_timeout=7200,
            model_router_transient_extra_attempts=0,
            variant_total_model_router_attempt_budget=20,
            variant_total_model_router_sec_budget=360,
            enable_option_claim_relation_query_planner=True,
            parallel_workers=4,
            model_router_per_attempt_timeout=90,
            model_router_reasoning_effort="low",
            model_router_subprocess_calls=True,
            model_router_no_byte_timeout_sec=120,
            model_router_backoff_base_sec=1.25,
            model_router_global_concurrency=2,
            model_router_global_concurrency_dir="/tmp/hle-slots",
            model_router_global_slot_ttl_sec=1800,
            model_router_global_slot_wait_sec=2400,
            recursive_selection_model_call_budget=2,
            recursive_selection_wallclock_budget_sec=180,
        )
        self.assertEqual(env["MODEL_ROUTER_ATTEMPTS"], "7")
        self.assertEqual(env["MODEL_ROUTER_TRANSIENT_EXTRA_ATTEMPTS"], "0")
        self.assertEqual(env["HLE_RECURSIVE_CHILD_MODEL_ROUTER_ATTEMPTS"], "1")
        self.assertEqual(env["HLE_RECURSIVE_CHILD_MODEL_ROUTER_TRANSIENT_EXTRA_ATTEMPTS"], "0")
        self.assertEqual(env["HLE_VARIANT_RELATION_COMPARATOR_MODEL_CALL_MIN_REMAINING_SEC"], "60")
        self.assertEqual(env["HLE_SOURCE_GROUNDED_OPTION_CLAIM_VERIFIER_CANDIDATE_LIMIT"], "2")
        self.assertEqual(env["HLE_SOURCE_GROUNDED_OPTION_CLAIM_RETRY_TOP_K"], "1")
        self.assertEqual(env["HLE_SOURCE_GROUNDED_OPTION_CLAIM_MISSING_MODEL_RETRY_LIMIT"], "1")
        self.assertEqual(env["HLE_SOURCE_GROUNDED_OPTION_CLAIM_SOURCE_QUALITY_CHALLENGER_LIMIT"], "1")
        self.assertEqual(env["HLE_LOW_SUPPORT_OPTION_CLAIM_SOURCE_VERIFIER_LIMIT"], "1")
        self.assertEqual(env["HLE_ZERO_QUALITY_SWEEP_GAP_OPTION_CLAIM_SOURCE_VERIFIER_LIMIT"], "1")
        self.assertNotIn("HLE_ENABLE_SOURCE_VERIFIER_SEMANTIC_GENERIC_BACKOFF_ATTEMPT_PRESSURE", env)
        self.assertNotIn("HLE_SOURCE_VERIFIER_SEMANTIC_GENERIC_BACKOFF_ATTEMPT_PRESSURE_MIN_ATTEMPTS", env)
        self.assertNotIn("HLE_SOURCE_VERIFIER_SEMANTIC_GENERIC_BACKOFF_MIN_ATTEMPTS", env)
        self.assertEqual(env["HLE_VARIANT_TOTAL_MODEL_ROUTER_ATTEMPT_BUDGET"], "20")
        self.assertEqual(env["HLE_VARIANT_TOTAL_MODEL_ROUTER_SEC_BUDGET"], "360.0")
        self.assertEqual(env["HLE_ENABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER"], "1")
        self.assertNotIn("HLE_DISABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER", env)
        self.assertEqual(env["MODEL_ROUTER_TIMEOUT"], "7200")
        self.assertEqual(env["MODEL_ROUTER_PER_ATTEMPT_TIMEOUT"], "90")
        self.assertEqual(env["MODEL_ROUTER_REASONING_EFFORT"], "low")
        self.assertEqual(env["MODEL_ROUTER_SUBPROCESS_CALLS"], "1")
        self.assertEqual(env["MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC"], "120")
        self.assertEqual(env["HLE_PARALLEL_SHARD_WORKERS"], "4")
        self.assertEqual(env["MODEL_ROUTER_BACKOFF_BASE_SEC"], "1.25")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_CONCURRENCY"], "2")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR"], "/tmp/hle-slots")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC"], "1800")
        self.assertEqual(env["MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC"], "2400")
        self.assertEqual(env["HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET"], "2")
        self.assertEqual(env["HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC"], "180.0")
        policy = model_router_policy_from_env(env)
        self.assertEqual(policy["subprocess_calls"], "1")
        self.assertEqual(policy["reasoning_effort"], "low")
        self.assertEqual(policy["subprocess_no_byte_timeout_sec"], "120")
        self.assertEqual(policy["recursive_child_attempts"], "1")
        self.assertEqual(policy["recursive_child_transient_extra_attempts"], "0")
        self.assertEqual(policy["parallel_shard_workers"], "4")
        self.assertEqual(policy["variant_total_model_router_attempt_budget"], "20")
        self.assertEqual(policy["variant_total_model_router_sec_budget"], "360.0")
        self.assertEqual(policy["recursive_selection_model_call_budget"], "2")
        self.assertEqual(policy["recursive_selection_wallclock_budget_sec"], "180.0")
        self.assertEqual(policy["variant_relation_comparator_model_call_min_remaining_sec"], "60")
        self.assertFalse(policy["raw_content_persisted"])
        source_policy = source_policy_from_env(env)
        self.assertEqual(source_policy["source_grounded_option_claim_verifier_candidate_limit"], "2")
        self.assertEqual(source_policy["source_grounded_option_claim_verifier_model_call_limit"], "2")
        self.assertEqual(source_policy["source_grounded_option_claim_retry_top_k"], "1")
        self.assertEqual(source_policy["source_grounded_option_claim_missing_model_retry_limit"], "1")
        self.assertEqual(source_policy["source_grounded_option_claim_source_quality_challenger_limit"], "1")
        self.assertEqual(source_policy["low_support_option_claim_source_verifier_limit"], "1")
        self.assertEqual(source_policy["zero_quality_sweep_gap_option_claim_source_verifier_limit"], "1")
        self.assertIsNone(source_policy["source_verifier_semantic_generic_backoff_attempt_pressure_env"])
        self.assertIsNone(
            source_policy["source_verifier_semantic_generic_backoff_attempt_pressure_min_attempts"]
        )
        self.assertIsNone(source_policy["source_verifier_semantic_generic_backoff_min_attempts"])
        configured_values = " ".join(
            env[key]
            for key in (
                "MODEL_ROUTER_ATTEMPTS",
                "MODEL_ROUTER_TRANSIENT_EXTRA_ATTEMPTS",
                "HLE_VARIANT_TOTAL_MODEL_ROUTER_ATTEMPT_BUDGET",
                "HLE_VARIANT_TOTAL_MODEL_ROUTER_SEC_BUDGET",
                "MODEL_ROUTER_TIMEOUT",
                "MODEL_ROUTER_PER_ATTEMPT_TIMEOUT",
                "MODEL_ROUTER_REASONING_EFFORT",
                "MODEL_ROUTER_SUBPROCESS_CALLS",
                "MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC",
                "HLE_PARALLEL_SHARD_WORKERS",
                "HLE_VARIANT_RELATION_COMPARATOR_MODEL_CALL_MIN_REMAINING_SEC",
                "HLE_SOURCE_GROUNDED_OPTION_CLAIM_VERIFIER_CANDIDATE_LIMIT",
                "HLE_SOURCE_GROUNDED_OPTION_CLAIM_RETRY_TOP_K",
                "HLE_SOURCE_GROUNDED_OPTION_CLAIM_MISSING_MODEL_RETRY_LIMIT",
                "HLE_SOURCE_GROUNDED_OPTION_CLAIM_SOURCE_QUALITY_CHALLENGER_LIMIT",
                "HLE_LOW_SUPPORT_OPTION_CLAIM_SOURCE_VERIFIER_LIMIT",
                "HLE_ZERO_QUALITY_SWEEP_GAP_OPTION_CLAIM_SOURCE_VERIFIER_LIMIT",
                "MODEL_ROUTER_BACKOFF_BASE_SEC",
                "MODEL_ROUTER_GLOBAL_CONCURRENCY",
                "MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR",
                "MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC",
                "MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC",
                "HLE_RECURSIVE_SELECTION_MODEL_CALL_BUDGET",
                "HLE_RECURSIVE_SELECTION_WALLCLOCK_BUDGET_SEC",
            )
        )
        self.assertNotIn("sk-", configured_values)
        self.assertNotIn("hf_", configured_values)

    def test_runner_env_relation_query_planner_explicit_toggle_clears_conflict(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HLE_ENABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER": "1",
                "HLE_DISABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER": "1",
                "HLE_ENABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR": "1",
                "HLE_DISABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR": "1",
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL": "1",
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL": "1",
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT": "1",
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT": "1",
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE": "1",
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE": "1",
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT": "1",
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT": "1",
            },
            clear=False,
        ):
            env = build_runner_env(
                model_router_attempts=None,
                model_router_timeout=None,
                enable_option_claim_relation_query_planner=True,
                enable_option_claim_relation_span_comparator=True,
                enable_option_claim_source_cache_corpus_backfill=True,
                enable_option_claim_source_verifier_repair_context=True,
                enable_option_claim_source_verifier_acceptance_quality_gate=True,
                enable_option_claim_source_verifier_structured_context=True,
            )
            self.assertEqual(env["HLE_ENABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER"], "1")
            self.assertNotIn("HLE_DISABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER", env)
            self.assertEqual(env["HLE_ENABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR"], "1")
            self.assertNotIn("HLE_DISABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR", env)
            self.assertEqual(
                env["HLE_ENABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL"],
                "1",
            )
            self.assertEqual(
                env["HLE_ENABLE_SOURCE_CACHE_ANSWER_BEARING_OPTION_CLAIM_RETRY"],
                "1",
            )
            self.assertNotIn(
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL",
                env,
            )
            self.assertNotIn(
                "HLE_DISABLE_SOURCE_CACHE_ANSWER_BEARING_OPTION_CLAIM_RETRY",
                env,
            )
            self.assertEqual(env["HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT"], "1")
            self.assertNotIn("HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT", env)
            self.assertEqual(
                env["HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE"],
                "1",
            )
            self.assertNotIn(
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE",
                env,
            )
            self.assertEqual(
                env["HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT"],
                "1",
            )
            self.assertNotIn(
                "HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT",
                env,
            )

            env = build_runner_env(
                model_router_attempts=None,
                model_router_timeout=None,
                disable_option_claim_relation_query_planner=True,
                disable_option_claim_relation_span_comparator=True,
                disable_option_claim_source_cache_corpus_backfill=True,
                disable_option_claim_source_verifier_repair_context=True,
                disable_option_claim_source_verifier_acceptance_quality_gate=True,
                disable_option_claim_source_verifier_structured_context=True,
            )
            self.assertEqual(env["HLE_DISABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER"], "1")
            self.assertNotIn("HLE_ENABLE_OPTION_CLAIM_RELATION_QUERY_PLANNER", env)
            self.assertEqual(env["HLE_DISABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR"], "1")
            self.assertNotIn("HLE_ENABLE_OPTION_CLAIM_RELATION_SPAN_COMPARATOR", env)
            self.assertEqual(
                env["HLE_DISABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL"],
                "1",
            )
            self.assertNotIn(
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_CACHE_CORPUS_BACKFILL",
                env,
            )
            self.assertEqual(env["HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT"], "1")
            self.assertNotIn("HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_REPAIR_CONTEXT", env)
            self.assertEqual(
                env["HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE"],
                "1",
            )
            self.assertNotIn(
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_ACCEPTANCE_QUALITY_GATE",
                env,
            )
            self.assertEqual(
                env["HLE_DISABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT"],
                "1",
            )
            self.assertNotIn(
                "HLE_ENABLE_OPTION_CLAIM_SOURCE_VERIFIER_STRUCTURED_CONTEXT",
                env,
            )

    def test_runner_env_does_not_default_recursive_child_batch_cap(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            env = build_runner_env(
                model_router_attempts=None,
                model_router_timeout=None,
            )
        self.assertNotIn("HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC", env)
        self.assertNotIn("HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC", env)

        with patch.dict(os.environ, {"HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC": "90"}, clear=True):
            env = build_runner_env(
                model_router_attempts=None,
                model_router_timeout=None,
        )
        self.assertEqual(env["HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC"], "90")

    def test_model_router_policy_logs_recursive_child_prompt_kind_limit(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            env = build_runner_env(
                model_router_attempts=None,
                model_router_timeout=None,
            )
        self.assertNotIn("HLE_RECURSIVE_CHILD_PROMPT_KIND_LIMIT", env)
        self.assertIsNone(
            model_router_policy_from_env(env)["recursive_child_prompt_kind_limit"]
        )

        with patch.dict(
            os.environ,
            {"HLE_RECURSIVE_CHILD_PROMPT_KIND_LIMIT": "4"},
            clear=True,
        ):
            env = build_runner_env(
                model_router_attempts=None,
                model_router_timeout=None,
            )
        self.assertEqual(env["HLE_RECURSIVE_CHILD_PROMPT_KIND_LIMIT"], "4")
        self.assertEqual(
            model_router_policy_from_env(env)["recursive_child_prompt_kind_limit"],
            "4",
        )

    def test_live_model_preflight_fails_fast_without_model_key(self) -> None:
        with patch("assumption_os.hle_parallel_shard_runner.subprocess.run") as run:
            payload = run_live_model_preflight(
                models="gpt-5.4-mini",
                env={},
                timeout_sec=5,
            )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["rows"][0]["error_type"], "RuntimeError")
        self.assertIn("missing", payload["rows"][0]["error_label"])
        run.assert_not_called()

    def test_model_router_primary_key_present_ignores_fallback_only(self) -> None:
        self.assertFalse(model_router_primary_key_present({}))
        self.assertFalse(model_router_primary_key_present({"MODEL_ROUTER_FALLBACK_API_KEYS": "sk-fallback"}))
        self.assertTrue(model_router_primary_key_present({"GPT5_API_KEY": "sk-primary"}))
        self.assertTrue(model_router_primary_key_present({"RUOLI_GPT_KEY": "sk-primary"}))
        self.assertTrue(model_router_primary_key_present({"OPENAI_API_KEY": "sk-primary"}))

    def test_live_model_preflight_redacts_model_key_from_error(self) -> None:
        secret = "secret-model-key"
        completed = subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr=f"HTTP Error 401 with {secret}",
        )
        with patch("assumption_os.hle_parallel_shard_runner.subprocess.run", return_value=completed):
            payload = run_live_model_preflight(
                models="gpt-5.4-mini",
                env={"RUOLI_GPT_KEY": secret},
                timeout_sec=5,
            )

        serialized = json.dumps(payload)
        self.assertFalse(payload["passed"])
        self.assertIn("[redacted]", serialized)
        self.assertNotIn(secret, serialized)
        self.assertFalse(payload["raw_content_persisted"])

    def test_live_model_preflight_supports_multi_probe_error_rate_gate(self) -> None:
        success = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
        failure = subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="",
            stderr="RemoteDisconnected: Remote end closed connection without response",
        )
        with patch(
            "assumption_os.hle_parallel_shard_runner.subprocess.run",
            side_effect=[success, failure, success],
        ):
            payload = run_live_model_preflight(
                models="gpt-5.4-mini",
                env={"RUOLI_GPT_KEY": "secret"},
                timeout_sec=5,
                probe_count=3,
                max_error_rate=0.34,
            )

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["probe_count"], 3)
        summary = payload["summary"]["by_model"]["gpt-5.4-mini"]
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["endpoint_retryable_error_count"], 1)
        self.assertEqual(summary["error_rate"], 0.3333)

        with patch(
            "assumption_os.hle_parallel_shard_runner.subprocess.run",
            side_effect=[success, failure, success],
        ):
            strict_payload = run_live_model_preflight(
                models="gpt-5.4-mini",
                env={"RUOLI_GPT_KEY": "secret"},
                timeout_sec=5,
                probe_count=3,
                max_error_rate=0.0,
            )
        self.assertFalse(strict_payload["passed"])

    def test_live_model_preflight_can_probe_with_long_prompt(self) -> None:
        success = subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )
        with patch(
            "assumption_os.hle_parallel_shard_runner.subprocess.run",
            return_value=success,
        ) as run:
            payload = run_live_model_preflight(
                models="gpt-5.4-mini",
                env={"RUOLI_GPT_KEY": "secret"},
                timeout_sec=5,
                prompt_chars=2048,
                max_tokens=32,
            )

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["prompt_char_count"], 2048)
        self.assertEqual(payload["max_tokens"], 32)
        stdin_payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(len(stdin_payload["prompt"]), 2048)
        self.assertEqual(stdin_payload["max_tokens"], 32)
        self.assertFalse(payload["raw_content_persisted"])
        self.assertNotIn("Synthetic HLE-style", json.dumps(payload))

    def test_runner_env_defaults_to_local_hle_and_cache_only_sources_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "hle_dataset_cache" / "test"
            cache_path = root / "hle_evidence_source_cache"
            dataset_path.mkdir(parents=True)
            cache_path.mkdir(parents=True)
            with patch(
                "assumption_os.hle_parallel_shard_runner.DEFAULT_HLE_DATASET_LOCAL_PATH",
                dataset_path,
            ), patch(
                "assumption_os.hle_parallel_shard_runner.DEFAULT_HLE_EVIDENCE_SOURCE_CACHE_DIR",
                cache_path,
            ), patch.dict(os.environ, {}, clear=True):
                env = build_runner_env(model_router_attempts=None, model_router_timeout=None)

        self.assertEqual(env["HLE_DATASET_LOCAL_PATH"], str(dataset_path))
        self.assertEqual(env["HLE_EVIDENCE_SOURCE_CACHE_DIR"], str(cache_path))
        self.assertEqual(env["HLE_EVIDENCE_SOURCE_CACHE_ONLY"], "1")
        self.assertEqual(env["HLE_SOURCE_SEARCH_CACHE_ONLY"], "1")
        self.assertEqual(env["HLE_DISABLE_LIVE_SOURCE_SEARCH"], "1")
        self.assertEqual(env["HLE_ALLOW_LIVE_SOURCE_SEARCH"], "0")

    def test_hle_offline_defaults_preserve_explicit_source_policy_and_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "hle_dataset_cache" / "test"
            cache_path = root / "hle_evidence_source_cache"
            dataset_path.mkdir(parents=True)
            cache_path.mkdir(parents=True)
            with patch(
                "assumption_os.hle_parallel_shard_runner.DEFAULT_HLE_DATASET_LOCAL_PATH",
                dataset_path,
            ), patch(
                "assumption_os.hle_parallel_shard_runner.DEFAULT_HLE_EVIDENCE_SOURCE_CACHE_DIR",
                cache_path,
            ):
                env = {
                    "HLE_DATASET_LOCAL_PATH": "/custom/hle",
                    "HLE_EVIDENCE_SOURCE_CACHE_DIR": "/custom/cache",
                    "HLE_ALLOW_LIVE_SOURCE_SEARCH": "1",
                }
                apply_hle_offline_defaults(env)

                disabled_env = {"HLE_DISABLE_LOCAL_HLE_DEFAULTS": "1"}
                apply_hle_offline_defaults(disabled_env)

        self.assertEqual(env["HLE_DATASET_LOCAL_PATH"], "/custom/hle")
        self.assertEqual(env["HLE_EVIDENCE_SOURCE_CACHE_DIR"], "/custom/cache")
        self.assertNotIn("HLE_EVIDENCE_SOURCE_CACHE_ONLY", env)
        self.assertNotIn("HLE_SOURCE_SEARCH_CACHE_ONLY", env)
        self.assertNotIn("HLE_DISABLE_LIVE_SOURCE_SEARCH", env)
        self.assertEqual(disabled_env, {"HLE_DISABLE_LOCAL_HLE_DEFAULTS": "1"})

    def test_live_network_defaults_stabilize_execute_live_without_overriding_user_values(self) -> None:
        args = Namespace(
            execute_live=True,
            eval_id="eval/live",
            parallel_workers=8,
            model_router_attempts=None,
            model_router_transient_extra_attempts=None,
            model_router_per_attempt_timeout=None,
            model_router_subprocess_calls=None,
            disable_model_router_subprocess_calls=False,
            model_router_no_byte_timeout_sec=None,
            model_router_backoff_base_sec=None,
            model_router_global_concurrency=None,
            model_router_global_concurrency_dir="",
            model_router_global_slot_ttl_sec=None,
            model_router_global_slot_wait_sec=None,
        )
        apply_live_network_defaults(args)

        self.assertEqual(args.model_router_attempts, 8)
        self.assertEqual(args.model_router_transient_extra_attempts, 0)
        self.assertEqual(args.model_router_per_attempt_timeout, 180.0)
        self.assertTrue(args.model_router_subprocess_calls)
        self.assertEqual(args.model_router_no_byte_timeout_sec, 180.0)
        self.assertEqual(args.model_router_backoff_base_sec, 1.5)
        self.assertEqual(args.model_router_global_concurrency, 4)
        self.assertIn("eval_live", args.model_router_global_concurrency_dir)
        self.assertEqual(args.model_router_global_slot_ttl_sec, 7200.0)
        self.assertEqual(args.model_router_global_slot_wait_sec, 7200.0)

        single_parallel = Namespace(
            execute_live=True,
            eval_id="eval/single",
            parallel_workers=1,
            agent_child_mode="parallel_quorum",
            model_router_attempts=None,
            model_router_transient_extra_attempts=None,
            model_router_per_attempt_timeout=None,
            model_router_subprocess_calls=None,
            disable_model_router_subprocess_calls=False,
            model_router_no_byte_timeout_sec=None,
            model_router_backoff_base_sec=None,
            model_router_global_concurrency=None,
            model_router_global_concurrency_dir="",
            model_router_global_slot_ttl_sec=None,
            model_router_global_slot_wait_sec=None,
        )
        apply_live_network_defaults(single_parallel)
        self.assertEqual(single_parallel.model_router_global_concurrency, 2)

        single_serial = Namespace(
            execute_live=True,
            eval_id="eval/single-serial",
            parallel_workers=1,
            agent_child_mode="serial",
            model_router_attempts=None,
            model_router_transient_extra_attempts=None,
            model_router_per_attempt_timeout=None,
            model_router_subprocess_calls=None,
            disable_model_router_subprocess_calls=False,
            model_router_no_byte_timeout_sec=None,
            model_router_backoff_base_sec=None,
            model_router_global_concurrency=None,
            model_router_global_concurrency_dir="",
            model_router_global_slot_ttl_sec=None,
            model_router_global_slot_wait_sec=None,
        )
        apply_live_network_defaults(single_serial)
        self.assertEqual(single_serial.model_router_global_concurrency, 1)

        explicit = Namespace(
            execute_live=True,
            eval_id="eval",
            parallel_workers=8,
            agent_child_mode="parallel_quorum",
            model_router_attempts=3,
            model_router_transient_extra_attempts=4,
            model_router_per_attempt_timeout=45,
            model_router_subprocess_calls=True,
            disable_model_router_subprocess_calls=False,
            model_router_no_byte_timeout_sec=75,
            model_router_backoff_base_sec=0.5,
            model_router_global_concurrency=2,
            model_router_global_concurrency_dir="/tmp/custom-slots",
            model_router_global_slot_ttl_sec=30,
            model_router_global_slot_wait_sec=60,
        )
        apply_live_network_defaults(explicit)

        self.assertEqual(explicit.model_router_attempts, 3)
        self.assertEqual(explicit.model_router_transient_extra_attempts, 4)
        self.assertEqual(explicit.model_router_per_attempt_timeout, 45)
        self.assertTrue(explicit.model_router_subprocess_calls)
        self.assertEqual(explicit.model_router_no_byte_timeout_sec, 75)
        self.assertEqual(explicit.model_router_backoff_base_sec, 0.5)
        self.assertEqual(explicit.model_router_global_concurrency, 2)
        self.assertEqual(explicit.model_router_global_concurrency_dir, "/tmp/custom-slots")
        self.assertEqual(explicit.model_router_global_slot_ttl_sec, 30)
        self.assertEqual(explicit.model_router_global_slot_wait_sec, 60)

        dry = Namespace(
            execute_live=False,
            eval_id="eval",
            parallel_workers=8,
            agent_child_mode="parallel_quorum",
            model_router_attempts=None,
            model_router_transient_extra_attempts=None,
            model_router_per_attempt_timeout=None,
            model_router_subprocess_calls=None,
            disable_model_router_subprocess_calls=False,
            model_router_no_byte_timeout_sec=None,
            model_router_backoff_base_sec=None,
            model_router_global_concurrency=None,
            model_router_global_concurrency_dir="",
            model_router_global_slot_ttl_sec=None,
            model_router_global_slot_wait_sec=None,
        )
        apply_live_network_defaults(dry)

        self.assertIsNone(dry.model_router_attempts)
        self.assertIsNone(dry.model_router_transient_extra_attempts)
        self.assertIsNone(dry.model_router_per_attempt_timeout)
        self.assertIsNone(dry.model_router_subprocess_calls)
        self.assertIsNone(dry.model_router_no_byte_timeout_sec)
        self.assertIsNone(dry.model_router_global_concurrency)

        disabled = Namespace(
            execute_live=True,
            eval_id="eval",
            parallel_workers=2,
            agent_child_mode="parallel_quorum",
            model_router_attempts=None,
            model_router_transient_extra_attempts=None,
            model_router_per_attempt_timeout=None,
            model_router_subprocess_calls=None,
            disable_model_router_subprocess_calls=True,
            model_router_no_byte_timeout_sec=None,
            model_router_backoff_base_sec=None,
            model_router_global_concurrency=None,
            model_router_global_concurrency_dir="",
            model_router_global_slot_ttl_sec=None,
            model_router_global_slot_wait_sec=None,
        )
        apply_live_network_defaults(disabled)
        self.assertFalse(disabled.model_router_subprocess_calls)
        self.assertEqual(disabled.model_router_no_byte_timeout_sec, 180.0)

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

    def test_failure_diagnostics_bucket_candidate_generation_missed_gold(self) -> None:
        rows = [
            _row(
                "p1",
                "assumption_agent_recursive_verify",
                False,
                component_efficacy={
                    "flags": {
                        "candidate_generation_missed_gold": True,
                        "candidate_generation_missed_gold_with_sweep_coverage": True,
                        "missing_model_option_source_retry_scheduled": True,
                        "mc_option_claim_source_verifier_cross_selection_blocked": True,
                        "gold_option_source_verifier_attempted": True,
                        "gold_option_source_verifier_accepted": False,
                        "gold_option_source_verifier_direct_source_insufficient": True,
                        "gold_option_source_verifier_indirect_or_generic": True,
                    },
                    "selection": {"selection_method": "verified_or_abstain_direct_fallback"},
                },
            ),
            _row("p1", "raw", False),
            _row("p1", "hipporag_baseline", False),
        ]

        diagnostics = build_failure_diagnostics(rows=rows)

        self.assertEqual(diagnostics["agent_failure_buckets"]["candidate_generation_missed_gold"], 1)
        self.assertEqual(
            diagnostics["agent_failure_buckets"]["candidate_generation_missed_gold_with_sweep_coverage"],
            1,
        )
        self.assertEqual(
            diagnostics["agent_failure_buckets"]["missing_model_option_source_retry_unhelpful"],
            1,
        )
        self.assertEqual(
            diagnostics["agent_failure_buckets"]["source_verifier_cross_selection_blocked"],
            1,
        )
        self.assertEqual(
            diagnostics["agent_failure_buckets"]["gold_option_source_verifier_unaccepted"],
            1,
        )
        self.assertEqual(
            diagnostics["agent_failure_buckets"]["gold_option_direct_source_insufficient"],
            1,
        )
        self.assertEqual(
            diagnostics["agent_failure_buckets"]["gold_option_source_indirect_or_generic"],
            1,
        )
        self.assertEqual(diagnostics["agent_failure_buckets"]["multiple_choice_selection_failed"], 1)

    def test_failure_diagnostics_summarize_source_directness_gaps(self) -> None:
        rows = [
            _row(
                "p1",
                "assumption_agent_recursive_verify",
                False,
                component_efficacy={
                    "flags": {
                        "candidate_generation_missed_gold": True,
                        "missing_model_option_source_retry_scheduled": True,
                        "missing_model_option_source_retry_success": False,
                        "low_support_exhaustive_missing_model_retry_used": True,
                        "mc_option_claim_source_verifier_used": True,
                        "mc_option_claim_source_verifier_repair_context_used": True,
                        "mc_option_claim_source_verifier_repair_context_found_spans": True,
                        "mc_option_claim_source_verifier_structured_context_used": True,
                        "mc_option_claim_source_verifier_acceptance_quality_gate_blocked": True,
                        "mc_option_claim_evidence_candidate_emitted": False,
                        "mc_option_claim_local_relation_query_expansion_used": True,
                        "mc_option_claim_sweep_gap_local_relation_backfill_used": True,
                        "mc_option_claim_span_directness_verifier_used": True,
                        "mc_option_claim_span_directness_verifier_accepted": False,
                        "mc_option_claim_span_directness_lexical_unique_but_generic": True,
                        "mc_option_claim_relation_span_comparator_used": True,
                        "mc_option_claim_relation_span_comparator_accepted": False,
                        "mc_option_claim_candidate_direct_relation_span_extractor_used": False,
                        "mc_option_claim_candidate_direct_relation_span_directness_accepted": False,
                        "mc_option_claim_contrastive_adjudicator_used": True,
                        "mc_option_claim_contrastive_adjudicator_accepted": False,
                        "mc_option_claim_contrastive_relation_matrix_returned": True,
                        "mc_option_claim_contrastive_structured_relation_audit_used": True,
                        "mc_option_claim_contrastive_structured_relation_audit_hard_blocked": True,
                        "gold_option_source_verifier_attempted": True,
                        "gold_option_source_verifier_accepted": False,
                        "gold_option_source_verifier_direct_source_insufficient": True,
                    },
                    "mc_option_claim_evidence_verifier": {
                        "relation_query_planner_status": "disabled",
                        "source_verifier_rejection_reason_counts": {
                            "no_selected_label_generic": 3,
                        },
                        "source_verifier_repair_context_status_counts": {
                            "activated": 2,
                        },
                        "source_verifier_repair_context_reason_counts": {
                            "candidate_relation_repair_context_available": 2,
                        },
                        "source_verifier_structured_context_status_counts": {
                            "activated": 2,
                        },
                        "source_verifier_structured_context_reason_counts": {
                            "target_relation_outline_available": 2,
                        },
                        "source_verifier_acceptance_quality_gate_reason_counts": {
                            "missing_programmatic_source_quality_signal": 1,
                        },
                        "source_quality_directness_promotion_detail": {
                            "status": "blocked",
                            "reason": "no_span_directness_direct_candidates",
                            "rejection_counts": {
                                "missing_candidate_direct_relation_span": 2,
                                "not_span_direct": 2,
                            },
                        },
                        "span_directness_verifier_status": "blocked_not_direct_relation",
                        "span_directness_verifier_reason": "no_candidate_span_direct_relation",
                        "relation_span_comparator_status": "blocked_not_direct_relation",
                        "relation_span_comparator_reason": "no_direct_relation_span_candidate",
                        "candidate_direct_relation_span_extractor_status": "activated",
                        "candidate_direct_relation_span_directness_verifier_status": (
                            "blocked_not_direct_relation"
                        ),
                        "contrastive_adjudicator_reason": "not_direct_high_confidence",
                        "contrastive_adjudicator_direct_relation_candidate_count": 0,
                        "contrastive_adjudicator_selected_structured_relation_hard_block_reason": (
                            "source_verifier_generic"
                        ),
                        "contrastive_adjudicator_structured_relation_matrix": [
                            {
                                "option_hash": "candidate-a",
                                "hard_block_reason": "source_verifier_generic",
                            }
                        ],
                    },
                    "selection": {"selection_method": "verified_or_abstain_direct_fallback"},
                },
            ),
            _row("p1", "raw", False),
            _row("p1", "hipporag_baseline", False),
        ]

        diagnostics = build_failure_diagnostics(rows=rows)

        buckets = diagnostics["source_directness_failure_buckets"]
        self.assertEqual(buckets["relation_query_planner_not_activated"], 1)
        self.assertEqual(buckets["source_verifier_used"], 1)
        self.assertEqual(buckets["source_verifier_no_candidate_emitted"], 1)
        self.assertEqual(buckets["source_verifier_repair_context_used"], 1)
        self.assertEqual(buckets["source_verifier_repair_context_found_spans"], 1)
        self.assertEqual(buckets["source_verifier_structured_context_used"], 1)
        self.assertEqual(buckets["source_verifier_acceptance_quality_gate_blocked"], 1)
        self.assertEqual(buckets["missing_model_source_retry_unhelpful"], 1)
        self.assertEqual(buckets["local_relation_query_expansion_found_docs"], 1)
        self.assertEqual(buckets["sweep_gap_local_relation_backfill_found_docs"], 1)
        self.assertEqual(buckets["source_quality_promotion_no_direct_span"], 1)
        self.assertEqual(buckets["span_directness_verifier_rejected"], 1)
        self.assertEqual(buckets["span_directness_lexical_unique_but_generic"], 1)
        self.assertEqual(buckets["relation_span_comparator_used"], 1)
        self.assertEqual(buckets["relation_span_comparator_rejected"], 1)
        self.assertEqual(buckets["candidate_direct_relation_span_directness_rejected"], 1)
        self.assertEqual(buckets["contrastive_relation_matrix_no_direct_candidate"], 1)
        self.assertEqual(buckets["contrastive_structured_relation_audit_used"], 1)
        self.assertEqual(buckets["contrastive_structured_relation_audit_hard_blocked"], 1)
        self.assertEqual(buckets["gold_option_source_verifier_unaccepted"], 1)

        reasons = diagnostics["source_directness_reason_counts"]
        self.assertEqual(
            reasons["source_verifier_rejection_reason"]["no_selected_label_generic"],
            3,
        )
        self.assertEqual(
            reasons["source_verifier_repair_context_status"]["activated"],
            2,
        )
        self.assertEqual(
            reasons["source_verifier_repair_context_reason"][
                "candidate_relation_repair_context_available"
            ],
            2,
        )
        self.assertEqual(
            reasons["source_verifier_structured_context_status"]["activated"],
            2,
        )
        self.assertEqual(
            reasons["source_verifier_structured_context_reason"][
                "target_relation_outline_available"
            ],
            2,
        )
        self.assertEqual(
            reasons["source_verifier_acceptance_quality_gate_reason"][
                "missing_programmatic_source_quality_signal"
            ],
            1,
        )
        self.assertEqual(
            reasons["source_quality_directness_promotion_reason"][
                "no_span_directness_direct_candidates"
            ],
            1,
        )
        self.assertEqual(reasons["source_quality_directness_rejection"]["not_span_direct"], 2)
        self.assertEqual(
            reasons["relation_span_comparator_status"]["blocked_not_direct_relation"],
            1,
        )
        self.assertEqual(
            reasons["relation_span_comparator_reason"]["no_direct_relation_span_candidate"],
            1,
        )
        self.assertEqual(
            reasons["contrastive_adjudicator_reason"]["not_direct_high_confidence"],
            1,
        )
        self.assertEqual(
            reasons["contrastive_structured_relation_hard_block"]["source_verifier_generic"],
            1,
        )


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


def _agent_multi_call_same_model_ce() -> dict[str, object]:
    return {
        "kind": "assumption_agent_recursive_verify",
        "flags": {
            "recursive_child_validation_activated": True,
            "claim_verifier_activated": True,
        },
        "recursive": {
            "status": "activated",
            "base_model": "gpt-5.4-mini",
            "child_model": "gpt-5.4-mini",
            "planned_child_count": 4,
            "child_count": 4,
            "answered_child_count": 4,
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
