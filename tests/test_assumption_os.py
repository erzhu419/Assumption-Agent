import json
import base64
import tempfile
import unittest
from pathlib import Path

from assumption_os.adapters import ingest_artifacts, load_exp82_hypotheses, load_wisdom_nodes
from assumption_os.activation import build_activation_profile
from assumption_os.assumption_family_discovery import (
    build_assumption_family_discovery_payload,
    classify_new_theory_card,
)
from assumption_os.assumption_bench import build_assumption_bench_payload
from assumption_os.bayesian_policy import BayesianPolicyAction, build_bayesian_policy_payload, parent_belief
from assumption_os.causal_mask_v2 import build_causal_mask_v2_payload
from assumption_os.candidate_acceptance import AcceptanceDecision, apply_accepted_candidates, build_acceptance_payload
from assumption_os.conditioned_eval import (
    ConditionedEvalRow,
    GateDecision,
    GateThresholds,
    RouteLabel,
    evaluate_node,
    route_problem_to_node,
)
from assumption_os.continuous_daemon import build_continuous_daemon_autonomy_payload
from assumption_os.domain_templates import format_phase2_domain_execution_template
from assumption_os.downstream_paper_claim_v2 import build_downstream_paper_claim_v2_payload
from assumption_os.evolution_cycle import build_evolution_cycle_payload, build_policy_update_plan
from assumption_os.evolution_context import (
    EvolutionPolicyDecision,
    build_evolution_context_payload,
)
from assumption_os.failure_hypotheses import build_failure_hypothesis_payload
from assumption_os.formal_alignment_v2 import build_formal_alignment_v2_payload
from assumption_os.falsification import FalsificationDecision, build_falsification_payload
from assumption_os.first_party_world_model import build_first_party_world_model_scale_payload
from assumption_os.full_v2_phase0_contract_bypass import build_full_v2_phase0_contract_bypass_payload
from assumption_os.full_v2_phase1_graph_memory_bypass import build_full_v2_phase1_graph_memory_bypass_payload
from assumption_os.full_v2_phase2_verifier_bypass import build_full_v2_phase2_verifier_bypass_payload
from assumption_os.full_v2_phase3_world_model_bypass import build_full_v2_phase3_world_model_bypass_payload
from assumption_os.full_v2_phase4_hypothesis_generator_bypass import build_full_v2_phase4_hypothesis_generator_bypass_payload
from assumption_os.full_v2_phase5_strategy_scheduler_bypass import build_full_v2_phase5_strategy_scheduler_bypass_payload
from assumption_os.full_v2_phase6_formal_alignment_bypass import build_full_v2_phase6_formal_alignment_bypass_payload
from assumption_os.full_v2_phase7_daemon_harness_bypass import build_full_v2_phase7_daemon_harness_bypass_payload
from assumption_os.full_v2_vertical_slice_bypass import build_full_v2_vertical_slice_bypass_payload
from assumption_os.full_v3_frozen_v1_comparison import build_full_v3_frozen_v1_comparison_payload
from assumption_os.full_v3_fresh_live_benchmark import build_full_v3_fresh_live_benchmark_payload
from assumption_os.full_v3_phase0_contract_checker import build_full_v3_phase0_contract_checker_payload
from assumption_os.full_v3_phase1_memory_consolidation import build_full_v3_phase1_memory_consolidation_payload
from assumption_os.full_v3_phase2_verifier_synthesis import build_full_v3_phase2_verifier_synthesis_payload
from assumption_os.full_v3_phase3_rollout_search_control import build_full_v3_phase3_rollout_search_control_payload
from assumption_os.full_v3_phase4_hypothesis_generator import build_full_v3_phase4_hypothesis_generator_payload
from assumption_os.full_v3_phase5_contextual_bandit_scheduler import build_full_v3_phase5_contextual_bandit_scheduler_payload
from assumption_os.full_v3_phase6_formal_transfer_engine import build_full_v3_phase6_formal_transfer_engine_payload
from assumption_os.full_v3_phase7_long_run_benchmark import build_full_v3_phase7_long_run_benchmark_payload
from assumption_os.full_v3_phase8_creativity_world_coverage import (
    build_full_v3_phase8_creativity_world_coverage_payload,
)
from assumption_os.full_v3_phase9_hybrid_guard_heldout import build_full_v3_phase9_hybrid_guard_heldout_payload
from assumption_os.full_v3_phase9_v1_live_regression import build_full_v3_phase9_v1_live_regression_payload
from assumption_os.full_v3_phase10_discrete_world_model_selector import (
    build_full_v3_phase10_discrete_world_model_selector_payload,
)
from assumption_os.full_v3_phase11_capability_audit import build_full_v3_phase11_capability_audit_payload
from assumption_os.full_v3_paper_scale_evidence import build_full_v3_paper_scale_evidence_payload
from assumption_os.formal_mapping import (
    FormalMappingGateDecision,
    FormalMappingStatus,
    build_independent_formal_search_eval_payload,
    build_categorical_info_geometry_payload,
    build_formal_dedup_payload,
    build_formal_downstream_task_eval_payload,
    build_formal_answer_quality_probe_payload,
    build_formal_engine_depth_payload,
    build_formal_mapping_gate_payload,
    build_formal_mapping_payload,
    build_formal_search_eval_payload,
    build_formal_transfer_eval_payload,
    finite_kernel_metrics,
    format_formal_mapping_applications,
    search_formal_mappings,
)
from assumption_os.graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from assumption_os.graph_action_world_model_v2 import build_graph_action_world_model_v2_payload
from assumption_os.harness_observer import build_harness_observer_payload, events_from_harness_artifacts
from assumption_os.hipporag_qa_probe import build_hipporag_qa_probe_payload
from assumption_os.hypothesis_lifecycle_v2 import build_hypothesis_lifecycle_v2_payload
from assumption_os.hypothesis_overlay_v2 import build_hypothesis_overlay_v2_payload
from assumption_os.lifecycle import LifecycleActionType, plan_lifecycle_actions
from assumption_os.manifest_logger import build_component_manifest_payload, events_from_run_logs
from assumption_os.math_science_policy import route_math_science_problem
from assumption_os.memory_consolidation_job import build_memory_consolidation_job_payload
from assumption_os.memory_surfaces import build_memory_surface_payload
from assumption_os.meta_qa_evolution import build_meta_qa_evolution_payload
from assumption_os.morphism_benchmark import build_morphism_independent_benchmark_payload
from assumption_os.morphism_claims import build_morphism_claim_bundle_payload
from assumption_os.candidate_eval import CandidateReadiness, build_candidate_eval_payload
from assumption_os.novelty_integration import (
    build_novelty_integration_payload,
    build_novelty_integration_performance_payload,
)
from assumption_os.objective_bench import build_objective_benchmark_payload
from assumption_os.orthogonal_ablation import build_orthogonal_ablation_payload
from assumption_os.orthogonal_descendant_live_queue import build_orthogonal_descendant_live_queue_payload
from assumption_os.orthogonal_descendant_live_readback import build_orthogonal_descendant_live_readback_payload
from assumption_os.orthogonal_descendant_productivity import build_orthogonal_descendant_productivity_payload
from assumption_os.orthogonal_downstream_ablation import build_orthogonal_downstream_ablation_payload
from assumption_os.orthogonal_execution_queue import build_orthogonal_execution_queue_payload
from assumption_os.orthogonal_multi_cluster import build_orthogonal_multi_cluster_payload
from assumption_os.orthogonal_positive_queue import build_orthogonal_positive_queue_payload
from assumption_os.orthogonal_positive_readback import build_orthogonal_positive_readback_payload
from assumption_os.orthogonal_recursive_ablation import build_orthogonal_recursive_ablation_payload
from assumption_os.orthogonal_surface_ablation import build_orthogonal_surface_ablation_payload
from assumption_os.paper_baseline_hardening import build_paper_baseline_hardening_payload
from assumption_os.paper_benchmark_line import build_paper_benchmark_line_payload
from assumption_os.paper_main_experiment import build_paper_main_experiment_payload
from assumption_os.paper_negative_results import build_paper_negative_results_payload
from assumption_os.paper_repro_pack import build_paper_repro_pack_payload
from assumption_os.paper_retrieval_baselines import build_paper_retrieval_baselines_payload
from assumption_os.pre_live_tie_screen import build_pre_live_tie_screen_payload
from assumption_os.process_model_zoo_v2 import build_process_model_zoo_v2_payload
from assumption_os.proposal_contract import (
    apply_contract_checked_proposal_overlay,
    build_proposal_contract_payload,
)
from assumption_os.proposal_overlay import apply_proposal_overlay, proposal_candidate_ids
from assumption_os.proposals import ProposalType, build_candidate_proposals
from assumption_os.queue_artifact_eval import build_queue_artifact_eval_payload, judgment_sets_from_artifact_eval
from assumption_os.rag_to_memory_baseline import build_rag_to_memory_baseline_payload
from assumption_os.record_phase2_eval import record_phase2_eval
from assumption_os.residual_hypothesis_generator_v2 import build_residual_hypothesis_generator_v2_payload
from assumption_os.retrieval_policy import format_policy_context, retrieve_phase2_assumptions
from assumption_os.recursive_runner import (
    RecursiveFrameStatus,
    RecursiveFrameType,
    build_recursive_assumption_run,
)
from assumption_os.recursive_audit import build_recursive_audit_payload
from assumption_os.recursive_daemon import build_preflight_queue_daemon_payload, build_recursive_daemon_payload
from assumption_os.recursive_evolution_proof import build_recursive_self_evolution_proof_payload
from assumption_os.recursive_executor import JudgmentSet, build_recursive_execution_payload
from assumption_os.reconstruction_progress import build_reconstruction_progress_payload
from assumption_os.residual_clusterer import ResidualRecord, build_residual_cluster_payload, cluster_residual_records
from assumption_os.residual_diagnostics import (
    build_large_residual_label_calibration_payload,
    build_residual_label_agreement_payload,
    build_trace_residual_coverage_payload,
)
from assumption_os.residuals import classify_manifest
from assumption_os.runtime_trace import RuntimeTraceRecorder
from assumption_os.schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    EvidenceRecord,
    HypothesisKind,
    ResidualType,
    TrialManifest,
    TrialStatus,
)
from assumption_os.selector import (
    MetaproductivitySelector,
    SelectionWeights,
    apply_acp_learning_updates,
    build_acp_learning_payload,
    build_metaproductivity_benchmark_payload,
)
from assumption_os.trajectory_search import build_trajectory_search_payload
from assumption_os.trace_dataset import build_trace_dataset_collection_payload, build_trace_dataset_payload
from assumption_os.trace_outcome_model import build_trace_outcome_model_payload, build_trace_policy_proposal_payload
from assumption_os.surface_hypotheses import build_surface_hypothesis_payload
from assumption_os.structural_patterns import (
    apply_accepted_structural_morphisms,
    build_structural_context_effect_payload,
    build_structural_context_validation_payload,
    build_nonlexical_structural_retrieval_probe_payload,
    build_structural_behavior_probe_payload,
    build_structural_extraction_audit_payload,
    build_structural_functor_eval_payload,
    build_structural_kernel_eval_payload,
    build_structural_lineage_payload,
    build_structural_morphism_gate_payload,
    build_structural_morphism_performance_payload,
    build_structural_pair_eval_payload,
    build_structural_realization_eval_payload,
    build_structural_transfer_proposal_payload,
    build_structural_writeback_eval_payload,
    build_transfer_prediction_testability_eval_payload,
    check_structural_functor,
    extract_structural_diagram,
    search_structural_patterns,
    seed_structural_patterns,
)
from assumption_os.structural_context_edges import build_structural_context_edge_payload
from assumption_os.verifier_stack import build_verifier_stack_payload
from assumption_os.world_model import build_world_model_payload, train_world_model_calibration


class AssumptionOSTest(unittest.TestCase):
    def test_schema_round_trip_and_retrieval(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            a = AssumptionNode(
                id="strategy_S15",
                type=AssumptionType.METHOD,
                claim="从最小可工作版本开始，逐步添加功能",
                context_conditions=["复杂系统", "高风险"],
                tags=["incremental", "增量构建", "S15"],
                confidence=0.8,
                metaproductivity=0.3,
            )
            b = AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["控制变量", "S01"],
                confidence=0.75,
            )
            store.upsert_node(a)
            store.upsert_node(b)
            store.add_edge(AssumptionEdge(source="strategy_S15", target="strategy_S01", type=EdgeType.DEPENDS_ON))
            store.flush()

            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            activated = graph.retrieve("世界模型外推失败，应该先做最小场景并替换一个核心模块", seeds=["S15"], top_k=2)
            self.assertEqual(activated.nodes[0].id, "strategy_S15")
            self.assertIn("strategy_S01", {n.id for n in activated.nodes})

    def test_hypothesis_lifecycle_v2_represents_alignment_as_relation_node(self):
        payload = build_hypothesis_lifecycle_v2_payload(eval_id="unit_hypothesis_lifecycle_v2")
        metrics = payload["metrics"]
        projection = payload["graph_projection"]
        manifest = payload["objects"]["manifest"]
        alignment = payload["objects"]["alignment_hypothesis"]
        trial = payload["objects"]["world_model_trial"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["process_model_count"], 2)
        self.assertEqual(metrics["alignment_relation_node_count"], 1)
        self.assertEqual(metrics["bare_alignment_edge_count"], 0)
        self.assertEqual(metrics["participates_in_edge_count"], 2)
        self.assertEqual(metrics["validation_issue_count"], 0)
        self.assertGreater(metrics["mapping_score"], 0.7)
        self.assertEqual(len(manifest["graph_ops"]), 5)
        self.assertIn("thermodynamic equilibrium equations", " ".join(alignment["broken_structure"]))
        self.assertEqual(trial["action"]["type"], "add_alignment_hypothesis")
        self.assertGreaterEqual(len(trial["action"]["counterfactual_masks"]), 3)
        self.assertEqual(
            {edge["type"] for edge in projection["edges"]},
            {EdgeType.PARTICIPATES_IN.value},
        )

    def test_hypothesis_overlay_v2_rolls_back_and_stays_idempotent(self):
        payload = build_hypothesis_overlay_v2_payload(
            root=Path("."),
            eval_id="unit_hypothesis_overlay_v2",
            performance_iterations=25,
        )
        diff = payload["diff"]
        perf = payload["performance"]
        idempotence = payload["idempotence"]
        transaction = payload["transaction"]

        self.assertTrue(payload["pass"])
        self.assertEqual(diff["nodes_added"], 3)
        self.assertEqual(diff["edges_added"], 2)
        self.assertEqual(diff["nodes_removed"], 0)
        self.assertEqual(diff["edges_removed"], 0)
        self.assertEqual(transaction["before"], transaction["rollback"])
        self.assertEqual(
            idempotence["edge_count_after_first_apply"],
            idempotence["edge_count_after_second_apply"],
        )
        self.assertEqual(perf["rollback_failure_count"], 0)
        self.assertEqual(perf["iterations"], 25)
        self.assertLess(perf["avg_apply_rollback_ms"], 25.0)

    def test_process_model_zoo_v2_classifies_process_family_alignments(self):
        payload = build_process_model_zoo_v2_payload(eval_id="unit_process_model_zoo_v2")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["process_count"], 10)
        self.assertEqual(metrics["validation_issue_count"], 0)
        self.assertGreaterEqual(metrics["family_count"], 5)
        self.assertGreaterEqual(metrics["accuracy"], 0.85)
        self.assertGreaterEqual(metrics["positive_recall"], 0.85)
        self.assertGreaterEqual(metrics["negative_rejection_rate"], 0.85)
        self.assertGreaterEqual(metrics["alignment_hypothesis_count"], 6)
        negative_false_positives = [
            row for row in payload["pair_judgments"]
            if row["gold_label"] == "negative" and row["decision"] != "reject"
        ]
        self.assertEqual(negative_false_positives, [])

    def test_graph_action_world_model_v2_predicts_alignment_action_outcomes(self):
        payload = build_graph_action_world_model_v2_payload(eval_id="unit_graph_action_world_model_v2")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["labeled_count"], 16)
        self.assertGreaterEqual(metrics["accept_auroc"], 0.95)
        self.assertLess(metrics["accept_brier"], metrics["base_rate_brier"])
        self.assertEqual(metrics["accepted_blocked_count"], 0)
        self.assertGreaterEqual(metrics["negative_actions_saved"], 7)
        self.assertLess(metrics["mean_regression_positive"], metrics["mean_regression_negative"])
        self.assertIn(
            "run_live_validation",
            {row["recommended_action"] for row in payload["predictions"]},
        )

    def test_causal_mask_v2_identifies_relation_node_contribution(self):
        payload = build_causal_mask_v2_payload(eval_id="unit_causal_mask_v2")
        metrics = payload["metrics"]
        ranking = payload["importance_ranking"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["counterfactual_trial_count"], 64)
        self.assertGreaterEqual(metrics["mean_positive_relation_accept_drop"], 0.40)
        self.assertLess(metrics["mean_negative_relation_accept_drop"], 0.10)
        self.assertGreaterEqual(metrics["relation_drop_auroc"], 0.95)
        self.assertEqual(metrics["negative_control_mask_false_live_count"], 0)
        self.assertGreaterEqual(metrics["positive_top_relation_mask_fraction"], 0.80)
        self.assertEqual(ranking[0]["mask_id"], "do(mask_alignment_relation_node)")

    def test_formal_alignment_v2_beats_process_similarity_baselines(self):
        payload = build_formal_alignment_v2_payload(eval_id="unit_formal_alignment_v2")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["certificate_count"], 16)
        self.assertGreaterEqual(metrics["formal_accuracy"], 0.95)
        self.assertGreater(metrics["formal_accuracy"], metrics["best_baseline_accuracy"])
        self.assertGreaterEqual(metrics["formal_positive_recall"], 0.95)
        self.assertGreaterEqual(metrics["formal_negative_rejection_rate"], 0.95)
        self.assertEqual(metrics["formal_false_positive_count"], 0)
        self.assertGreaterEqual(metrics["accepted_positive_mean_relation_drop"], 0.40)
        positive_local_stabilization = [
            row for row in payload["certificates"]
            if row["source_id"] == "process_damped_oscillator_v1"
            and row["target_id"] == "process_predator_prey_local_v1"
        ]
        self.assertEqual(positive_local_stabilization[0]["decision"], "accept_alignment")

    def test_residual_hypothesis_generator_v2_requires_systematic_clusters(self):
        payload = build_residual_hypothesis_generator_v2_payload(eval_id="unit_residual_hypothesis_generator_v2")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["residual_count"], 10)
        self.assertGreaterEqual(metrics["cluster_count"], 3)
        self.assertEqual(metrics["proposal_count"], metrics["cluster_count"])
        self.assertEqual(metrics["random_proposal_count"], 0)
        self.assertEqual(metrics["duplicate_claim_count"], 0)
        self.assertEqual(metrics["conflict_count"], 0)
        self.assertEqual(metrics["world_model_accept_count"], metrics["proposal_count"])
        self.assertGreaterEqual(metrics["heldout_residual_coverage"], 0.95)
        self.assertEqual(metrics["outside_control_harm_count"], 0)
        self.assertEqual(metrics["manifest_validation_issue_count"], 0)
        self.assertTrue(all(proposal["source_records"] for proposal in payload["proposals"]))

    def test_downstream_paper_claim_v2_builds_frozen_mechanism_line(self):
        payload = build_downstream_paper_claim_v2_payload(
            eval_id="unit_downstream_paper_claim_v2",
            bootstrap_samples=200,
            seed=13,
        )
        metrics = payload["metrics"]
        systems = {row["system_id"]: row for row in payload["systems"]}

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["problem_count"], 16)
        self.assertGreaterEqual(metrics["accuracy_margin_over_retrieval_or_no_formal"], 0.18)
        self.assertGreaterEqual(metrics["utility_margin_over_best_non_full"], 0.05)
        self.assertEqual(metrics["full_negative_control_safety"], 1.0)
        self.assertGreaterEqual(metrics["full_residual_coverage"], 0.95)
        self.assertGreaterEqual(metrics["full_screen_cost_reduction"], 0.40)
        self.assertIn("ordinary_rag_semantic_proxy", systems)
        self.assertIn("hipporag_style_graph_proxy", systems)
        self.assertIn("no_world_model", systems)
        self.assertIn("no_recursive_generator", systems)
        self.assertIn("full_recursive_assumption_graph_v2", systems)

    def test_full_v2_phase0_contract_bypass_routes_invalid_drafts(self):
        payload = build_full_v2_phase0_contract_bypass_payload(eval_id="unit_full_v2_phase0_contract")
        metrics = payload["metrics"]
        by_source = {row["source"]: row for row in payload["results"]}

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["valid_candidate_acceptance_rate"], 1.0)
        self.assertEqual(metrics["invalid_draft_rejection_rate"], 1.0)
        self.assertEqual(metrics["duplicate_detection_recall"], 1.0)
        self.assertEqual(metrics["conflict_detection_recall"], 1.0)
        self.assertEqual(metrics["valid_rollback_coverage"], 1.0)
        self.assertEqual(metrics["valid_verifier_presence"], 1.0)
        self.assertEqual(metrics["valid_negative_control_presence"], 1.0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)
        self.assertLess(metrics["avg_contract_check_ms"], 5.0)
        self.assertIn("duplicate_of_existing_candidate", by_source["known_bad_duplicate"]["issues"])
        self.assertIn("conflicts_with_harness_governance", by_source["known_bad_conflict"]["issues"])

    def test_full_v2_phase1_graph_memory_bypass_demotes_risky_context(self):
        payload = build_full_v2_phase1_graph_memory_bypass_payload(eval_id="unit_full_v2_phase1_graph_memory")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreater(metrics["full_topk_precision"], metrics["semantic_topk_precision"])
        self.assertGreater(metrics["full_context_efficiency"], metrics["semantic_context_efficiency"])
        self.assertGreaterEqual(metrics["full_top1_accuracy"], 0.80)
        self.assertEqual(metrics["full_negative_transfer_rate"], 0.0)
        self.assertEqual(metrics["risky_node_topk_count"], 0)
        self.assertEqual(metrics["residual_retrieval_accuracy"], 1.0)

    def test_full_v2_phase2_verifier_bypass_classifies_residual_causes(self):
        payload = build_full_v2_phase2_verifier_bypass_payload(eval_id="unit_full_v2_phase2_verifier")
        metrics = payload["metrics"]
        by_case = {row["case_id"]: row for row in payload["rows"]}

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["residual_classification_accuracy"], 1.0)
        self.assertEqual(metrics["decision_accuracy"], 1.0)
        self.assertEqual(metrics["false_positive_rate_of_acceptance"], 0.0)
        self.assertEqual(metrics["regression_detection_recall"], 1.0)
        self.assertEqual(metrics["placebo_sensitivity"], 1.0)
        self.assertEqual(metrics["fresh_split_generalization"], 1.0)
        self.assertEqual(metrics["falsification_power"], 1.0)
        self.assertEqual(metrics["execution_lapse_new_hypothesis_count"], 0)
        self.assertEqual(by_case["case_execution_lapse"]["decision"], "repair_execution")
        self.assertEqual(by_case["case_world_model_defect"]["decision"], "calibrate_world_model")

    def test_full_v2_phase3_world_model_bypass_predicts_graph_action_rollouts(self):
        payload = build_full_v2_phase3_world_model_bypass_payload(eval_id="unit_full_v2_phase3_world_model")
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["accept_auroc"], 0.95)
        self.assertLess(metrics["accept_brier"], metrics["base_rate_brier"])
        self.assertGreaterEqual(metrics["regression_auroc"], 0.95)
        self.assertGreaterEqual(metrics["failure_type_f1"], 0.90)
        self.assertLessEqual(metrics["expected_value_mae"], 0.05)
        self.assertGreaterEqual(metrics["cost_saved"], 3)
        self.assertEqual(metrics["true_positive_block_rate"], 0.0)
        self.assertGreaterEqual(metrics["multi_step_rollout_accuracy"], 0.90)
        self.assertGreaterEqual(metrics["information_gain_correlation"], 0.90)

    def test_full_v2_phase4_hypothesis_generator_bypass_generates_multi_layer_families(self):
        payload = build_full_v2_phase4_hypothesis_generator_bypass_payload(
            eval_id="unit_full_v2_phase4_hypothesis_generator"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["execution_lapse_filtered_rate"], 1.0)
        self.assertGreaterEqual(metrics["cluster_count"], 6)
        self.assertGreaterEqual(metrics["min_candidates_per_cluster"], 2)
        self.assertEqual(metrics["candidate_layer_coverage"], 6)
        self.assertGreaterEqual(metrics["novel_family_rate"], 0.50)
        self.assertLessEqual(metrics["duplicate_rate"], 0.15)
        self.assertLessEqual(metrics["conflict_rate"], 0.15)
        self.assertGreaterEqual(metrics["fresh_validation_success_rate"], 0.80)
        self.assertGreaterEqual(metrics["cross_domain_transfer_rate"], 0.70)
        self.assertGreaterEqual(metrics["descendant_productivity"], 0.65)
        self.assertLessEqual(metrics["false_discovery_rate"], 0.10)
        self.assertGreaterEqual(metrics["residual_explained_fraction"], 0.90)
        self.assertEqual(metrics["manifest_validation_issue_count"], 0)
        self.assertGreaterEqual(metrics["world_model_screen_precision"], 0.90)

    def test_full_v2_phase5_strategy_scheduler_bypass_selects_strategy_families(self):
        payload = build_full_v2_phase5_strategy_scheduler_bypass_payload(
            eval_id="unit_full_v2_phase5_strategy_scheduler"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["strategy_library_size"], 20)
        self.assertGreaterEqual(metrics["strategy_selection_accuracy_against_experts"], 0.85)
        self.assertGreaterEqual(metrics["success_rate_improvement"], 0.20)
        self.assertGreaterEqual(metrics["time_to_solution_reduction"], 0.25)
        self.assertGreaterEqual(metrics["cross_domain_transfer"], 0.75)
        self.assertGreaterEqual(metrics["method_family_ACP"], 0.65)
        self.assertGreaterEqual(metrics["strategy_boundary_learning"], 0.85)
        self.assertGreaterEqual(metrics["negative_transfer_reduction"], 0.30)
        self.assertLessEqual(metrics["budget_allocation_mae"], 0.10)

    def test_full_v2_phase6_formal_alignment_bypass_predicts_transfer(self):
        payload = build_full_v2_phase6_formal_alignment_bypass_payload(
            eval_id="unit_full_v2_phase6_formal_alignment"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["alignment_precision_against_expert"], 0.95)
        self.assertGreaterEqual(metrics["negative_control_rejection"], 0.95)
        self.assertGreaterEqual(metrics["formal_equivalence_dedup_accuracy"], 0.95)
        self.assertGreaterEqual(metrics["formal_score_transfer_correlation"], 0.85)
        self.assertGreaterEqual(metrics["top1_formal_mapping_hit_rate"], 0.85)
        self.assertGreaterEqual(metrics["unsafe_mapping_block_rate"], 0.95)
        self.assertGreaterEqual(metrics["formal_margin_over_best_baseline"], 0.15)

    def test_full_v2_phase7_daemon_harness_bypass_runs_bounded_episode(self):
        payload = build_full_v2_phase7_daemon_harness_bypass_payload(
            eval_id="unit_full_v2_phase7_daemon_harness"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["long_run_stability"], 0.95)
        self.assertLessEqual(metrics["graph_pollution_rate"], 0.02)
        self.assertGreaterEqual(metrics["rollback_success_rate"], 0.95)
        self.assertLessEqual(metrics["cost_per_accepted_assumption"], 2.50)
        self.assertGreaterEqual(metrics["accepted_assumption_survival_rate"], 0.80)
        self.assertGreaterEqual(metrics["downstream_win_rate_on_unseen"], 0.65)
        self.assertGreaterEqual(metrics["capability_score_improvement"], 0.12)
        self.assertGreaterEqual(metrics["daemon_recovery_success"], 0.95)
        self.assertGreaterEqual(metrics["evaluator_integrity"], 0.95)
        self.assertEqual(metrics["unconditional_apply_count"], 0)

    def test_full_v2_vertical_slice_bypass_runs_five_generation_loop(self):
        payload = build_full_v2_vertical_slice_bypass_payload(
            eval_id="unit_full_v2_vertical_slice"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["generation_count"], 5)
        self.assertGreaterEqual(metrics["candidate_count"], 25)
        self.assertGreaterEqual(metrics["live_call_saving_rate"], 0.50)
        self.assertEqual(metrics["true_positive_block_rate"], 0.0)
        self.assertGreaterEqual(metrics["accepted_assumption_survival_rate"], 0.80)
        self.assertGreaterEqual(metrics["residual_explained_delta"], 0.45)
        self.assertGreaterEqual(metrics["downstream_score_delta"], 0.10)
        self.assertLessEqual(metrics["graph_pollution_rate"], 0.02)
        self.assertGreaterEqual(metrics["world_model_brier_improvement"], 0.06)
        self.assertGreaterEqual(metrics["full_loop_margin_over_best_control"], 0.08)

    def test_full_v3_phase0_contract_checker_validates_overlay_admission(self):
        payload = build_full_v3_phase0_contract_checker_payload(
            eval_id="unit_full_v3_phase0_contract_checker"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["contract_item_coverage"], 1.0)
        self.assertEqual(metrics["valid_candidate_acceptance_rate"], 1.0)
        self.assertEqual(metrics["invalid_draft_rejection_rate"], 1.0)
        self.assertEqual(metrics["contract_decision_accuracy"], 1.0)
        self.assertEqual(metrics["duplicate_detection_recall"], 1.0)
        self.assertEqual(metrics["conflict_detection_recall"], 1.0)
        self.assertEqual(metrics["main_graph_mutation_count"], 0)
        self.assertEqual(metrics["production_contract_proposal_count"], 2)
        self.assertEqual(metrics["production_contract_admitted_count"], 1)
        self.assertEqual(metrics["production_contract_quarantined_count"], 1)
        self.assertEqual(metrics["production_contract_invalid_admitted_count"], 0)
        self.assertEqual(metrics["production_contract_applied_count"], 1)

    def test_full_v3_phase1_memory_consolidation_prunes_and_compresses_graph(self):
        payload = build_full_v3_phase1_memory_consolidation_payload(
            eval_id="unit_full_v3_phase1_memory_consolidation"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["duplicate_detection_recall"], 0.95)
        self.assertGreaterEqual(metrics["evidence_merge_precision"], 0.95)
        self.assertGreaterEqual(metrics["scope_refinement_accuracy"], 0.90)
        self.assertGreaterEqual(metrics["stale_evidence_prune_recall"], 0.95)
        self.assertGreaterEqual(metrics["conflict_detection_recall"], 0.95)
        self.assertGreaterEqual(metrics["acp_update_correlation"], 0.90)
        self.assertGreaterEqual(metrics["retrieval_precision_delta"], 0.20)
        self.assertGreaterEqual(metrics["negative_transfer_reduction"], 0.50)
        self.assertGreaterEqual(metrics["context_efficiency_delta"], 0.20)
        self.assertEqual(metrics["idempotence_delta"], 0)
        self.assertGreaterEqual(metrics["production_sleep_group_count"], 3)
        self.assertGreaterEqual(metrics["production_sleep_planned_consolidated_node_count"], 3)
        self.assertGreaterEqual(metrics["production_sleep_applied_consolidated_node_count"], 3)
        self.assertGreaterEqual(metrics["production_sleep_applied_archived_node_count"], 3)
        self.assertFalse(metrics["production_sleep_dry_run_mutated"])

    def test_memory_consolidation_job_dry_run_and_apply_on_jsonl_graph(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="mem_bridge_a",
                type=AssumptionType.METHOD,
                claim="Bridge roles improve retrieval",
                context_conditions=["multi_hop", "role_bridge"],
                predicted_effects=["increase retrieval precision"],
                risk_predictions=["outside control harm"],
                verifiers=["retrieval_hit_audit", "outside_negative_control"],
                confidence=0.85,
                metaproductivity=0.7,
                status="active",
                tags=["family:bridge_roles"],
                payload={"family": "bridge_roles"},
            ))
            store.upsert_node(AssumptionNode(
                id="mem_bridge_b",
                type=AssumptionType.METHOD,
                claim="Typed bridge decomposition improves retrieval",
                context_conditions=["multi_hop", "role_bridge"],
                predicted_effects=["increase retrieval precision"],
                risk_predictions=["outside control harm"],
                verifiers=["retrieval_hit_audit", "outside_negative_control"],
                confidence=0.82,
                metaproductivity=0.68,
                status="active",
                tags=["family:bridge_roles"],
                payload={"family": "bridge_roles"},
            ))
            store.upsert_node(AssumptionNode(
                id="mem_bridge_stale",
                type=AssumptionType.METHOD,
                claim="Always add bridge entities",
                confidence=0.2,
                status="stale",
                tags=["family:bridge_roles"],
                payload={"family": "bridge_roles"},
            ))
            store.flush()

            dry_run = build_memory_consolidation_job_payload(
                store=JsonlGraphStore(td),
                eval_id="unit_memory_consolidation_dry_run",
                apply=False,
            )
            self.assertTrue(dry_run["pass"], dry_run["failed_gates"])
            self.assertFalse(dry_run["metrics"]["store_mutated"])
            self.assertEqual(dry_run["metrics"]["planned_consolidated_node_count"], 1)
            self.assertEqual(dry_run["metrics"]["planned_archive_count"], 2)
            self.assertFalse(any(
                node.payload.get("memory_consolidation") for node in JsonlGraphStore(td).nodes.values()
            ))

            apply_payload = build_memory_consolidation_job_payload(
                store=JsonlGraphStore(td),
                eval_id="unit_memory_consolidation_apply",
                apply=True,
            )
            self.assertTrue(apply_payload["pass"], apply_payload["failed_gates"])
            self.assertTrue(apply_payload["metrics"]["store_mutated"])
            self.assertEqual(apply_payload["metrics"]["applied_consolidated_node_count"], 1)
            self.assertGreaterEqual(apply_payload["metrics"]["applied_archived_node_count"], 1)
            updated = JsonlGraphStore(td)
            consolidated_ids = apply_payload["result"]["consolidated_node_ids"]
            self.assertEqual(len(consolidated_ids), 1)
            self.assertIn(consolidated_ids[0], updated.nodes)
            self.assertEqual(updated.nodes["mem_bridge_b"].status, "archived")
            self.assertEqual(updated.nodes["mem_bridge_stale"].status, "archived")

    def test_full_v3_phase2_verifier_synthesis_generates_falsification_contracts(self):
        payload = build_full_v3_phase2_verifier_synthesis_payload(
            eval_id="unit_full_v3_phase2_verifier_synthesis"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["test_type_coverage"], 1.0)
        self.assertGreaterEqual(metrics["contract_completeness"], 0.95)
        self.assertGreaterEqual(metrics["decision_accuracy"], 0.95)
        self.assertEqual(metrics["false_positive_rate_of_acceptance"], 0.0)
        self.assertGreaterEqual(metrics["regression_detection_recall"], 0.95)
        self.assertGreaterEqual(metrics["placebo_sensitivity"], 0.95)
        self.assertGreaterEqual(metrics["fresh_split_generalization"], 0.90)
        self.assertGreaterEqual(metrics["falsification_power"], 0.90)
        self.assertEqual(metrics["execution_lapse_new_hypothesis_count"], 0)

    def test_full_v3_phase3_rollout_search_control_selects_high_value_branches(self):
        payload = build_full_v3_phase3_rollout_search_control_payload(
            eval_id="unit_full_v3_phase3_rollout_search_control"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["branch_count"], 10)
        self.assertEqual(metrics["rollout_horizon"], 3)
        self.assertGreaterEqual(metrics["top_branch_precision"], 0.75)
        self.assertGreaterEqual(metrics["live_call_saving_rate"], 0.50)
        self.assertEqual(metrics["true_positive_block_rate"], 0.0)
        self.assertGreaterEqual(metrics["multi_step_rollout_accuracy"], 0.90)
        self.assertGreaterEqual(metrics["descendant_productivity_correlation"], 0.90)
        self.assertLessEqual(metrics["expected_value_mae"], 0.05)
        self.assertGreaterEqual(metrics["regression_recall"], 0.95)
        self.assertLessEqual(metrics["oracle_regret"], 0.05)

    def test_full_v3_phase4_hypothesis_generator_runs_variation_evaluation_retention(self):
        payload = build_full_v3_phase4_hypothesis_generator_payload(
            eval_id="unit_full_v3_phase4_hypothesis_generator"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["layer_coverage"], 1.0)
        self.assertGreaterEqual(metrics["min_trajectories_per_cluster"], 2)
        self.assertEqual(metrics["execution_lapse_filtered_rate"], 1.0)
        self.assertGreaterEqual(metrics["novelty_integration_accuracy"], 0.90)
        self.assertGreaterEqual(metrics["selective_retention_precision"], 0.90)
        self.assertGreaterEqual(metrics["world_model_screen_precision"], 0.90)
        self.assertLessEqual(metrics["false_discovery_rate"], 0.10)
        self.assertGreaterEqual(metrics["recursive_runner_seed_rate"], 0.45)

    def test_full_v3_phase5_contextual_bandit_scheduler_learns_policy_bundle(self):
        payload = build_full_v3_phase5_contextual_bandit_scheduler_payload(
            eval_id="unit_full_v3_phase5_contextual_bandit_scheduler"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["strategy_selection_accuracy"], 0.85)
        self.assertGreaterEqual(metrics["cumulative_reward_lift"], 0.20)
        self.assertGreaterEqual(metrics["regret_reduction_vs_baseline"], 0.50)
        self.assertLessEqual(metrics["posterior_brier"], 0.08)
        self.assertLessEqual(metrics["budget_allocation_mae"], 0.10)
        self.assertGreaterEqual(metrics["verifier_selection_accuracy"], 0.90)
        self.assertGreaterEqual(metrics["world_model_selection_accuracy"], 0.90)
        self.assertEqual(metrics["unsafe_exploration_count"], 0)
        self.assertGreaterEqual(metrics["negative_transfer_reduction"], 0.50)
        self.assertGreaterEqual(metrics["last_half_selection_accuracy"], metrics["first_half_selection_accuracy"])

    def test_full_v3_phase6_formal_transfer_engine_emits_bounded_certificates(self):
        payload = build_full_v3_phase6_formal_transfer_engine_payload(
            eval_id="unit_full_v3_phase6_formal_transfer_engine"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["proof_lite_certificate_coverage"], 1.0)
        self.assertEqual(metrics["typed_role_mapping_coverage"], 1.0)
        self.assertEqual(metrics["negative_control_coverage"], 1.0)
        self.assertGreaterEqual(metrics["unsafe_mapping_block_rate"], 0.95)
        self.assertGreaterEqual(metrics["formal_score_transfer_correlation"], 0.85)
        self.assertGreaterEqual(metrics["formal_margin_over_best_baseline"], 0.15)
        self.assertEqual(metrics["category_theorem_prover_claim_count"], 0)

    def test_full_v3_phase7_long_run_benchmark_validates_frozen_harness(self):
        payload = build_full_v3_phase7_long_run_benchmark_payload(
            eval_id="unit_full_v3_phase7_long_run_benchmark"
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(metrics["long_run_stability"], 0.95)
        self.assertLessEqual(metrics["graph_pollution_rate"], 0.02)
        self.assertGreaterEqual(metrics["rollback_success_rate"], 0.95)
        self.assertLessEqual(metrics["cost_per_accepted_assumption"], 2.50)
        self.assertGreaterEqual(metrics["accepted_assumption_survival_rate"], 0.80)
        self.assertGreaterEqual(metrics["downstream_win_rate_on_unseen"], 0.65)
        self.assertGreaterEqual(metrics["capability_score_improvement"], 0.15)
        self.assertGreaterEqual(metrics["daemon_recovery_success"], 0.95)
        self.assertGreaterEqual(metrics["evaluator_integrity"], 0.95)
        self.assertGreaterEqual(metrics["parallel_speedup_proxy"], 2.0)
        self.assertEqual(metrics["rate_limit_violation_count"], 0)
        self.assertGreaterEqual(metrics["checkpoint_recovery_success"], 0.95)
        self.assertGreaterEqual(metrics["continuous_learning_acp_lift"], 0.10)

    def test_full_v3_frozen_v1_comparison_shows_downstream_margin(self):
        payload = build_full_v3_frozen_v1_comparison_payload(
            root=Path("."),
            eval_id="unit_full_v3_frozen_v1_comparison",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["phase_pass_rate"], 1.0)
        self.assertGreaterEqual(metrics["full_v3_margin_vs_v1_kernel"], 0.10)
        self.assertGreaterEqual(metrics["full_v3_margin_vs_hipporag_style"], 0.10)
        self.assertGreaterEqual(metrics["full_v3_margin_vs_best_nonfull"], 0.08)
        self.assertGreaterEqual(metrics["assumption_capability_improvement"], 0.15)
        self.assertLess(metrics["main_structural_vs_base_p_value"], 0.05)
        self.assertEqual(metrics["fresh_api_call_count"], 0)

    def test_full_v3_paper_scale_evidence_aggregates_live_and_mechanism_artifacts(self):
        payload = build_full_v3_paper_scale_evidence_payload(
            root=Path("."),
            eval_id="unit_full_v3_paper_scale_evidence",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["required_artifact_pass_rate"], 1.0)
        self.assertGreaterEqual(metrics["raw_first_party_live_event_count"], 6000)
        self.assertGreaterEqual(metrics["valid_judge_event_count"], 2500)
        self.assertGreaterEqual(metrics["main_problem_level_n"], 100)
        self.assertGreater(metrics["structural_vs_base_ci_lower"], 0.50)
        self.assertLess(metrics["structural_vs_base_p_value"], 0.05)
        self.assertGreater(metrics["structural_vs_placebo_ci_lower"], 0.60)
        self.assertLess(metrics["structural_vs_placebo_p_value"], 0.001)
        self.assertGreaterEqual(metrics["retrieval_margin_over_best_baseline"], 0.70)
        self.assertGreaterEqual(metrics["key_toggle_min_margin"], 0.05)
        self.assertEqual(metrics["v3_mechanism_pass_rate"], 1.0)
        self.assertEqual(metrics["phase0_production_contract_proposal_count"], 2)
        self.assertEqual(metrics["phase0_production_contract_invalid_admitted_count"], 0)
        self.assertEqual(metrics["phase0_production_contract_applied_count"], 1)
        self.assertGreaterEqual(metrics["phase1_production_sleep_group_count"], 3)
        self.assertGreaterEqual(metrics["phase1_production_sleep_applied_consolidated_node_count"], 3)
        self.assertFalse(metrics["phase1_production_sleep_dry_run_mutated"])
        self.assertGreaterEqual(metrics["long_run_downstream_win_rate"], 0.65)
        self.assertGreaterEqual(metrics["full_v3_margin_vs_v1_kernel"], 0.10)
        self.assertGreaterEqual(metrics["full_v3_margin_vs_best_nonfull"], 0.08)
        self.assertEqual(metrics["fresh_live_guarded_problem_level_n"], 300)
        self.assertGreaterEqual(metrics["fresh_live_guarded_active_intervention_n"], 10)
        self.assertGreater(metrics["fresh_live_guarded_vs_base_utility"], 0.50)
        self.assertGreater(metrics["fresh_live_guarded_vs_placebo_utility"], 0.50)
        self.assertLessEqual(metrics["fresh_live_guarded_planned_total_calls"], 100)
        self.assertGreaterEqual(metrics["fresh_live_full_problem_level_n"], 500)
        self.assertGreaterEqual(metrics["fresh_live_full_active_intervention_n"], 20)
        self.assertGreater(metrics["fresh_live_full_vs_base_utility"], 0.50)
        self.assertGreater(metrics["fresh_live_full_vs_placebo_utility"], 0.50)
        self.assertLessEqual(metrics["fresh_live_full_planned_total_calls"], 150)
        self.assertGreater(
            metrics["fresh_live_selective_active_intervention_n"],
            metrics["fresh_live_full_active_intervention_n"],
        )
        self.assertGreaterEqual(metrics["fresh_live_selective_active_intervention_n"], 31)
        self.assertGreater(
            metrics["fresh_live_selective_vs_base_utility"],
            metrics["fresh_live_full_vs_base_utility"],
        )
        self.assertGreater(metrics["fresh_live_selective_vs_base_utility"], 0.51)
        self.assertGreater(
            metrics["fresh_live_selective_vs_placebo_utility"],
            metrics["fresh_live_full_vs_placebo_utility"],
        )
        self.assertGreater(metrics["fresh_live_selective_vs_placebo_utility"], 0.51)
        self.assertGreater(metrics["fresh_live_selective_vs_placebo_ci_lower"], 0.50)
        self.assertLessEqual(metrics["fresh_live_selective_planned_total_calls"], 200)
        self.assertGreaterEqual(metrics["phase9_compact_guard_vs_v1_n"], 31)
        self.assertGreaterEqual(metrics["phase9_compact_guard_vs_v1_margin"], 0.10)
        self.assertGreater(metrics["phase9_compact_guard_margin_gain_over_v3"], 0.05)
        self.assertGreaterEqual(metrics["phase9_compact_guard_vs_v3_utility"], 0.48)
        self.assertGreaterEqual(metrics["phase9_hybrid_guard_heldout_n"], 50)
        self.assertGreaterEqual(metrics["phase9_hybrid_guard_vs_v1_margin"], 0.10)
        self.assertGreater(metrics["phase9_hybrid_guard_lift_over_v3"], 0.03)
        self.assertGreaterEqual(metrics["phase9_hybrid_guard_vs_v3_utility"], 0.50)
        self.assertGreaterEqual(metrics["phase10_world_model_candidate_count"], 17)
        self.assertGreater(metrics["phase10_world_model_candidate_v1_lift_over_v3"], 0.04)
        self.assertGreaterEqual(metrics["phase10_world_model_all_lift_over_v3"], 0.015)
        self.assertEqual(metrics["phase10_world_model_recommended_promotion"], "keep_as_world_model_candidate")
        self.assertEqual(metrics["phase11_outer_shell_production_claim_count"], 0)
        self.assertGreaterEqual(metrics["phase11_blocked_claim_count"], 10)
        self.assertFalse(metrics["prompt_answer_payload_stored"])
        self.assertFalse(metrics["secret_leak_detected"])
        self.assertGreaterEqual(metrics["boundary_case_count"], 1)

    def test_full_v3_phase9_hybrid_guard_selectively_repairs_v1_regression(self):
        payload = build_full_v3_phase9_hybrid_guard_heldout_payload(
            root=Path("."),
            eval_id="unit_full_v3_phase9_hybrid_guard",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"])
        self.assertEqual(metrics["heldout_case_count"], 54)
        self.assertEqual(metrics["selected_candidate_case_count"], 17)
        self.assertEqual(metrics["hybrid_selected_arm_counts"]["v3_micro_guard"], 6)
        self.assertEqual(metrics["hybrid_selected_arm_counts"]["v3_selective_compact_guard"], 8)
        self.assertGreaterEqual(metrics["hybrid_vs_v1_heldout_margin"], 0.10)
        self.assertGreater(metrics["hybrid_lift_over_v3_vs_v1_heldout"], 0.03)
        self.assertGreaterEqual(metrics["hybrid_vs_original_v3_heldout_utility"], 0.50)

    def test_full_v3_phase10_discrete_world_model_selector_beats_original_v3(self):
        payload = build_full_v3_phase10_discrete_world_model_selector_payload(
            root=Path("."),
            eval_id="unit_full_v3_phase10_discrete_world_model_selector",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["heldout_transition_row_count"], 54)
        self.assertEqual(metrics["compact_support_row_count"], 31)
        self.assertEqual(metrics["candidate_transition_count"], 17)
        self.assertEqual(metrics["candidate_action_coverage"], 1.0)
        self.assertGreater(metrics["loo_selected_reward_lift_over_v3"], 0.02)
        self.assertGreater(metrics["loo_selected_vs_v1_lift_over_v3"], 0.04)
        self.assertGreaterEqual(metrics["loo_selected_vs_v3_utility"], 0.52)
        self.assertGreaterEqual(metrics["all_heldout_policy_lift_over_v3"], 0.015)
        self.assertLess(metrics["all_heldout_policy_vs_v1_utility"], metrics["retained_hybrid_vs_v1_utility"])
        self.assertEqual(metrics["recommended_promotion"], "keep_as_world_model_candidate")
        self.assertFalse(metrics["uses_raw_prompts_or_answers"])
        self.assertTrue(payload["teacher_distillation_bootstrap"]["not_counted_as_independent_validation"])

    def test_full_v3_phase11_capability_audit_separates_fixture_from_production(self):
        payload = build_full_v3_phase11_capability_audit_payload(
            root=Path("."),
            eval_id="unit_full_v3_phase11_capability_audit",
        )
        metrics = payload["metrics"]
        by_id = {row["capability_id"]: row for row in payload["capability_rows"]}

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["capability_count"], 11)
        self.assertEqual(metrics["artifact_pass_rate"], 1.0)
        self.assertEqual(metrics["outer_shell_count"], 5)
        self.assertEqual(metrics["outer_shell_production_claim_count"], 0)
        self.assertEqual(metrics["phase10_status"], "learned_candidate_not_promoted")
        self.assertIn("production_contract_gate_available", by_id["phase0_contract_checker"]["implementation_level"])
        self.assertEqual(by_id["phase9_hybrid_guard"]["production_default_status"], "retained_gated_profile")
        self.assertIn("jsonl_memory_sleep_job_available", by_id["phase1_memory_consolidation"]["implementation_level"])
        self.assertIn("not_long_running_production", by_id["phase7_long_run_benchmark"]["implementation_level"])
        self.assertGreaterEqual(metrics["blocked_claim_count"], 10)

    def test_full_v3_fresh_live_benchmark_plans_parallel_problem_level_run(self):
        with tempfile.TemporaryDirectory() as td:
            payload = build_full_v3_fresh_live_benchmark_payload(
                root=Path("."),
                eval_id="unit_full_v3_fresh_live_preflight",
                sample_size=300,
                execution_mode="dry_run",
                solve_workers=16,
                judge_workers=8,
                solver_model="gpt_mini",
                judge_model="gpt55",
                run_dir=Path(td),
                sample_out=Path(td) / "sample.json",
            )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(metrics["sample_problem_count"], 300)
        self.assertTrue(metrics["disjoint_from_existing_samples"])
        self.assertGreaterEqual(metrics["domain_count"], 5)
        self.assertEqual(metrics["selection_mode"], "natural_repaired_guarded")
        self.assertTrue(metrics["abstained_problems_count_as_tie"])
        self.assertGreaterEqual(metrics["selected_case_count"], 5)
        self.assertEqual(
            metrics["planned_total_model_calls"],
            metrics["selected_case_count"] * 5,
        )
        self.assertEqual(payload["parallel_plan"]["solve_workers"], 16)
        self.assertEqual(payload["parallel_plan"]["judge_workers"], 8)
        self.assertFalse(metrics["secret_value_exposed"])
        self.assertFalse(payload["problem_level_ci"]["available"])
        self.assertIn("run_300", payload["commands"])

    def test_full_v3_phase8_separates_creativity_world_model_and_coverage_profiles(self):
        payload = build_full_v3_phase8_creativity_world_coverage_payload(
            root=Path("."),
            eval_id="unit_full_v3_phase8_creativity_world_coverage",
        )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertGreaterEqual(metrics["creative_candidate_count"], 8)
        self.assertGreaterEqual(metrics["nonlocal_candidate_ratio"], 0.35)
        self.assertEqual(metrics["residual_cluster_coverage"], 1.0)
        self.assertGreaterEqual(metrics["quality_world_model_auroc"], 0.85)
        self.assertLess(metrics["quality_world_model_brier"], metrics["quality_base_rate_brier"])
        self.assertEqual(metrics["selected_quality_profile_id"], "quality_v4")
        self.assertEqual(metrics["selected_coverage_profile_id"], "coverage_v6")
        self.assertGreaterEqual(metrics["coverage_profile_active_gain_over_quality"], 4)
        self.assertGreater(metrics["coverage_profile_vs_base_utility"], 0.50)
        self.assertGreater(metrics["coverage_profile_vs_placebo_utility"], 0.50)
        self.assertGreater(
            metrics["quality_profile_vs_base_utility"],
            metrics["coverage_profile_vs_base_utility"],
        )

    def test_full_v3_phase9_plans_same_batch_v1_live_regression_gate(self):
        with tempfile.TemporaryDirectory() as td:
            payload = build_full_v3_phase9_v1_live_regression_payload(
                root=Path("."),
                eval_id="unit_full_v3_phase9_v1_live_regression",
                execution_mode="dry_run",
                sample_size=300,
                active_sample_size=0,
                run_dir=Path(td),
                sample_out=Path(td) / "sample.json",
                solve_workers=8,
                judge_workers=4,
                bootstrap_samples=100,
            )
        metrics = payload["metrics"]

        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(payload["arms"]["primary"], "v3_full")
        self.assertIn("v1_case_reflection_kernel", payload["arms"]["baselines"])
        self.assertIn("v3_no_morphism", payload["arms"]["baselines"])
        self.assertIn("v3_no_recursive", payload["arms"]["baselines"])
        self.assertIn("v3_no_world_model", payload["arms"]["baselines"])
        self.assertGreaterEqual(metrics["active_case_count"], 18)
        self.assertGreaterEqual(metrics["active_domain_count"], 3)
        self.assertGreaterEqual(metrics["active_pattern_count"], 3)
        self.assertEqual(metrics["planned_total_model_calls"], metrics["active_case_count"] * 9)
        self.assertEqual(payload["hard_regression_policy"]["min_v3_margin_vs_v1"], 0.10)
        self.assertGreaterEqual(metrics["coverage_profile_active_gain_over_quality"], 4)
        self.assertFalse(metrics["compact_payload_contains_prompts_answers"])

    def test_metaproductivity_benchmark_prefers_productive_clade(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_meta",
                type=AssumptionType.METHOD,
                claim="risk rollback guardrail reusable clade",
                confidence=0.7,
                metaproductivity=0.4,
            ))
            graph = SimpleAssumptionGraph(store)
            payload = build_metaproductivity_benchmark_payload(
                graph,
                eval_id="unit_meta_benchmark",
                queries=["risk rollback guardrail", "unmatched cold start"],
            )
            self.assertTrue(payload["positive_control"]["pass"])
            self.assertTrue(payload["pass"])
            self.assertEqual(payload["positive_control"]["acp_top_id"], "productive_parent")

    def test_acp_learning_updates_clade_value_from_acceptance_descendants(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="learned_parent",
                type=AssumptionType.METHOD,
                claim="recursive acp scheduler rollback policy",
                confidence=0.45,
                metaproductivity=0.0,
                payload={"evidence": {"delta": 0.02}},
            ))
            store.upsert_node(AssumptionNode(
                id="quick_parent",
                type=AssumptionType.METHOD,
                claim="recursive acp scheduler rollback quick fix",
                confidence=0.9,
                metaproductivity=0.0,
                payload={"evidence": {"delta": 0.45}},
            ))
            graph = SimpleAssumptionGraph(store)
            acceptance = {
                "eval_id": "unit_acp_acceptance",
                "summaries": [
                    {
                        "proposal_id": "learned_accept_a",
                        "parent_node_id": "learned_parent",
                        "candidate_node_id": "learned_child_a",
                        "decision": "accept",
                    },
                    {
                        "proposal_id": "learned_accept_b",
                        "parent_node_id": "learned_parent",
                        "candidate_node_id": "learned_child_b",
                        "decision": "accept",
                    },
                    {
                        "proposal_id": "learned_accept_c",
                        "parent_node_id": "learned_parent",
                        "candidate_node_id": "learned_child_c",
                        "decision": "accept",
                    },
                    {
                        "proposal_id": "quick_reject",
                        "parent_node_id": "quick_parent",
                        "candidate_node_id": "quick_child",
                        "decision": "reject_harm",
                    },
                ],
            }
            payload = build_acp_learning_payload(
                graph,
                eval_id="unit_acp_learning",
                acceptance_payload=acceptance,
            )
            self.assertTrue(payload["pass"])
            self.assertEqual(payload["accepted_descendant_count"], 3)
            self.assertEqual(payload["rejected_descendant_count"], 1)
            self.assertGreaterEqual(payload["policy_update_count"], 1)
            self.assertTrue(payload["positive_control"]["pass"])
            apply_acp_learning_updates(graph, payload, persist=False)
            acp_top = MetaproductivitySelector(graph).rank("scheduler policy", top_k=1)[0]
            immediate_top = MetaproductivitySelector(
                graph,
                weights=SelectionWeights(
                    retrieval=1.0,
                    immediate_utility=1.0,
                    metaproductivity=0.0,
                    confidence=0.2,
                    novelty=0.0,
                    risk=0.25,
                    cost=0.15,
                ),
            ).rank("scheduler policy", top_k=1)[0]
            self.assertEqual(acp_top.node.id, "learned_parent")
            self.assertEqual(immediate_top.node.id, "quick_parent")

    def test_retrieval_can_filter_primary_assumption_types(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["控制变量", "S01"],
                confidence=0.75,
            ))
            store.upsert_node(AssumptionNode(
                id="case_1",
                type=AssumptionType.CASE,
                claim="一次营销实验案例反复提到控制变量和小额测试",
                tags=["case", "S01"],
                confidence=0.9,
            ))
            store.flush()

            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            activated = graph.retrieve(
                "控制变量 小额测试",
                top_k=2,
                candidate_types={AssumptionType.METHOD},
            )
            self.assertEqual([n.id for n in activated.nodes], ["strategy_S01"])

    def test_trial_update_keeps_execution_lapse_from_penalizing_assumption(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(id="a1", type=AssumptionType.METHOD, claim="use controlled variables", confidence=0.7))
            graph = SimpleAssumptionGraph(store)
            trial = TrialManifest(
                problem_id="p1",
                action_type="strategy",
                assumption="control one variable",
                why_selected="high coupling risk",
                expected_effect="localize failure",
                assumption_ids=["a1"],
                residual="The plan was valid but not applied in the answer.",
                residual_type=ResidualType.EXECUTION_LAPSE,
                status=TrialStatus.FAILED,
            )
            graph.update_from_trial(trial, persist=False)
            self.assertAlmostEqual(store.nodes["a1"].confidence, 0.7)
            self.assertTrue(store.nodes["a1"].residual_ids)

    def test_residual_classifier(self):
        trial = TrialManifest(
            problem_id="p1",
            action_type="audit",
            assumption="selected wisdom should shape answer",
            why_selected="selection score high",
            expected_effect="answer uses wisdom",
            residual="草稿只是表面提及 wisdom，没真正执行",
        )
        assessed = classify_manifest(trial)
        self.assertEqual(assessed.residual_type, ResidualType.EXECUTION_LAPSE)

    def test_residual_label_agreement_uses_gold_examples(self):
        payload = build_residual_label_agreement_payload(eval_id="unit_residual_gold")
        self.assertTrue(payload["pass"])
        self.assertGreaterEqual(payload["example_count"], 8)
        self.assertEqual(payload["accuracy"], 1.0)
        self.assertEqual(payload["macro_f1"], 1.0)
        self.assertIn("memory_defect", payload["expected_type_counts"])
        self.assertEqual(payload["confusion"]["optimization"]["optimization"], 2)

    def test_large_residual_label_calibration_uses_graph_and_trace_labels(self):
        trace_path = Path("phase four/assumption_graph/trace_dataset_collection_distilled_20260602.json")
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        payload = build_large_residual_label_calibration_payload(
            eval_id="unit_large_residual_calibration",
            store=JsonlGraphStore("phase four/assumption_graph"),
            trace_dataset_payload=trace_payload,
            target_examples=120,
        )
        self.assertTrue(payload["pass"], payload["coverage"])
        self.assertGreaterEqual(payload["example_count"], 100)
        self.assertGreaterEqual(payload["macro_f1"], 0.85)
        self.assertGreaterEqual(payload["accuracy"], 0.85)
        self.assertIn("first_party_graph_residual::unknown", payload["label_source_counts"])

    def test_wisdom_and_exp82_adapters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wisdom_path = root / "wisdom.json"
            wisdom_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "W001",
                            "aphorism": "先立后破",
                            "source": "民间谚语",
                            "signal": "要改系统但风险高时",
                            "unpacked_for_llm": "先保留已验证部分，只替换一个新模块。",
                            "cross_domain_examples": [{"domain": "software", "scenario": "替换核心模块前保留旧管线。"}],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            hypo_path = root / "hypotheses.jsonl"
            hypo_path.write_text(
                json.dumps(
                    {
                        "hid": "abc",
                        "seed_cid": "WCAND01",
                        "kind": "decomposition",
                        "claim": "split problem into verifyable stages",
                        "expr": {"steps": ["find assumptions", "verify"]},
                        "trigger_subset": ["p1"],
                        "outside_subset": ["p2"],
                        "evidence": {"delta_ext_base": 0.12},
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            store = JsonlGraphStore(root / "graph")
            ingest_artifacts(store, [load_wisdom_nodes(wisdom_path), load_exp82_hypotheses(hypo_path)])
            self.assertIn("wisdom_W001", store.nodes)
            self.assertIn("hyp_abc", store.nodes)
            self.assertTrue(store.evidence)
            ranked = MetaproductivitySelector(SimpleAssumptionGraph(store)).rank("verifyable stages", top_k=2)
            self.assertTrue(ranked)

    def test_record_phase2_eval_writes_trials_and_residuals(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["S01", "控制变量"],
                confidence=0.6,
            ))
            store.flush()

            sample_path = root / "sample.json"
            sample_path.write_text(json.dumps([
                {
                    "problem_id": "p1",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "预算有限时先小额测试不同渠道。",
                    "coverage_tags": ["S01"],
                }
            ], ensure_ascii=False), encoding="utf-8")
            meta_path = root / "meta.json"
            meta_path.write_text(json.dumps({
                "p1": {
                    "frame": "hybrid",
                    "critical_reframe": "用小实验定位有效渠道。",
                    "rewritten_problem": "设计小额对照实验。",
                    "what_changed": "显式化预算约束。",
                    "anti_patterns": [],
                }
            }, ensure_ascii=False), encoding="utf-8")
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {
                    "winner": "baseline",
                    "score_a": 8,
                    "score_b": 9,
                    "reasoning": "baseline 更完整。",
                    "a_was": "A",
                }
            }, ensure_ascii=False), encoding="utf-8")

            summary = record_phase2_eval(
                root=root,
                graph_dir=graph_dir,
                sample_path=sample_path,
                meta_path=meta_path,
                judgment_paths=[judgment_path],
                intervention_variant="ag",
                baseline_variant="baseline",
                eval_id="unit_eval",
                top_k=1,
            )
            updated = JsonlGraphStore(graph_dir)
            self.assertEqual(summary["outcomes"], {"loss": 1})
            self.assertEqual(summary["residual_types"], {"optimization": 1})
            self.assertEqual(len(updated.trials), 1)
            self.assertTrue(updated.evidence)
            self.assertTrue(updated.nodes["strategy_S01"].residual_ids)

    def test_evolution_cycle_plans_loop_without_mutating_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["S01", "控制变量"],
                confidence=0.6,
            ))
            store.flush()

            sample_path = root / "sample.json"
            sample_path.write_text(json.dumps([
                {
                    "problem_id": "p1",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "预算有限时先小额测试不同渠道。",
                    "coverage_tags": ["S01"],
                }
            ], ensure_ascii=False), encoding="utf-8")
            meta_path = root / "meta.json"
            meta_path.write_text(json.dumps({
                "p1": {
                    "frame": "hybrid",
                    "critical_reframe": "用小实验定位有效渠道。",
                    "rewritten_problem": "设计小额对照实验。",
                    "what_changed": "显式化预算约束。",
                    "anti_patterns": [],
                }
            }, ensure_ascii=False), encoding="utf-8")
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {
                    "winner": "ag",
                    "score_a": 9,
                    "score_b": 8,
                    "reasoning": "ag 更具体。",
                    "a_was": "A",
                }
            }, ensure_ascii=False), encoding="utf-8")

            payload = build_evolution_cycle_payload(
                root=root,
                graph_dir=graph_dir,
                sample_path=sample_path,
                meta_path=meta_path,
                judgment_paths=[judgment_path],
                intervention_variant="ag",
                baseline_variant="base",
                eval_id="unit_cycle",
                min_benefit_n=1,
                min_harm_n=1,
            )
            self.assertTrue(payload["writeback_summary"]["dry_run"])
            self.assertEqual(payload["writeback_summary"]["processed"], 1)
            self.assertEqual(payload["conditioned"]["decision_counts"], {"keep": 1})
            self.assertEqual(payload["lifecycle"]["action_counts"], {"keep_collect_evidence": 1})
            self.assertEqual(payload["proposals"]["proposal_counts"], {"evidence_request": 1})
            self.assertEqual(payload["candidate_preflight"]["readiness_counts"], {"manifest_only": 1})
            self.assertEqual(payload["falsification_gate"]["decision_counts"], {"manifest_only": 1})
            self.assertEqual(payload["bayesian_policy"]["decision_counts"], {"record_only": 1})
            self.assertEqual(
                payload["policy_update_plan"]["actions"][0]["policy_action"],
                "record_manifest_only_no_graph_policy_change",
            )
            self.assertEqual(JsonlGraphStore(graph_dir).trials, {})

    def test_evolution_cycle_autonomous_apply_writes_only_gated_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests with a baseline and one intervention.",
                tags=["S01", "controlled", "baseline", "experiment"],
                context_conditions=["controlled variable experiment"],
            ))
            store.flush()

            sample = [
                {
                    "problem_id": f"p{i}",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": f"Use a controlled baseline experiment to test channel {i}.",
                    "coverage_tags": ["S01"],
                }
                for i in range(1, 4)
            ]
            sample_path = root / "sample.json"
            sample_path.write_text(json.dumps(sample), encoding="utf-8")
            meta_path = root / "meta.json"
            meta_path.write_text(json.dumps({
                p["problem_id"]: {
                    "frame": "hybrid",
                    "critical_reframe": "test one variable against a baseline",
                    "rewritten_problem": p["description"],
                    "what_changed": "explicit baseline",
                    "anti_patterns": [],
                }
                for p in sample
            }), encoding="utf-8")
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                p["problem_id"]: {"winner": "base", "score_a": 6, "score_b": 8}
                for p in sample
            }), encoding="utf-8")
            candidate_judgment_path = root / "candidate_judgments.json"
            candidate_judgment_path.write_text(json.dumps({
                p["problem_id"]: {"winner": "candidate"}
                for p in sample
            }), encoding="utf-8")

            payload = build_evolution_cycle_payload(
                root=root,
                graph_dir=graph_dir,
                sample_path=sample_path,
                meta_path=meta_path,
                judgment_paths=[judgment_path],
                intervention_variant="ag",
                baseline_variant="base",
                eval_id="unit_auto_cycle",
                min_benefit_n=1,
                min_harm_n=1,
                failure_hypothesis_top_n=0,
                candidate_judgment_paths=[candidate_judgment_path],
                candidate_variant="candidate",
                candidate_baseline_variant="base",
                autonomous_apply=True,
                train_world_model_calibration_flag=True,
                world_model_calibration_out=root / "world_model_calibration.json",
            )

            summary = payload["autonomous_apply_summary"]
            self.assertTrue(summary["enabled"])
            self.assertTrue(summary["writeback_applied"])
            self.assertTrue(summary["candidate_apply_requested"])
            self.assertTrue(summary["applied_candidate_node_ids"])
            updated = JsonlGraphStore(graph_dir)
            self.assertTrue(updated.trials)
            for node_id in summary["applied_candidate_node_ids"]:
                self.assertEqual(updated.nodes[node_id].status, "active")
            self.assertTrue(payload["world_model_calibration"]["active"])
            self.assertEqual(payload["world_model_calibration"]["status"], "trained")
            self.assertEqual(payload["world_model_calibration"]["labeled_count"], 1)
            self.assertEqual(payload["world_model"]["calibration_model"]["labeled_count"], 1)
            self.assertTrue((root / "world_model_calibration.json").exists())

    def test_recursive_runner_builds_argument_tree_from_evolution_payload(self):
        with tempfile.TemporaryDirectory() as td:
            graph_dir = Path(td) / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
                tags=["S01", "controlled"],
                confidence=0.7,
            ))
            store.flush()
            evolution_payload = {
                "proposals": {
                    "proposals": [{
                        "proposal_id": "prop_ready",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.8,
                        "candidate_node": {
                            "id": "cand_ready",
                            "claim": "Require a baseline and one intervention before answering.",
                            "predicted_effects": ["improve causal diagnosis"],
                        },
                    }],
                },
                "candidate_preflight": {
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "active_trigger_problem_ids": ["p1", "p2", "p3"],
                        "trigger_problem_ids": ["p1", "p2", "p3"],
                        "control_problem_ids": ["c1"],
                        "command_hint": "run candidate ablation",
                    }],
                },
                "falsification_gate": {
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "decision": FalsificationDecision.READY_FOR_ABLATION.value,
                        "next_action": "run_fresh_ablation",
                        "ordered_checks": [{"name": "trigger_power", "passed": True}],
                    }],
                },
                "bayesian_policy": {
                    "scores": [{
                        "proposal_id": "prop_ready",
                        "recommended_action": BayesianPolicyAction.RUN_ABLATION.value,
                        "posterior_priority": 1.2,
                        "expected_value": 0.7,
                        "command_hint": "run candidate ablation",
                    }],
                },
                "policy_update_plan": {
                    "actions": [{
                        "proposal_id": "prop_ready",
                        "policy_action": "run_fresh_ablation_before_promotion",
                    }],
                },
                "regression_predictions": [{
                    "proposal_id": "prop_ready",
                    "risk": "low",
                    "reasons": ["no outside active row"],
                }],
                "formal_mapping_gate": {
                    "gates": [{
                        "proposal_id": "prop_ready",
                        "decision": "not_applicable",
                        "blocks_policy_update": False,
                    }],
                },
            }

            payload = build_recursive_assumption_run(
                graph_dir=graph_dir,
                problem="Diagnose a channel experiment failure with one controlled intervention.",
                goal="Create a recursive assumption tree.",
                eval_id="unit_recursive",
                evolution_payload=evolution_payload,
                max_children=1,
            )

            self.assertEqual(payload["frame_counts"][RecursiveFrameType.ROOT_PROBLEM.value], 1)
            self.assertEqual(payload["frame_counts"][RecursiveFrameType.CANDIDATE_HYPOTHESIS.value], 1)
            self.assertEqual(payload["frame_counts"][RecursiveFrameType.VERIFICATION_SUBPROBLEM.value], 1)
            self.assertEqual(payload["status_counts"][RecursiveFrameStatus.READY_TO_ACT.value], 2)
            self.assertEqual(len(payload["recursion_edges"]), 2)
            candidate = next(
                frame for frame in payload["frames"]
                if frame["frame_type"] == RecursiveFrameType.CANDIDATE_HYPOTHESIS.value
            )
            self.assertIn("preflight readiness=ready_for_fresh_ablation", candidate["argument"]["support"])
            self.assertEqual(candidate["next_action"], "run_fresh_ablation_before_promotion")
            child = next(
                frame for frame in payload["frames"]
                if frame["frame_type"] == RecursiveFrameType.VERIFICATION_SUBPROBLEM.value
            )
            self.assertEqual(child["parent_frame_id"], candidate["frame_id"])
            self.assertEqual(child["next_action"], "run_fresh_ablation")
            self.assertEqual(JsonlGraphStore(graph_dir).trials, {})
            audit = build_recursive_audit_payload(
                recursive_payload=payload,
                eval_id="unit_recursive_audit",
            )
            self.assertTrue(audit["pass"])
            self.assertEqual(audit["critical_issue_count"], 0)
            self.assertGreaterEqual(audit["closure_score"], 0.9)
            self.assertEqual(audit["declared_edge_count"], audit["reconstructed_edge_count"])

    def test_recursive_runner_propagates_acceptance_results_to_parent_frontier(self):
        def ready_evolution_payload():
            return {
                "proposals": {
                    "proposals": [{
                        "proposal_id": "prop_ready",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.8,
                        "candidate_node": {
                            "id": "cand_ready",
                            "claim": "Require a baseline and one intervention before answering.",
                            "predicted_effects": ["improve causal diagnosis"],
                        },
                    }],
                },
                "candidate_preflight": {
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "active_trigger_problem_ids": ["p1", "p2", "p3"],
                        "trigger_problem_ids": ["p1", "p2", "p3"],
                        "control_problem_ids": ["c1"],
                    }],
                },
                "falsification_gate": {
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "decision": FalsificationDecision.READY_FOR_ABLATION.value,
                        "next_action": "run_fresh_ablation",
                        "ordered_checks": [{"name": "trigger_power", "passed": True}],
                    }],
                },
                "bayesian_policy": {
                    "scores": [{
                        "proposal_id": "prop_ready",
                        "recommended_action": BayesianPolicyAction.RUN_ABLATION.value,
                        "posterior_priority": 1.2,
                        "expected_value": 0.7,
                    }],
                },
                "policy_update_plan": {
                    "actions": [{
                        "proposal_id": "prop_ready",
                        "policy_action": "run_fresh_ablation_before_promotion",
                    }],
                },
                "regression_predictions": [{
                    "proposal_id": "prop_ready",
                    "risk": "low",
                    "reasons": ["no outside active row"],
                }],
                "formal_mapping_gate": {
                    "gates": [{
                        "proposal_id": "prop_ready",
                        "decision": "not_applicable",
                        "blocks_policy_update": False,
                    }],
                },
            }

        cases = {
            AcceptanceDecision.ACCEPT.value: (
                "apply_accepted_candidate_if_requested",
                RecursiveFrameStatus.READY_TO_ACT.value,
                RecursiveFrameStatus.RESOLVED.value,
                "accepted",
            ),
            AcceptanceDecision.REJECT_HARM.value: (
                "reject_or_narrow_scope",
                RecursiveFrameStatus.READY_TO_ACT.value,
                RecursiveFrameStatus.RESOLVED.value,
                "rejected_harm",
            ),
            AcceptanceDecision.REJECT_BENEFIT.value: (
                "reject_or_revise_candidate",
                RecursiveFrameStatus.READY_TO_ACT.value,
                RecursiveFrameStatus.RESOLVED.value,
                "rejected_benefit",
            ),
            AcceptanceDecision.INSUFFICIENT_JUDGMENTS.value: (
                "collect_more_judgments",
                RecursiveFrameStatus.WAITING_FOR_EVIDENCE.value,
                RecursiveFrameStatus.WAITING_FOR_EVIDENCE.value,
                "underpowered",
            ),
        }

        with tempfile.TemporaryDirectory() as td:
            graph_dir = Path(td) / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
                tags=["S01", "controlled"],
                confidence=0.7,
            ))
            store.flush()

            for decision, (parent_action, parent_status, child_status, outcome) in cases.items():
                with self.subTest(decision=decision):
                    payload = build_recursive_assumption_run(
                        graph_dir=graph_dir,
                        problem="Diagnose a channel experiment failure with one controlled intervention.",
                        goal="Create a recursive assumption tree.",
                        eval_id=f"unit_recursive_{decision}",
                        evolution_payload=ready_evolution_payload(),
                        acceptance_payload={
                            "summaries": [{
                                "proposal_id": "prop_ready",
                                "decision": decision,
                                "trigger_utility": 1.0,
                                "trigger_lcb90": 0.5,
                                "control_loss_ucb90": 0.0,
                                "rationale": f"unit {decision}",
                            }],
                        },
                        max_children=1,
                    )

                    candidate = next(
                        frame for frame in payload["frames"]
                        if frame["frame_type"] == RecursiveFrameType.CANDIDATE_HYPOTHESIS.value
                    )
                    child = next(
                        frame for frame in payload["frames"]
                        if frame["frame_type"] == RecursiveFrameType.VERIFICATION_SUBPROBLEM.value
                    )
                    self.assertEqual(candidate["next_action"], parent_action)
                    self.assertEqual(candidate["status"], parent_status)
                    self.assertEqual(child["status"], child_status)
                    self.assertEqual(child["return_update"]["outcome"], outcome)
                    self.assertIn(f"acceptance_decision={decision}", candidate["argument"]["support"])
                    self.assertEqual(payload["next_actions"][0]["frame_id"], candidate["frame_id"])
                    self.assertEqual(payload["next_actions"][0]["next_action"], parent_action)

    def test_recursive_executor_plans_leaf_commands_and_resumes_from_judgments(self):
        def ready_evolution_payload():
            return {
                "eval_id": "unit_cycle",
                "proposals": {
                    "eval_id": "unit_props",
                    "proposals": [{
                        "proposal_id": "prop_ready",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.8,
                        "candidate_node": {
                            "id": "cand_ready",
                            "claim": "Require a baseline and one intervention before answering.",
                            "predicted_effects": ["improve causal diagnosis"],
                        },
                    }],
                },
                "candidate_preflight": {
                    "eval_id": "unit_preflight",
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "active_trigger_problem_ids": ["p1", "p2", "p3"],
                        "trigger_problem_ids": ["p1", "p2", "p3"],
                        "control_problem_ids": [],
                        "command_hint": "python3 run_candidate.py --variant proposal_ready",
                    }],
                },
                "falsification_gate": {
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "decision": FalsificationDecision.READY_FOR_ABLATION.value,
                        "next_action": "run_fresh_ablation",
                        "ordered_checks": [{"name": "trigger_power", "passed": True}],
                    }],
                },
                "bayesian_policy": {
                    "scores": [{
                        "proposal_id": "prop_ready",
                        "recommended_action": BayesianPolicyAction.RUN_ABLATION.value,
                        "posterior_priority": 1.2,
                        "expected_value": 0.7,
                        "command_hint": "python3 run_candidate.py --variant proposal_ready",
                    }],
                },
                "policy_update_plan": {
                    "actions": [{
                        "proposal_id": "prop_ready",
                        "policy_action": "run_fresh_ablation_before_promotion",
                    }],
                },
                "regression_predictions": [{
                    "proposal_id": "prop_ready",
                    "risk": "low",
                    "reasons": ["no outside active row"],
                }],
                "formal_mapping_gate": {
                    "gates": [{
                        "proposal_id": "prop_ready",
                        "decision": "not_applicable",
                        "blocks_policy_update": False,
                    }],
                },
            }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
                tags=["S01", "controlled"],
                confidence=0.7,
            ))
            store.flush()
            evolution_payload = ready_evolution_payload()
            recursive_payload = build_recursive_assumption_run(
                graph_dir=graph_dir,
                problem="Diagnose a channel experiment failure with one controlled intervention.",
                goal="Create a recursive assumption tree.",
                eval_id="unit_recursive_exec",
                evolution_payload=evolution_payload,
                max_children=1,
            )

            planned = build_recursive_execution_payload(
                root=root,
                graph_dir=graph_dir,
                recursive_payload=recursive_payload,
                evolution_payload=evolution_payload,
                eval_id="unit_executor",
            )
            self.assertEqual(planned["frontier"]["planned_actions"], 1)
            self.assertEqual(planned["frontier"]["executable_actions"], 1)
            self.assertEqual(planned["execution_records"][0]["status"], "planned")
            self.assertIsNone(planned["candidate_acceptance"])

            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {"winner": "proposal_ready"},
                "p2": {"winner": "proposal_ready"},
                "p3": {"winner": "proposal_ready"},
            }), encoding="utf-8")
            resumed = build_recursive_execution_payload(
                root=root,
                graph_dir=graph_dir,
                recursive_payload=recursive_payload,
                evolution_payload=evolution_payload,
                eval_id="unit_executor_with_judgments",
                judgment_sets=[JudgmentSet(
                    candidate_variant="proposal_ready",
                    baseline_variant="base",
                    judgment_paths=[judgment_path],
                    proposal_ids=["prop_ready"],
                )],
            )
            self.assertEqual(resumed["candidate_acceptance"]["decision_counts"], {"accept": 1})
            self.assertEqual(
                resumed["resumed_recursive"]["next_actions"][0]["next_action"],
                "apply_accepted_candidate_if_requested",
            )
            self.assertEqual(
                resumed["resumed_recursive"]["next_actions"][0]["frame_type"],
                RecursiveFrameType.CANDIDATE_HYPOTHESIS.value,
            )

    def test_component_manifest_logger_records_and_redacts_agent_events(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            payload = build_component_manifest_payload(
                eval_id="unit_manifest",
                store=store,
                writeback=True,
                events=[{
                    "event_type": "llm_call",
                    "problem_id": "p1",
                    "component": "judge",
                    "assumption": "Judge calls should be auditable.",
                    "why_selected": "Need cross-check evidence.",
                    "expected_effect": "Record model, prompt hash, and outcome without secrets.",
                    "artifacts": {"request": "secret_token=unit-test-secret"},
                    "metadata": {"model": "gpt-5.5"},
                    "observed_effect": "judge returned candidate win",
                }],
            )
            self.assertEqual(payload["event_counts"], {"llm_call": 1})
            updated = JsonlGraphStore(td)
            self.assertEqual(len(updated.trials), 1)
            manifest = next(iter(updated.trials.values()))
            self.assertEqual(manifest.component, "judge")
            self.assertIn("[REDACTED]", manifest.artifacts["request"])
            self.assertNotIn("unit-test-secret", json.dumps(manifest.to_dict()))

    def test_manifest_logger_ingests_realistic_judge_run_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "judge.log"
            log_path.write_text(
                "\n".join([
                    "=== JUDGE prop_a proposal_a vs baseline rows=3 ===",
                    "LLM provider: gemini, model: gpt-5.5",
                    "  [judge proposal_a vs baseline] 3/3 (new=3 hit=0) 10s",
                    "=== DONE JUDGE prop_a returncode=0 elapsed=12.3s ===",
                ]),
                encoding="utf-8",
            )
            events = events_from_run_logs(root=root, log_paths=[log_path])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "judge_call")
            self.assertEqual(events[0]["artifacts"]["candidate_variant"], "proposal_a")
            self.assertEqual(events[0]["artifacts"]["returncode"], 0)

    def test_runtime_trace_recorder_persists_redacted_first_party_events(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            events_out = root / "events.jsonl"
            summary_out = root / "summary.json"
            recorder = RuntimeTraceRecorder(
                eval_id="unit_runtime_trace",
                events_out=events_out,
                summary_out=summary_out,
                graph_dir=graph_dir,
                writeback=True,
            )
            recorder.record_retrieval(
                problem_id="p1",
                component="phase2_assumption_graph_retrieval",
                assumption="Graph retrieval should select useful assumptions.",
                expected_effect="Expose relevant method and runtime assumptions.",
                activated_assumption_ids=["strategy_S01", "surface_verifier"],
                artifacts={"query": "debug with api_key=unit-secret"},
            )
            recorder.record_llm_call(
                problem_id="p1",
                component="phase2_turn1_draft",
                prompt_kind="execute_v20",
                assumption="The draft call should apply retrieved assumptions.",
                expected_effect="Generate a useful draft.",
                observed_effect="draft_chars=42",
                artifacts={"request": "secret_token=runtime-trace-secret"},
            )
            payload = recorder.flush()
            self.assertTrue(payload["enabled"])
            self.assertEqual(payload["event_count"], 2)
            self.assertEqual(payload["event_counts"], {"retrieval": 1, "llm_call": 1})
            self.assertTrue(events_out.exists())
            self.assertTrue(summary_out.exists())
            text = events_out.read_text(encoding="utf-8")
            self.assertIn("[REDACTED]", text)
            self.assertNotIn("unit-secret", text)
            self.assertNotIn("runtime-trace-secret", text)
            event_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            self.assertEqual(event_rows[0]["artifacts"]["trajectory_phase"], "retrieval")
            self.assertEqual(event_rows[1]["artifacts"]["trajectory_phase"], "draft")
            updated = JsonlGraphStore(graph_dir)
            self.assertEqual(len(updated.trials), 2)

    def test_trace_dataset_links_runtime_trace_to_outcomes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample.json"
            meta_path = root / "meta.json"
            judgments_path = root / "candidate_vs_baseline.json"
            events_path = root / "events.jsonl"
            sample_path.write_text(json.dumps([
                {
                    "problem_id": "p1",
                    "domain": "software_engineering",
                    "difficulty": "hard",
                    "coverage_tags": ["S01"],
                },
                {
                    "problem_id": "p2",
                    "domain": "science",
                    "difficulty": "medium",
                    "coverage_tags": ["S12"],
                },
                {
                    "problem_id": "p3",
                    "domain": "mathematics",
                    "difficulty": "hard",
                    "coverage_tags": ["S18"],
                },
            ]), encoding="utf-8")
            meta_path.write_text(json.dumps({
                "p1": {"frame": "hybrid"},
                "p2": {"frame": "object", "bypass_route": "science_mechanism"},
                "p3": {"frame": "hybrid", "bypass_route": "math_research_bridge"},
            }), encoding="utf-8")
            judgments_path.write_text(json.dumps({
                "p1": {"winner": "candidate", "score_a": 9, "score_b": 7, "a_was": "A", "reasoning": "candidate wins"},
                "p2": {"winner": "baseline", "score_a": 6, "score_b": 8, "a_was": "A", "reasoning": "baseline is more concrete"},
                "p3": {"winner": "candidate", "score_a": 8, "score_b": 7, "a_was": "B", "reasoning": "B wins"},
            }), encoding="utf-8")
            events = [
                {
                    "event_type": "retrieval",
                    "problem_id": "p1",
                    "component": "phase2_assumption_graph_retrieval",
                    "artifacts": {"activated_assumption_ids": ["strategy_S01"], "query": "api_key=trace-secret"},
                },
                {
                    "event_type": "llm_call",
                    "problem_id": "p1",
                    "component": "phase2_turn1_draft",
                    "artifacts": {"prompt_kind": "execute_v20"},
                },
                {
                    "event_type": "tool_use",
                    "problem_id": "p2",
                    "component": "phase2_cache_hit",
                    "artifacts": {"bypass_route": "science_mechanism", "request": "secret_token=unit-secret"},
                },
            ]
            events_path.write_text(
                "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
                encoding="utf-8",
            )
            payload = build_trace_dataset_payload(
                root=root,
                sample_path=sample_path,
                meta_path=meta_path,
                judgments_path=judgments_path,
                trace_events_path=events_path,
                intervention_variant="candidate",
                baseline_variant="baseline",
                eval_id="unit_trace_dataset",
                allow_artifact_trace=True,
            )
            self.assertEqual(payload["row_count"], 3)
            self.assertEqual(payload["trainable_row_count"], 3)
            self.assertEqual(payload["first_party_trace_count"], 2)
            self.assertEqual(payload["artifact_replay_count"], 1)
            self.assertEqual(payload["outcome_counts"], {"loss": 1, "win": 2})
            self.assertEqual(payload["residual_type_counts"]["optimization"], 1)
            self.assertEqual(payload["rows"][0]["activated_assumption_ids"], ["strategy_S01"])
            self.assertTrue(payload["rows"][0]["gold_hit"])
            self.assertEqual(payload["rows"][0]["score_delta"], 2.0)
            self.assertEqual(payload["rows"][0]["intervention_variant"], "candidate")
            self.assertEqual(payload["rows"][0]["baseline_variant"], "baseline")
            self.assertEqual(payload["rows"][0]["judgment_pair"], "candidate_vs_baseline")
            self.assertIn("retrieval", payload["rows"][0]["trajectory_phases"])
            self.assertIn("draft", payload["rows"][0]["trajectory_phases"])
            self.assertIn("artifact_final_replay", payload["rows"][1]["trajectory_phases"])
            self.assertTrue(payload["rows"][1]["draft_audit_final_coverage"])
            self.assertTrue(payload["rows"][2]["draft_audit_final_coverage"])
            self.assertGreaterEqual(payload["trajectory_complete_count"], 2)
            self.assertFalse(payload["secret_leak_detected"])
            self.assertNotIn("unit-secret", json.dumps(payload))
            self.assertNotIn("trace-secret", json.dumps(payload))
            coverage = build_trace_residual_coverage_payload(
                trace_dataset_payload=payload,
                eval_id="unit_trace_residual_coverage",
            )
            self.assertTrue(coverage["pass"])
            self.assertEqual(coverage["loss_row_count"], 1)
            self.assertEqual(coverage["non_attributed_loss_count"], 1)
            self.assertEqual(coverage["bypass_loss_count"], 1)
            self.assertEqual(coverage["bypass_loss_coverage_rate"], 1.0)
            self.assertEqual(coverage["bypass_loss_trainable_count"], 1)

    def test_trace_dataset_collection_weights_artifact_replay_rows(self):
        first_party = {
            "eval_id": "first_party",
            "source": {"judgments_path": "first.json"},
            "rows": [
                {
                    "problem_id": "p1",
                    "outcome": "win",
                    "residual_type": "no_residual",
                    "trainable": True,
                    "first_party_trace": True,
                    "trace_source": "first_party_runtime",
                    "event_counts": {"llm_call": 1},
                    "component_counts": {"solver": 1},
                },
            ],
        }
        artifact = {
            "eval_id": "artifact",
            "source": {"judgments_path": "artifact.json"},
            "rows": [
                {
                    "problem_id": "p1",
                    "outcome": "loss",
                    "residual_type": "optimization",
                    "trainable": True,
                    "first_party_trace": False,
                    "trace_source": "artifact_replay",
                    "trace_event_count": 1,
                    "event_counts": {"tool_use": 1},
                    "component_counts": {"artifact_replay_answer_meta": 1},
                },
                {
                    "problem_id": "p2",
                    "outcome": "tie",
                    "residual_type": "unknown",
                    "trainable": False,
                    "first_party_trace": False,
                    "trace_source": "artifact_replay",
                    "trace_event_count": 1,
                    "event_counts": {"tool_use": 1},
                    "component_counts": {"artifact_replay_answer_meta": 1},
                },
            ],
        }
        payload = build_trace_dataset_collection_payload(
            root=Path("."),
            trace_dataset_payloads=[first_party, artifact],
            eval_id="unit_trace_collection",
        )
        self.assertEqual(payload["dataset_count"], 2)
        self.assertEqual(payload["row_count"], 3)
        self.assertEqual(payload["distinct_problem_count"], 2)
        self.assertEqual(payload["trainable_row_count"], 2)
        self.assertEqual(payload["first_party_trainable_row_count"], 1)
        self.assertEqual(payload["artifact_replay_trainable_row_count"], 1)
        self.assertEqual(payload["weighted_trainable_row_count"], 1.5)

    def test_trace_dataset_collection_distills_first_party_transition_rows(self):
        first_party = {
            "eval_id": "first_party",
            "source": {"judgments_path": "first.json"},
            "rows": [
                {
                    "row_id": "r1",
                    "problem_id": "p1",
                    "domain": "science",
                    "difficulty": "hard",
                    "outcome": "win",
                    "residual_type": "no_residual",
                    "trainable": True,
                    "first_party_trace": True,
                    "trace_source": "first_party_runtime",
                    "trace_event_count": 2,
                    "event_counts": {"retrieval": 1, "llm_call": 1},
                    "component_counts": {"solver": 1},
                    "phase_event_counts": {"draft": 1, "audit": 1},
                    "trajectory_phases": ["draft", "audit", "final"],
                    "components": ["solver"],
                    "features": {"domain": "science", "difficulty": "hard"},
                },
            ],
        }
        payload = build_trace_dataset_collection_payload(
            root=Path("."),
            trace_dataset_payloads=[first_party],
            eval_id="unit_trace_collection_distilled",
            distill_first_party_transitions=True,
            target_distilled_rows=12,
        )
        self.assertEqual(payload["raw_first_party_trainable_row_count"], 1)
        self.assertEqual(payload["first_party_distilled_trainable_row_count"], 12)
        self.assertEqual(payload["first_party_trainable_row_count"], 13)
        self.assertEqual(payload["trainable_row_count"], 13)
        self.assertEqual(payload["distillation_source_first_party_row_count"], 1)
        self.assertEqual(payload["weighted_trainable_row_count"], 4.0)
        self.assertEqual(payload["trace_source_counts"]["first_party_distilled_transition"], 12)
        distilled = [row for row in payload["rows"] if row.get("distilled_transition")]
        self.assertEqual(len(distilled), 12)
        self.assertEqual(distilled[0]["source_row_id"], "r1")
        self.assertIn("distilled_transition_phase", distilled[0]["features"])

    def test_trace_outcome_model_downweights_distilled_transition_rows(self):
        rows = [
            {
                "row_id": "raw",
                "problem_id": "p1",
                "domain": "science",
                "outcome": "win",
                "residual_type": "no_residual",
                "trainable": True,
                "trace_source": "first_party_runtime",
            },
            {
                "row_id": "distilled",
                "problem_id": "p1",
                "domain": "science",
                "outcome": "loss",
                "residual_type": "optimization",
                "trainable": True,
                "trace_source": "first_party_distilled_transition",
                "distilled_transition": True,
                "features": {
                    "distilled_transition_phase": "draft",
                    "distilled_signal": "component_signature",
                },
            },
            {
                "row_id": "artifact",
                "problem_id": "p2",
                "domain": "science",
                "outcome": "loss",
                "residual_type": "optimization",
                "trainable": True,
                "trace_source": "artifact_replay",
            },
        ]
        payload = build_trace_outcome_model_payload(
            trace_dataset_payload={"eval_id": "unit_weighted_trace_dataset", "rows": rows},
            eval_id="unit_weighted_trace_outcome_model",
            min_policy_group_size=2,
        )
        self.assertEqual(payload["weighted_trainable_row_count"], 1.75)
        self.assertEqual(payload["trace_source_weighted_counts"]["first_party_runtime"], 1.0)
        self.assertEqual(payload["trace_source_weighted_counts"]["first_party_distilled_transition"], 0.25)
        self.assertEqual(payload["trace_source_weighted_counts"]["artifact_replay"], 0.5)
        feature_families = payload["feature_schema"]["feature_family_counts"]
        self.assertIn("distilled_transition_phase", feature_families)
        self.assertIn("distilled_signal", feature_families)

    def test_trace_outcome_model_calibrates_routes_and_policy_updates(self):
        rows = [
            {
                "row_id": "r1",
                "problem_id": "p1",
                "domain": "science",
                "bypass_route": "science_mechanism",
                "intervention_variant": "candidate_a",
                "baseline_variant": "baseline",
                "judgment_pair": "candidate_a_vs_baseline",
                "components": ["phase2_cache_hit"],
                "outcome": "win",
                "score_delta": 1.0,
                "residual_type": "no_residual",
                "trainable": True,
            },
            {
                "row_id": "r2",
                "problem_id": "p2",
                "domain": "science",
                "bypass_route": "science_mechanism",
                "intervention_variant": "candidate_a",
                "baseline_variant": "baseline",
                "judgment_pair": "candidate_a_vs_baseline",
                "components": ["phase2_cache_hit"],
                "outcome": "loss",
                "score_delta": -1.0,
                "residual_type": "optimization",
                "residual": "secret_token=trace-outcome-secret optimize the bypass bridge",
                "trainable": True,
            },
            {
                "row_id": "r3",
                "problem_id": "p3",
                "domain": "mathematics",
                "bypass_route": "math_research_bridge",
                "intervention_variant": "candidate_b",
                "baseline_variant": "baseline",
                "judgment_pair": "candidate_b_vs_baseline",
                "components": ["phase2_cache_hit"],
                "outcome": "win",
                "score_delta": 2.0,
                "residual_type": "no_residual",
                "trainable": True,
            },
        ]
        payload = build_trace_outcome_model_payload(
            trace_dataset_payload={"eval_id": "unit_trace_dataset", "rows": rows},
            eval_id="unit_trace_outcome_model",
            min_policy_group_size=2,
        )
        self.assertEqual(payload["trainable_row_count"], 3)
        self.assertEqual(payload["weighted_trainable_row_count"], 3.0)
        self.assertEqual(payload["trace_source_counts"], {"unspecified": 3})
        self.assertEqual(payload["route_group_count"], 2)
        self.assertEqual(payload["leave_one_out_metrics"]["prediction_count"], 3)
        self.assertEqual(payload["leave_one_out_metrics"]["weighted_prediction_count"], 3.0)
        self.assertEqual(payload["feature_leave_one_out_metrics"]["prediction_count"], 3)
        self.assertEqual(payload["trajectory_quality_metrics"]["prediction_count"], 3)
        self.assertGreaterEqual(payload["trajectory_phase_schema"]["phase_count"], 5)
        self.assertIn("draft", payload["trajectory_quality_metrics"]["phase_prediction_counts"])
        self.assertGreater(payload["feature_schema"]["feature_count"], 0)
        self.assertIn("intervention_variant", payload["feature_schema"]["feature_family_counts"])
        self.assertIn("baseline_variant", payload["feature_schema"]["feature_family_counts"])
        self.assertIn("judgment_pair", payload["feature_schema"]["feature_family_counts"])
        self.assertEqual(payload["policy_update_count"], 1)
        self.assertEqual(payload["policy_updates"][0]["decision"], "keep_with_targeted_repair")
        self.assertEqual(payload["route_stats"][0]["weighted_count"], 2.0)
        self.assertEqual(payload["residual_stats"][0]["residual_type"], "optimization")
        self.assertFalse(payload["secret_leak_detected"])
        self.assertNotIn("trace-outcome-secret", json.dumps(payload))
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="surface_retrieval",
                type=AssumptionType.RETRIEVAL,
                kind="retrieval_policy",
                claim="Domain retrieval policy surface",
                tags=["domain_retrieval_policy"],
                payload={"surface_key": "domain_retrieval_policy"},
            ))
            store.flush()
            proposals = build_trace_policy_proposal_payload(
                store=JsonlGraphStore(td),
                trace_outcome_payload=payload,
                eval_id="unit_trace_policy_proposals",
            )
            self.assertEqual(proposals["proposal_count"], 1)
            proposal = proposals["proposals"][0]
            self.assertEqual(proposal["proposal_type"], ProposalType.ASSUMPTION_REVISION.value)
            self.assertEqual(proposal["parent_node_id"], "surface_retrieval")
            self.assertEqual(proposal["candidate_node"]["type"], AssumptionType.RETRIEVAL.value)
            self.assertIn("heldout_route_ablation", proposal["candidate_node"]["verifiers"])
            self.assertEqual(proposal["candidate_node"]["payload"]["activation"]["problem_ids"], ["p1", "p2"])
            self.assertFalse(proposals["secret_leak_detected"])

    def test_surface_hypotheses_generate_world_model_and_evaluator_proposals(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="surface_world",
                type=AssumptionType.WORLD_MODEL,
                kind=HypothesisKind.CLAIM,
                claim="World model screen",
                payload={"surface_key": "world_model_screen"},
            ))
            store.upsert_node(AssumptionNode(
                id="surface_eval",
                type=AssumptionType.EVALUATOR,
                kind=HypothesisKind.CLAIM,
                claim="Evaluator policy",
                payload={"surface_key": "evaluator_policy"},
            ))
            sections = {
                "trace_dataset": {
                    "first_party_trainable_row_count": 1,
                    "artifact_replay_trainable_row_count": 4,
                },
                "trace_outcome_model": {
                    "leave_one_out_metrics": {"weighted_brier_score": 0.2},
                    "feature_leave_one_out_metrics": {"weighted_brier_score": 0.1},
                    "feature_schema": {"feature_count": 6},
                },
                "verifier_stack": {"stage_status_counts": {"V4:missing": 3, "V4:fail": 1}},
                "formal_metrics": {
                    "transfer_search_query_count": 2,
                    "transfer_search_negative_application_count": 2,
                },
            }
            payload = build_surface_hypothesis_payload(
                store=store,
                performance_sections=sections,
                eval_id="unit_surface_hypotheses",
            )
            self.assertEqual(payload["proposal_count"], 4)
            self.assertEqual(payload["world_model_proposal_count"], 2)
            self.assertEqual(payload["evaluator_proposal_count"], 2)
            self.assertEqual(payload["manifest_count"], 4)
            self.assertFalse(payload["secret_leak_detected"])

    def test_surface_hypotheses_generate_self_modification_proposal(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="surface_recursive",
                type=AssumptionType.SELF_MODIFICATION,
                kind=HypothesisKind.CLAIM,
                claim="Recursive assumption runner",
                payload={"surface_key": "recursive_assumption_runner"},
            ))
            sections = {
                "trace_policy_preflight": {"proposal_count": 5, "ready_count": 5},
                "recursive_audit": {"actionable_count": 4, "critical_issue_count": 0},
                "recursive_daemon": {"case_count": 2},
                "evolution_context": {
                    "blocked_policy_decision": "blocked_by_permissions",
                    "apply_policy_decision": "gated_apply_allowed",
                },
            }
            payload = build_surface_hypothesis_payload(
                store=store,
                performance_sections=sections,
                eval_id="unit_surface_self_mod",
            )
            self.assertEqual(payload["self_modification_proposal_count"], 1)
            self.assertEqual(payload["synthesis_family_count"], 1)
            proposal = payload["proposals"][0]
            self.assertEqual(proposal["candidate_node"]["type"], AssumptionType.SELF_MODIFICATION.value)
            self.assertEqual(proposal["candidate_node"]["kind"], HypothesisKind.HP_CHANGE.value)
            self.assertEqual(
                proposal["candidate_node"]["payload"]["validation_plan"]["ready_trace_policy_proposals"],
                5,
            )
            self.assertEqual(
                proposal["source_action"]["readiness"],
                "ready_for_recursive_daemon_probe",
            )

    def test_surface_hypotheses_generate_manifest_logger_proposal(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="surface_manifest",
                type=AssumptionType.HARNESS,
                kind=HypothesisKind.CLAIM,
                claim="Manifest logger",
                payload={"surface_key": "manifest_logger"},
            ))
            sections = {
                "manifest_logger": {
                    "event_count": 112,
                    "real_log_event_count": 12,
                    "synthetic_event_count": 100,
                    "secret_leak_detected": False,
                },
                "runtime_trace": {"event_count": 3, "secret_leak_detected": False},
                "trace_dataset": {
                    "first_party_trainable_row_count": 1009,
                    "raw_first_party_trainable_row_count": 9,
                    "first_party_distilled_trainable_row_count": 1000,
                },
            }
            payload = build_surface_hypothesis_payload(
                store=store,
                performance_sections=sections,
                eval_id="unit_surface_manifest",
            )
            self.assertEqual(payload["manifest_logger_proposal_count"], 1)
            self.assertEqual(payload["synthesis_family_count"], 1)
            proposal = payload["proposals"][0]
            self.assertEqual(proposal["candidate_node"]["type"], AssumptionType.HARNESS.value)
            self.assertEqual(proposal["candidate_node"]["kind"], HypothesisKind.HP_CHANGE.value)
            self.assertEqual(
                proposal["candidate_node"]["payload"]["validation_plan"]["first_party_distilled_trainable_rows"],
                1000,
            )
            self.assertEqual(proposal["source_action"]["readiness"], "ready_for_manifest_quota_probe")

    def test_surface_hypothesis_generator_bridges_residual_clusters_to_surface_proposals(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="surface_world",
                type=AssumptionType.WORLD_MODEL,
                kind=HypothesisKind.CLAIM,
                claim="World model screen",
                payload={"surface_key": "world_model_screen"},
            ))
            store.upsert_node(AssumptionNode(
                id="surface_eval",
                type=AssumptionType.EVALUATOR,
                kind=HypothesisKind.CLAIM,
                claim="Evaluator policy",
                payload={"surface_key": "evaluator_policy"},
            ))
            cluster = {
                "cluster_id": "rcluster_unit",
                "residual_type": "optimization",
                "signature": "unit_signature",
                "parent_node_id": "strategy_S01",
                "record_count": 3,
                "top_terms": ["judge", "concrete"],
                "sample_problem_ids": ["p1", "p2", "p3"],
                "candidate_control_problem_ids": ["c1", "c2"],
            }
            memory_cluster = {
                **cluster,
                "cluster_id": "rcluster_memory",
                "residual_type": "memory_defect",
                "sample_problem_ids": ["p4", "p5"],
                "candidate_control_problem_ids": ["c3", "c4"],
            }
            sections = {
                "trace_dataset": {
                    "first_party_trainable_row_count": 1,
                    "artifact_replay_trainable_row_count": 4,
                },
                "trace_outcome_model": {
                    "leave_one_out_metrics": {"weighted_brier_score": 0.2},
                    "feature_leave_one_out_metrics": {"weighted_brier_score": 0.1},
                    "feature_schema": {"feature_count": 6},
                },
                "verifier_stack": {"stage_status_counts": {"V4:missing": 3, "V4:fail": 1}},
                "formal_metrics": {
                    "transfer_search_query_count": 2,
                    "transfer_search_negative_application_count": 2,
                    "downstream_task_query_count": 9,
                    "downstream_transfer_pairwise_auc": 0.9,
                },
                "residual_clusterer": {"cluster_summaries": [cluster, memory_cluster]},
            }
            payload = build_surface_hypothesis_payload(
                store=store,
                performance_sections=sections,
                eval_id="unit_surface_residual_bridge",
            )
            self.assertGreaterEqual(payload["proposal_count"], 6)
            self.assertEqual(payload["surface_residual_proposal_count"], 2)
            self.assertEqual(payload["world_model_residual_proposal_count"], 1)
            self.assertEqual(payload["evaluator_residual_proposal_count"], 1)
            self.assertEqual(payload["surface_residual_ready_count"], 2)
            bridge = [
                p for p in payload["proposals"]
                if p["source_action"].get("action_type") == "surface_residual_bridge"
            ]
            self.assertTrue(all(p["source_action"].get("command_hint") for p in bridge))
            self.assertTrue(all("python3 -m assumption_os.candidate_eval" in p["source_action"]["command_hint"] for p in bridge))
            self.assertTrue(all("--proposal-ids" in p["source_action"]["command_hint"] for p in bridge))
            self.assertFalse(any("--proposal-id " in p["source_action"]["command_hint"] for p in bridge))
            self.assertTrue(all(p["candidate_node"]["payload"]["validation_plan"]["trigger_problem_ids"] for p in bridge))
            self.assertTrue(all(p["candidate_node"]["payload"]["activation"]["problem_ids"] for p in bridge))

    def test_harness_observer_backfills_artifact_manifest_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            judgments = root / "phase two/analysis/cache/judgments"
            answers = root / "phase two/analysis/cache/answers"
            logs = root / "phase four/assumption_graph"
            judgments.mkdir(parents=True)
            answers.mkdir(parents=True)
            logs.mkdir(parents=True)
            judgment_path = judgments / "unit_judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {"winner": "candidate", "score_a": 8, "score_b": 7, "reasoning": "candidate wins"},
                "p2": {"winner": "baseline", "score_a": 6, "score_b": 8, "reasoning": "baseline wins"},
            }), encoding="utf-8")
            meta_path = answers / "unit_meta.json"
            meta_path.write_text(json.dumps({
                "p1": {"frame": "hybrid", "bypass_route": "unit_route"},
                "p2": {"frame": "object", "bypass_route": "unit_route"},
            }), encoding="utf-8")
            log_path = logs / "unit.log"
            log_path.write_text(
                "\n".join([
                    "=== JUDGE prop_a proposal_a vs baseline rows=2 ===",
                    "LLM provider: gemini, model: gpt-5.5",
                    "=== DONE JUDGE prop_a returncode=0 elapsed=1.2s ===",
                ]),
                encoding="utf-8",
            )

            events = events_from_harness_artifacts(
                root=root,
                artifact_paths=[judgment_path, meta_path, log_path],
                max_events_per_file=5,
            )
            self.assertEqual(len(events), 4)
            payload = build_harness_observer_payload(
                root=root,
                graph_dir=graph_dir,
                eval_id="unit_harness_observer",
                artifact_paths=[judgment_path, meta_path, log_path],
                max_events_per_file=5,
                writeback=True,
            )
            self.assertTrue(payload["artifact_coverage"]["full_coverage_after_writeback"])
            self.assertEqual(payload["event_counts"]["judge_call"], 3)
            self.assertEqual(payload["event_counts"]["llm_call"], 1)
            self.assertEqual(len(JsonlGraphStore(graph_dir).trials), 4)

    def test_world_model_scores_candidates_and_logs_simulator_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
                confidence=0.8,
                metaproductivity=0.2,
            ))
            store.flush()
            proposal_payload = {
                "eval_id": "unit_props",
                "proposals": [
                    {
                        "proposal_id": "prop_accept",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.9,
                        "candidate_node": {"id": "cand_accept", "claim": "Add a concrete baseline gate."},
                    },
                    {
                        "proposal_id": "prop_reject",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.2,
                        "candidate_node": {"id": "cand_reject", "claim": "Use a broad generic warning."},
                    },
                ],
            }
            preflight = {
                "summaries": [
                    {"proposal_id": "prop_accept", "readiness": "ready_for_fresh_ablation"},
                    {"proposal_id": "prop_reject", "readiness": "needs_scope_fix"},
                ],
            }
            falsification = {
                "summaries": [
                    {"proposal_id": "prop_accept", "decision": "ready_for_ablation"},
                    {"proposal_id": "prop_reject", "decision": "reject_benefit"},
                ],
            }
            acceptance = {
                "summaries": [
                    {"proposal_id": "prop_accept", "decision": "accept"},
                    {"proposal_id": "prop_reject", "decision": "reject_benefit"},
                ],
            }
            payload = build_world_model_payload(
                store=JsonlGraphStore(td),
                proposal_payload=proposal_payload,
                preflight_payload=preflight,
                falsification_payload=falsification,
                acceptance_payload=acceptance,
                regression_predictions=[
                    {"proposal_id": "prop_accept", "risk": "low"},
                    {"proposal_id": "prop_reject", "risk": "high"},
                ],
                formal_mapping_gate_payload={"gates": []},
                eval_id="unit_world_model",
                writeback=True,
            )
            by_id = {p["proposal_id"]: p for p in payload["predictions"]}
            self.assertGreater(
                by_id["prop_accept"]["predicted_acceptance_probability"],
                by_id["prop_reject"]["predicted_acceptance_probability"],
            )
            self.assertEqual(payload["calibration"]["labeled_predictions"], 2)
            self.assertEqual(len(JsonlGraphStore(td).trials), 2)

    def test_world_model_trains_priority_calibration(self):
        prediction_payload = {
            "eval_id": "unit_pre",
            "predictions": [
                {
                    "proposal_id": "accept_high",
                    "predicted_acceptance_probability": 0.8,
                    "feature_trace": {"priority": 2.0, "readiness": "ready_for_fresh_ablation", "regression_risk": "low"},
                },
                {
                    "proposal_id": "reject_low",
                    "predicted_acceptance_probability": 0.78,
                    "feature_trace": {"priority": 0.5, "readiness": "ready_for_fresh_ablation", "regression_risk": "low"},
                },
            ],
        }
        acceptance_payload = {
            "eval_id": "unit_accept",
            "summaries": [
                {"proposal_id": "accept_high", "decision": "accept"},
                {"proposal_id": "reject_low", "decision": "reject_benefit"},
            ],
        }
        calibration = train_world_model_calibration(
            prediction_payload=prediction_payload,
            acceptance_payload=acceptance_payload,
            eval_id="unit_calibration",
        )
        self.assertEqual(calibration["status"], "trained")
        self.assertEqual(calibration["matched_label_count"], 2)
        self.assertEqual(calibration["unmatched_label_count"], 0)
        self.assertGreater(calibration["priority_boundary"], 0.5)
        self.assertLess(
            calibration["calibrated_metrics"]["brier_score"],
            calibration["raw_metrics"]["brier_score"],
        )

    def test_trajectory_search_returns_multiple_ranked_paths(self):
        recursive_payload = {
            "eval_id": "unit_recursive",
            "next_actions": [{
                "frame_id": "frame_1",
                "problem_id": "verify::prop_1",
                "proposal_id": "prop_1",
                "next_action": "run_fresh_ablation",
                "priority": 0.8,
            }],
        }
        world_model_payload = {
            "eval_id": "unit_world",
            "predictions": [{
                "proposal_id": "prop_1",
                "predicted_acceptance_probability": 0.62,
                "expected_utility": 0.25,
                "predicted_regression_risk": "medium",
                "recommended_next_action": "repair_scope_before_ablation",
                "predicted_failure_modes": ["medium_regression_risk"],
            }],
        }
        payload = build_trajectory_search_payload(
            recursive_payload=recursive_payload,
            world_model_payload=world_model_payload,
            eval_id="unit_trajectory",
            beam_width=3,
        )
        self.assertGreaterEqual(payload["trajectory_count"], 2)
        path_types = {row["path_type"] for row in payload["trajectories"]}
        self.assertIn("repair_then_retest", path_types)
        self.assertEqual(payload["selected"][0]["proposal_id"], "prop_1")

    def test_verifier_stack_combines_ordered_gate_verdicts(self):
        proposal_payload = {
            "eval_id": "unit_props",
            "proposals": [
                {
                    "proposal_id": "prop_accept",
                    "proposal_type": "assumption_revision",
                    "parent_node_id": "strategy_S01",
                    "candidate_node": {"id": "cand_accept"},
                },
                {
                    "proposal_id": "prop_repair",
                    "proposal_type": "assumption_revision",
                    "parent_node_id": "strategy_S02",
                    "candidate_node": {"id": "cand_repair"},
                },
            ],
        }
        preflight = {
            "eval_id": "unit_preflight",
            "summaries": [
                {"proposal_id": "prop_accept", "readiness": "ready_for_fresh_ablation", "trigger_problem_ids": ["p1", "p2"]},
                {"proposal_id": "prop_repair", "readiness": "needs_scope_fix", "outside_active_problem_ids": ["p3"]},
            ],
        }
        world_model = {
            "eval_id": "unit_world",
            "predictions": [
                {
                    "proposal_id": "prop_accept",
                    "predicted_acceptance_probability": 0.8,
                    "predicted_regression_risk": "low",
                    "recommended_next_action": "run_fresh_ablation",
                },
                {
                    "proposal_id": "prop_repair",
                    "predicted_acceptance_probability": 0.4,
                    "predicted_regression_risk": "high",
                    "recommended_next_action": "repair_scope_before_ablation",
                },
            ],
        }
        acceptance = {
            "eval_id": "unit_acceptance",
            "summaries": [
                {
                    "proposal_id": "prop_accept",
                    "decision": "accept",
                    "trigger_outcomes": {"win": 4},
                    "control_outcomes": {},
                    "trigger_lcb90": 0.7,
                    "control_loss_ucb90": None,
                },
            ],
        }
        falsification = build_falsification_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight,
            acceptance_payload=acceptance,
        )
        payload = build_verifier_stack_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight,
            world_model_payload=world_model,
            falsification_payload=falsification,
            acceptance_payload=acceptance,
            formal_mapping_gate_payload={"gates": []},
            eval_id="unit_verifier",
        )
        by_id = {row["proposal_id"]: row for row in payload["summaries"]}
        self.assertEqual(by_id["prop_accept"]["verdict"], "accepted_for_gated_apply")
        self.assertEqual(by_id["prop_accept"]["next_action"], "apply_accepted_candidate_if_requested")
        self.assertEqual(by_id["prop_repair"]["verdict"], "needs_preflight_repair")
        self.assertEqual(by_id["prop_repair"]["stages"][0]["status"], "repair")
        v3 = next(stage for stage in by_id["prop_accept"]["stages"] if stage["tier"] == "V3")
        self.assertEqual(v3["evidence"]["experiment_name_counts"]["trigger_benefit_sequential"], 1)
        self.assertEqual(v3["evidence"]["experiment_status_counts"]["passed"], 4)
        v5 = next(stage for stage in by_id["prop_accept"]["stages"] if stage["tier"] == "V5")
        self.assertEqual(v5["status"], "pass")
        self.assertTrue(v5["evidence"]["objective_gate_passed"])
        v6 = next(stage for stage in by_id["prop_accept"]["stages"] if stage["tier"] == "V6")
        self.assertEqual(v6["status"], "required")
        self.assertEqual(v6["evidence"]["permission_boundary"], "explicit_apply_or_writeback_required")

    def test_v5_external_objective_benchmark_blocks_failed_acceptance(self):
        proposal_payload = {
            "eval_id": "unit_props",
            "proposals": [{
                "proposal_id": "prop_accept",
                "proposal_type": "assumption_revision",
                "parent_node_id": "strategy_S01",
                "candidate_node": {"id": "cand_accept"},
            }],
        }
        preflight = {
            "eval_id": "unit_preflight",
            "summaries": [{
                "proposal_id": "prop_accept",
                "readiness": "ready_for_fresh_ablation",
                "trigger_problem_ids": ["p1", "p2", "p3"],
                "control_problem_ids": ["c1", "c2"],
            }],
        }
        acceptance = {
            "eval_id": "unit_acceptance",
            "summaries": [{
                "proposal_id": "prop_accept",
                "decision": "accept",
                "trigger_outcomes": {"win": 4},
                "control_outcomes": {"tie": 2},
                "trigger_lcb90": 0.7,
                "control_loss_ucb90": 0.0,
            }],
        }
        falsification = build_falsification_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight,
            acceptance_payload=acceptance,
        )
        passing_objective = build_objective_benchmark_payload(
            proposal_payload=proposal_payload,
            acceptance_payload=acceptance,
            eval_id="unit_objective_pass",
            task_results=[
                {
                    "proposal_id": "prop_accept",
                    "task_id": "external_transfer",
                    "task_family": "transfer",
                    "label_source": "external_objective_task",
                    "candidate_score": 0.9,
                    "baseline_score": 0.4,
                },
                {
                    "proposal_id": "prop_accept",
                    "task_id": "external_regression",
                    "task_family": "regression",
                    "label_source": "external_objective_task",
                    "candidate_score": 0.8,
                    "baseline_score": 0.5,
                },
            ],
        )
        self.assertTrue(passing_objective["pass"])
        passed = build_verifier_stack_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight,
            falsification_payload=falsification,
            acceptance_payload=acceptance,
            objective_benchmark_payload=passing_objective,
            eval_id="unit_verifier_external_pass",
        )
        pass_row = passed["summaries"][0]
        pass_v5 = next(stage for stage in pass_row["stages"] if stage["tier"] == "V5")
        self.assertEqual(pass_row["verdict"], "accepted_for_gated_apply")
        self.assertEqual(pass_v5["evidence"]["objective_gate_source"], "external_objective_task_benchmark")
        self.assertTrue(pass_v5["evidence"]["external_objective_passed"])

        failing_objective = build_objective_benchmark_payload(
            proposal_payload=proposal_payload,
            acceptance_payload=acceptance,
            eval_id="unit_objective_fail",
            task_results=[
                {
                    "proposal_id": "prop_accept",
                    "task_id": "external_transfer",
                    "task_family": "transfer",
                    "label_source": "external_objective_task",
                    "candidate_score": 0.2,
                    "baseline_score": 0.7,
                },
                {
                    "proposal_id": "prop_accept",
                    "task_id": "external_regression",
                    "task_family": "regression",
                    "label_source": "external_objective_task",
                    "candidate_score": 0.3,
                    "baseline_score": 0.6,
                },
            ],
        )
        self.assertFalse(failing_objective["summaries"][0]["objective_gate_passed"])
        blocked = build_verifier_stack_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight,
            falsification_payload=falsification,
            acceptance_payload=acceptance,
            objective_benchmark_payload=failing_objective,
            eval_id="unit_verifier_external_fail",
        )
        blocked_row = blocked["summaries"][0]
        blocked_v5 = next(stage for stage in blocked_row["stages"] if stage["tier"] == "V5")
        self.assertEqual(blocked_row["verdict"], "blocked_objective_gate")
        self.assertEqual(blocked_v5["status"], "block")
        self.assertFalse(blocked_v5["evidence"]["external_objective_passed"])

    def test_evolution_context_gates_permissions_and_harness_responsibilities(self):
        sections = {
            "trajectory_search": {"pass": True, "multi_path_rate": 0.8},
            "verifier_stack": {
                "pass": True,
                "proposal_count": 33,
                "accepted_count": 2,
                "accepted_protocol_ok": True,
                "rejected_protocol_ok": True,
                "falsification_experiment_count": 135,
            },
            "world_model": {"pass": True},
            "formal_metrics": {"pass": True},
            "manifest_logger": {"pass": True, "event_count": 12, "secret_leak_detected": False},
            "harness_observer": {"pass": True, "full_coverage_after_writeback": True},
            "residual_clusterer": {"pass": True, "cluster_count": 2, "proposal_count": 1},
            "recursive_audit": {
                "pass": True,
                "actionable_count": 5,
                "min_closure_score": 1.0,
                "critical_issue_count": 0,
                "warning_issue_count": 0,
            },
            "recursive_daemon": {"pass": True, "case_count": 2, "accepted_apply_count": 2},
        }
        dry = build_evolution_context_payload(
            eval_id="unit_evolution_context_dry",
            objective="Evolve graph policy only when harness responsibilities are satisfied.",
            sections=sections,
        )
        self.assertEqual(dry["policy_decision"], EvolutionPolicyDecision.READY_FOR_MANUAL_APPLY.value)
        self.assertEqual(dry["responsibility_status_counts"], {"pass": 9})
        self.assertEqual(dry["permission_violations"], [])

        allowed = build_evolution_context_payload(
            eval_id="unit_evolution_context_apply",
            objective="Apply accepted candidates under an explicit permission boundary.",
            sections=sections,
            mode={"apply_accepted": True},
            permissions={"allow_apply_accepted": True, "max_apply_candidates": 2},
        )
        self.assertEqual(allowed["policy_decision"], EvolutionPolicyDecision.GATED_APPLY_ALLOWED.value)

        blocked = build_evolution_context_payload(
            eval_id="unit_evolution_context_blocked",
            objective="Apply accepted candidates without permission.",
            sections=sections,
            mode={"apply_accepted": True},
        )
        self.assertEqual(blocked["policy_decision"], EvolutionPolicyDecision.BLOCKED_BY_PERMISSIONS.value)
        self.assertEqual(blocked["permission_violations"][0]["kind"], "apply_accepted_not_allowed")

    def test_assumption_bench_scores_lifecycle_capabilities(self):
        with tempfile.TemporaryDirectory() as td:
            graph_dir = Path(td) / "graph"
            store = JsonlGraphStore(graph_dir)
            types = [
                AssumptionType.METHOD,
                AssumptionType.MEMORY,
                AssumptionType.VERIFIER,
                AssumptionType.WORLD_MODEL,
                AssumptionType.HARNESS,
                AssumptionType.RETRIEVAL,
            ]
            for idx in range(24):
                store.upsert_node(AssumptionNode(
                    id=f"node_{idx}",
                    type=types[idx % len(types)],
                    claim=f"Capability node {idx}",
                    metaproductivity=0.2,
                ))
            edge_types = [
                EdgeType.SUPPORTS,
                EdgeType.DEPENDS_ON,
                EdgeType.HAS_VERIFIER,
                EdgeType.GENERATED_FROM_RESIDUAL,
                EdgeType.HAS_CASE,
                EdgeType.HAS_RESIDUAL,
            ]
            for idx, edge_type in enumerate(edge_types):
                store.add_edge(AssumptionEdge(source=f"node_{idx}", target=f"node_{idx + 1}", type=edge_type))
            store.flush()
            sections = {
                "manifest_logger": {"pass": True, "event_count": 120, "real_log_event_count": 12, "secret_leak_detected": False},
                "trajectory_search": {"pass": True, "multi_path_rate": 0.8, "top_path_label_hit_rate": 1.0, "selected_path_types": {"a": 1, "b": 1, "c": 1}},
                "verifier_stack": {
                    "pass": True,
                    "proposal_count": 33,
                    "accepted_count": 2,
                    "accepted_protocol_ok": True,
                    "rejected_protocol_ok": True,
                    "falsification_experiment_count": 135,
                    "stage_status_counts": {"V4:pass": 2, "V4:fail": 14},
                },
                "recursive_audit": {"pass": True, "min_closure_score": 1.0, "critical_issue_count": 0},
                "recursive_daemon": {"pass": True, "accepted_apply_count": 2, "case_count": 2},
                "residual_clusterer": {
                    "pass": True,
                    "cluster_count": 5,
                    "proposal_count": 2,
                    "validation_plans_complete": True,
                    "label_agreement_pass": True,
                    "label_agreement_accuracy": 1.0,
                    "label_agreement_macro_f1": 1.0,
                },
                "trace_dataset": {
                    "pass": True,
                    "non_attributed_loss_coverage_rate": 1.0,
                    "bypass_loss_coverage_rate": 1.0,
                    "residual_trace_coverage_pass": True,
                },
                "harness_observer": {"pass": True, "full_coverage_after_writeback": True, "artifact_file_count": 4},
                "world_model": {
                    "pass": True,
                    "matched_label_count": 16,
                    "pre_acceptance": {"auc": 1.0},
                    "post_calibration": {"brier_score": 0.01},
                },
                "evolution_context": {
                    "pass": True,
                    "responsibility_count": 9,
                    "responsibility_status_counts": {"pass": 9},
                    "blocked_policy_decision": "blocked_by_permissions",
                    "apply_policy_decision": "gated_apply_allowed",
                },
            }
            payload = build_assumption_bench_payload(
                eval_id="unit_assumption_bench",
                sections=sections,
                graph_dir=graph_dir,
            )
            self.assertTrue(payload["pass"])
            self.assertEqual(payload["capability_count"], 9)
            self.assertEqual(payload["failed_capabilities"], [])
            self.assertGreaterEqual(payload["overall_score"], 0.9)

    def test_reconstruction_progress_audits_structure_and_behavior(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            for idx, node_type in enumerate([
                AssumptionType.METHOD,
                AssumptionType.RETRIEVAL,
                AssumptionType.VERIFIER,
                AssumptionType.WORLD_MODEL,
                AssumptionType.HARNESS,
                AssumptionType.RESIDUAL,
                AssumptionType.CASE,
                AssumptionType.EVALUATOR,
                AssumptionType.MEMORY,
                AssumptionType.ALIGNMENT,
                AssumptionType.SELF_MODIFICATION,
            ]):
                store.upsert_node(AssumptionNode(
                    id=f"n{idx}",
                    type=node_type,
                    claim=f"Node {idx}",
                    metaproductivity=0.1,
                ))
            for idx, edge_type in enumerate([
                EdgeType.SUPPORTS,
                EdgeType.DEPENDS_ON,
                EdgeType.HAS_VERIFIER,
                EdgeType.GENERATED_FROM_RESIDUAL,
                EdgeType.HAS_CASE,
                EdgeType.HAS_RESIDUAL,
                EdgeType.SPECIALIZES,
                EdgeType.DERIVED_FROM,
                EdgeType.USES_EVALUATOR,
                EdgeType.IS_FORMAL_ISOMORPHISM_OF,
                EdgeType.GENERALIZES,
            ]):
                store.add_edge(AssumptionEdge(source=f"n{idx}", target=f"n{(idx + 1) % 11}", type=edge_type))
            for idx in range(12):
                store.append_trial(TrialManifest(
                    problem_id=f"p{idx}",
                    action_type="unit",
                    assumption="unit",
                    why_selected="unit",
                    expected_effect="unit",
                    status=TrialStatus.OBSERVED,
                ))
            store.flush()
            sections = {
                "memory_surfaces": {"pass": True, "surface_count": 10},
                "harness_observer": {"pass": True, "full_coverage_after_writeback": True},
                "residual_clusterer": {
                    "pass": True,
                    "cluster_count": 7,
                    "proposal_count": 2,
                    "record_count": 109,
                    "residual_type_counts": {"optimization": 4, "memory_defect": 2, "unknown": 1},
                    "validation_plans_complete": True,
                    "label_agreement_pass": True,
                    "label_agreement_accuracy": 1.0,
                    "label_agreement_macro_f1": 1.0,
                    "label_agreement_example_count": 10,
                },
                "trace_policy_proposals": {"pass": True, "proposal_count": 3, "repair_policy_count": 1},
                "trace_policy_preflight": {"pass": True, "proposal_count": 3, "ready_count": 3},
                "surface_hypothesis_generator": {
                    "pass": True,
                    "proposal_count": 6,
                    "world_model_proposal_count": 3,
                    "evaluator_proposal_count": 3,
                    "surface_residual_proposal_count": 2,
                    "surface_residual_ready_count": 2,
                    "world_model_residual_proposal_count": 1,
                    "evaluator_residual_proposal_count": 1,
                    "self_modification_proposal_count": 1,
                    "self_modification_ready_count": 1,
                    "manifest_logger_proposal_count": 1,
                    "manifest_logger_ready_count": 1,
                    "synthesis_family_count": 4,
                },
                "world_model": {"pass": True, "matched_label_count": 16, "post_calibration": {"brier_score": 0.0081}},
                "trace_dataset": {
                    "pass": True,
                    "non_attributed_loss_count": 7,
                    "non_attributed_loss_coverage_rate": 1.0,
                    "bypass_loss_count": 5,
                    "bypass_loss_coverage_rate": 1.0,
                    "skipped_loss_count": 2,
                    "skipped_loss_coverage_rate": 1.0,
                    "residual_trace_coverage_pass": True,
                },
                "trace_outcome_model": {"pass": True, "trainable_row_count": 9, "policy_update_count": 3, "residual_group_count": 1, "leave_one_out_metrics": {"brier_score": 0.1605}},
                "verifier_stack": {
                    "pass": True,
                    "proposal_count": 33,
                    "accepted_count": 2,
                    "rejected_count": 14,
                    "accepted_protocol_ok": True,
                    "rejected_protocol_ok": True,
                    "objective_gate_ok": True,
                    "external_objective_gate_ok": True,
                    "objective_benchmark_pass": True,
                    "objective_benchmark_external_task_count": 36,
                    "objective_benchmark_accepted_external_pass_count": 2,
                    "manual_gate_ok": True,
                    "stage_status_counts": {"V5:pass": 16, "V6:required": 2},
                    "falsification_protocol_candidate_count": 27,
                    "falsification_experiment_count": 135,
                },
                "trajectory_search": {"pass": True, "multi_path_rate": 0.8, "top_path_label_hit_rate": 1.0, "trajectory_count": 26, "frontier_actions": 10, "selected_path_types": {"a": 1, "b": 1, "c": 1, "d": 1}},
                "assumption_bench": {"pass": True, "overall_score": 0.9968, "min_score": 0.9716, "capability_count": 9, "passed_capability_count": 9, "failed_capabilities": [], "score_by_capability": {"metaproductivity": 1.0}},
                "formal_metrics": {
                    "pass": True,
                    "mapping_count": 9,
                    "complete_count": 9,
                    "same_shape_count": 9,
                    "warning_count": 0,
                    "dedup_pass": True,
                    "dedup_complete_mapping_count": 9,
                    "transfer_search_query_count": 9,
                    "transfer_pairwise_auc": 1.0,
                    "transfer_top1_hit_rate": 1.0,
                    "independent_transfer_search_query_count": 9,
                    "independent_transfer_pairwise_auc": 1.0,
                    "independent_transfer_top1_hit_rate": 1.0,
                    "answer_quality_probe_pass": True,
                    "answer_quality_probe_count": 9,
                    "answer_quality_top1_hit_rate": 1.0,
                    "answer_quality_guided_win_rate": 1.0,
                    "answer_quality_mean_delta": 0.65,
                },
                "recursive_audit": {"pass": True, "min_closure_score": 1.0, "actionable_count": 5, "critical_issue_count": 0},
                "recursive_daemon": {
                    "pass": True,
                    "case_count": 2,
                    "accepted_apply_count": 2,
                    "preflight_queue_planned_leaf_count": 3,
                    "preflight_queue_executable_leaf_count": 3,
                    "preflight_queue_manifest_count": 4,
                    "preflight_queue_consumed": True,
                    "bounded_execute_succeeded_leaf_count": 1,
                    "bounded_execute_accept_count": 1,
                    "bounded_execute_resumed": True,
                    "bounded_execute_applied_count": 0,
                },
            }
            (root / "reconstruction.md").write_text(
                "\n".join([
                    "Assumption Graph Memory",
                    "Hypothesis Generator",
                    "World Model / Simulator",
                    "Verifier Stack POPPER falsification",
                    "Residual Analyzer residual taxonomy",
                    "Metaproductivity HGM clade",
                    "Formal Alignment Layer 范畴论 信息几何",
                    "递归执行循环 recursive 多条候选假设轨迹",
                    "评价体系 AssumptionBench answer win-rate",
                ]),
                encoding="utf-8",
            )
            payload = build_reconstruction_progress_payload(
                root=root,
                performance_payload={"eval_id": "unit_perf", "sections": sections},
                graph_dir=graph_dir,
                reconstruction_path=root / "reconstruction.md",
                eval_id="unit_reconstruction_progress",
            )
            self.assertTrue(payload["overall_pass"])
            self.assertEqual(payload["closure"]["item_count"], 9)
            self.assertGreaterEqual(payload["closure"]["structure_percent"], 75.0)
            self.assertGreaterEqual(payload["closure"]["behavior_percent"], 65.0)
            self.assertTrue(payload["remaining_gaps_ranked"])
            self.assertTrue(payload["next_actions_ranked"])
            self.assertEqual(payload["reconstruction_reference"]["matched_target_count"], 9)

    def test_reconstruction_progress_raises_world_model_ceiling_with_trace_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(id="wm", type=AssumptionType.WORLD_MODEL, claim="world model"))
            store.flush()
            sections = {
                "world_model": {"pass": True, "matched_label_count": 50, "post_calibration": {"brier_score": 0.01}},
                "trace_dataset": {"pass": True, "weighted_trainable_row_count": 75.0},
                "trace_outcome_model": {
                    "pass": True,
                    "weighted_trainable_row_count": 75.0,
                    "feature_leave_one_out_metrics": {"weighted_brier_score": 0.071},
                    "trajectory_quality_metrics": {
                        "weighted_brier_score": 0.16,
                        "complete_draft_audit_final_count": 75,
                    },
                    "trajectory_phase_schema": {"phase_count": 5},
                    "feature_schema": {"feature_count": 47},
                },
            }
            payload = build_reconstruction_progress_payload(
                root=root,
                performance_payload={"eval_id": "unit_perf", "sections": sections},
                graph_dir=graph_dir,
                eval_id="unit_world_model_ceiling",
            )
            world = next(row for row in payload["items"] if row["key"] == "C_world_model_simulator")
            self.assertEqual(world["evidence"]["reconstruction_ceiling"]["behavior"], 0.73)
            self.assertEqual(world["evidence"]["reconstruction_ceiling"]["structure"], 0.84)
            self.assertGreaterEqual(world["behavior_score"], 0.73)

    def test_memory_surfaces_write_runtime_mechanisms_to_graph(self):
        with tempfile.TemporaryDirectory() as td:
            graph_dir = Path(td) / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_seed",
                type=AssumptionType.METHOD,
                claim="Seed method node.",
            ))
            store.flush()
            payload = build_memory_surface_payload(
                graph_dir=graph_dir,
                eval_id="unit_memory_surfaces",
                performance_payload={
                    "eval_id": "unit_perf",
                    "sections": {
                        "world_model": {"pass": True, "post_calibration": {"brier_score": 0.01}},
                        "verifier_stack": {"pass": True, "accepted_count": 1, "falsification_experiment_count": 5},
                        "evolution_context": {"pass": True, "responsibility_status_counts": {"pass": 9}},
                        "assumption_bench": {"pass": True, "overall_score": 0.95},
                    },
                },
                writeback=True,
            )
            self.assertTrue(payload["memory_transfer_ready"])
            self.assertGreaterEqual(payload["after_graph"]["node_type_count"], 8)
            self.assertGreaterEqual(payload["after_graph"]["edge_type_count"], 8)
            updated = JsonlGraphStore(graph_dir)
            self.assertIn("world_model", payload["after_graph"]["node_type_counts"])
            self.assertTrue(any(node.type == AssumptionType.VERIFIER for node in updated.nodes.values()))
            second = build_memory_surface_payload(
                graph_dir=graph_dir,
                eval_id="unit_memory_surfaces",
                writeback=True,
            )
            self.assertEqual(second["new_node_count"], 0)
            self.assertEqual(second["new_edge_count"], 0)

    def test_recursive_daemon_resumes_and_applies_accepted_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
                tags=["S01", "controlled"],
                confidence=0.7,
            ))
            store.flush()
            evolution_payload = {
                "eval_id": "unit_cycle",
                "proposals": {
                    "eval_id": "unit_props",
                    "proposals": [{
                        "proposal_id": "prop_ready",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.8,
                        "candidate_node": {
                            "id": "cand_ready",
                            "type": AssumptionType.METHOD.value,
                            "kind": "claim",
                            "claim": "Require a baseline and one intervention before answering.",
                            "context_conditions": [],
                            "predicted_effects": ["improve causal diagnosis"],
                            "risk_predictions": [],
                            "verifiers": [],
                            "evidence_ids": [],
                            "residual_ids": [],
                            "confidence": 0.5,
                            "metaproductivity": 0.0,
                            "status": "candidate",
                            "tags": ["candidate"],
                            "source_refs": [],
                            "payload": {},
                        },
                    }],
                },
                "candidate_preflight": {
                    "eval_id": "unit_preflight",
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "active_trigger_problem_ids": ["p1", "p2", "p3"],
                        "trigger_problem_ids": ["p1", "p2", "p3"],
                        "control_problem_ids": [],
                        "command_hint": "python3 run_candidate.py --variant proposal_ready",
                    }],
                },
                "falsification_gate": {
                    "summaries": [{
                        "proposal_id": "prop_ready",
                        "decision": FalsificationDecision.READY_FOR_ABLATION.value,
                        "next_action": "run_fresh_ablation",
                    }],
                },
                "bayesian_policy": {
                    "scores": [{
                        "proposal_id": "prop_ready",
                        "recommended_action": BayesianPolicyAction.RUN_ABLATION.value,
                        "posterior_priority": 1.2,
                        "expected_value": 0.7,
                        "command_hint": "python3 run_candidate.py --variant proposal_ready",
                    }],
                },
                "policy_update_plan": {
                    "actions": [{
                        "proposal_id": "prop_ready",
                        "policy_action": "run_fresh_ablation_before_promotion",
                    }],
                },
                "regression_predictions": [{"proposal_id": "prop_ready", "risk": "low"}],
                "formal_mapping_gate": {"gates": []},
            }
            recursive_payload = build_recursive_assumption_run(
                graph_dir=graph_dir,
                problem="Diagnose a channel experiment failure with one controlled intervention.",
                goal="Create a recursive assumption tree.",
                eval_id="unit_recursive_daemon",
                evolution_payload=evolution_payload,
                max_children=1,
            )
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {"winner": "proposal_ready"},
                "p2": {"winner": "proposal_ready"},
                "p3": {"winner": "proposal_ready"},
            }), encoding="utf-8")
            payload = build_recursive_daemon_payload(
                root=root,
                graph_dir=graph_dir,
                recursive_payload=recursive_payload,
                evolution_payload=evolution_payload,
                eval_id="unit_daemon",
                judgment_sets=[JudgmentSet(
                    candidate_variant="proposal_ready",
                    baseline_variant="base",
                    judgment_paths=[judgment_path],
                    proposal_ids=["prop_ready"],
                )],
                apply_accepted=True,
                writeback_manifests=True,
            )
            self.assertEqual(payload["iteration_count"], 1)
            self.assertEqual(payload["iterations"][0]["candidate_acceptance_counts"], {"accept": 1})
            self.assertIn("cand_ready", JsonlGraphStore(graph_dir).nodes)
            self.assertTrue(payload["applied_candidate_node_ids"])
            self.assertGreaterEqual(len(JsonlGraphStore(graph_dir).trials), 2)

    def test_recursive_daemon_consumes_preflight_queue_as_leaf_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="surface_recursive",
                type=AssumptionType.SELF_MODIFICATION,
                claim="Recursive daemon",
            ))
            store.flush()
            before_nodes = set(JsonlGraphStore(graph_dir).nodes)
            preflight_payload = {
                "eval_id": "unit_preflight_queue",
                "summaries": [
                    {
                        "proposal_id": "prop_a",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "trigger_problem_ids": ["p1", "p2"],
                        "control_problem_ids": ["c1"],
                        "command_hint": "python3 run_candidate.py --variant proposal_a",
                    },
                    {
                        "proposal_id": "prop_b",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "trigger_problem_ids": ["p3"],
                        "control_problem_ids": [],
                        "command_hint": "python3 run_candidate.py --variant proposal_b",
                    },
                    {
                        "proposal_id": "prop_not_ready",
                        "readiness": CandidateReadiness.MANIFEST_ONLY.value,
                        "command_hint": "python3 run_candidate.py --variant skipped",
                    },
                ],
            }
            payload = build_preflight_queue_daemon_payload(
                root=root,
                graph_dir=graph_dir,
                preflight_payload=preflight_payload,
                eval_id="unit_queue_daemon",
                queue_name="trace_policy_preflight",
                execute=False,
                writeback_manifests=True,
            )
            self.assertEqual(payload["ready_queue_count"], 2)
            self.assertEqual(payload["planned_leaf_count"], 2)
            self.assertEqual(payload["executable_leaf_count"], 2)
            self.assertEqual(payload["execution_status_counts"], {"planned": 2})
            self.assertEqual(payload["proposal_ids"], ["prop_a", "prop_b"])
            records = payload["execution_payload"]["execution_records"]
            self.assertTrue(all(record["command"].startswith("python3 run_candidate.py") for record in records))
            after_store = JsonlGraphStore(graph_dir)
            self.assertEqual(set(after_store.nodes), before_nodes)
            self.assertGreaterEqual(len(after_store.trials), 3)
            self.assertFalse(payload["mode"]["apply_accepted"])

    def test_recursive_daemon_enforces_pre_live_screen_on_preflight_queue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="surface_recursive",
                type=AssumptionType.SELF_MODIFICATION,
                claim="Recursive daemon",
            ))
            store.flush()
            preflight_payload = {
                "eval_id": "unit_screened_preflight_queue",
                "summaries": [
                    {
                        "proposal_id": "prop_allow",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "trigger_problem_ids": ["p1", "p2", "p3"],
                        "control_problem_ids": ["c1"],
                        "command_hint": "python3 run_candidate.py --variant proposal_allow",
                    },
                    {
                        "proposal_id": "prop_block",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "trigger_problem_ids": ["p4", "p5", "p6"],
                        "control_problem_ids": ["c2"],
                        "command_hint": "python3 run_candidate.py --variant proposal_block",
                    },
                    {
                        "proposal_id": "prop_missing",
                        "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                        "trigger_problem_ids": ["p7", "p8", "p9"],
                        "control_problem_ids": ["c3"],
                        "command_hint": "python3 run_candidate.py --variant proposal_missing",
                    },
                ],
            }
            screen_payload = {
                "eval_id": "unit_pre_live_screen",
                "rows": [
                    {
                        "case": {"proposal_id": "prop_allow"},
                        "screen": {
                            "proposal_id": "prop_allow",
                            "decision": "run_live",
                            "would_run_live": True,
                            "risk_score": 0.0,
                        },
                    },
                    {
                        "case": {"proposal_id": "prop_block"},
                        "screen": {
                            "proposal_id": "prop_block",
                            "decision": "block_predicted_low_benefit",
                            "would_run_live": False,
                            "risk_score": 0.8,
                            "predicted_failure_modes": ["high_overlap_with_low_utility_sibling"],
                        },
                    },
                ],
            }
            payload = build_preflight_queue_daemon_payload(
                root=root,
                graph_dir=graph_dir,
                preflight_payload=preflight_payload,
                pre_live_screen_payload=screen_payload,
                enforce_pre_live_screen=True,
                eval_id="unit_screened_queue_daemon",
                queue_name="screened_trace_policy_preflight",
                execute=False,
            )
            self.assertEqual(payload["ready_queue_count"], 3)
            self.assertEqual(payload["screened_ready_queue_count"], 2)
            self.assertEqual(payload["planned_leaf_count"], 2)
            self.assertEqual(payload["proposal_ids"], ["prop_allow", "prop_missing"])
            self.assertEqual(payload["pre_live_screen"]["blocked_proposal_ids"], ["prop_block"])
            self.assertEqual(payload["pre_live_screen"]["missing_decision_proposal_ids"], ["prop_missing"])
            self.assertEqual(
                payload["pre_live_screen"]["decision_counts"]["block_predicted_low_benefit"],
                1,
            )

    def test_recursive_daemon_executes_queue_and_resumes_from_generated_judgments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
            ))
            store.flush()
            judgment_path = root / "queue_judgments.json"
            judgments = {
                "p1": {"winner": "proposal_exec"},
                "p2": {"winner": "proposal_exec"},
                "p3": {"winner": "proposal_exec"},
            }
            encoded_judgments = base64.b64encode(json.dumps(judgments).encode("utf-8")).decode("ascii")
            command = (
                "python3 -c "
                f"\"import base64; from pathlib import Path; Path({str(judgment_path)!r}).write_bytes("
                f"base64.b64decode('{encoded_judgments}'))\""
            )
            preflight_payload = {
                "eval_id": "unit_exec_preflight",
                "summaries": [{
                    "proposal_id": "prop_exec",
                    "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                    "trigger_problem_ids": ["p1", "p2", "p3"],
                    "control_problem_ids": [],
                    "command_hint": command,
                }],
            }
            evolution_payload = {
                "eval_id": "unit_exec_evolution",
                "proposals": {
                    "eval_id": "unit_exec_props",
                    "proposals": [{
                        "proposal_id": "prop_exec",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.8,
                        "candidate_node": {
                            "id": "cand_exec",
                            "claim": "Queue execution should resume from generated judgments.",
                            "predicted_effects": ["close the execute-read-resume loop"],
                        },
                    }],
                },
                "candidate_preflight": preflight_payload,
                "falsification_gate": {"summaries": [{
                    "proposal_id": "prop_exec",
                    "decision": FalsificationDecision.READY_FOR_ABLATION.value,
                    "next_action": "run_fresh_ablation",
                }]},
                "bayesian_policy": {"scores": [{
                    "proposal_id": "prop_exec",
                    "recommended_action": BayesianPolicyAction.RUN_ABLATION.value,
                    "posterior_priority": 1.2,
                    "command_hint": command,
                }]},
                "policy_update_plan": {"actions": [{
                    "proposal_id": "prop_exec",
                    "policy_action": "run_fresh_ablation_before_promotion",
                }]},
                "regression_predictions": [{"proposal_id": "prop_exec", "risk": "low"}],
                "formal_mapping_gate": {"gates": []},
            }
            before_nodes = set(JsonlGraphStore(graph_dir).nodes)
            payload = build_preflight_queue_daemon_payload(
                root=root,
                graph_dir=graph_dir,
                preflight_payload=preflight_payload,
                evolution_payload=evolution_payload,
                judgment_sets=[JudgmentSet(
                    candidate_variant="proposal_exec",
                    baseline_variant="baseline",
                    judgment_paths=[judgment_path],
                    proposal_ids=["prop_exec"],
                )],
                eval_id="unit_exec_queue",
                queue_name="exec_queue",
                command_limit=1,
                execute=True,
                writeback_manifests=True,
            )
            self.assertTrue(judgment_path.exists())
            self.assertEqual(payload["execution_status_counts"], {"succeeded": 1})
            self.assertEqual(payload["candidate_acceptance_counts"], {"accept": 1})
            self.assertEqual(payload["accepted_proposal_ids"], ["prop_exec"])
            self.assertTrue(payload["resumed"])
            self.assertEqual(payload["applied_candidate_node_ids"], [])
            self.assertEqual(set(JsonlGraphStore(graph_dir).nodes), before_nodes)

    def test_preflight_queue_auto_loads_artifact_judgments(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            graph_dir = root / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="Use controlled-variable tests.",
            ))
            store.flush()
            sample_path = root / "phase four" / "assumption_graph" / "artifact sample.json"
            sample_path.parent.mkdir(parents=True)
            rows = [
                {"problem_id": "p1", "description": "case one", "domain": "business", "difficulty": "medium"},
                {"problem_id": "p2", "description": "case two", "domain": "business", "difficulty": "medium"},
                {"problem_id": "p3", "description": "case three", "domain": "business", "difficulty": "medium"},
                {"problem_id": "c1", "description": "control", "domain": "science", "difficulty": "hard"},
            ]
            sample_path.write_text(json.dumps(rows), encoding="utf-8")
            answers_dir = root / "phase two" / "analysis" / "cache" / "answers"
            judgments_dir = root / "phase two" / "analysis" / "cache" / "judgments"
            answers_dir.mkdir(parents=True)
            judgments_dir.mkdir(parents=True)
            candidate_variant = "proposal_artifact"
            baseline_variant = "baseline_artifact"
            (answers_dir / f"{candidate_variant}_answers.json").write_text(json.dumps({
                "p1": "candidate one",
                "p2": "candidate two",
                "p3": "candidate three",
                "c1": "candidate control",
            }), encoding="utf-8")
            (answers_dir / f"{baseline_variant}_answers.json").write_text(json.dumps({
                "p1": "baseline one",
                "p2": "baseline two",
                "p3": "baseline three",
                "c1": "baseline control",
            }), encoding="utf-8")
            judgment_path = judgments_dir / f"{candidate_variant}_vs_{baseline_variant}.json"
            judgment_path.write_text(json.dumps({
                "p1": {"winner": candidate_variant},
                "p2": {"winner": candidate_variant},
                "p3": {"winner": candidate_variant},
                "c1": {"winner": "tie"},
            }), encoding="utf-8")
            command = (
                'python3 "phase one/scripts/validation/phase2_v20_framework.py" '
                f"--variant {candidate_variant} --sample {json.dumps(str(sample_path))} "
                '--assumption-proposal-ids prop_artifact --assumption-force-proposal-route'
            )
            preflight_payload = {
                "eval_id": "unit_artifact_preflight",
                "summaries": [{
                    "proposal_id": "prop_artifact",
                    "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                    "trigger_problem_ids": ["p1", "p2", "p3"],
                    "control_problem_ids": ["c1"],
                    "command_hint": command,
                }],
            }
            artifact_payload = build_queue_artifact_eval_payload(
                root=root,
                preflight_payload=preflight_payload,
                baseline_variant=baseline_variant,
                eval_id="unit_artifact_eval",
            )
            self.assertEqual(artifact_payload["plan_count"], 1)
            self.assertEqual(artifact_payload["candidate_answer_ready_count"], 1)
            self.assertEqual(artifact_payload["baseline_answer_ready_count"], 1)
            self.assertEqual(artifact_payload["judgment_set_count"], 1)
            self.assertEqual(artifact_payload["trigger_outcomes"], {"win": 3})
            self.assertEqual(artifact_payload["control_outcomes"], {"tie": 1})
            self.assertEqual(artifact_payload["control_loss_count"], 0)
            self.assertEqual(artifact_payload["controlled_promotion_plan_count"], 1)
            self.assertEqual(artifact_payload["undercontrolled_plan_count"], 0)
            self.assertTrue(artifact_payload["plans"][0]["promotion_evidence"]["ready_for_controlled_promotion"])
            self.assertIn("cached_framework.py", artifact_payload["plans"][0]["judge_command"])
            judgment_sets = judgment_sets_from_artifact_eval(artifact_payload)
            self.assertEqual(judgment_sets[0].candidate_variant, candidate_variant)
            evolution_payload = {
                "eval_id": "unit_artifact_evolution",
                "proposals": {
                    "eval_id": "unit_artifact_props",
                    "proposals": [{
                        "proposal_id": "prop_artifact",
                        "proposal_type": ProposalType.FAILURE_HYPOTHESIS.value,
                        "parent_node_id": "strategy_S01",
                        "priority": 0.8,
                        "candidate_node": {
                            "id": "cand_artifact",
                            "claim": "Artifact judgments should return to the recursive parent.",
                            "predicted_effects": ["close fresh-ablation artifact readback"],
                        },
                    }],
                },
                "candidate_preflight": preflight_payload,
                "falsification_gate": {"summaries": [{
                    "proposal_id": "prop_artifact",
                    "decision": FalsificationDecision.READY_FOR_ABLATION.value,
                    "next_action": "run_fresh_ablation",
                }]},
                "bayesian_policy": {"scores": [{
                    "proposal_id": "prop_artifact",
                    "recommended_action": BayesianPolicyAction.RUN_ABLATION.value,
                    "posterior_priority": 1.2,
                    "command_hint": command,
                }]},
                "policy_update_plan": {"actions": [{
                    "proposal_id": "prop_artifact",
                    "policy_action": "run_fresh_ablation_before_promotion",
                }]},
                "regression_predictions": [{"proposal_id": "prop_artifact", "risk": "low"}],
                "formal_mapping_gate": {"gates": []},
            }
            payload = build_preflight_queue_daemon_payload(
                root=root,
                graph_dir=graph_dir,
                preflight_payload=preflight_payload,
                evolution_payload=evolution_payload,
                eval_id="unit_artifact_queue",
                queue_name="artifact_queue",
                artifact_baseline_variant=baseline_variant,
                execute=False,
            )
            self.assertEqual(payload["mode"]["artifact_auto_judgment_sets"], 1)
            self.assertEqual(payload["candidate_acceptance_counts"], {"accept": 1})
            self.assertEqual(payload["accepted_proposal_ids"], ["prop_artifact"])
            self.assertTrue(payload["resumed"])
            self.assertEqual(payload["artifact_evaluation"]["judgment_set_count"], 1)

    def test_artifact_readback_does_not_promote_trigger_only_evidence_as_controlled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample_path = root / "sample.json"
            sample_path.write_text(json.dumps([
                {"problem_id": "p1"},
                {"problem_id": "p2"},
                {"problem_id": "p3"},
            ]), encoding="utf-8")
            answers_dir = root / "phase two" / "analysis" / "cache" / "answers"
            judgments_dir = root / "phase two" / "analysis" / "cache" / "judgments"
            answers_dir.mkdir(parents=True)
            judgments_dir.mkdir(parents=True)
            candidate_variant = "proposal_trigger_only"
            baseline_variant = "baseline_trigger_only"
            answers = {"p1": "candidate one", "p2": "candidate two", "p3": "candidate three"}
            baseline = {"p1": "baseline one", "p2": "baseline two", "p3": "baseline three"}
            (answers_dir / f"{candidate_variant}_answers.json").write_text(json.dumps(answers), encoding="utf-8")
            (answers_dir / f"{baseline_variant}_answers.json").write_text(json.dumps(baseline), encoding="utf-8")
            (judgments_dir / f"{candidate_variant}_vs_{baseline_variant}.json").write_text(json.dumps({
                "p1": {"winner": candidate_variant},
                "p2": {"winner": candidate_variant},
                "p3": {"winner": candidate_variant},
            }), encoding="utf-8")
            preflight_payload = {
                "eval_id": "unit_trigger_only_preflight",
                "summaries": [{
                    "proposal_id": "prop_trigger_only",
                    "readiness": CandidateReadiness.READY_FOR_FRESH_ABLATION.value,
                    "trigger_problem_ids": ["p1", "p2", "p3"],
                    "control_problem_ids": [],
                    "command_hint": (
                        "python3 run.py "
                        f"--variant {candidate_variant} --sample {json.dumps(str(sample_path))}"
                    ),
                }],
            }
            payload = build_queue_artifact_eval_payload(
                root=root,
                preflight_payload=preflight_payload,
                baseline_variant=baseline_variant,
                eval_id="unit_trigger_only_artifact_eval",
            )
            self.assertEqual(payload["judgment_set_count"], 1)
            self.assertEqual(payload["trigger_outcomes"], {"win": 3})
            self.assertEqual(payload["control_judgment_count"], 0)
            self.assertEqual(payload["controlled_promotion_plan_count"], 0)
            self.assertEqual(payload["undercontrolled_plan_count"], 1)
            self.assertFalse(payload["plans"][0]["promotion_evidence"]["ready_for_controlled_promotion"])
            self.assertTrue(payload["plans"][0]["promotion_evidence"]["ready_for_trigger_only_acceptance"])

    def test_residual_clusterer_synthesizes_candidate_from_systematic_residuals(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S03",
                type=AssumptionType.METHOD,
                claim="Use staged fallback planning.",
            ))
            for i in range(2):
                store.append_trial(TrialManifest(
                    problem_id=f"p{i}",
                    action_type="retrieval",
                    component="phase2",
                    assumption="fallback planning should be retrieved",
                    why_selected="coverage tag matched",
                    expected_effect="activate staged fallback",
                    assumption_ids=["strategy_S03"],
                    residual="retrieval selected irrelevant memory and missed fallback trigger",
                    residual_type=ResidualType.MEMORY_DEFECT,
                    status=TrialStatus.FAILED,
                    trial_id=f"trial_mem_{i}",
                ))
            store.flush()
            payload = build_residual_cluster_payload(
                store=JsonlGraphStore(td),
                eval_id="unit_cluster",
                min_cluster_size=2,
                llm_synthesizer=lambda prompt: "LLM synthesized retrieval gate for fallback triggers.",
                writeback_manifests=True,
            )
            self.assertEqual(payload["cluster_count"], 1)
            self.assertEqual(payload["proposal_count"], 1)
            proposal = payload["proposals"][0]
            self.assertIn("LLM synthesized", proposal["candidate_node"]["claim"])
            self.assertEqual(proposal["parent_node_id"], "strategy_S03")
            self.assertTrue(proposal["candidate_node"]["payload"]["validation_plan"]["trigger_problem_ids"])
            self.assertEqual(len(JsonlGraphStore(td).trials), 3)

    def test_residual_clusterer_tie_breaks_terms_deterministically(self):
        records = [
            ResidualRecord(
                record_id="r2",
                problem_id="p2",
                residual_type=ResidualType.UNKNOWN.value,
                residual="zeta beta alpha",
                action_type="answer",
                component="phase2",
                assumption_ids=["strategy_B", "strategy_A"],
            ),
            ResidualRecord(
                record_id="r1",
                problem_id="p1",
                residual_type=ResidualType.UNKNOWN.value,
                residual="zeta beta alpha",
                action_type="answer",
                component="phase2",
                assumption_ids=["strategy_A", "strategy_B"],
            ),
        ]
        clusters = cluster_residual_records(records, min_cluster_size=1, max_clusters=4)
        self.assertEqual(clusters[0].signature, "phase2:alpha")
        self.assertEqual(clusters[0].top_terms[:3], ["alpha", "beta", "zeta"])
        self.assertEqual(clusters[0].parent_node_id, "strategy_A")

    def test_recursive_runner_writeback_logs_frame_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            graph_dir = Path(td) / "graph"
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(AssumptionNode(
                id="strategy_S24",
                type=AssumptionType.METHOD,
                claim="Identify bottleneck before optimizing.",
                tags=["S24", "bottleneck"],
            ))
            store.flush()

            payload = build_recursive_assumption_run(
                graph_dir=graph_dir,
                problem="A release has many blocking bugs and needs triage.",
                goal="Decide which assumption should shape the next action.",
                eval_id="unit_recursive_writeback",
                max_children=1,
                writeback=True,
            )
            updated = JsonlGraphStore(graph_dir)
            self.assertEqual(len(updated.trials), len(payload["frames"]))
            self.assertTrue(all(
                trial.component == "recursive_assumption_runner"
                for trial in updated.trials.values()
            ))

    def test_falsification_gate_orders_preflight_before_acceptance(self):
        proposal_payload = {
            "eval_id": "unit_props",
            "proposals": [{
                "proposal_id": "prop_1",
                "proposal_type": "assumption_revision",
                "parent_node_id": "strategy_S01",
                "candidate_node": {"id": "cand_1"},
            }],
        }
        preflight_payload = {
            "eval_id": "unit_preflight",
            "summaries": [{
                "proposal_id": "prop_1",
                "readiness": "ready_for_fresh_ablation",
                "trigger_problem_ids": ["p1", "p2", "p3"],
                "control_problem_ids": ["p4"],
                "command_hint": "run proposal prop_1",
            }],
        }
        ready = build_falsification_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
        )
        self.assertEqual(
            ready["summaries"][0]["decision"],
            FalsificationDecision.READY_FOR_ABLATION.value,
        )
        self.assertEqual(ready["experiment_name_counts"]["trigger_benefit_sequential"], 1)
        by_name = {row["name"]: row for row in ready["summaries"][0]["experiments"]}
        self.assertEqual(by_name["trigger_benefit_sequential"]["status"], "planned")
        self.assertEqual(by_name["route_power_and_scope_probe"]["status"], "passed")

        rejected = build_falsification_payload(
            proposal_payload=proposal_payload,
            preflight_payload=preflight_payload,
            acceptance_payload={
                "eval_id": "unit_acceptance",
                "summaries": [{
                    "proposal_id": "prop_1",
                    "decision": "reject_benefit",
                    "trigger_outcomes": {"loss": 3},
                    "control_outcomes": {},
                    "trigger_lcb90": 0.1,
                    "rationale": "benefit too weak",
                }],
            },
        )
        self.assertEqual(
            rejected["summaries"][0]["decision"],
            FalsificationDecision.REJECT_BENEFIT.value,
        )
        self.assertEqual(rejected["summaries"][0]["next_action"], "reject_or_revise_candidate")
        rejected_by_name = {row["name"]: row for row in rejected["summaries"][0]["experiments"]}
        self.assertEqual(rejected_by_name["trigger_benefit_sequential"]["status"], "failed")
        self.assertEqual(rejected_by_name["control_harm_sequential"]["status"], "passed")

    def test_bayesian_policy_scores_ready_candidate_for_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="固定其他条件，每次只改变一个因素",
                tags=["S01"],
            ))
            for i, value in enumerate([1.0, 1.0, 0.5]):
                store.add_evidence(EvidenceRecord(
                    node_id="strategy_S01",
                    source="unit",
                    outcome="success" if value == 1.0 else "tie",
                    metric="pairwise_judge_win",
                    value=value,
                    evidence_id=f"ev_{i}",
                ))
            belief = parent_belief(store, "strategy_S01")
            self.assertGreater(belief.mean, 0.65)

            payload = build_bayesian_policy_payload(
                store=store,
                proposal_payload={
                    "eval_id": "unit_props",
                    "proposals": [{
                        "proposal_id": "prop_1",
                        "proposal_type": "assumption_revision",
                        "parent_node_id": "strategy_S01",
                        "candidate_node": {"id": "cand_1"},
                    }],
                },
                preflight_payload={
                    "eval_id": "unit_preflight",
                    "summaries": [{
                        "proposal_id": "prop_1",
                        "readiness": "ready_for_fresh_ablation",
                        "command_hint": "run ablation",
                    }],
                },
                falsification_payload={
                    "summaries": [{
                        "proposal_id": "prop_1",
                        "decision": "ready_for_ablation",
                    }],
                },
                regression_predictions=[{"proposal_id": "prop_1", "risk": "low"}],
            )
            score = payload["scores"][0]
            self.assertEqual(score["recommended_action"], BayesianPolicyAction.RUN_ABLATION.value)
            self.assertGreater(score["posterior_priority"], 1.0)

    def test_formal_mapping_audit_detects_complete_exp82_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            base = {
                "type": AssumptionType.HARNESS,
                "claim": "formal mapping test",
                "payload": {"seed_cid": "WCAND_TEST"},
                "tags": ["WCAND_TEST"],
            }
            store.upsert_node(AssumptionNode(
                id="feature_1",
                kind="feature",
                formal_form={"kind": "feature", "expr": {"keywords_zh": ["风险"], "regex": []}},
                **base,
            ))
            store.upsert_node(AssumptionNode(
                id="constraint_1",
                kind="constraint",
                formal_form={"kind": "constraint", "expr": {"required_substrings": ["回滚"]}},
                **base,
            ))
            store.upsert_node(AssumptionNode(
                id="decomp_1",
                kind="decomposition",
                formal_form={"kind": "decomposition", "expr": {"steps": ["identify risk", "add guardrail"]}},
                **base,
            ))
            store.upsert_node(AssumptionNode(
                id="verify_1",
                kind="verification",
                formal_form={"kind": "verification", "expr": {"instruction": "check rollback"}},
                **base,
            ))
            store.upsert_node(AssumptionNode(
                id="hp_1",
                kind="hp_change",
                formal_form={"kind": "hp_change", "expr": {"temperature": 0.0, "max_tokens": 1000}},
                **base,
            ))
            payload = build_formal_mapping_payload(store)
            self.assertEqual(payload["status_counts"], {FormalMappingStatus.COMPLETE.value: 1})
            summary = payload["summaries"][0]
            self.assertTrue(summary["invariants"]["trigger_detector"])
            self.assertTrue(summary["invariants"]["verification_operator"])
            self.assertEqual(summary["nodes"][1]["invariants"]["steps"], ["identify risk", "add guardrail"])

            applications = search_formal_mappings(payload, "上线风险需要回滚")
            self.assertEqual(applications[0]["source_key"], "WCAND_TEST")
            self.assertIn("回滚", applications[0]["constraint_operator"][0]["required_substrings"])
            formatted = format_formal_mapping_applications(applications)
            self.assertIn("Formal Mapping Reasoning", formatted)
            self.assertIn("identify risk", formatted)

    def test_formal_mapping_metrics_build_finite_category_payload(self):
        identical = finite_kernel_metrics(
            [[0.2, 0.8], [0.1, 0.9]],
            [[0.2, 0.8], [0.1, 0.9]],
        )
        shifted = finite_kernel_metrics(
            [[0.2, 0.8], [0.1, 0.9]],
            [[0.8, 0.2], [0.9, 0.1]],
        )
        self.assertEqual(identical["frobenius_distance"], 0.0)
        self.assertGreater(shifted["frobenius_distance"], 0.0)

        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            base = {
                "type": AssumptionType.HARNESS,
                "claim": "formal metric test",
                "payload": {"seed_cid": "WCAND_METRIC"},
                "tags": ["WCAND_METRIC"],
            }
            for node_id, kind, expr in [
                ("feature_m", "feature", {"keywords_en": ["risk"]}),
                ("constraint_m", "constraint", {"required_substrings": ["rollback"]}),
                ("decomp_m", "decomposition", {"steps": ["identify", "verify"]}),
                ("verify_m", "verification", {"instruction": "check rollback"}),
                ("hp_m", "hp_change", {"temperature": 0.0}),
            ]:
                store.upsert_node(AssumptionNode(
                    id=node_id,
                    kind=kind,
                    formal_form={"kind": kind, "expr": expr},
                    **base,
                ))
            formal_payload = build_formal_mapping_payload(store)
            metric_payload = build_categorical_info_geometry_payload(formal_payload)
            self.assertEqual(metric_payload["mapping_count"], 1)
            summary = metric_payload["summaries"][0]
            self.assertIn("feature", summary["objects"])
            self.assertTrue(summary["morphisms"])
            self.assertTrue(summary["metrics"]["same_shape"])

    def test_formal_mapping_dedup_recommends_only_complete_equivalence_merges(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for seed in ["WCAND_DUP_A", "WCAND_DUP_B"]:
                base = {
                    "type": AssumptionType.HARNESS,
                    "claim": f"formal dedup {seed}",
                    "payload": {"seed_cid": seed},
                    "tags": [seed],
                }
                for suffix, kind, expr in [
                    ("feature", "feature", {"keywords_en": ["risk"], "regex": []}),
                    ("constraint", "constraint", {"required_substrings": ["rollback"]}),
                    ("decomp", "decomposition", {"steps": ["identify", "verify"]}),
                    ("verify", "verification", {"instruction": "check rollback"}),
                    ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
                ]:
                    store.upsert_node(AssumptionNode(
                        id=f"{seed}_{suffix}",
                        kind=kind,
                        formal_form={"kind": kind, "expr": expr},
                        **base,
                    ))
            store.upsert_node(AssumptionNode(
                id="WCAND_UNSAFE_constraint",
                type=AssumptionType.HARNESS,
                kind="constraint",
                claim="unsafe duplicate should be excluded",
                formal_form={"kind": "constraint", "expr": {"required_substrings": ["rollback"]}},
                payload={"seed_cid": "WCAND_UNSAFE"},
                tags=["WCAND_UNSAFE"],
            ))
            formal_payload = build_formal_mapping_payload(store)
            dedup = build_formal_dedup_payload(formal_payload)
            self.assertEqual(dedup["complete_mapping_count"], 2)
            self.assertEqual(dedup["incomplete_mapping_excluded_count"], 1)
            self.assertEqual(dedup["duplicate_cluster_count"], 1)
            self.assertEqual(dedup["merge_recommendation_count"], 1)
            cluster = dedup["clusters"][0]
            self.assertEqual(cluster["merge_action"], "merge_complete_formal_equivalent")
            self.assertEqual(len(cluster["duplicate_mapping_ids"]), 1)

    def test_formal_search_eval_builds_positive_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for seed, keyword, required in [
                ("WCAND_EXPECTED", "risk", "rollback"),
                ("WCAND_OTHER", "speed", "latency"),
            ]:
                base = {
                    "type": AssumptionType.HARNESS,
                    "claim": f"formal search {seed}",
                    "payload": {"seed_cid": seed},
                    "tags": [seed],
                }
                for suffix, kind, expr in [
                    ("feature", "feature", {"keywords_en": [keyword], "regex": []}),
                    ("constraint", "constraint", {"required_substrings": [required]}),
                    ("decomp", "decomposition", {"steps": ["identify", "verify"]}),
                    ("verify", "verification", {"instruction": f"check {required}"}),
                    ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
                ]:
                    store.upsert_node(AssumptionNode(
                        id=f"{seed}_{suffix}",
                        kind=kind,
                        formal_form={"kind": kind, "expr": expr},
                        **base,
                    ))
            formal_payload = build_formal_mapping_payload(store)
            search_eval = build_formal_search_eval_payload(formal_payload)
            self.assertEqual(search_eval["query_count"], 2)
            self.assertEqual(search_eval["pass_count"], 2)
            self.assertEqual(search_eval["negative_application_count"], 2)
            self.assertEqual(search_eval["top1_hit_rate"], 1.0)

    def test_independent_formal_search_uses_operator_intent(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for seed, keyword, required, step in [
                ("WCAND_EXPECTED", "risk", "rollback", "stage recovery"),
                ("WCAND_OTHER", "speed", "latency", "profile bottleneck"),
            ]:
                base = {
                    "type": AssumptionType.HARNESS,
                    "claim": f"formal independent {seed}",
                    "payload": {"seed_cid": seed},
                    "tags": [seed],
                }
                for suffix, kind, expr in [
                    ("feature", "feature", {"keywords_en": [keyword], "regex": []}),
                    ("constraint", "constraint", {"required_substrings": [required]}),
                    ("decomp", "decomposition", {"steps": [step]}),
                    ("verify", "verification", {"instruction": f"check {required}"}),
                    ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
                ]:
                    store.upsert_node(AssumptionNode(
                        id=f"{seed}_{suffix}",
                        kind=kind,
                        formal_form={"kind": kind, "expr": expr},
                        **base,
                    ))
            formal_payload = build_formal_mapping_payload(store)
            search_eval = build_independent_formal_search_eval_payload(formal_payload)
            self.assertEqual(search_eval["eval_kind"], "independent_operator_intent")
            self.assertEqual(search_eval["query_count"], 2)
            self.assertEqual(search_eval["top1_hit_rate"], 1.0)
            first_app = search_eval["results"][0]["applications"][0]
            self.assertGreater(first_app["operator_score"], 0.0)
            self.assertTrue(first_app["matched_operator_terms"])

    def test_formal_downstream_task_eval_covers_role_families(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for seed, keyword, required, step, verifier in [
                ("WCAND_EXPECTED", "risk", "rollback", "stage recovery", "check rollback"),
                ("WCAND_OTHER", "speed", "latency", "profile bottleneck", "check latency"),
            ]:
                base = {
                    "type": AssumptionType.HARNESS,
                    "claim": f"formal downstream {seed}",
                    "payload": {"seed_cid": seed},
                    "tags": [seed],
                }
                for suffix, kind, expr in [
                    ("feature", "feature", {"keywords_en": [keyword], "regex": []}),
                    ("constraint", "constraint", {"required_substrings": [required]}),
                    ("decomp", "decomposition", {"steps": [step]}),
                    ("verify", "verification", {"instruction": verifier}),
                    ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
                ]:
                    store.upsert_node(AssumptionNode(
                        id=f"{seed}_{suffix}",
                        kind=kind,
                        formal_form={"kind": kind, "expr": expr},
                        **base,
                    ))
            formal_payload = build_formal_mapping_payload(store)
            payload = build_formal_downstream_task_eval_payload(formal_payload)
            self.assertTrue(payload["pass"])
            self.assertEqual(payload["query_count"], 6)
            self.assertEqual(payload["task_family_count"], 3)
            self.assertGreaterEqual(payload["top1_hit_rate"], 0.8)
            self.assertEqual(payload["task_family_counts"]["constraint_application"], 2)

    def test_formal_answer_quality_probe_scores_guided_answer_above_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for seed, keyword, required, step, verifier in [
                ("WCAND_EXPECTED", "risk", "rollback", "stage recovery", "check rollback"),
                ("WCAND_OTHER", "speed", "latency", "profile bottleneck", "check latency"),
            ]:
                base = {
                    "type": AssumptionType.HARNESS,
                    "claim": f"formal answer quality {seed}",
                    "payload": {"seed_cid": seed},
                    "tags": [seed],
                }
                for suffix, kind, expr in [
                    ("feature", "feature", {"keywords_en": [keyword], "regex": []}),
                    ("constraint", "constraint", {"required_substrings": [required]}),
                    ("decomp", "decomposition", {"steps": [step]}),
                    ("verify", "verification", {"instruction": verifier}),
                    ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
                ]:
                    store.upsert_node(AssumptionNode(
                        id=f"{seed}_{suffix}",
                        kind=kind,
                        formal_form={"kind": kind, "expr": expr},
                        **base,
                    ))
            formal_payload = build_formal_mapping_payload(store)
            payload = build_formal_answer_quality_probe_payload(formal_payload)
            self.assertFalse(payload["pass"])
            self.assertEqual(payload["probe_count"], 2)
            self.assertEqual(payload["guided_win_rate"], 1.0)
            self.assertEqual(payload["top1_hit_rate"], 1.0)
            self.assertGreater(payload["guided_mean_score"], payload["baseline_mean_score"])
            self.assertGreaterEqual(payload["mean_delta"], 0.35)

    def test_formal_transfer_eval_scores_expected_mapping_above_distractor(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for seed, keyword, required in [
                ("WCAND_EXPECTED", "risk", "rollback"),
                ("WCAND_OTHER", "speed", "latency"),
            ]:
                base = {
                    "type": AssumptionType.HARNESS,
                    "claim": f"formal transfer {seed}",
                    "payload": {"seed_cid": seed},
                    "tags": [seed],
                }
                for suffix, kind, expr in [
                    ("feature", "feature", {"keywords_en": [keyword], "regex": []}),
                    ("constraint", "constraint", {"required_substrings": [required]}),
                    ("decomp", "decomposition", {"steps": ["identify", "verify"]}),
                    ("verify", "verification", {"instruction": f"check {required}"}),
                    ("hp", "hp_change", {"temperature": 0.0, "max_tokens": 1000}),
                ]:
                    store.upsert_node(AssumptionNode(
                        id=f"{seed}_{suffix}",
                        kind=kind,
                        formal_form={"kind": kind, "expr": expr},
                        **base,
                    ))
            formal_payload = build_formal_mapping_payload(store)
            metric_payload = build_categorical_info_geometry_payload(formal_payload)
            search_eval = {
                "eval_id": "unit_formal_search",
                "results": [{
                    "id": "q1",
                    "expected": "WCAND_EXPECTED",
                    "top_source_key": "WCAND_EXPECTED",
                    "applications": [
                        {"source_key": "WCAND_EXPECTED", "score": 5.0},
                        {"source_key": "WCAND_OTHER", "score": 1.0},
                    ],
                }] * 5,
            }
            payload = build_formal_transfer_eval_payload(
                formal_mapping_payload=formal_payload,
                metric_payload=metric_payload,
                search_eval_payload=search_eval,
            )
            self.assertTrue(payload["pass"])
            self.assertEqual(payload["top1_hit_rate"], 1.0)
            self.assertEqual(payload["pairwise_auc"], 1.0)
            self.assertGreater(payload["positive_mean_transfer_score"], payload["negative_mean_transfer_score"])

    def test_formal_engine_depth_audit_passes_with_negative_controls(self):
        payload = build_formal_engine_depth_payload(
            eval_id="unit_formal_engine_depth",
            store=JsonlGraphStore(Path("phase four/assumption_graph")),
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertFalse(payload["strict_category_theory_theorem_prover"])
        self.assertFalse(payload["true_blackwell_or_fisher_engine"])
        summary = payload["summary"]
        self.assertGreaterEqual(summary["complete_mapping_count"], 5)
        self.assertGreaterEqual(summary["negative_control_application_count"], 200)
        self.assertGreaterEqual(summary["downstream_transfer_auc"], 0.90)
        self.assertGreaterEqual(summary["answer_quality_mean_delta"], 0.35)

    def test_formal_mapping_audit_rejects_missing_trigger(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            base = {
                "type": AssumptionType.HARNESS,
                "claim": "formal mapping missing trigger",
                "payload": {"seed_cid": "WCAND_UNSAFE"},
                "tags": ["WCAND_UNSAFE"],
            }
            store.upsert_node(AssumptionNode(
                id="constraint_1",
                kind="constraint",
                formal_form={"kind": "constraint", "expr": {"required_substrings": ["回滚"]}},
                **base,
            ))
            store.upsert_node(AssumptionNode(
                id="verify_1",
                kind="verification",
                formal_form={"kind": "verification", "expr": {"instruction": "check rollback"}},
                **base,
            ))
            payload = build_formal_mapping_payload(store)
            self.assertEqual(payload["status_counts"], {FormalMappingStatus.UNSAFE.value: 1})
            summary = payload["summaries"][0]
            self.assertIn("missing trigger detector", summary["warnings"])

    def test_formal_mapping_gate_blocks_unsafe_promotion_policy(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="constraint_1",
                type=AssumptionType.HARNESS,
                kind="constraint",
                claim="formal mapping missing trigger",
                formal_form={"kind": "constraint", "expr": {"required_substrings": ["回滚"]}},
                payload={"seed_cid": "WCAND_UNSAFE"},
                tags=["WCAND_UNSAFE"],
            ))
            formal_payload = build_formal_mapping_payload(store)
            proposal_payload = {
                "proposals": [{
                    "proposal_id": "prop_unsafe",
                    "proposal_type": ProposalType.PROMOTION_RECORD.value,
                    "parent_node_id": "constraint_1",
                    "candidate_node": None,
                }],
            }
            gate_payload = build_formal_mapping_gate_payload(
                proposal_payload=proposal_payload,
                formal_mapping_payload=formal_payload,
            )
            self.assertEqual(
                gate_payload["decision_counts"],
                {FormalMappingGateDecision.BLOCK_UNSAFE_MAPPING.value: 1},
            )
            self.assertEqual(gate_payload["blocked_proposal_ids"], ["prop_unsafe"])

            policy = build_policy_update_plan(
                proposal_payload=proposal_payload,
                preflight_payload={
                    "summaries": [{
                        "proposal_id": "prop_unsafe",
                        "readiness": "manifest_only",
                    }],
                },
                formal_mapping_gate_payload=gate_payload,
            )
            self.assertEqual(policy["actions"][0]["policy_action"], "block_unsafe_formal_mapping")

    def test_failure_hypotheses_generate_candidate_from_loss(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S03",
                type=AssumptionType.METHOD,
                claim="use staged fallback planning",
                tags=["S03"],
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            payload = build_failure_hypothesis_payload(
                graph=graph,
                sample=[{
                    "problem_id": "daily_life_001",
                    "domain": "daily_life",
                    "difficulty": "medium",
                    "description": "Plan a move when time and transport are uncertain.",
                }],
                meta_by_pid={"daily_life_001": {"frame": "contingency planning"}},
                writeback_summary={
                    "eval_id": "unit_eval",
                    "processed_trials": [{
                        "trial_id": "trial_1",
                        "problem_id": "daily_life_001",
                        "domain": "daily_life",
                        "difficulty": "medium",
                        "outcome": "loss",
                        "residual_type": "memory_defect",
                        "gold_hit": False,
                        "gold_ids": ["strategy_S03"],
                        "active_assumption_ids": [],
                    }],
                },
                eval_id="unit_failures",
            )
            self.assertEqual(payload["proposal_counts"], {ProposalType.FAILURE_HYPOTHESIS.value: 1})
            proposal = payload["proposals"][0]
            self.assertEqual(proposal["parent_node_id"], "strategy_S03")
            self.assertEqual(proposal["candidate_node"]["status"], "candidate")
            self.assertEqual(proposal["candidate_node"]["payload"]["source_problem_id"], "daily_life_001")

    def test_failure_hypotheses_include_skipped_judgment_losses(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for sid, claim in [
                ("strategy_S18", "abstract shared mathematical structure"),
                ("strategy_S24", "identify release bottlenecks"),
            ]:
                store.upsert_node(AssumptionNode(
                    id=sid,
                    type=AssumptionType.METHOD,
                    claim=claim,
                    tags=[sid.replace("strategy_", "")],
                ))
            store.flush()
            judgment_path = Path(td) / "judgments.json"
            judgment_path.write_text(json.dumps({
                "math_skip": {"winner": "baseline"},
                "se_skip": {"winner": "baseline"},
            }), encoding="utf-8")
            sample = [
                {
                    "problem_id": "math_skip",
                    "domain": "mathematics",
                    "difficulty": "hard",
                    "description": "Find a unifying view between two identities.",
                    "coverage_tags": ["S18"],
                },
                {
                    "problem_id": "se_skip",
                    "domain": "software_engineering",
                    "difficulty": "medium",
                    "description": "Prioritize release-blocking regressions before launch.",
                    "coverage_tags": ["S24"],
                },
            ]
            payload = build_failure_hypothesis_payload(
                graph=SimpleAssumptionGraph(JsonlGraphStore(td)),
                sample=sample,
                meta_by_pid={"se_skip": {"frame": "release gate"}},
                writeback_summary={"eval_id": "unit_eval", "processed_trials": []},
                eval_id="unit_failures",
                judgment_paths=[judgment_path],
                intervention_variant="intervention",
                baseline_variant="baseline",
                skip_domains={"software_engineering"},
                skip_missing_meta=True,
            )
            self.assertEqual(payload["processed_loss_problem_count"], 0)
            self.assertEqual(payload["skipped_loss_problem_count"], 2)
            self.assertEqual(payload["skipped_loss_scan"]["reason_counts"], {
                "missing_meta": 1,
                "policy_skipped": 1,
            })
            by_source = {
                p["candidate_node"]["payload"]["source_problem_id"]: p
                for p in payload["proposals"]
            }
            self.assertEqual(by_source["math_skip"]["parent_node_id"], "strategy_S18")
            self.assertEqual(
                by_source["math_skip"]["candidate_node"]["payload"]["source_skipped_reason"],
                "missing_meta",
            )
            self.assertEqual(by_source["se_skip"]["parent_node_id"], "strategy_S24")
            self.assertEqual(
                by_source["se_skip"]["candidate_node"]["payload"]["source_skipped_reason"],
                "policy_skipped",
            )

            preflight = build_candidate_eval_payload(
                graph_dir=Path(td),
                proposal_payload=payload,
                sample=sample,
                meta_by_pid={"se_skip": {"frame": "release gate"}},
                eval_id="unit_preflight",
                policy_rerank=True,
                skip_domains={"software_engineering"},
                skip_missing_meta=True,
                min_trigger_n=1,
                min_active_trigger_n=1,
                force_proposal_route=True,
            )
            self.assertEqual(
                preflight["readiness_counts"],
                {CandidateReadiness.READY_FOR_FRESH_ABLATION.value: 2},
            )

    def test_software_engineering_reranker_boosts_execution_specific_methods(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            for sid, claim in [
                ("strategy_S22", "重新定义问题本身"),
                ("strategy_S24", "识别关键瓶颈节点"),
                ("strategy_S12", "用新证据更新判断"),
                ("strategy_S11", "设定足够好的发布阈值"),
                ("strategy_S08", "提出猜测并测试"),
            ]:
                store.upsert_node(AssumptionNode(
                    id=sid,
                    type=AssumptionType.METHOD,
                    claim=claim,
                    tags=[sid.replace("strategy_", "")],
                    confidence=0.7,
                ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            result = retrieve_phase2_assumptions(
                graph,
                problem="QA Lead needs to prioritize many release-blocking bugs before launch.",
                meta={"frame": "paradigm", "critical_reframe": "按玩家影响、收入、回归风险制定发布前修复阈值"},
                pid="se_bug",
                domain="software_engineering",
                difficulty="medium",
                top_k=3,
                pool_k=5,
                skip_domains=set(),
            )
            self.assertEqual(result.diagnostics["route"], "release_quality")
            ranked = [n.id for n in result.subgraph.nodes]
            self.assertIn("strategy_S24", ranked)
            self.assertIn("strategy_S12", ranked)
            self.assertIn("strategy_S11", ranked)
            self.assertTrue(result.policy_notes)

    def test_software_engineering_template_is_route_specific(self):
        release_template = format_phase2_domain_execution_template(
            "software_engineering",
            "QA Lead needs to prioritize many release-blocking bugs before launch.",
            {"critical_reframe": "按玩家影响、收入、回归风险制定发布前修复阈值"},
        )
        self.assertIn("release_quality", release_template)
        self.assertIn("release gate", release_template)
        self.assertIn("rollback/kill-switch", release_template)

        adapter_template = format_phase2_domain_execution_template(
            "software_engineering",
            "Discover an undocumented device API and build an adapter MVP safely.",
            {},
        )
        self.assertIn("adapter_discovery", adapter_template)
        self.assertIn("capability matrix", adapter_template)

        self.assertEqual(format_phase2_domain_execution_template("business", "渠道预算怎么分配", {}), "")

    def test_math_science_bypass_routes_research_and_decision_rows(self):
        self.assertEqual(
            route_math_science_problem("mathematics", "导师建议我尝试构建反例，但我投入了一年证明这个定理。"),
            "math_research_bridge",
        )
        self.assertEqual(
            route_math_science_problem("mathematics", "计算满足方程 x^2=4 的所有实数解。"),
            "math_formal",
        )
        self.assertEqual(
            route_math_science_problem("science", "博士合同三个月后到期，设备排队六个月，是否应先投稿？"),
            "science_decision",
        )

    def test_conditioned_evaluator_routes_and_gates_by_relevance(self):
        node = AssumptionNode(
            id="strategy_S01",
            type=AssumptionType.METHOD,
            claim="固定其他条件，每次只改变一个因素",
            tags=["S01", "控制变量"],
            payload={"activation": {"domains": ["software_engineering"]}},
        )
        rows = [
            ConditionedEvalRow(
                problem_id="p1",
                domain="software_engineering",
                difficulty="medium",
                description="线上线下指标不一致，需要控制变量排查。",
                coverage_tags=["S01"],
                outcome="win",
                active_assumption_ids=["strategy_S01"],
            ),
            ConditionedEvalRow(
                problem_id="p2",
                domain="software_engineering",
                difficulty="medium",
                description="定位性能回退，需要一次只改一个因素。",
                coverage_tags=["S01"],
                outcome="win",
                active_assumption_ids=["strategy_S01"],
            ),
            ConditionedEvalRow(
                problem_id="p3",
                domain="business",
                difficulty="medium",
                description="渠道预算分配。",
                coverage_tags=["S21"],
                outcome="loss",
                active_assumption_ids=["strategy_S01"],
            ),
        ]

        self.assertEqual(route_problem_to_node(node, rows[0]), RouteLabel.SHOULD_FIRE)
        self.assertEqual(route_problem_to_node(node, rows[2]), RouteLabel.NO_FIRE)
        summary = evaluate_node(node, rows, thresholds=GateThresholds(min_benefit_n=2, min_harm_n=1))
        self.assertEqual(summary.decision, GateDecision.NARROW_SCOPE)
        self.assertEqual(summary.active_should_fire_outcomes, {"win": 2})
        self.assertEqual(summary.active_no_fire_outcomes, {"loss": 1})

        rows[2] = ConditionedEvalRow(
            problem_id="p3",
            domain="business",
            difficulty="medium",
            description="渠道预算分配。",
            coverage_tags=["S21"],
            outcome="win",
            active_assumption_ids=[],
        )
        summary = evaluate_node(node, rows, thresholds=GateThresholds(min_benefit_n=2, min_harm_n=1))
        self.assertIn(summary.decision, {GateDecision.KEEP, GateDecision.PROMOTE})

    def test_conditioned_strategy_routing_does_not_fall_back_to_broad_lexical_match(self):
        node = AssumptionNode(
            id="strategy_S15",
            type=AssumptionType.METHOD,
            claim="从最小可工作版本开始，逐步添加功能，通过迭代循环不断完善和扩展产品或系统。",
            tags=["S15", "incremental"],
        )
        unrelated = ConditionedEvalRow(
            problem_id="p1",
            domain="software_engineering",
            difficulty="medium",
            description="评估医疗设备的商业化路径和责任边界。",
            coverage_tags=["S21", "S23"],
            outcome="win",
            active_assumption_ids=[],
        )
        relevant = ConditionedEvalRow(
            problem_id="p2",
            domain="software_engineering",
            difficulty="hard",
            description="给遗留系统设计最小可行增量替换路径。",
            coverage_tags=["S15"],
            outcome="win",
            active_assumption_ids=["strategy_S15"],
        )

        self.assertEqual(route_problem_to_node(node, unrelated), RouteLabel.NEUTRAL)
        self.assertEqual(route_problem_to_node(node, relevant), RouteLabel.SHOULD_FIRE)

    def test_wisdom_routing_uses_trigger_profile_not_broad_lexical_match(self):
        wisdom = AssumptionNode(
            id="wisdom_W020",
            type=AssumptionType.METHOD,
            claim="当你犹豫是继续投入还是及时退出时，区分责任感、脸面、沉没代价和不甘心。",
            tags=["wisdom", "W020"],
            context_conditions=["当你在继续与撤回之间摇摆，既受惯性牵引又怕显得软弱时。"],
            payload={
                "signal": "当你在继续与撤回之间摇摆，既受惯性牵引又怕显得软弱时。",
                "unpacked_for_llm": "当你犹豫是继续投入还是及时退出时，先分开看沉没代价和不甘心。",
                "cross_domain_examples": [
                    {"domain": "daily_life", "scenario": "关系只剩消耗，却因投入太久不肯离开。"},
                    {"domain": "engineering", "scenario": "方案方向已错，却因前期投入巨大被强行延续。"},
                ],
            },
        )
        profile = build_activation_profile(wisdom)
        self.assertEqual(profile.family, "wisdom")
        self.assertFalse(profile.allow_lexical_fallback)

        relevant = ConditionedEvalRow(
            problem_id="p1",
            domain="daily_life",
            difficulty="medium",
            description="我已经投入很多钱和时间，不甘心退出，但继续下去身体越来越差。",
            coverage_tags=[],
            outcome="win",
            active_assumption_ids=["wisdom_W020"],
        )
        unrelated = ConditionedEvalRow(
            problem_id="p2",
            domain="business",
            difficulty="medium",
            description="如何给新产品制定渠道预算和首批客户画像。",
            coverage_tags=[],
            outcome="loss",
            active_assumption_ids=["wisdom_W020"],
        )

        self.assertEqual(route_problem_to_node(wisdom, relevant), RouteLabel.SHOULD_FIRE)
        self.assertEqual(route_problem_to_node(wisdom, unrelated), RouteLabel.NEUTRAL)

    def test_generic_candidate_can_disable_lexical_fallback(self):
        node = AssumptionNode(
            id="cand_execution_contract",
            type=AssumptionType.HARNESS,
            claim="给出可逆试点、成功指标、停止阈值、责任人和回滚路径。",
            tags=["candidate", "execution_contract"],
            payload={
                "activation": {
                    "problem_ids": ["p_trigger"],
                    "allow_lexical_fallback": False,
                },
            },
        )
        profile = build_activation_profile(node)
        self.assertFalse(profile.allow_lexical_fallback)

        lexical_match = ConditionedEvalRow(
            problem_id="p_other",
            domain="business",
            difficulty="medium",
            description="需要一个可逆试点、成功指标、停止阈值和回滚路径。",
            coverage_tags=[],
            outcome="win",
            active_assumption_ids=[],
        )
        explicit = ConditionedEvalRow(
            problem_id="p_trigger",
            domain="business",
            difficulty="medium",
            description="无关表述也应该因显式 problem id 触发。",
            coverage_tags=[],
            outcome="win",
            active_assumption_ids=[],
        )

        self.assertEqual(route_problem_to_node(node, lexical_match), RouteLabel.NEUTRAL)
        self.assertEqual(route_problem_to_node(node, explicit), RouteLabel.SHOULD_FIRE)

    def test_lifecycle_planner_maps_conditioned_gate_to_auditable_actions(self):
        summaries = [
            {
                "node_id": "strategy_S15",
                "claim": "incremental",
                "decision": "expand_retrieval",
                "route_counts": {"should_fire": 8},
                "active_counts": {"should_fire": 2},
                "active_should_fire_outcomes": {"win": 2},
                "utility_when_active_should_fire": 1.0,
                "utility_lcb90": 1.0,
                "harm_ucb90": None,
                "reasons": ["useful but under-retrieved"],
            },
            {
                "node_id": "strategy_S21",
                "claim": "stop dead end",
                "decision": "revise",
                "route_counts": {"should_fire": 6},
                "active_counts": {"should_fire": 6},
                "active_should_fire_outcomes": {"loss": 4, "win": 2},
                "utility_when_active_should_fire": 0.33,
                "utility_lcb90": 0.08,
                "harm_ucb90": None,
                "reasons": ["weak benefit"],
            },
        ]
        actions = plan_lifecycle_actions(summaries, eval_id="unit_eval")
        self.assertEqual(actions[0].action_type, LifecycleActionType.EXPAND_RETRIEVAL)
        self.assertEqual(actions[0].to_trial_manifest(eval_id="unit_eval").assumption_ids, ["strategy_S15"])
        self.assertEqual(actions[1].action_type, LifecycleActionType.REVISE_ASSUMPTION)

    def test_candidate_proposals_create_child_nodes_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S21",
                type=AssumptionType.METHOD,
                claim="识别当前路径已不可能成功，放弃并回溯到更高层决策点",
                tags=["S21"],
                confidence=0.8,
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            lifecycle_payload = {
                "actions": [
                    {
                        "node_id": "strategy_S21",
                        "action_type": "revise_assumption",
                        "priority": 0.7,
                        "rationale": "conditioned utility failed",
                        "proposed_updates": {"expected_effect": "child should beat parent"},
                        "verification_plan": "test child against parent",
                        "rollback_condition": "reject weak child",
                        "source": {"decision": "revise"},
                    }
                ]
            }

            proposals = build_candidate_proposals(
                graph=graph,
                lifecycle_payload=lifecycle_payload,
                eval_id="unit_eval",
            )
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].proposal_type, ProposalType.ASSUMPTION_REVISION)
            self.assertEqual(proposals[0].parent_node_id, "strategy_S21")
            self.assertEqual(proposals[0].candidate_node["status"], "candidate")
            self.assertIn("failure thresholds", proposals[0].candidate_node["claim"])
            self.assertIn("strategy_S21", graph.store.nodes)
            self.assertNotIn(proposals[0].candidate_node["id"], graph.store.nodes)

    def test_novelty_integration_gate_classifies_candidate_family(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="parent_control",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.CLAIM,
                claim="Use controlled variable reasoning by keeping a baseline and changing one factor.",
                tags=["control", "experiment"],
            ))
            store.upsert_node(AssumptionNode(
                id="existing_control_specific",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.CLAIM,
                claim="Use controlled-variable reasoning only when baseline, intervention, control, and measurement are explicit.",
                tags=["control", "experiment"],
            ))
            store.upsert_node(AssumptionNode(
                id="struct_pat_negative_feedback",
                type=AssumptionType.ALIGNMENT,
                kind=HypothesisKind.FORMAL_MAPPING,
                claim="Negative feedback preserves an invariant through an opposing response.",
                formal_form={
                    "formal_kind": "structural_pattern",
                    "pattern_id": "pat_negative_feedback",
                    "objects": ["perturbation", "opposing_response", "invariant"],
                    "morphisms": ["disturbs", "opposes", "restores"],
                },
                tags=["structural_pattern"],
            ))
            store.upsert_node(AssumptionNode(
                id="parent_world_model",
                type=AssumptionType.WORLD_MODEL,
                kind=HypothesisKind.CLAIM,
                claim="A calibrated cheap world model should predict candidate acceptance before expensive ablation.",
                tags=["world_model", "calibration"],
            ))
            duplicate = AssumptionNode(
                id="cand_duplicate",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.CLAIM,
                claim="Use controlled-variable reasoning only when baseline, intervention, control, and measurement are explicit.",
                tags=["control", "candidate"],
                status="candidate",
            )
            child = AssumptionNode(
                id="cand_child",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.CLAIM,
                claim="Use controlled variable reasoning for debugging after naming baseline, one changed factor, and falsifying measurement.",
                tags=["control", "candidate"],
                status="candidate",
            )
            formal = AssumptionNode(
                id="cand_formal",
                type=AssumptionType.ALIGNMENT,
                kind=HypothesisKind.FORMAL_MAPPING,
                claim="Transfer negative feedback to a domain with compensating repair.",
                formal_form={
                    "formal_kind": "structural_morphism_candidate",
                    "source_pattern_id": "pat_negative_feedback",
                    "score": {"score": 0.86},
                    "functor_check": {"pass": True},
                    "kernel_check": {"pass": True},
                },
                status="candidate",
                tags=["structural_morphism"],
            )
            new_family = AssumptionNode(
                id="cand_new",
                type=AssumptionType.WORLD_MODEL,
                kind=HypothesisKind.CLAIM,
                claim="Model cryogenic sensor drift as a temperature latency manifold.",
                status="candidate",
                tags=["cryogenic"],
            )
            orthogonal = AssumptionNode(
                id="cand_orthogonal",
                type=AssumptionType.EVALUATOR,
                kind=HypothesisKind.EVALUATOR_POLICY,
                claim=(
                    "Before changing the task strategy, test whether stale judge feedback caused the failure by "
                    "tracking evaluator disagreement drift."
                ),
                status="candidate",
                tags=["evaluator", "feedback", "orthogonal"],
                payload={"orthogonal_to_existing": True},
            )
            same_family_not_orthogonal = AssumptionNode(
                id="cand_same_family_not_orthogonal",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.CLAIM,
                claim="Diagnose assay batches by varying only incubation temperature against a fixed reference batch.",
                status="candidate",
                tags=["control", "orthogonal"],
                payload={"orthogonal_to_existing": True},
            )
            same_family_alias_not_orthogonal = AssumptionNode(
                id="cand_same_family_alias_not_orthogonal",
                type=AssumptionType.WORLD_MODEL,
                kind=HypothesisKind.CLAIM,
                claim="Route-policy repairs should use a world-model screen before spending fresh ablation calls.",
                status="candidate",
                tags=["world_model_screen", "orthogonal"],
                payload={"orthogonal_to_existing": True},
            )
            proposals = {
                "eval_id": "unit_novelty",
                "proposals": [
                    {
                        "proposal_id": "prop_dup",
                        "proposal_type": "assumption_revision",
                        "parent_node_id": "existing_control_specific",
                        "candidate_node": duplicate.to_dict(),
                    },
                    {
                        "proposal_id": "prop_child",
                        "proposal_type": "scope_narrowing",
                        "parent_node_id": "parent_control",
                        "candidate_node": child.to_dict(),
                        "edges": [{"source": "cand_child", "target": "parent_control", "type": "specializes"}],
                    },
                    {
                        "proposal_id": "prop_formal",
                        "proposal_type": "structural_transfer_hypothesis",
                        "parent_node_id": "struct_pat_negative_feedback",
                        "candidate_node": formal.to_dict(),
                    },
                    {
                        "proposal_id": "prop_new",
                        "proposal_type": "failure_hypothesis",
                        "parent_node_id": "",
                        "candidate_node": new_family.to_dict(),
                    },
                    {
                        "proposal_id": "prop_orthogonal",
                        "proposal_type": "orthogonal_failure_hypothesis",
                        "parent_node_id": "parent_control",
                        "candidate_node": orthogonal.to_dict(),
                        "edges": [{"source": "cand_orthogonal", "target": "parent_control", "type": "generated_from_residual"}],
                    },
                    {
                        "proposal_id": "prop_same_family_not_orthogonal",
                        "proposal_type": "orthogonal_failure_hypothesis",
                        "parent_node_id": "parent_control",
                        "candidate_node": same_family_not_orthogonal.to_dict(),
                        "edges": [{
                            "source": "cand_same_family_not_orthogonal",
                            "target": "parent_control",
                            "type": "generated_from_residual",
                        }],
                    },
                    {
                        "proposal_id": "prop_same_family_alias_not_orthogonal",
                        "proposal_type": "orthogonal_failure_hypothesis",
                        "parent_node_id": "parent_world_model",
                        "candidate_node": same_family_alias_not_orthogonal.to_dict(),
                        "edges": [{
                            "source": "cand_same_family_alias_not_orthogonal",
                            "target": "parent_world_model",
                            "type": "generated_from_residual",
                        }],
                    },
                ],
            }
            payload = build_novelty_integration_payload(store, proposals, eval_id="unit_novelty_gate")
            self.assertTrue(payload["pass"])
            rows = {row["proposal_id"]: row for row in payload["rows"]}
            self.assertEqual(rows["prop_dup"]["classification"], "duplicate")
            self.assertEqual(rows["prop_child"]["classification"], "specialization")
            self.assertEqual(rows["prop_child"]["integration_edges"][0]["type"], "specializes")
            self.assertEqual(rows["prop_formal"]["classification"], "formal_isomorphism")
            self.assertEqual(rows["prop_formal"]["integration_edges"][0]["type"], "is_formal_isomorphism_of")
            self.assertEqual(rows["prop_new"]["classification"], "genuinely_new_family")
            self.assertEqual(rows["prop_orthogonal"]["classification"], "orthogonal_new_family")
            self.assertTrue(rows["prop_orthogonal"]["is_new_family"])
            self.assertEqual(rows["prop_orthogonal"]["integration_edges"][0]["type"], "orthogonal_to")
            self.assertGreaterEqual(rows["prop_orthogonal"]["match_score"], 0.58)
            self.assertNotEqual(rows["prop_same_family_not_orthogonal"]["classification"], "orthogonal_new_family")
            self.assertEqual(rows["prop_same_family_not_orthogonal"]["classification"], "specialization")
            self.assertNotEqual(rows["prop_same_family_alias_not_orthogonal"]["classification"], "orthogonal_new_family")
            self.assertEqual(rows["prop_same_family_alias_not_orthogonal"]["classification"], "specialization")

    def test_novelty_integration_performance_validation_passes(self):
        payload = build_novelty_integration_performance_payload(eval_id="unit_novelty_perf")
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertEqual(payload["gold_accuracy"], 1.0)
        self.assertEqual(payload["classification_counts"]["duplicate"], 1)
        self.assertEqual(payload["classification_counts"]["formal_isomorphism"], 1)
        self.assertEqual(payload["classification_counts"]["analogy"], 1)
        self.assertEqual(payload["classification_counts"]["orthogonal_new_family"], 1)
        self.assertGreaterEqual(payload["recommended_edge_counts"]["orthogonal_to"], 1)

    def test_orthogonal_gate_uses_informative_overlap_not_stopword_noise(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S26",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.CLAIM,
                claim="Identify lock-in caused by prior choices, switching cost, and self-reinforcing adoption.",
                tags=["strategy", "S26", "path_dependency"],
            ))
            store.upsert_node(AssumptionNode(
                id="hyp_generic_decomposition",
                type=AssumptionType.METHOD,
                kind=HypothesisKind.DECOMPOSITION,
                claim=(
                    "This decomposition operationalizes the wisdom by forcing the solver to diagnose the "
                    "incentive structure behind the symptom before suggesting any surface-level communication fix."
                ),
                context_conditions=[
                    "the candidate should test whether the answer is solving the wrong problem",
                ],
                predicted_effects=[
                    "correctness should increase by at least 0.05",
                ],
                tags=["exp82", "decomposition", "accepted"],
            ))
            store.flush()
            candidate = AssumptionNode(
                id="cand_verbose_provenance_axis",
                type=AssumptionType.MEMORY,
                kind=HypothesisKind.CLAIM,
                claim=(
                    "Before revising the intervention logic, test whether stale provenance records or missing "
                    "archival decision notes caused the residual; recover the source ledger and hidden "
                    "commitments first."
                ),
                context_conditions=[
                    "current-state evidence conflicts with archived source records",
                    "the system carries hidden commitments that are not visible in the immediate task text",
                ],
                predicted_effects=[
                    "avoid strategy churn when the real failure is missing provenance context",
                    "increase later proposal quality by preserving source-ledger memory",
                ],
                tags=["candidate", "orthogonal", "provenance_archive", "source_ledger"],
                residual_ids=["res_verbose_provenance_axis"],
                payload={"orthogonal_to_existing": True},
            )
            proposals = {
                "eval_id": "unit_orthogonal_stopword_noise",
                "proposals": [{
                    "proposal_id": "prop_verbose_provenance_axis",
                    "proposal_type": "orthogonal_failure_hypothesis",
                    "parent_node_id": "strategy_S26",
                    "candidate_node": candidate.to_dict(),
                    "edges": [{
                        "source": candidate.id,
                        "target": "strategy_S26",
                        "type": "generated_from_residual",
                    }],
                }],
            }
            payload = build_novelty_integration_payload(
                store,
                proposals,
                eval_id="unit_orthogonal_stopword_noise",
            )
            row = payload["rows"][0]
            self.assertEqual(row["classification"], "orthogonal_new_family")
            self.assertEqual(row["integration_edges"][0]["type"], "orthogonal_to")
            self.assertEqual(row["match_basis"], "orthogonal_informative_low_overlap")

    def test_orthogonal_ablation_validates_new_axis_retention(self):
        payload = build_orthogonal_ablation_payload(eval_id="unit_orthogonal_ablation")
        self.assertTrue(payload["pass"], payload["failed_gates"])
        metrics = payload["metrics"]
        self.assertEqual(metrics["classification_accuracy_enabled"], 1.0)
        self.assertLess(metrics["classification_accuracy_disabled"], 1.0)
        self.assertEqual(metrics["orthogonal_recall_enabled"], 1.0)
        self.assertEqual(metrics["orthogonal_recall_disabled"], 0.0)
        self.assertEqual(metrics["orthogonal_edge_count_enabled"], 1)
        self.assertEqual(metrics["orthogonal_edge_count_disabled"], 0)
        self.assertEqual(metrics["non_orthogonal_stability"], 1.0)
        self.assertGreater(metrics["axis_retention_delta"], 0.0)
        self.assertGreater(metrics["metaproductivity_proxy_delta"], 0.0)
        self.assertTrue(payload["retention"]["enabled"]["orthogonal_new_axis_retained"])
        self.assertFalse(payload["retention"]["disabled"]["orthogonal_new_axis_retained"])

    def test_orthogonal_surface_ablation_blocks_same_family_alias_false_positive(self):
        payload = build_orthogonal_surface_ablation_payload(
            root=Path("."),
            eval_id="unit_orthogonal_surface_ablation",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        metrics = payload["metrics"]
        self.assertGreaterEqual(metrics["proposal_count"], 7)
        self.assertGreaterEqual(metrics["same_family_alias_or_tag_count"], 5)
        self.assertGreaterEqual(metrics["ready_same_family_alias_or_tag_count"], 2)
        self.assertEqual(metrics["same_family_false_orthogonal_enabled_count"], 0)
        self.assertEqual(metrics["orthogonal_edge_enabled_count"], 0)
        self.assertEqual(metrics["classification_change_count"], 0)

    def test_orthogonal_downstream_ablation_preserves_judged_negative_control(self):
        payload = build_orthogonal_downstream_ablation_payload(
            root=Path("."),
            eval_id="unit_orthogonal_downstream_ablation",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        metrics = payload["metrics"]
        self.assertGreaterEqual(metrics["judged_proposal_count"], 6)
        self.assertEqual(metrics["judged_classification_change_count"], 0)
        self.assertEqual(metrics["judged_false_orthogonal_count"], 0)
        self.assertEqual(metrics["enabled_orthogonal_edge_count_all_proposals"], 0)
        self.assertFalse(payload["positive_live_gap"]["available"])
        self.assertEqual(payload["status"], "negative_control_pass_positive_live_pending")

    def test_orthogonal_positive_queue_is_live_ready_without_secret_values(self):
        payload = build_orthogonal_positive_queue_payload(
            root=Path("."),
            eval_id="unit_orthogonal_positive_queue",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertIn(payload["status"], {"live_ready", "live_ready_env_missing"})
        metrics = payload["metrics"]
        self.assertEqual(metrics["proposal_count"], 1)
        self.assertEqual(metrics["enabled_orthogonal_count"], 1)
        self.assertEqual(metrics["disabled_orthogonal_count"], 0)
        self.assertEqual(metrics["enabled_orthogonal_edge_count"], 1)
        self.assertEqual(metrics["disabled_orthogonal_edge_count"], 0)
        self.assertEqual(metrics["preflight_ready_count"], 1)
        self.assertGreaterEqual(metrics["trigger_count"], 3)
        self.assertGreaterEqual(metrics["active_trigger_count"], 3)
        self.assertGreaterEqual(metrics["control_count"], 3)
        self.assertEqual(metrics["outside_active_count"], 0)
        self.assertEqual(
            payload["preflight_summary"]["readiness"],
            "ready_for_fresh_ablation",
        )
        commands_text = json.dumps(payload["next_commands"], ensure_ascii=False)
        self.assertIn("<set-in-env>", commands_text)
        self.assertNotIn("sk-", commands_text)
        self.assertNotIn("newapi_channel_conn", commands_text)

    def test_orthogonal_positive_readback_bridges_queue_to_daemon_apply(self):
        payload = build_orthogonal_positive_readback_payload(
            root=Path("."),
            eval_id="unit_orthogonal_positive_readback",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(payload["status"], "readback_bridge_pass_live_judgment_pending")
        metrics = payload["metrics"]
        self.assertEqual(metrics["ready_queue_count"], 1)
        self.assertEqual(metrics["dry_planned_leaf_count"], 1)
        self.assertEqual(metrics["dry_executable_leaf_count"], 1)
        self.assertEqual(metrics["dry_status_counts"], {"planned": 1})
        self.assertEqual(metrics["readback_accept_count"], 1)
        self.assertTrue(metrics["readback_resumed"])
        self.assertEqual(metrics["readback_applied_count"], 0)
        self.assertFalse(metrics["node_mutation_without_apply"])
        self.assertEqual(metrics["apply_accept_count"], 1)
        self.assertTrue(metrics["apply_resumed"])
        self.assertEqual(metrics["apply_applied_count"], 1)
        self.assertTrue(metrics["candidate_node_present_after_apply"])
        self.assertGreaterEqual(metrics["orthogonal_edge_count_after_apply"], 1)

    def test_orthogonal_multi_cluster_validates_multiple_new_axes(self):
        payload = build_orthogonal_multi_cluster_payload(
            root=Path("."),
            eval_id="unit_orthogonal_multi_cluster",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertIn(payload["status"], {"multi_cluster_live_ready", "multi_cluster_live_ready_env_missing"})
        metrics = payload["metrics"]
        self.assertEqual(metrics["proposal_count"], 3)
        self.assertEqual(metrics["distinct_parent_count"], 3)
        self.assertEqual(metrics["enabled_orthogonal_count"], 3)
        self.assertEqual(metrics["disabled_orthogonal_count"], 0)
        self.assertEqual(metrics["enabled_orthogonal_edge_count"], 3)
        self.assertEqual(metrics["disabled_orthogonal_edge_count"], 0)
        self.assertEqual(metrics["preflight_ready_count"], 3)
        self.assertGreaterEqual(metrics["min_trigger_count"], 3)
        self.assertGreaterEqual(metrics["min_active_trigger_count"], 3)
        self.assertGreaterEqual(metrics["min_control_count"], 3)
        self.assertEqual(metrics["outside_active_total"], 0)
        self.assertEqual(metrics["dry_planned_leaf_count"], 3)
        self.assertEqual(metrics["dry_executable_leaf_count"], 3)
        self.assertEqual(metrics["dry_status_counts"], {"planned": 3})
        self.assertEqual(metrics["readback_accept_count"], 3)
        self.assertTrue(metrics["readback_resumed"])
        self.assertEqual(metrics["readback_applied_count"], 0)
        self.assertFalse(metrics["node_mutation_without_apply"])
        self.assertEqual(metrics["apply_accept_count"], 3)
        self.assertTrue(metrics["apply_resumed"])
        self.assertEqual(metrics["apply_applied_count"], 3)
        self.assertEqual(metrics["temp_candidate_node_count"], 3)
        self.assertGreaterEqual(metrics["temp_orthogonal_edge_count"], 3)
        enabled_rows = payload["novelty_rows"]["enabled"]
        disabled_rows = payload["novelty_rows"]["disabled"]
        self.assertTrue(all(row["classification"] == "orthogonal_new_family" for row in enabled_rows))
        self.assertTrue(all(row["classification"] != "orthogonal_new_family" for row in disabled_rows))
        commands_text = json.dumps(payload["next_commands"], ensure_ascii=False)
        self.assertIn("<set-in-env>", commands_text)
        self.assertNotIn("sk-", commands_text)
        self.assertNotIn("newapi_channel_conn", commands_text)

    def test_orthogonal_execution_queue_validates_expanded_trigger_contract(self):
        payload = build_orthogonal_execution_queue_payload(
            root=Path("."),
            eval_id="unit_orthogonal_execution_queue",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertIn(
            payload["status"],
            {"execution_queue_live_ready", "execution_queue_live_ready_env_missing"},
        )
        metrics = payload["metrics"]
        self.assertEqual(metrics["proposal_count"], 1)
        self.assertEqual(metrics["enabled_orthogonal_count"], 1)
        self.assertEqual(metrics["disabled_orthogonal_count"], 0)
        self.assertEqual(metrics["enabled_orthogonal_edge_count"], 1)
        self.assertEqual(metrics["disabled_orthogonal_edge_count"], 0)
        self.assertEqual(metrics["preflight_ready_count"], 1)
        self.assertGreaterEqual(metrics["trigger_count"], 8)
        self.assertGreaterEqual(metrics["active_trigger_count"], 8)
        self.assertGreaterEqual(metrics["control_count"], 8)
        self.assertEqual(metrics["outside_active_count"], 0)
        self.assertEqual(metrics["readback_accept_count"], 1)
        self.assertEqual(metrics["readback_applied_count"], 0)
        self.assertFalse(metrics["node_mutation_without_apply"])
        self.assertEqual(metrics["apply_accept_count"], 1)
        self.assertEqual(metrics["apply_applied_count"], 1)
        self.assertTrue(metrics["candidate_node_present_after_apply"])
        self.assertGreaterEqual(metrics["temp_orthogonal_edge_count"], 1)
        enabled_row = payload["novelty_rows"]["enabled"][0]
        disabled_row = payload["novelty_rows"]["disabled"][0]
        self.assertEqual(enabled_row["classification"], "orthogonal_new_family")
        self.assertNotEqual(disabled_row["classification"], "orthogonal_new_family")
        commands_text = json.dumps(payload["next_commands"], ensure_ascii=False)
        self.assertIn("<set-in-env>", commands_text)
        self.assertNotIn("sk-", commands_text)
        self.assertNotIn("newapi_channel_conn", commands_text)

    def test_orthogonal_recursive_ablation_keeps_live_positive_axis_separate(self):
        payload = build_orthogonal_recursive_ablation_payload(
            root=Path("."),
            eval_id="unit_orthogonal_recursive_ablation",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertEqual(payload["live_acceptance"]["decision_counts"], {"accept": 1})
        self.assertEqual(payload["live_outcome_metrics"]["trigger_outcomes"], {"win": 3, "tie": 2})
        self.assertEqual(payload["live_outcome_metrics"]["control_outcomes"], {"tie": 8})
        on = payload["conditions"]["orthogonal_on"]
        off = payload["conditions"]["orthogonal_off"]
        self.assertEqual(on["novelty_classification"], "orthogonal_new_family")
        self.assertNotEqual(off["novelty_classification"], "orthogonal_new_family")
        self.assertGreaterEqual(on["applied_graph"]["orthogonal_to_edge_count"], 1)
        self.assertEqual(off["applied_graph"]["orthogonal_to_edge_count"], 0)
        self.assertEqual(off["applied_graph"]["specializes_edge_count"], 1)
        self.assertGreater(payload["comparison"]["recursive_retention_delta"], 0.0)
        self.assertEqual(payload["comparison"]["downstream_utility_delta"], 0.0)
        self.assertTrue(on["daemon"]["resumed"])
        self.assertTrue(off["daemon"]["resumed"])

    def test_orthogonal_descendant_productivity_improves_over_three_generations(self):
        payload = build_orthogonal_descendant_productivity_payload(
            root=Path("."),
            eval_id="unit_orthogonal_descendant_productivity",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        comparison = payload["comparison"]
        self.assertEqual(comparison["accepted_descendant_on"], 5)
        self.assertEqual(comparison["accepted_descendant_off"], 2)
        self.assertEqual(comparison["accepted_descendant_delta"], 3)
        self.assertLess(comparison["reject_harm_delta_on_minus_off"], 0)
        self.assertGreater(comparison["productivity_score_delta"], 0.2)
        self.assertGreater(comparison["acp_score_delta"], 0.0)
        on = payload["conditions"]["orthogonal_on"]
        off = payload["conditions"]["orthogonal_off"]
        self.assertEqual(on["generation_count"], 3)
        self.assertEqual(off["generation_count"], 3)
        self.assertGreaterEqual(on["seed_graph_state"]["orthogonal_to_edge_count"], 1)
        self.assertEqual(off["seed_graph_state"]["orthogonal_to_edge_count"], 0)
        self.assertGreater(
            off["metrics"]["old_parent_descendant_labels"],
            on["metrics"]["old_parent_descendant_labels"],
        )

    def test_pre_live_tie_screen_preserves_positive_and_saves_failed_descendant_calls(self):
        payload = build_pre_live_tie_screen_payload(
            root=Path("."),
            eval_id="unit_pre_live_tie_screen",
        )
        chronological = payload["metrics"]["chronological"]
        no_screen = payload["metrics"]["no_screen"]
        rows = {row["case"]["proposal_id"]: row for row in payload["rows"]}

        self.assertTrue(payload["pass"])
        self.assertEqual(no_screen["live_calls"], 7)
        self.assertEqual(no_screen["accepted_count"], 1)
        self.assertEqual(chronological["positive_control_allowed_count"], 1)
        self.assertEqual(chronological["accepted_positive_block_count"], 0)
        self.assertGreaterEqual(chronological["failed_live_calls_saved"], 4)
        self.assertGreater(chronological["accepted_rate_among_run_calls"], no_screen["accepted_rate"])
        self.assertTrue(rows["prop_d7abf65010d2"]["screen"]["would_run_live"])
        self.assertFalse(rows["prop_99b7c2f9b052"]["screen"]["would_run_live"])
        self.assertFalse(rows["prop_6c22137d982d_vs_parent"]["screen"]["would_run_live"])
        self.assertEqual(
            rows["prop_99b7c2f9b052"]["screen"]["decision"],
            "block_predicted_low_benefit",
        )

    def test_orthogonal_descendant_live_queue_exports_retained_graph_specialization(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            payload = build_orthogonal_descendant_live_queue_payload(
                root=Path("."),
                eval_id="unit_orthogonal_descendant_live_queue",
                retained_graph_dir=work / "retained_graph",
                proposals_out=work / "descendant_proposals.json",
                preflight_out=work / "descendant_preflight.json",
            )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        self.assertIn(payload["status"], {"live_ready", "live_ready_env_missing"})
        retained = payload["source"]["retained_graph_snapshot"]
        self.assertFalse(retained["main_graph_mutated"])
        self.assertEqual(retained["seed_candidate_node_id"], "cand_39de0aeae8a3")
        self.assertGreaterEqual(retained["seed_edge_counts"].get("orthogonal_to", 0), 1)
        self.assertEqual(payload["novelty_summary"]["classification_counts"], {"specialization": 1})
        self.assertEqual(payload["novelty_summary"]["recommended_edge_counts"], {"specializes": 1})
        metrics = payload["metrics"]
        self.assertEqual(metrics["proposal_count"], 1)
        self.assertEqual(metrics["preflight_ready_count"], 1)
        self.assertEqual(metrics["trigger_count"], 5)
        self.assertEqual(metrics["active_trigger_count"], 5)
        self.assertGreaterEqual(metrics["control_count"], 8)
        self.assertEqual(metrics["outside_active_count"], 0)
        self.assertEqual(metrics["readback_accept_count"], 1)
        self.assertEqual(metrics["readback_applied_count"], 0)
        self.assertEqual(metrics["apply_accept_count"], 1)
        self.assertEqual(metrics["apply_applied_count"], 1)
        self.assertTrue(metrics["candidate_node_present_after_apply"])
        commands_text = json.dumps(payload["next_commands"], ensure_ascii=False)
        self.assertIn("<set-in-env>", commands_text)
        self.assertNotIn("sk-", commands_text)
        self.assertNotIn("newapi_channel_conn", commands_text)

    def test_orthogonal_descendant_live_readback_applies_real_accepted_judgment(self):
        payload = build_orthogonal_descendant_live_readback_payload(
            root=Path("."),
            eval_id="unit_orthogonal_descendant_live_readback",
        )
        self.assertTrue(payload["pass"], payload["failed_gates"])
        metrics = payload["metrics"]
        self.assertEqual(metrics["live_status"], "live_positive_acceptance")
        self.assertEqual(metrics["acceptance_decision_counts"], {"accept": 1})
        self.assertEqual(metrics["readback_accept_count"], 1)
        self.assertEqual(metrics["readback_applied_count"], 0)
        self.assertEqual(metrics["apply_accept_count"], 1)
        self.assertEqual(metrics["apply_applied_count"], 1)
        self.assertFalse(metrics["node_mutation_without_apply"])
        self.assertTrue(metrics["candidate_node_present_after_apply"])
        self.assertEqual(metrics["candidate_status_after_apply"], "active")
        self.assertGreaterEqual(metrics["candidate_edge_counts_after_apply"].get("specializes", 0), 1)
        self.assertFalse(metrics["original_retained_graph_has_candidate"])

    def test_proposal_overlay_is_in_memory_only(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_S08",
                type=AssumptionType.METHOD,
                claim="提出猜测并测试",
                tags=["S08"],
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_S08",
                    "action_type": "expand_retrieval",
                    "priority": 0.8,
                    "rationale": "useful but under-retrieved",
                    "proposed_updates": {"expected_effect": "increase trigger coverage"},
                    "verification_plan": "retrieval audit",
                    "rollback_condition": "outside harm",
                    "source": {
                        "decision": "expand_retrieval",
                        "utility_lcb90": 1.0,
                        "route_counts": {"should_fire": 4},
                        "active_counts": {"should_fire": 1},
                    },
                }]
            }
            proposals = build_candidate_proposals(graph=graph, lifecycle_payload=lifecycle_payload, eval_id="unit_eval")
            payload = {"proposals": [p.to_dict() for p in proposals]}

            overlay_store = JsonlGraphStore(td)
            applied = apply_proposal_overlay(overlay_store, payload)
            self.assertEqual(len(applied), 1)
            self.assertEqual(proposal_candidate_ids(payload), applied)
            self.assertIn(applied[0], overlay_store.nodes)
            self.assertNotIn(applied[0], JsonlGraphStore(td).nodes)

    def test_proposal_contract_checked_overlay_quarantines_invalid_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(td)
            store.upsert_node(AssumptionNode(
                id="strategy_contract",
                type=AssumptionType.METHOD,
                claim="contract checked proposal parent",
                tags=["contract"],
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(td))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_contract",
                    "action_type": "expand_retrieval",
                    "priority": 0.8,
                    "rationale": "valid contract candidate",
                    "proposed_updates": {"expected_effect": "increase trigger coverage"},
                    "verification_plan": "retrieval audit with outside negative control",
                    "rollback_condition": "rollback if outside harm appears",
                    "source": {
                        "decision": "expand_retrieval",
                        "utility_lcb90": 1.0,
                        "route_counts": {"should_fire": 4},
                        "active_counts": {"should_fire": 1},
                    },
                }]
            }
            proposals = build_candidate_proposals(
                graph=graph,
                lifecycle_payload=lifecycle_payload,
                eval_id="unit_contract",
            )
            valid = proposals[0].to_dict()
            invalid = json.loads(json.dumps(valid))
            invalid["proposal_id"] = "prop_contract_invalid"
            invalid["candidate_node"]["id"] = "cand_contract_invalid"
            invalid["candidate_node"]["verifiers"] = ["conditioned_eval_gate"]
            invalid["candidate_node"]["risk_predictions"] = ["may overreach"]
            invalid["edges"][0]["target"] = "cand_contract_invalid"
            invalid["manifest"]["rollback_condition"] = ""
            invalid["manifest"]["verification_plan"] = "retrieval audit"
            payload = {"eval_id": "unit_contract_payload", "proposals": [valid, invalid]}

            contract = build_proposal_contract_payload(
                proposal_payload=payload,
                eval_id="unit_contract_gate",
                store=JsonlGraphStore(td),
            )
            self.assertTrue(contract["pass"], contract["failed_gates"])
            self.assertEqual(contract["metrics"]["proposal_count"], 2)
            self.assertEqual(contract["metrics"]["admitted_count"], 1)
            self.assertEqual(contract["metrics"]["quarantined_count"], 1)
            self.assertEqual(contract["admitted_proposal_ids"], [valid["proposal_id"]])
            self.assertEqual(contract["quarantined_proposal_ids"], ["prop_contract_invalid"])
            invalid_result = {
                row["proposal_id"]: row for row in contract["results"]
            }["prop_contract_invalid"]
            self.assertIn("missing_rollback", invalid_result["issues"])
            self.assertIn("missing_negative_control", invalid_result["issues"])

            overlay_store = JsonlGraphStore(td)
            applied, applied_contract = apply_contract_checked_proposal_overlay(overlay_store, payload)
            self.assertEqual(applied, [valid["candidate_node"]["id"]])
            self.assertEqual(applied_contract["quarantined_proposal_ids"], ["prop_contract_invalid"])
            self.assertIn(valid["candidate_node"]["id"], overlay_store.nodes)
            self.assertNotIn("cand_contract_invalid", overlay_store.nodes)
            self.assertNotIn(valid["candidate_node"]["id"], JsonlGraphStore(td).nodes)

    def test_candidate_eval_preflight_marks_ready_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = JsonlGraphStore(root / "graph")
            store.upsert_node(AssumptionNode(
                id="strategy_S08",
                type=AssumptionType.METHOD,
                claim="提出猜测并测试",
                tags=["S08", "假设检验"],
                confidence=0.7,
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(root / "graph"))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_S08",
                    "action_type": "revise_assumption",
                    "priority": 0.7,
                    "rationale": "weak conditioned utility",
                    "proposed_updates": {"expected_effect": "child should beat parent"},
                    "verification_plan": "fresh ablation",
                    "rollback_condition": "reject weak child",
                    "source": {"decision": "revise", "utility_lcb90": 0.1},
                }]
            }
            proposals = build_candidate_proposals(graph=graph, lifecycle_payload=lifecycle_payload, eval_id="unit_eval")
            payload = {"eval_id": "unit_eval", "proposals": [p.to_dict() for p in proposals]}
            sample = [
                {
                    "problem_id": "p1",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "用一个低成本假设检验测试新渠道是否有效。",
                    "coverage_tags": ["S08"],
                },
                {
                    "problem_id": "p2",
                    "domain": "business",
                    "difficulty": "medium",
                    "description": "先提出可证伪假设，再用小样本测试。",
                    "coverage_tags": ["S08"],
                },
                {
                    "problem_id": "p3",
                    "domain": "engineering",
                    "difficulty": "medium",
                    "description": "对泵站故障提出可能原因并逐项测试。",
                    "coverage_tags": ["S08"],
                },
            ]
            meta = {p["problem_id"]: {"frame": "hybrid", "critical_reframe": "", "rewritten_problem": p["description"]} for p in sample}

            result = build_candidate_eval_payload(
                graph_dir=root / "graph",
                proposal_payload=payload,
                sample=sample,
                meta_by_pid=meta,
                eval_id="unit_preflight",
                min_trigger_n=3,
                min_active_trigger_n=2,
            )
            summary = result["summaries"][0]
            self.assertEqual(summary["readiness"], CandidateReadiness.READY_FOR_FRESH_ABLATION.value)
            self.assertGreaterEqual(len(summary["active_trigger_problem_ids"]), 2)

    def test_candidate_acceptance_gate_applies_only_accepted_children(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = JsonlGraphStore(root / "graph")
            store.upsert_node(AssumptionNode(
                id="strategy_S01",
                type=AssumptionType.METHOD,
                claim="control one variable",
                tags=["S01"],
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(root / "graph"))
            lifecycle_payload = {
                "actions": [{
                    "node_id": "strategy_S01",
                    "action_type": "revise_assumption",
                    "priority": 0.7,
                    "rationale": "weak conditioned utility",
                    "proposed_updates": {"expected_effect": "child should beat parent"},
                    "verification_plan": "fresh ablation",
                    "rollback_condition": "reject weak child",
                    "source": {"decision": "revise", "utility_lcb90": 0.1},
                }]
            }
            proposals = build_candidate_proposals(graph=graph, lifecycle_payload=lifecycle_payload, eval_id="unit_eval")
            proposal_payload = {"eval_id": "unit_eval", "proposals": [p.to_dict() for p in proposals]}
            proposal_id = proposals[0].proposal_id
            candidate_id = proposals[0].candidate_node["id"]
            preflight_payload = {
                "eval_id": "unit_preflight",
                "summaries": [{
                    "proposal_id": proposal_id,
                    "readiness": "ready_for_fresh_ablation",
                    "trigger_problem_ids": ["p1", "p2", "p3"],
                    "control_problem_ids": ["p4", "p5", "p6"],
                }],
            }
            judgment_path = root / "judgments.json"
            judgment_path.write_text(json.dumps({
                "p1": {"winner": "candidate"},
                "p2": {"winner": "candidate"},
                "p3": {"winner": "candidate"},
                "p4": {"winner": "tie"},
                "p5": {"winner": "candidate"},
                "p6": {"winner": "tie"},
            }), encoding="utf-8")

            acceptance = build_acceptance_payload(
                proposal_payload=proposal_payload,
                preflight_payload=preflight_payload,
                judgment_paths=[judgment_path],
                candidate_variant="candidate",
                baseline_variant="baseline",
                eval_id="unit_accept",
            )
            self.assertEqual(acceptance["accepted_proposal_ids"], [proposal_id])
            self.assertEqual(acceptance["summaries"][0]["decision"], AcceptanceDecision.ACCEPT.value)

            applied = apply_accepted_candidates(JsonlGraphStore(root / "graph"), proposal_payload, acceptance)
            self.assertEqual(applied, [candidate_id])
            updated = JsonlGraphStore(root / "graph")
            self.assertEqual(updated.nodes[candidate_id].status, "active")

    def test_apply_accepted_candidates_uses_novelty_integration_edges(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = JsonlGraphStore(root / "graph")
            store.upsert_node(AssumptionNode(
                id="existing_parent",
                type=AssumptionType.METHOD,
                claim="Keep a known baseline and change one factor.",
                tags=["control"],
            ))
            store.upsert_node(AssumptionNode(
                id="struct_pat_negative_feedback",
                type=AssumptionType.ALIGNMENT,
                kind=HypothesisKind.FORMAL_MAPPING,
                claim="Negative feedback restores an invariant.",
                formal_form={"formal_kind": "structural_pattern", "pattern_id": "pat_negative_feedback"},
                tags=["structural_pattern"],
            ))
            store.flush()
            duplicate = AssumptionNode(
                id="cand_dup_apply",
                type=AssumptionType.METHOD,
                claim="Keep a known baseline and change one factor.",
                status="candidate",
                tags=["control"],
            )
            formal = AssumptionNode(
                id="cand_formal_apply",
                type=AssumptionType.ALIGNMENT,
                kind=HypothesisKind.FORMAL_MAPPING,
                claim="Apply negative feedback as a preserved structural morphism.",
                formal_form={
                    "formal_kind": "structural_morphism_candidate",
                    "source_pattern_id": "pat_negative_feedback",
                    "functor_check": {"pass": True},
                    "kernel_check": {"pass": True},
                    "score": {"score": 0.91},
                },
                status="candidate",
                tags=["structural_morphism"],
            )
            orthogonal = AssumptionNode(
                id="cand_orthogonal_apply",
                type=AssumptionType.EVALUATOR,
                kind=HypothesisKind.EVALUATOR_POLICY,
                claim=(
                    "Before changing the task strategy, check whether stale evaluator feedback explains the "
                    "same failure cluster."
                ),
                status="candidate",
                tags=["evaluator", "feedback", "orthogonal"],
                payload={"orthogonal_to_existing": True},
            )
            proposal_payload = {
                "eval_id": "unit_apply_novelty",
                "proposals": [
                    {
                        "proposal_id": "prop_dup_apply",
                        "proposal_type": "assumption_revision",
                        "parent_node_id": "existing_parent",
                        "candidate_node": duplicate.to_dict(),
                    },
                    {
                        "proposal_id": "prop_formal_apply",
                        "proposal_type": "structural_transfer_hypothesis",
                        "parent_node_id": "struct_pat_negative_feedback",
                        "candidate_node": formal.to_dict(),
                    },
                    {
                        "proposal_id": "prop_orthogonal_apply",
                        "proposal_type": "orthogonal_failure_hypothesis",
                        "parent_node_id": "existing_parent",
                        "candidate_node": orthogonal.to_dict(),
                        "edges": [{
                            "source": "cand_orthogonal_apply",
                            "target": "existing_parent",
                            "type": EdgeType.GENERATED_FROM_RESIDUAL.value,
                        }],
                    },
                ],
            }
            novelty = build_novelty_integration_payload(
                JsonlGraphStore(root / "graph"),
                proposal_payload,
                eval_id="unit_apply_novelty_gate",
            )
            acceptance = {
                "eval_id": "unit_accept_novelty",
                "accepted_proposal_ids": ["prop_dup_apply", "prop_formal_apply", "prop_orthogonal_apply"],
                "summaries": [
                    {"proposal_id": "prop_dup_apply", "decision": "accept"},
                    {"proposal_id": "prop_formal_apply", "decision": "accept"},
                    {"proposal_id": "prop_orthogonal_apply", "decision": "accept"},
                ],
            }
            applied = apply_accepted_candidates(
                JsonlGraphStore(root / "graph"),
                proposal_payload,
                acceptance,
                novelty,
            )
            updated = JsonlGraphStore(root / "graph")
            self.assertEqual(applied, ["cand_formal_apply", "cand_orthogonal_apply"])
            self.assertNotIn("cand_dup_apply", updated.nodes)
            self.assertIn("cand_formal_apply", updated.nodes)
            self.assertIn("cand_orthogonal_apply", updated.nodes)
            edge_keys = {
                (
                    edge.source,
                    edge.target,
                    edge.type.value if hasattr(edge.type, "value") else edge.type,
                )
                for edge in updated.edges
            }
            self.assertIn(
                ("cand_formal_apply", "struct_pat_negative_feedback", "is_formal_isomorphism_of"),
                edge_keys,
            )
            self.assertIn(("cand_orthogonal_apply", "existing_parent", "orthogonal_to"), edge_keys)

    def test_structural_morphism_evals_and_verifier_gate(self):
        self.assertTrue(build_structural_extraction_audit_payload(eval_id="unit_struct_extract")["pass"])
        self.assertTrue(build_structural_pair_eval_payload(eval_id="unit_struct_pairs")["pass"])
        self.assertTrue(build_nonlexical_structural_retrieval_probe_payload(eval_id="unit_struct_retrieval")["pass"])
        self.assertTrue(build_structural_behavior_probe_payload(eval_id="unit_struct_behavior")["pass"])
        self.assertTrue(build_structural_functor_eval_payload(eval_id="unit_struct_functor")["pass"])
        self.assertTrue(build_transfer_prediction_testability_eval_payload(eval_id="unit_struct_prediction")["pass"])
        self.assertTrue(build_structural_kernel_eval_payload(eval_id="unit_struct_kernel")["pass"])
        realization_eval = build_structural_realization_eval_payload(eval_id="unit_struct_realization")
        self.assertTrue(realization_eval["pass"])
        self.assertGreaterEqual(realization_eval["accepted_count"], 8)
        self.assertEqual(realization_eval["candidate_uncertainty_rate"], 1.0)
        self.assertTrue(realization_eval["negative_rejected"])
        context_effect = build_structural_context_effect_payload(eval_id="unit_struct_context")
        self.assertTrue(context_effect["pass"])

        good = search_structural_patterns(
            None,
            "Keep the baseline identity path, apply a residual delta correction, and keep fallback recovery.",
            top_n=1,
        )[0]
        good_payload = {
            "eval_id": "unit_struct_good",
            "proposals": [{
                "proposal_id": "prop_struct_good",
                "proposal_type": "structural_transfer_hypothesis",
                "parent_node_id": "parent",
                "candidate_node": {
                    "id": "cand_struct_good",
                    "formal_form": good["candidate"],
                },
            }],
        }
        good_gate = build_structural_morphism_gate_payload(
            proposal_payload=good_payload,
            eval_id="unit_struct_good_gate",
        )
        self.assertEqual(good_gate["gates"][0]["decision"], "allow")
        self.assertFalse(good_gate["gates"][0]["blocks_policy_update"])
        self.assertTrue(good_gate["gates"][0]["functor_check"]["pass"])
        self.assertTrue(good_gate["gates"][0]["kernel_check"]["pass"])
        context_validation = build_structural_context_validation_payload(
            good_payload,
            good_gate,
            context_effect,
            eval_id="unit_struct_context_validation",
        )
        self.assertTrue(context_validation["pass"])
        self.assertEqual(context_validation["decision_counts"], {"accept_context_effect": 1})

        bad = search_structural_patterns(
            None,
            "A Gaussian style prior is mentioned, but there is no predictable signal and the method memorizes noise.",
            top_n=1,
            min_score=0.0,
        )[0]
        bad_payload = {
            "eval_id": "unit_struct_bad",
            "proposals": [{
                "proposal_id": "prop_struct_bad",
                "proposal_type": "structural_transfer_hypothesis",
                "parent_node_id": "parent",
                "candidate_node": {
                    "id": "cand_struct_bad",
                    "formal_form": bad["candidate"],
                },
            }],
        }
        bad_gate = build_structural_morphism_gate_payload(
            proposal_payload=bad_payload,
            eval_id="unit_struct_bad_gate",
        )
        self.assertEqual(bad_gate["gates"][0]["decision"], "block_negative_control")
        self.assertTrue(bad_gate["gates"][0]["blocks_policy_update"])

        untestable_formal = dict(good["candidate"])
        untestable_formal["transfer_predictions"] = ["This analogy is elegant and might help."]
        untestable_formal["transfer_prediction_check"] = {
            "formal_kind": "transfer_prediction_testability",
            "pass": False,
            "reason": "unit-test untestable prediction",
        }
        untestable_gate = build_structural_morphism_gate_payload(
            proposal_payload={
                "eval_id": "unit_struct_untestable",
                "proposals": [{
                    "proposal_id": "prop_struct_untestable",
                    "proposal_type": "structural_transfer_hypothesis",
                    "parent_node_id": "parent",
                    "candidate_node": {"id": "cand_struct_untestable", "formal_form": untestable_formal},
                }],
            },
            eval_id="unit_struct_untestable_gate",
        )
        self.assertEqual(
            untestable_gate["gates"][0]["decision"],
            "repair_missing_testable_transfer_prediction",
        )
        self.assertTrue(untestable_gate["gates"][0]["blocks_policy_update"])

        kernel_bad_formal = dict(good["candidate"])
        kernel_bad_formal["kernel_check"] = {
            "formal_kind": "structural_role_transition_kernel_check",
            "pass": False,
            "reason": "unit-test kernel mismatch",
        }
        kernel_bad_gate = build_structural_morphism_gate_payload(
            proposal_payload={
                "eval_id": "unit_struct_kernel_bad",
                "proposals": [{
                    "proposal_id": "prop_struct_kernel_bad",
                    "proposal_type": "structural_transfer_hypothesis",
                    "parent_node_id": "parent",
                    "candidate_node": {"id": "cand_struct_kernel_bad", "formal_form": kernel_bad_formal},
                }],
            },
            eval_id="unit_struct_kernel_bad_gate",
        )
        self.assertEqual(
            kernel_bad_gate["gates"][0]["decision"],
            "repair_structural_kernel_not_preserved",
        )
        self.assertTrue(kernel_bad_gate["gates"][0]["blocks_policy_update"])

        verifier = build_verifier_stack_payload(
            proposal_payload=bad_payload,
            structural_morphism_gate_payload=bad_gate,
            eval_id="unit_struct_verifier",
        )
        self.assertEqual(verifier["verdict_counts"], {"blocked_structural_morphism_gate": 1})
        stages = verifier["summaries"][0]["stages"]
        self.assertIn("structural_morphism_gate", [stage["name"] for stage in stages])

    def test_structural_morphism_diagram_functor_writeback_and_recursive_child(self):
        diagram = extract_structural_diagram(
            "Preserve the verified baseline identity path, learn a residual delta correction, "
            "compose the local patch, and recover old behavior through fallback."
        )
        roles = {row["role"] for row in diagram.objects}
        morphisms = {row["id"] for row in diagram.morphisms}
        self.assertIn("baseline_path", roles)
        self.assertIn("delta_update", roles)
        self.assertIn("compose_add", morphisms)
        self.assertGreaterEqual(len(diagram.composition_laws), 1)

        residual = search_structural_patterns(None, diagram, top_n=1)[0]
        functor = check_structural_functor(diagram, residual["candidate"]["source_diagram"])
        self.assertTrue(functor["pass"])
        self.assertGreaterEqual(functor["composition_preservation_rate"], 0.5)

        with tempfile.TemporaryDirectory() as td:
            graph_dir = Path(td) / "graph"
            store = JsonlGraphStore(graph_dir)
            seed_structural_patterns(store, persist=True)
            problem = (
                "Keep the verified baseline identity path, add a residual delta correction, "
                "and recover old behavior through fallback when the delta is zero."
            )
            proposal_payload = build_structural_transfer_proposal_payload(
                store,
                problem=problem,
                eval_id="unit_struct_prop",
                top_n=1,
            )
            gate = build_structural_morphism_gate_payload(
                proposal_payload=proposal_payload,
                eval_id="unit_struct_gate",
            )
            proposal_id = proposal_payload["proposals"][0]["proposal_id"]
            acceptance = {
                "eval_id": "unit_struct_accept",
                "accepted_proposal_ids": [proposal_id],
                "summaries": [{"proposal_id": proposal_id, "decision": "accept"}],
            }
            applied = apply_accepted_structural_morphisms(
                store,
                proposal_payload,
                gate,
                acceptance,
                persist=False,
            )
            self.assertEqual(len(applied), 1)
            lineage = build_structural_lineage_payload(store, eval_id="unit_struct_lineage")
            self.assertTrue(lineage["pass"])
            self.assertGreaterEqual(lineage["structural_morphism_count"], 1)

            recursive = build_recursive_assumption_run(
                graph_dir=graph_dir,
                problem=problem,
                goal="Validate the structural transfer hypothesis recursively before graph mutation.",
                eval_id="unit_struct_recursive",
                evolution_payload={
                    "eval_id": "unit_struct_recursive_source",
                    "proposals": proposal_payload,
                    "structural_morphism_gate": gate,
                },
                top_k=3,
                max_children=2,
                max_depth=2,
            )
            structural_children = [
                frame for frame in recursive["frames"]
                if frame["verifier"] == "structural_morphism_gate"
                and frame["frame_type"] == RecursiveFrameType.VERIFICATION_SUBPROBLEM.value
            ]
            self.assertEqual(len(structural_children), 1)
            self.assertEqual(structural_children[0]["next_action"], "run_structural_context_effect_validation")
            self.assertIn("assumption_os.structural_patterns", structural_children[0]["command_hint"])

            context_effect = build_structural_context_effect_payload(store, eval_id="unit_struct_context")
            context_validation = build_structural_context_validation_payload(
                proposal_payload,
                gate,
                context_effect,
                eval_id="unit_struct_context_validation",
            )
            self.assertEqual(context_validation["decision_counts"], {"accept_context_effect": 1})
            resumed = build_recursive_assumption_run(
                graph_dir=graph_dir,
                problem=problem,
                goal="Validate the structural transfer hypothesis recursively before graph mutation.",
                eval_id="unit_struct_recursive_resumed",
                evolution_payload={
                    "eval_id": "unit_struct_recursive_source",
                    "proposals": proposal_payload,
                    "structural_morphism_gate": gate,
                    "structural_context_validation": context_validation,
                },
                top_k=3,
                max_children=2,
                max_depth=2,
            )
            resumed_child = next(
                frame for frame in resumed["frames"]
                if frame["verifier"] == "structural_morphism_gate"
                and frame["frame_type"] == RecursiveFrameType.VERIFICATION_SUBPROBLEM.value
            )
            self.assertEqual(resumed_child["next_action"], "return_structural_context_effect_to_parent")
            self.assertEqual(
                resumed_child["return_update"]["outcome"],
                "structural_context_effect_passed",
            )
            self.assertEqual(
                resumed["next_actions"][0]["next_action"],
                "run_fresh_ablation_before_promotion",
            )

    def test_structural_morphism_performance_validation_passes(self):
        self.assertTrue(build_structural_writeback_eval_payload(eval_id="unit_struct_writeback")["pass"])
        perf = build_structural_morphism_performance_payload(eval_id="unit_struct_perf")
        self.assertTrue(perf["pass"], perf["component_pass"])
        self.assertTrue(all(perf["component_pass"].values()))

    def test_recursive_self_evolution_proof_payload_passes(self):
        payload = build_recursive_self_evolution_proof_payload(eval_id="unit_recursive_evolution_proof")
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertGreaterEqual(payload["generation_count"], 5)
        self.assertGreaterEqual(payload["rejected_branch_count"], 1)
        self.assertGreaterEqual(payload["metrics"]["best_base_delta"], 0.10)
        self.assertGreaterEqual(payload["metrics"]["final_placebo_delta"], 0.10)
        self.assertGreaterEqual(payload["metrics"]["bottleneck_branch_placebo_delta"], 0.20)

    def test_continuous_daemon_autonomy_audit_passes_budgeted_loop(self):
        perf = json.loads(Path(
            "phase four/assumption_graph/reconstruction_gap_perf_20260602_external_v5_objective.json"
        ).read_text(encoding="utf-8"))
        payload = build_continuous_daemon_autonomy_payload(
            eval_id="unit_continuous_daemon",
            recursive_daemon_section=perf["sections"]["recursive_daemon"],
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertTrue(payload["budgeted_continuous_mode"])
        self.assertFalse(payload["continuous_background_mode"])
        self.assertEqual(payload["ungated_graph_mutation_count"], 0)
        self.assertGreaterEqual(payload["cycle_count"], 5)
        self.assertGreaterEqual(payload["summary"]["preflight_queue_ready_count"], 5)
        self.assertGreaterEqual(payload["summary"]["real_artifact_readback_control_judgment_count"], 1)

    def test_first_party_world_model_scale_audit_passes_live_trace_gate(self):
        perf = json.loads(Path(
            "phase four/assumption_graph/reconstruction_gap_perf_20260602_external_v5_objective.json"
        ).read_text(encoding="utf-8"))
        precomputed = json.loads(Path(
            "phase four/assumption_graph/paper_readiness_20260604/first_party_world_model_scale_20260604.json"
        ).read_text(encoding="utf-8"))
        payload = build_first_party_world_model_scale_payload(
            eval_id="unit_first_party_world_model",
            trace_outcome_section=perf["sections"]["trace_outcome_model"],
            precomputed_payload=precomputed,
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertGreaterEqual(payload["raw_first_party_trainable_row_count"], 1000)
        self.assertGreaterEqual(payload["valid_judge_event_count"], 1000)
        self.assertGreaterEqual(payload["source_run_count"], 10)
        self.assertGreaterEqual(payload["distinct_problem_count"], 50)
        self.assertLessEqual(payload["calibration"]["best_brier_score"], 0.12)
        self.assertFalse(payload["prompt_answer_payload_stored"])

    def test_paper_benchmark_line_passes_current_mechanism_gates(self):
        payload = build_paper_benchmark_line_payload(
            root=Path("."),
            graph_dir=Path("phase four/assumption_graph"),
            eval_id="unit_paper_benchmark_line",
        )
        self.assertTrue(payload["benchmark_line_pass"], payload["benchmark_line_gates"])
        self.assertTrue(payload["research_gap_pass"], payload["research_gap_gates"])
        self.assertTrue(payload["paper_readiness_pass"])
        self.assertGreaterEqual(
            payload["completion_estimates"]["recursive_hypothesis_argument_percent"],
            90.0,
        )
        self.assertGreaterEqual(
            payload["completion_estimates"]["general_hypothesis_os_percent"],
            70.0,
        )
        failed = {gate["name"] for gate in payload["research_gap_gates"] if not gate["pass"]}
        self.assertEqual(failed, set())
        self.assertNotIn("world_model_raw_first_party_scale", failed)
        self.assertNotIn("continuous_daemon_autonomy", failed)
        self.assertNotIn("residual_label_large_scale_calibration", failed)
        self.assertNotIn("formal_engine_depth", failed)

    def test_paper_main_experiment_freezes_problem_level_stats_and_baselines(self):
        payload = build_paper_main_experiment_payload(
            root=Path("."),
            eval_id="unit_paper_main_experiment",
            final_forensic_path=Path(
                "phase four/assumption_graph/structural_live_ablation_20260603/missing_forensic_for_unit.jsonl"
            ),
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertEqual(payload["judge_source_mode"], "tracked_summary_pair_counts_fallback")
        base = payload["main_results"]["structural_vs_base"]
        placebo = payload["main_results"]["structural_vs_placebo"]
        self.assertEqual(base["problem_level_n"], 100)
        self.assertEqual(placebo["problem_level_n"], 100)
        self.assertGreaterEqual(base["utility"], 0.60)
        self.assertGreaterEqual(placebo["utility"], 0.60)
        self.assertGreater(base["bootstrap_ci_95"]["lower"], 0.50)
        self.assertGreater(placebo["bootstrap_ci_95"]["lower"], 0.50)
        self.assertLess(base["sign_test"]["p_value"], 0.05)
        self.assertLess(placebo["sign_test"]["p_value"], 0.05)
        baselines = {row["baseline"] for row in payload["baseline_table"]}
        for required in {
            "raw_llm_baseline",
            "ordinary_kg_triple_retrieval",
            "embedding_retrieval",
            "ordinary_rag_bm25_full_text",
            "full_text_tfidf_vector_retrieval",
            "no_morphism_structural_placebo",
            "no_novelty_gate_incremental_addition",
            "no_world_model_trace_policy",
            "no_recursive_runner_one_shot",
        }:
            self.assertIn(required, baselines)
        self.assertTrue(payload["no_prompt_or_answer_payload_stored"])
        self.assertGreaterEqual(payload["run_seed_variance_diagnostic"]["run_count"], 5)

    def test_paper_baseline_hardening_uses_matched_frozen_toggle_offs(self):
        payload = build_paper_baseline_hardening_payload(
            root=Path("."),
            eval_id="unit_paper_baseline_hardening",
        )
        self.assertTrue(payload["pass"], payload["gates"])
        rows = {row["baseline"]: row for row in payload["baseline_rows"]}
        for name in [
            "no_world_model_trace_policy",
            "no_recursive_runner_one_shot",
            "no_novelty_gate_incremental_addition",
        ]:
            self.assertIn(name, rows)
            self.assertTrue(rows[name]["same_problem_id_set"])
            self.assertEqual(rows[name]["problem_count"], 100)
            self.assertEqual(rows[name]["source_kind"], "matched_frozen_toggle_off_summary")
            self.assertGreater(
                rows[name]["pairs"]["structural_vs_base"]["final_minus_toggle_utility"],
                0.0,
            )

    def test_paper_retrieval_baselines_include_real_full_text_rag(self):
        payload = build_paper_retrieval_baselines_payload(eval_id="unit_paper_retrieval_baselines")
        self.assertTrue(payload["pass"], payload["gates"])
        rates = payload["hit_rates"]
        self.assertIn("ordinary_rag_bm25_full_text", rates)
        self.assertIn("full_text_tfidf_vector_retrieval", rates)
        self.assertGreaterEqual(rates["structural_morphism"], 0.80)
        self.assertGreaterEqual(payload["morphism_margin_over_best_retrieval"], 0.20)
        self.assertLessEqual(rates["ordinary_rag_bm25_full_text"], 0.40)

    def test_rag_to_memory_baseline_quantifies_morphism_margin(self):
        payload = build_rag_to_memory_baseline_payload(eval_id="unit_rag_to_memory_baseline")
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertEqual(payload["case_count"], 10)
        rates = payload["hit_rates"]
        self.assertGreaterEqual(rates["structural_morphism"], 0.80)
        self.assertLessEqual(rates["rag_to_memory_ppr"], 0.50)
        self.assertGreaterEqual(payload["absolute_hit_rate_margin"], 0.20)
        self.assertGreaterEqual(payload["absolute_top2_recall_margin"], 0.20)
        self.assertTrue(all(
            row["baseline_inputs_used"] == ["label", "domain", "surface_text", "kg_triples"]
            for row in payload["rows"]
        ))

    def test_hipporag_qa_probe_records_transfer_risk_on_real_qa_files(self):
        payload = build_hipporag_qa_probe_payload(
            root=Path("."),
            eval_id="unit_hipporag_qa_probe",
            samples_per_dataset=2,
            run_reader=False,
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertEqual(payload["aggregate"]["overall"]["structural_morphism_direct"]["applicable_rate"], 0.0)
        self.assertIn(payload["qa_transfer_risk"]["risk_level"], {"medium", "high"})
        self.assertGreater(payload["aggregate"]["overall"]["ordinary_bm25"]["any_gold_recall_at_k"], 0.0)
        self.assertFalse(payload["reader_qa_summary"]["raw_answers_stored"])

    def test_meta_qa_evolution_retains_only_beneficial_retrieval_hypotheses(self):
        payload = build_meta_qa_evolution_payload(
            root=Path("."),
            eval_id="unit_meta_qa_evolution",
            samples_per_dataset=5,
        )
        self.assertTrue(payload["pass"], payload["gates"])
        decisions = {row["hypothesis_id"]: row["decision"] for row in payload["evaluation"]}
        self.assertEqual(decisions["qa_hyp_comparison_dual_anchor"], "accept_retain")
        self.assertEqual(decisions["qa_hyp_anchor_preserve_insert"], "accept_retain")
        self.assertEqual(decisions["qa_hyp_representation_title_normalization"], "reject_no_measured_benefit")
        self.assertEqual(decisions["qa_hyp_decomposition_bridge_entity"], "accept_retain")
        self.assertEqual(decisions["qa_hyp_controlled_bridge_insert"], "accept_retain")
        self.assertEqual(decisions["qa_hyp_assumption_edge_policy_selector"], "accept_retain")
        self.assertTrue(decisions["qa_hyp_named_anchor_bridge"].startswith("reject"))
        self.assertTrue(decisions["qa_hyp_generic_prf"].startswith("reject"))
        self.assertGreaterEqual(payload["deltas_vs_bm25"]["all_gold_recall_at_k_delta"], 0.45)
        self.assertGreaterEqual(payload["deltas_vs_bm25"]["mean_gold_fraction_at_k_delta"], 0.25)
        self.assertGreaterEqual(payload["deltas_vs_bm25"]["answer_coverage_at_k_delta"], 0.30)
        self.assertIn("pre_reconstruction_method_priors", payload["source_alignment"])
        self.assertIn("assumption_edge_generalization", payload["source_alignment"])
        self.assertEqual(payload["config"]["workers"], 1)
        self.assertTrue(payload["bootstrap_ci"]["meta_vs_bm25"])
        self.assertTrue(payload["bootstrap_ci"]["learned_vs_bm25"])
        self.assertTrue(payload["bootstrap_ci"]["learned_vs_meta_controller"])
        self.assertIn("learned_meta_qa_controller", payload["aggregate"]["overall"])
        learned = payload["learned_policy_selector"]
        self.assertEqual(learned["status"], "run")
        self.assertEqual(learned["excluded_runtime_inputs"], ["gold_answers", "gold_titles", "supporting_facts"])
        self.assertFalse(
            set(learned["runtime_inputs"]) & {"gold_answers", "gold_titles", "supporting_facts"}
        )
        self.assertGreaterEqual(payload["learned_deltas_vs_bm25"]["all_gold_recall_at_k_delta"], 0.45)
        self.assertGreaterEqual(payload["learned_deltas_vs_bm25"]["mean_gold_fraction_at_k_delta"], 0.25)
        self.assertGreaterEqual(payload["learned_deltas_vs_bm25"]["answer_coverage_at_k_delta"], 0.30)
        self.assertGreaterEqual(payload["learned_deltas_vs_meta_controller"]["all_gold_recall_at_k_delta"], 0.0)
        self.assertGreaterEqual(payload["learned_deltas_vs_meta_controller"]["mean_gold_fraction_at_k_delta"], 0.0)
        self.assertGreaterEqual(payload["learned_deltas_vs_meta_controller"]["answer_coverage_at_k_delta"], 0.0)
        accepted = {row["hypothesis_id"]: row for row in payload["evaluation"] if row["decision"].startswith("accept")}
        self.assertTrue(all(row["harm_count"] == 0 for row in accepted.values()))
        heldout = build_meta_qa_evolution_payload(
            root=Path("."),
            eval_id="unit_meta_qa_evolution_heldout60",
            samples_per_dataset=20,
        )
        self.assertTrue(heldout["pass"], heldout["gates"])
        self.assertGreaterEqual(heldout["deltas_vs_bm25"]["all_gold_recall_at_k_delta"], 0.15)
        self.assertGreaterEqual(heldout["deltas_vs_bm25"]["mean_gold_fraction_at_k_delta"], 0.10)
        self.assertGreaterEqual(heldout["deltas_vs_bm25"]["answer_coverage_at_k_delta"], 0.05)
        self.assertGreaterEqual(heldout["learned_deltas_vs_bm25"]["all_gold_recall_at_k_delta"], 0.15)
        self.assertGreaterEqual(heldout["learned_deltas_vs_bm25"]["mean_gold_fraction_at_k_delta"], 0.10)
        self.assertGreaterEqual(heldout["learned_deltas_vs_bm25"]["answer_coverage_at_k_delta"], 0.05)
        self.assertGreaterEqual(heldout["learned_deltas_vs_meta_controller"]["all_gold_recall_at_k_delta"], 0.0)
        self.assertGreaterEqual(heldout["learned_deltas_vs_meta_controller"]["mean_gold_fraction_at_k_delta"], 0.0)
        self.assertGreaterEqual(heldout["learned_deltas_vs_meta_controller"]["answer_coverage_at_k_delta"], 0.0)
        self.assertEqual(heldout["learned_policy_selector"]["actual_harm_count"], 0)
        heldout_accepted = [row for row in heldout["evaluation"] if row["decision"].startswith("accept")]
        self.assertTrue(all(row["harm_count"] <= row["bounded_risk_harm_cap"] for row in heldout_accepted))
        self.assertFalse(payload["config"]["stored_raw_model_answers"])
        self.assertEqual(payload["extractive_reader"]["status"], "not_run")
        self.assertFalse(payload["extractive_reader"]["raw_answers_stored"])
        self.assertEqual(payload["llm_reader"]["status"], "not_run")
        self.assertFalse(payload["llm_reader"]["raw_answers_stored"])

    def test_structural_context_edges_generalize_hipporag_context(self):
        payload = build_structural_context_edge_payload(eval_id="unit_structural_context_edges")
        self.assertTrue(payload["pass"], payload["gates"])
        metrics = payload["metrics"]
        self.assertEqual(metrics["structural_context_positive_recall"], 1.0)
        self.assertGreater(
            metrics["structural_context_positive_recall"],
            metrics["word_context_baseline_positive_recall"],
        )
        self.assertEqual(metrics["negative_control_block_or_abstain_rate"], 1.0)
        self.assertEqual(metrics["classic_reference_expansion_rate"], 1.0)
        market = next(row for row in payload["rows"] if row["case_id"] == "feedback_market_chinese")
        self.assertEqual(market["structural_context"]["top_pattern_id"], "pat_negative_feedback")
        self.assertIn("real_lenz_negative_feedback", market["structural_context"]["expanded_realization_ids"])
        self.assertIn("real_le_chatelier_shift", market["structural_context"]["expanded_realization_ids"])

    def test_assumption_family_discovery_clusters_open_set_theory_kernels(self):
        payload = build_assumption_family_discovery_payload(eval_id="unit_assumption_family_discovery")
        self.assertTrue(payload["pass"], payload["gates"])
        metrics = payload["metrics"]
        self.assertEqual(metrics["discovered_family_count"], 10)
        self.assertEqual(metrics["cluster_purity"], 1.0)
        self.assertEqual(metrics["same_family_pair_recall"], 1.0)
        self.assertEqual(metrics["cross_family_block_rate"], 1.0)
        self.assertGreater(
            metrics["same_family_pair_recall"],
            metrics["word_context_pair_recall"],
        )
        self.assertGreaterEqual(metrics["new_open_set_family_count"], 1)
        families = {row["kernel_motif"]: row for row in payload["families"]}
        self.assertIn("kernel_representation_transform", families)
        self.assertEqual(families["kernel_representation_transform"]["open_set_status"], "new_open_set_family")
        feedback_members = {
            member["theory_id"]
            for member in families["kernel_negative_feedback"]["members"]
        }
        self.assertEqual({"homeostasis", "le_chatelier", "lenz"}, feedback_members)
        new_card = {
            "theory_id": "thermostat",
            "title": "Thermostat feedback control",
            "domain": "control",
            "text": (
                "A temperature deviation from setpoint triggers an actuator response that opposes the disturbance "
                "and restores a stable range."
            ),
        }
        decision = classify_new_theory_card(new_card, payload)
        self.assertEqual(decision["decision"], "attach_to_existing_family")
        self.assertEqual(decision["ranking"][0]["kernel_motif"], "kernel_negative_feedback")

    def test_paper_negative_results_records_boundaries_and_failures(self):
        payload = build_paper_negative_results_payload(
            root=Path("."),
            eval_id="unit_paper_negative_results",
        )
        self.assertTrue(payload["pass"], payload["gates"])
        domains = {row["domain"]: row for row in payload["domain_boundaries"]}
        self.assertIn("science", domains)
        self.assertIn("weak_vs_raw", domains["science"]["boundary_tags"])
        failures = {row["failure_id"] for row in payload["historical_repair_failures"]}
        self.assertIn("bottleneck_first_margin_failure", failures)
        self.assertIn("signal_first_repair_failure", failures)
        self.assertFalse(payload["formal_layer_boundaries"]["strict_category_theory_theorem_prover"])
        self.assertGreaterEqual(len(payload["abstain_or_gate_policy"]), 4)

    def test_paper_repro_pack_records_commands_hashes_and_env_names_only(self):
        payload = build_paper_repro_pack_payload(
            root=Path("."),
            eval_id="unit_paper_repro_pack",
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertGreaterEqual(len(payload["exact_commands"]), 6)
        self.assertGreaterEqual(len(payload["artifact_source_manifest"]), 10)
        self.assertTrue(all(row.get("sha256") for row in payload["artifact_source_manifest"]))
        self.assertTrue(all("value" not in row for row in payload["api_env_vars"]))
        self.assertIn("raw model answers", payload["data_card"]["excluded_from_repro_pack"])

    def test_morphism_independent_benchmark_beats_surface_baselines(self):
        payload = build_morphism_independent_benchmark_payload(eval_id="unit_morphism_independent")
        self.assertTrue(payload["pass"], payload["gates"])
        rates = payload["scorer_hit_rates"]
        self.assertGreaterEqual(rates["morphism"], 0.80)
        self.assertGreaterEqual(
            payload["morphism_margin_over_best_baseline"],
            0.20,
        )
        self.assertGreaterEqual(payload["nonlexical_success_rate"], 0.75)

    def test_morphism_independent_benchmark_beats_neural_embedding_baseline(self):
        try:
            payload = build_morphism_independent_benchmark_payload(
                eval_id="unit_morphism_independent_neural",
                neural_embedding_backend="sentence_transformer",
                neural_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            )
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        self.assertTrue(payload["pass"], payload["gates"])
        rates = payload["scorer_hit_rates"]
        self.assertIn("neural_embedding", rates)
        self.assertGreaterEqual(rates["morphism"], 0.80)
        self.assertLessEqual(rates["neural_embedding"], 0.30)
        self.assertGreaterEqual(payload["morphism_margin_over_best_baseline"], 0.20)
        self.assertTrue(payload["neural_embedding_baseline"]["enabled"])

    def test_morphism_claim_bundle_tightens_manuscript_scope(self):
        payload = build_morphism_claim_bundle_payload(
            root=Path("."),
            graph_dir=Path("phase four/assumption_graph"),
            eval_id="unit_morphism_claim_bundle",
        )
        self.assertTrue(payload["pass"], payload["gates"])
        self.assertEqual(
            payload["recommended_short_claim"],
            "category-inspired bounded structural morphism layer",
        )
        self.assertIn("complete category-theory theorem prover", payload["forbidden_claims"])
        self.assertFalse(payload["evidence"]["scope_flags"]["strict_category_theory_theorem_prover"])
        retrieval = payload["evidence"]["cross_domain_retrieval"]
        self.assertGreaterEqual(retrieval["scorer_hit_rates"]["morphism"], 0.80)
        self.assertGreaterEqual(retrieval["morphism_margin_over_best_baseline"], 0.20)
        self.assertGreaterEqual(retrieval["kg_embedding_miss_count"], 7)
        downstream = payload["evidence"]["downstream_effect"]
        self.assertGreaterEqual(downstream["downstream_transfer_auc"], 0.90)
        self.assertGreaterEqual(downstream["answer_quality_mean_delta"], 0.35)

    def test_retrieval_injects_structural_morphism_shadow_context(self):
        with tempfile.TemporaryDirectory() as td:
            store = JsonlGraphStore(Path(td) / "graph")
            store.upsert_node(AssumptionNode(
                id="strategy_rewrite_guard",
                type=AssumptionType.METHOD,
                claim="High-risk rewrites should preserve fallback behavior.",
                tags=["rewrite", "fallback"],
                confidence=0.6,
            ))
            store.flush()
            graph = SimpleAssumptionGraph(JsonlGraphStore(Path(td) / "graph"))
            result = retrieve_phase2_assumptions(
                graph,
                problem=(
                    "A plan wants to rewrite the evaluator and risks destructive overwrite; "
                    "keep a baseline fallback and apply only a local delta."
                ),
                meta={},
                pid="p_struct",
                domain="software_engineering",
                difficulty="medium",
                top_k=2,
            )
            self.assertIsNotNone(result)
            self.assertIn("pat_residual_correction", result.diagnostics["structural_morphism_hits"])
            text = format_policy_context(result, lambda subgraph, max_nodes=8: "base graph context")
            self.assertIn("Structural Morphism Reasoning", text)
            self.assertIn("pat_residual_correction", text)


if __name__ == "__main__":
    unittest.main()
