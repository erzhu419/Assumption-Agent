"""Build v10.2 read-only SHA256 integrity & study-classification errata index (Deliverable A4 & F16).

Writes:
  ``docs/evidence/v10_2_audit_remediation_20261116/errata_and_sha256_ledger_20261116.json``

Rules enforced:
  - Read-only SHA256 hashing over the 13 existing JSON receipts in
    ``docs/evidence/v10_world_structure_20261116/`` (confirming zero overwrites) and, if mounted,
    the small closed JSON summary files on ``/mnt/hegel-snapshot-cow`` (no broad disk scan;
    explicitly distinguishing SHA256 content hashes from prior size+mtime checks per A4).
  - Classifies each study by role (``historical_audit``, ``solver_or_tool_ablation``,
    ``workflow_multi_factor_study``, ``equal_tool_attribution``, ``frozen_prospective_lifecycle``)
    and exact independent physical-source count (F16).
  - Maps all 20 audit findings (``F01``–``F20``) from prior claim -> actual support -> corrected
    scientific claim -> remediation diff & verification receipt.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V10_EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"
V10_2_OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"
COW_ROOT = Path("/mnt/hegel-snapshot-cow")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


STUDY_CLASSIFICATION = {
    "paid_position_round1_census_20261116.json": {
        "study_role": "historical_read_only_workspace_census",
        "primary_question": "Read-only census of 135 historical workspaces / 229 position states checking whether any unanalyzed 2-assignment independent non-source site remained after Round 1",
        "independent_physical_sources": 0,
        "notes": "Audit/census only (0 remaining 2-assignment independent non-source sites); not a prospective evaluation.",
    },
    "v10_ablation_and_cross_scale_successor_20261116.json": {
        "study_role": "capability_ablation_and_cross_scale_fixture",
        "primary_question": "Equivalence reorganization preservation checks plus N=4 source formation -> 5 N=5 shifted-radius/reversed-direction targets comparing baseline, commitment-only, and full v10",
        "independent_physical_sources": 6,
        "notes": "Baseline disabled world_structure_commitments and transition_premise_audit; proves v10 mechanism capability on 5 targets (-27.69% target compute; 1,372,440 cumulative including source), whereas equal-tool knowledge attribution is isolated separately in v10.2 Study B1 & Deliverable C.",
    },
    "v10_cow_602_603_new_ledger_20261116.json": {
        "study_role": "historical_development_continuation_comparison",
        "primary_question": "Same-v10-tool cold vs warm continuation from historical 602 experience on known development apparatus 603",
        "independent_physical_sources": 2,
        "notes": "66.47% (199,912 -> 67,028) is accounted SearchBudget reduction, not wall-clock speedup or single-mechanism isolation.",
    },
    "v10_phase2a_expanded_revisions_20261116.json": {
        "study_role": "solver_and_premise_revision_capability_ablation",
        "primary_question": "Developer-authored conditional/non-conservative premise revision repertoire on N=4->N=5 pairs and N=3->4->5->6 chain",
        "independent_physical_sources": 6,
        "notes": "Tests runtime instantiation & revision within developer-authored revision branches.",
    },
    "v10_phase2b_open_grammar_llm_20261116.json": {
        "study_role": "workflow_multi_factor_llm_discovery_and_frozen_transfer",
        "primary_question": "Structured revision workflow (parent + roles + premise cue + local patch AST) vs flat anonymous LLM generation, followed by zero-call N=5 transfer",
        "independent_physical_sources": 4,
        "notes": "Stage A 4/4 vs 0/4 reflects the combined structured workflow, not isolated support-dependency alone (isolated in v10.2 Study B2). Source fresh48 confirmation passed, while saved research_method metadata recorded validation_rows=36 prior to v10.2 fix (F09).",
    },
    "v10_phase2c_active_threshold_and_pomdp_20261116.json": {
        "study_role": "active_probe_and_solver_capability_ablation",
        "primary_question": "Active continuous threshold bisection (4/4 vs passive 1/4) + first-order POMDP P_markov_state solver repertoire (6/6 vs Markov v10 0/6)",
        "independent_physical_sources": 8,
        "notes": "Active intervention + POMDP model-class capability ablation.",
    },
    "v10_phase3a_higher_order_and_coupled_pomdp_20261116.json": {
        "study_role": "solver_model_class_ablation",
        "primary_question": "AR(2) and spatially coupled linear POMDP solver repertoire vs Markov/AR(1) repertoire",
        "independent_physical_sources": 4,
        "notes": "Model-class/solver ablation; AST contract uses Program('accumulate', value=a1, name='ar2'|'ar2_lag1', children=(drive, const(a2))).",
    },
    "v10_phase3b_open_grammar_pomdp_llm_20261116.json": {
        "study_role": "workflow_multi_factor_llm_pomdp_discovery_and_frozen_transfer",
        "primary_question": "Structured POMDP LLM local AST workflow vs flat LLM, plus N=4->N=5 shifted-pole zero-call transfer",
        "independent_physical_sources": 4,
        "notes": "Multi-factor structured workflow result; confirmation metadata bug fixed in v10.2 script (F09).",
    },
    "v10_phase4a_curved_hypersurface_probing_20261116.json": {
        "study_role": "active_probe_and_boundary_capability_study",
        "primary_question": "Coupled 3-parameter curved-hypersurface active boundary design (7/8 vs passive 0/8 & 1D bisection 1/8) + N=4->N=5 transfer",
        "independent_physical_sources": 6,
        "notes": "Active boundary experiment design capability study.",
    },
    "v10_phase4b_continuum_pde_and_vdp_20261116.json": {
        "study_role": "solver_and_representation_capability_ablation_with_holdout_negative",
        "primary_question": "1D periodic ring discrete stencil discovery/transfer (8/8) + cubic Van der Pol limit-cycle POMDP solver (3/4; 990401 near-critical holdout negative)",
        "independent_physical_sources": 8,
        "notes": "Finite discrete lattice system with cyclic Z_N shift invariance (not continuum PDE limit, F18) + model-class ablation preserving 990401 passive-excitation failure.",
    },
    "v10_phase5a_manifold_and_graph_pde_20261116.json": {
        "study_role": "solver_and_representation_capability_ablation",
        "primary_question": "2D discrete torus (Z_Nx x Z_Ny translation invariance), 1D Neumann/Dirichlet chain, and finite graph-Laplacian solver capability vs disabled-solver baselines",
        "independent_physical_sources": 12,
        "notes": "12/12 vs 0/12 proves new manifold/graph solver capability on finite discrete systems (3x3->4x4, 6->9), NOT persistent-learning superiority under equal tools nor full Aut(T^2) / continuum PDE convergence (F03, F18).",
    },
    "v10_phase5b_stochastic_sde_and_resonance_20261116.json": {
        "study_role": "estimator_and_active_probe_capability_ablation",
        "primary_question": "Cohort 5B-1: 2SLS-IV vs naive OLS under process noise (2/4 native completed; 4/4 noiseless-twin transfer <= 2e-3); Cohort 5B-2: active resonance probing on near-critical VdP (4/4)",
        "independent_physical_sources": 8,
        "notes": "oracle_zero_noise_drift_mse in this receipt is noiseless-twin transfer MSE, NOT noisy-history conditional drift risk (re-evaluated with true noisy-history conditional drift in sde_and_proof_remediation_20261116.json, F08).",
    },
    "v10_phase5c_algebraic_proofs_and_impossibility_20261116.json": {
        "study_role": "supplied_model_solver_and_checker_qualification",
        "primary_question": "Exact rational/SymPy QQ invariant & Farkas LP certificate solver/checker on supplied linear models + 4 instances of boundary port-gain discrepancy repair",
        "independent_physical_sources": 8,
        "notes": "Errata (F01, F02, F05, F06, F07, F10, F11, F12, F14): (1) Lean output is draft_only, not compiled Lean theorems; (2) invariants verified on finite arities, not arbitrary N; (3) Cohort 5C-3 tests 4 parameter levels (0.60, 1.00, 1.25, 1.50) of one boundary port-gain mechanism; (4) 89.92% probe reduction was a nominal formula estimate 2(3N+1)+N(N-1), replaced in v10.2 Study B3 by executed equal-tool two-arm measurements; (5) state_bounds domain legality unified in v10.2.",
    },
}


FINDINGS_ERRATA = [
    {
        "id": "F01",
        "severity": "P0",
        "prior_claim_or_issue": "PotentialLowerBoundCertificate checked state_bounds while ReachabilityWitnessCertificate and BudgetedControlLab ignored state_bounds without proving reachable enclosure.",
        "actual_support_at_a66cd82b": "Confirmed bug: probe_certificate_domain.py accepted both an impossibility certificate and a winning witness on [-1, 0]^2 (pre_fix_a66cd82b_domain_probe_20261116.json).",
        "corrected_claim_and_remediation": "Unified via validate_control_task_semantics & verify_reachable_box_enclosure: physical_legality enforces state_bounds across lab/planner/witness/potential; analysis_region returns conditional_impossibility_domain_unverified (verified=False) unless enclosure is proved; unbounded never invents a default box.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/post_fix_domain_probe_20261116.json",
            "docs/evidence/v10_2_audit_remediation_20261116/sde_and_proof_remediation_20261116.json",
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_01_and_02_gpt6_domain_counterexample_never_accepts_opposite_certificates",
        ],
    },
    {
        "id": "F02",
        "severity": "P1",
        "prior_claim_or_issue": "Described Lean 4 string export as self-contained zero-sorry Lean proofs.",
        "actual_support_at_a66cd82b": "Exact arithmetic checks ran in Python Fraction & SymPy QQ; emit_lean4_certificate emitted draft definitions/comments without Lean compilation.",
        "corrected_claim_and_remediation": "Retracted 'Lean-verified proof' claim; tagged all Lean exports as lean_status='draft_only', lean_compiled=False, with trusted_computing_base='python_fractions_and_sympy_QQ'.",
        "verification_evidence": [
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin"
        ],
    },
    {
        "id": "F03",
        "severity": "P1",
        "prior_claim_or_issue": "Pooled solver/representation capability ablations (Phase 3A/4A/4B/5A) into persistent Hegel learning superiority.",
        "actual_support_at_a66cd82b": "Baseline arms disabled manifold_pde_discovery / ar2 / vdp flags alongside memory.",
        "corrected_claim_and_remediation": "Reclassified Phase 3A/4A/4B/5A as solver/model-class capability ablations; executed dedicated equal-tool 3-arm attribution (cold_full_tools vs archive_full_tools vs structured_full_tools) in Study B1.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json"
        ],
    },
    {
        "id": "F04",
        "severity": "P1",
        "prior_claim_or_issue": "Labeled Phase 2B/3B flat-LLM vs gated-LLM comparison as a single-factor support-dependency isolation.",
        "actual_support_at_a66cd82b": "Changed multiple factors simultaneously (parent commitment, resolved roles, violated premise cue, and local patch grammar vs full Program AST).",
        "corrected_claim_and_remediation": "Reclassified Phase 2B/3B Stage A as a combined structured-workflow comparison; added strict single-factor dependency isolation (Study B2) plus separate patch-vs-scratch factor table.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json",
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators",
        ],
    },
    {
        "id": "F05",
        "severity": "P1",
        "prior_claim_or_issue": "Described Cohort 5C-3 as repairing 4 distinct fault mechanisms (sink, coupling, inventory, clip).",
        "actual_support_at_a66cd82b": "Cohort 5C-3 tested 4 parameter values (0.60, 1.00, 1.25, 1.50) of boundary_port_gain.",
        "corrected_claim_and_remediation": "Corrected Phase 5C report to '4 parameter instances of boundary port gain'; additionally implemented & executed a genuine 4-mechanism equal-tool repair benchmark in Study B3.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json"
        ],
    },
    {
        "id": "F06",
        "severity": "P1",
        "prior_claim_or_issue": "Reported 89.92% probe reduction against untargeted baseline as an empirical result.",
        "actual_support_at_a66cd82b": "Baseline count came from nominal formula 2*(3N+1) + N*(N-1), not an executed agent.",
        "corrected_claim_and_remediation": "Relabeled 89.92% as a nominal full-scan formula estimate; executed real two-arm equal-tool repair agents in Study B3 measuring actual lab.probe_queries and SearchBudget compute.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json"
        ],
    },
    {
        "id": "F07",
        "severity": "P1",
        "prior_claim_or_issue": "Hardcoded rollout baseline certified_proof=False without access to witness verification.",
        "actual_support_at_a66cd82b": "0/8 vs 8/8 reflected solver + checker access difference, not equal-tool search.",
        "corrected_claim_and_remediation": "Added certify_rollout_baseline sharing SymbolicProofKernel.verify_solvability_certificate and reclassified 5C-2 as a solver/checker capability comparison.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/sde_and_proof_remediation_20261116.json"
        ],
    },
    {
        "id": "F08",
        "severity": "P1",
        "prior_claim_or_issue": "Equated noiseless-twin trajectory MSE (process_noise_std=0) with noisy-history conditional drift risk E[(m_hat(F_t) - E[Y_{t+1}|F_t])^2].",
        "actual_support_at_a66cd82b": "Turning off noise changes historical state/flow distribution F_t; native online completion was 2/4.",
        "corrected_claim_and_remediation": "Implemented evaluate_sde_conditional_drift_and_twin_metrics computing true noisy-history conditional drift MSE (exact + 64-rollout MC on E_noisy) separately from noiseless_twin_transfer_mse and live_noisy_trajectory_mse, with 2/4 native completion surfaced prominently.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/sde_and_proof_remediation_20261116.json",
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_17_sde_noisy_history_conditional_drift_distinct_from_noiseless_twin",
        ],
    },
    {
        "id": "F09",
        "severity": "P1",
        "prior_claim_or_issue": "After passing fresh48 confirmation, Phase 2B/3B scripts passed validation_loss and len(validation)=36 into remember_confirmed_selection.",
        "actual_support_at_a66cd82b": "Independent fresh48 gate WAS executed, but saved confirmation_loss/rows metadata referenced development validation.",
        "corrected_claim_and_remediation": "Fixed Phase 2B/3B scripts and remember_confirmed_selection to record distinct fit_rows, selection_rows, selection_loss, confirmation_rows=48, confirmation_loss=fresh48_mse, and deduplicated confirmation_ref.",
        "verification_evidence": [
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_09_11_and_f15_confirmation_identity_zero_weight_columns_and_ar2_ast"
        ],
    },
    {
        "id": "F10",
        "severity": "P1",
        "prior_claim_or_issue": "Phase 5C algebraic synthesis, verification, and LP proving did not accept or deduct SearchBudget.",
        "actual_support_at_a66cd82b": "Confirmed: Phase 5C functions had no budget parameter.",
        "corrected_claim_and_remediation": "Threaded SearchBudget across SymbolicProofKernel, synthesize_emergent_invariants, prove_task_solvability, BudgetedControlLab, certify_rollout_baseline, and repair_proof_from_discrepancy.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/sde_and_proof_remediation_20261116.json",
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators",
        ],
    },
    {
        "id": "F11",
        "severity": "P2",
        "prior_claim_or_issue": "Described finite-arity SymPy polynomial checks (e.g. N in [3,5,8]) as proving arbitrary N.",
        "actual_support_at_a66cd82b": "Verified symbolic polynomial identities at each tested finite N.",
        "corrected_claim_and_remediation": "Set all_arity_proved=False and reported verified_on_finite_arities explicitly.",
        "verification_evidence": [
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin"
        ],
    },
    {
        "id": "F12",
        "severity": "P2",
        "prior_claim_or_issue": "Directional moment monotonicity checked flux polynomial identity without enforcing nonnegative donor cone and active edge direction obligations.",
        "actual_support_at_a66cd82b": "Sign conditions existed only in comments/metadata.",
        "corrected_claim_and_remediation": "Bound explicit domain_obligations (donor_state_cone='nonnegative', active_edge_displacement_sign, and cone-generator positivity check in QQ) inside SymbolicProofKernel.",
        "verification_evidence": [
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin"
        ],
    },
    {
        "id": "F13",
        "severity": "P2",
        "prior_claim_or_issue": "Conflated developer-authored premise graph nodes & solver families with runtime theory invention.",
        "actual_support_at_a66cd82b": "Premise node ontology and solver families are developer-authored; runtime learns role mappings, parameter values, scoped commitments, and local AST combinations.",
        "corrected_claim_and_remediation": "Explicitly separated developer-authored repertoire from runtime-learned commitments and separated declared_premises, empirically_supported_premises, refuted_premises, and formal_obligations.",
        "verification_evidence": ["docs/hegel_v10_2_errata_and_audit_closure_20261116.md"],
    },
    {
        "id": "F14",
        "severity": "P2",
        "prior_claim_or_issue": "Phase 5C passed hardcoded system_spec/model dicts rather than runtime-learned commitments.",
        "actual_support_at_a66cd82b": "Phase 5C was a supplied_model solver/checker qualification test.",
        "corrected_claim_and_remediation": "Added explicit model_origin ('supplied_model' vs 'learned_model') and implemented + verified extract_control_model_from_learned_commitment from live AnonymousObjectLab research.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/sde_and_proof_remediation_20261116.json"
        ],
    },
    {
        "id": "F15",
        "severity": "P2",
        "prior_claim_or_issue": "Handoff dossier wrote AR(2) AST children as (const(a1), const(a2), drive) instead of actual implementation.",
        "actual_support_at_a66cd82b": "_ar2_role_term stores a1 in Program.value and children=(drive, const(a2)).",
        "corrected_claim_and_remediation": "Corrected all documentation to match actual Program('accumulate', value=a1, name='ar2'|'ar2_lag1', children=(drive, const(a2))) and locked via round-trip serialization test.",
        "verification_evidence": [
            "tests/test_v10_2_audit_acceptance.py::test_acceptance_09_11_and_f15_confirmation_identity_zero_weight_columns_and_ar2_ast"
        ],
    },
    {
        "id": "F16",
        "severity": "P2",
        "prior_claim_or_issue": "Cited 28 focused tests and 13 JSON receipts as full-repo regression & 13 prospective validations.",
        "actual_support_at_a66cd82b": "28 focused v10/historical capability tests; 13 receipts span historical audits, solver ablations, and multi-factor studies.",
        "corrected_claim_and_remediation": "Classified every receipt by exact study_role and independent_physical_sources in this ledger, and ran broader repository pytest suite separately from focused v10/v10.2 tests.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/errata_and_sha256_ledger_20261116.json"
        ],
    },
    {
        "id": "F17",
        "severity": "P2",
        "prior_claim_or_issue": "Described ordinary archive baselines reconstructing equivalent methods in 15 compute units as an 'isomorphism bug'.",
        "actual_support_at_a66cd82b": "Cheap baseline reconstruction from shared raw sources is a legitimate strong baseline and valid empirical tie/negative.",
        "corrected_claim_and_remediation": "Retracted 'isomorphism bug' framing; archive_full_tools in Deliverables B & C is explicitly permitted to reconstruct equivalent commitments on demand with reconstruction compute billed.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json",
            "docs/evidence/v10_2_audit_remediation_20261116/frozen_v4_lifecycle_20261116.json",
        ],
    },
    {
        "id": "F18",
        "severity": "P2",
        "prior_claim_or_issue": "Referred to cyclic grid shifts as full Aut(T^2) and cross-lattice transfer (3x3->4x4) as continuum PDE limit.",
        "actual_support_at_a66cd82b": "Tested discrete periodic translation subgroup Z_{N_x} x Z_{N_y} and finite discrete lattice/graph sizes.",
        "corrected_claim_and_remediation": "Tightened all mathematical claims to discrete translation subgroup Z_{N_x} x Z_{N_y} invariance and cross-lattice-size finite discrete dynamical systems.",
        "verification_evidence": ["docs/hegel_v10_2_errata_and_audit_closure_20261116.md"],
    },
    {
        "id": "F19",
        "severity": "positive",
        "prior_claim_or_issue": "Complete fitted-column cross-arity instantiation, non-zero/zero-weight column preservation, and premise-level revision.",
        "actual_support_at_a66cd82b": "Verified in source code (world_structure_commitments.py & premise_dependency_audit.py).",
        "corrected_claim_and_remediation": "Preserved intact and extended with empirical consistency gating (is_commitment_empirically_consistent) for 6-stage lifecycle return.",
        "verification_evidence": [
            "docs/evidence/v10_2_audit_remediation_20261116/frozen_v4_lifecycle_20261116.json"
        ],
    },
    {
        "id": "F20",
        "severity": "positive",
        "prior_claim_or_issue": "Same-v10-tool 602->603 warm vs cold compute reduction (199,912 -> 67,028) and zero-extra-LLM-call cross-arity reuse.",
        "actual_support_at_a66cd82b": "Supported by receipts on known development apparatuses.",
        "corrected_claim_and_remediation": "Preserved with precise scope (accounted SearchBudget compute reduction; source vs target costs separated).",
        "verification_evidence": [
            "docs/evidence/v10_world_structure_20261116/v10_cow_602_603_new_ledger_20261116.json",
            "docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json",
        ],
    },
]


def main() -> int:
    V10_2_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read-only SHA256 over all 13 untouched v10 receipts
    v10_files = sorted(V10_EVIDENCE_DIR.glob("*.json"))
    v10_hashes = []
    for p in v10_files:
        cls_info = STUDY_CLASSIFICATION.get(p.name, {})
        v10_hashes.append(
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
                **cls_info,
            }
        )

    # 2. Read-only SHA256 over designated small closed COW ledger files (if mounted; no broad scan!)
    cow_designated = [
        COW_ROOT / "closed_ledgers" / "v8_601_603_closed_summary.json",
        COW_ROOT / "evidence" / "hegel_603_three_arm_summary.json",
    ]
    cow_hashes = []
    if COW_ROOT.exists():
        # Check exact paths referenced in v10_cow_602_603_new_ledger_20261116.json if present
        cow_rec_path = V10_EVIDENCE_DIR / "v10_cow_602_603_new_ledger_20261116.json"
        if cow_rec_path.is_file():
            cow_rec = json.loads(cow_rec_path.read_text(encoding="utf-8"))
            guards = (cow_rec.get("cow_integrity_verification") or {}).get("guards_snapshot") or {}
            for raw_p in guards.keys():
                fp = Path(raw_p)
                if fp.is_file() and fp.stat().st_size <= 5_000_000 and fp not in cow_designated:
                    cow_designated.append(fp)
        for fp in cow_designated:
            if fp.is_file() and fp.stat().st_size <= 5_000_000:
                cow_hashes.append(
                    {
                        "path": str(fp),
                        "size_bytes": fp.stat().st_size,
                        "sha256": _sha256_file(fp),
                        "integrity_check_method": "read_only_sha256_content_digest",
                    }
                )

    # 3. Read-only SHA256 over new v10.2 remediation receipts
    v10_2_hashes = []
    for p in sorted(V10_2_OUT_DIR.glob("*.json")):
        if p.name == "errata_and_sha256_ledger_20261116.json":
            continue
        v10_2_hashes.append(
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }
        )

    ledger = {
        "ledger_id": "v10_2_errata_and_sha256_ledger_20261116",
        "generated_at": _utc_now(),
        "audited_commit": "a66cd82b22f444044dbda98f43782c96f833a4f0",
        "paid_llm_calls": 0,
        "untouched_v10_receipts_count": len(v10_hashes),
        "cow_mount_active_in_session": bool(cow_hashes),
        "integrity_note": "Prior COW check recorded size+mtime metadata only; this ledger records true read-only SHA256 content digests over all 13 untouched v10 receipts and all new v10.2 receipts (plus designated COW receipts whenever /mnt/hegel-snapshot-cow is mounted).",
        "untouched_v10_receipts_sha256_and_roles": v10_hashes,
        "closed_cow_small_ledger_files_sha256": cow_hashes,
        "new_v10_2_remediation_receipts_sha256": v10_2_hashes,
        "findings_remediation_matrix": FINDINGS_ERRATA,
    }
    out_path = V10_2_OUT_DIR / "errata_and_sha256_ledger_20261116.json"
    out_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (v10 receipts hashed={len(v10_hashes)}, cow files hashed={len(cow_hashes)}, v10.2 receipts={len(v10_2_hashes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
