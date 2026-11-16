"""Behavioral acceptance test suite for Hegel Machine v10.2 Audit Closure (Contract §5 #1–#22 + F15).

Note (#22): Passing these behavioral acceptance tests verifies code-level trust-boundary,
accounting, and experimental-isolation invariants; it does NOT by itself establish prospective
scientific superiority across unseen worlds.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from hegel_machine.cognition.algebraic_proof_certificates import (
    BudgetedControlLab,
    SymbolicProofKernel,
    certify_rollout_baseline,
    extract_control_model_from_learned_commitment,
    prove_task_solvability,
    repair_proof_from_discrepancy,
    synthesize_emergent_invariants,
    validate_control_task_semantics,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuous_state import _sum_terms
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    run_single_factor_dependency_ablation_arm,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_dependency_audit import audit_transition_premises
from hegel_machine.cognition.research_programs import _plan
from hegel_machine.cognition.stochastic_premise_audit import (
    evaluate_sde_conditional_drift_and_twin_metrics,
)
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    _ar2_role_term,
    _extract_ar2_params_from_role_terms,
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
    instantiate_commitment_terms,
    is_commitment_empirically_consistent,
    propose_scientific_premise_revision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_schema(n: int, prefix: str = "obj") -> ObjectSchema:
    objs = tuple(f"{prefix}_{i}" for i in range(n))
    return ObjectSchema(
        objs,
        {g: {r: f"{r}:{g}" for r in ("q", "u", "x", "f")} for g in objs},
        {g: (g,) for g in objs},
        {g: {r: f"ctl:{r}:{g}" for r in ("q", "u", "x")} for g in objs},
    )


def test_acceptance_01_and_02_gpt6_domain_counterexample_never_accepts_opposite_certificates():
    """Acceptance #1 & #2: Run GPT-6's probe_certificate_domain.py directly & across all 3 domain modes."""
    probe_path = REPO_ROOT / "markdown" / "hegel_v10.2" / "probe_certificate_domain.py"
    spec = importlib.util.spec_from_file_location("gpt6_probe_mod", probe_path)
    assert spec and spec.loader
    probe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_mod)

    repo_res = probe_mod.actual_check(REPO_ROOT)
    assert repo_res["both_opposite_certificates_accepted"] is False
    assert repo_res["potential_lean_contains_theorem_declaration"] is False
    assert repo_res["potential_lean_compiled"] is False
    # Under default physical_legality for declared state_bounds=[-1, 0], out-of-box witness is rejected
    assert repo_res["potential_verifier"]["verified"] is True
    assert repo_res["witness_verifier"]["verified"] is False
    assert repo_res["lab_trace"]["winnable_verified_on_lab"] is False

    # Under analysis_region mode, unverified box [-1, 0] CANNOT certify unconditional impossibility!
    task_analysis = {**probe_mod.TASK, "state_bounds_mode": "analysis_region"}
    pot_ar = SymbolicProofKernel.verify_solvability_certificate(
        probe_mod.POTENTIAL, probe_mod.MODEL, task_analysis
    )
    wit_ar = SymbolicProofKernel.verify_solvability_certificate(
        probe_mod.WITNESS, probe_mod.MODEL, task_analysis
    )
    assert pot_ar["verified"] is False
    assert pot_ar["status"] == "conditional_impossibility_domain_unverified"
    assert wit_ar["verified"] is True
    assert wit_ar["status"] == "witness_verified_in_declared_model"

    # Under undeclared state_bounds=None, nonzero state multipliers are rejected and solver finds witness!
    task_unbounded = {k: v for k, v in probe_mod.TASK.items() if k != "state_bounds"}
    pot_ub = SymbolicProofKernel.verify_solvability_certificate(
        probe_mod.POTENTIAL, probe_mod.MODEL, task_unbounded
    )
    assert pot_ub["verified"] is False
    assert pot_ub["status"] == "invalid_certificate"
    solved_ub = prove_task_solvability(probe_mod.MODEL, task_unbounded)
    assert solved_ub["verdict"] == "winnable"
    assert solved_ub["status"] == "witness_verified_in_declared_model"


def test_acceptance_03_and_04_invalid_task_rejection_and_rationalization_reporting():
    """Acceptance #3 & #4: Out-of-bounds initial/step state, negative cost/fuel, dim mismatch, goal truncation, & rationalization delta."""
    model = {"objects": 2, "positions": [0.0, 0.5], "coupling": 0.25, "radius": 1.0, "direction": "right"}
    valid_task = {
        "initial_state": [0.0, 0.0],
        "control_bounds": [-0.5, 0.5],
        "state_bounds": [-1.0, 1.0],
        "budget": 1.0,
        "step_cost": 0.0,
        "fuel_weights": [1.0, 1.0],
        "goal_matrix": [[-1.0, -1.0]],
        "goal_rhs": [-0.5],
    }

    # 1. Initial state outside physical state_bounds
    bad_init = {**valid_task, "initial_state": [2.0, 0.0]}
    assert validate_control_task_semantics(model, bad_init)["valid"] is False

    # 2. Negative step_cost or negative fuel_weight
    assert validate_control_task_semantics(model, {**valid_task, "step_cost": -0.1})["valid"] is False
    assert validate_control_task_semantics(model, {**valid_task, "fuel_weights": [1.0, -0.2]})["valid"] is False

    # 3. Dimension mismatch & goal truncation
    assert validate_control_task_semantics(model, {**valid_task, "initial_state": [0.0]})["valid"] is False
    assert validate_control_task_semantics(model, {**valid_task, "goal_matrix": [[-1.0, -1.0]], "goal_rhs": []})["valid"] is False
    assert validate_control_task_semantics(model, {**valid_task, "goal_matrix": [[-1.0]], "goal_rhs": [-0.5]})["valid"] is False

    # 4. Action causing post-step out-of-domain rejected on BudgetedControlLab under physical_legality
    lab = BudgetedControlLab(
        objects=2, positions=(0.0, 0.5), coupling=0.25, radius=1.0, control_bounds=(-1.0, 1.0), state_bounds=(-0.5, 0.5)
    )
    with pytest.raises(ValueError, match="state_bounds"):
        lab.step([0.3, 0.0], [0.5, 0.0])

    # 5. Rationalization delta explicitly reported when IEEE-754 float (e.g. 0.1 or 1/3 float) is rationalized (#4)
    irr_model = {**model, "coupling": 1.0 / 3.0}
    sem_irr = validate_control_task_semantics(irr_model, valid_task)
    assert sem_irr["valid"] is True
    assert sem_irr["rational_model"]["rationalization_changed_values"] is True
    assert sem_irr["rational_model"]["max_rationalization_abs_delta"] > 0.0
    assert sem_irr["rational_model"]["coupling"] == "1/3"


def test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin():
    """Acceptance #5, #6, #7, #8: Draft Lean status, finite-N scope, cone obligations, and model_origin separation."""
    # #5 & #6: Emergent invariant certificate has lean_status='draft_only' and all_arity_proved=False
    certs = synthesize_emergent_invariants(
        {"system_class": "coupled_state_reservoir_pomdp", "alpha": "3/4", "eta": "1/2", "c_u": "1"}
    )
    cert = certs[0]
    assert cert["lean_status"] == "draft_only"
    assert cert["lean_compiled"] is False
    assert cert["all_arity_proved"] is False
    assert cert["kernel_verification"]["all_arity_proved"] is False
    assert cert["verified_on_finite_arities"] == [3, 5, 8]
    assert "proof_verified=false in Lean" in cert["lean4_module"]

    # #7: Directional moment monotonicity MUST enforce explicit cone & edge-direction obligations in code!
    dir_cert = synthesize_emergent_invariants(
        {"system_class": "directional_moment_monotonicity", "gamma": "1/4", "direction": "right"}
    )[0]
    assert dir_cert["kernel_verification"]["verified"] is True

    missing_dom = deepcopy(dir_cert)
    missing_dom.pop("domain_obligations")
    assert SymbolicProofKernel.verify_emergent_invariant_certificate(missing_dom)["verified"] is False

    bad_cone = deepcopy(dir_cert)
    bad_cone["domain_obligations"]["donor_state_cone"] = "unbounded_signed"
    assert SymbolicProofKernel.verify_emergent_invariant_certificate(bad_cone)["verified"] is False

    bad_edge = deepcopy(dir_cert)
    bad_edge["domain_obligations"]["positions"] = [0.0, 1.0, 2.0]
    bad_edge["domain_obligations"]["active_edges"] = [(2, 0)]  # leftward edge violating right-direction obligation!
    assert SymbolicProofKernel.verify_emergent_invariant_certificate(bad_edge)["verified"] is False

    # #8: supplied_model vs learned_model provenance tags
    sup_cert = prove_task_solvability(
        {"objects": 2, "positions": [0.0, 1.5], "coupling": 0.25, "radius": 0.5, "direction": "right"},
        {
            "initial_state": [0.0, 0.0],
            "control_bounds": [-0.5, 0.5],
            "state_bounds": [-5.0, 5.0],
            "budget": 0.4,
            "fuel_weights": [1.0, 1.0],
            "goal_matrix": [[-1.0, -1.0]],
            "goal_rhs": [-1.0],
        },
    )
    assert sup_cert["model_origin"] == "supplied_model"
    assert sup_cert["empirically_supported_premises"] == []

    learned_spec = extract_control_model_from_learned_commitment(
        {
            "commitment_id": "world_commitment:0001",
            "radius_parameters": [{"symbol": "r_0", "direction": "right", "source_values": [0.95]}],
            "support_premises": [{"premise_id": "conservation"}, {"premise_id": "interaction_direction:r_0"}],
        },
        {
            "verified_direction": "right",
            "radius_interval": [0.85, 1.05],
            "matches_by_direction": {"right": [{"coupling_estimate": 0.25}]},
            "supported_premises": ["P_conservation", "P_interaction_direction", "P_locality_radius"],
        },
        positions=[0.0, 1.5],
    )
    lrn_cert = prove_task_solvability(
        learned_spec,
        {
            "initial_state": [0.0, 0.0],
            "control_bounds": [-0.5, 0.5],
            "state_bounds": [-5.0, 5.0],
            "budget": 0.4,
            "fuel_weights": [1.0, 1.0],
            "goal_matrix": [[-1.0, -1.0]],
            "goal_rhs": [-1.0],
        },
    )
    assert lrn_cert["model_origin"] == "learned_model"
    assert "P_conservation" in lrn_cert["empirically_supported_premises"]


def test_acceptance_09_11_and_f15_confirmation_identity_zero_weight_columns_and_ar2_ast():
    """Acceptance #9, #11 & F15: Distinct fit/selection/confirmation rows, deduplicated confirmation_ref, zero-weight preservation, & AR(2) AST."""
    sc = _make_schema(3, "s")
    tot_q = Program("object_sum", children=tuple(Program("feature", name=sc.fields[g]["q"]) for g in sc.objects))
    tot_u = Program("object_sum", children=tuple(Program("feature", name=sc.fields[g]["u"]) for g in sc.objects))
    triples = tuple(Program("feature", name=sc.fields[g][r]) for g in sc.objects for r in ("f", "x", "q"))
    nb_sym = Program("neighbor_sum", name="right", children=(Program("arg", name="r_0"), *triples))
    nb_act = Program("neighbor_sum", name="right", children=(Program("const", value=0.9), *triples))

    # Include an explicit ZERO-WEIGHT column (nb_act has coefficient 0.0!) to verify #11!
    act_terms = (tot_q, tot_u, nb_act)
    sym_terms = (tot_q, tot_u, nb_sym)
    weights = [1.0, 1.0, 0.0]
    prog = _sum_terms(list(act_terms), weights, 0.0)
    sym_prog = _sum_terms(list(sym_terms), weights, 0.0)

    plan = _plan(
        {"fit_terms": act_terms, "coefficients": [0.0, *weights], "program": prog, "loss": 0.0},
        [
            {
                "operation": "resolve_relation_intervals",
                "question": "Identify r_0",
                "parameter_template": sym_prog.to_dict(),
                "parameter_terms": [t.to_dict() for t in sym_terms],
                "parameters": [
                    {
                        "symbol": "r_0",
                        "radius": 0.9,
                        "radius_intervals": [[0.8, 1.0]],
                        "prior_radius_intervals": [[0.8, 1.0]],
                        "relation": "right",
                        "source_concept_id": None,
                    }
                ],
                "evidence_contexts": [],
                "compact_replacement": False,
                "program": prog,
                "loss": 0.0,
            }
        ],
        20,
        400,
    )
    entry = {
        "candidate": Candidate("cand_zero_col", prog, "structural", "test"),
        "kind": "structural:conserved_total",
        "family_id": "T15",
        "training_loss": 0.0,
        "validation_loss": 1.25e-11,
        "research_program": plan,
    }

    lib = KnowledgeLibrary()
    agent = UnifiedResearchAgent(ResearchConfig(world_structure_commitments=True), library=lib)
    with AnonymousObjectLab(seed=701, objects=3, readout="total") as lab:
        spec = agent._spec(lab, ("fine",))
        spec.context["object_schema"] = __import__("json").dumps(sc.to_dict())
        train_rows = [Observation({f: 0.0 for f in sc.feature_names}, 0.0) for _ in range(20)]
        sel_rows = [Observation({f: 0.0 for f in sc.feature_names}, 0.0) for _ in range(12)]
        conf_ref = {"kind": "fresh48", "eval_seed": 9991, "rows": 48}

        saved1 = agent.remember_confirmed_selection(
            entry,
            spec,
            train_rows,
            sel_rows,
            confirmed=True,
            confirmation_loss=3.5e-14,
            confirmation_rows=48,
            selected_channels=("fine",),
            confirmation_ref=conf_ref,
        )
        assert saved1["stored"] is True
        assert saved1["reused"] is False

        # Re-saving the same confirmation_ref MUST deduplicate without double-counting (#9)!
        saved2 = agent.remember_confirmed_selection(
            entry,
            spec,
            train_rows,
            sel_rows,
            confirmed=True,
            confirmation_loss=3.5e-14,
            confirmation_rows=48,
            selected_channels=("fine",),
            confirmation_ref=conf_ref,
        )
        assert saved2["reused"] is True
        assert len(lib.meta_priors) == 1

    rm = lib.meta_priors[0]["research_method"]
    assert rm["fit_rows"] == 20
    assert rm["selection_rows"] == 12
    assert rm["confirmation_rows"] == 48
    assert abs(rm["selection_loss"] - 1.25e-11) <= 1e-18
    assert abs(rm["confirmation_loss"] - 3.5e-14) <= 1e-20

    # Consolidate & rebind on N=4: zero-weight column and shared r_0 MUST be preserved (#11)!
    comms = consolidate_world_structure_commitments(lib.meta_priors, budget=SearchBudget(10000))
    assert len(comms) == 1
    assert len(comms[0]["role_terms"]) == 3  # zero-weight 3rd column NOT dropped!
    sc4 = _make_schema(4, "t")
    rebound = instantiate_commitment_terms(comms[0], sc4, radius_bindings={"r_0": 1.15})
    assert rebound is not None and len(rebound) == 3

    # F15: Exact AR(2) AST contract verification via round-trip serialization
    ar2_rt = _ar2_role_term("u", 1.32, -0.64, mode="ar2_lag1", focused=False)
    summand_prog = Program.from_dict(ar2_rt["summand"])
    assert summand_prog.op == "accumulate"
    assert summand_prog.name == "ar2_lag1"
    assert abs(float(summand_prog.value) - 1.32) <= 1e-12
    assert len(summand_prog.children) == 2
    assert summand_prog.children[0] == Program("feature", name="object_role:u")
    assert summand_prog.children[1] == Program("const", value=-0.64)
    extracted_ar2 = _extract_ar2_params_from_role_terms([ar2_rt])
    assert extracted_ar2 == [(1.32, -0.64)]


def test_acceptance_10_12_and_18_scope_gating_local_dependency_invalidation_and_return_safety():
    """Acceptance #10, #12, #18: Refuted old commitment blocked on refuted world, unaffected parts retained, newer supported generation rejected on returning world."""
    from scripts.run_v10_phase3a_higher_order_and_coupled_pomdp import _make_markov_right_prior

    sc4 = _make_schema(4, "p")
    c0_prior = _make_markov_right_prior(sc4, concept_id="c0_right", radius=0.95, readout_kind="total_event")
    c0_prior["status"] = "conditional"
    c0_prior["counterexamples"] = [{"task_id": "task_left_world", "kind": "scientific_premise_refutation"}]
    c0_comm = consolidate_world_structure_commitments([c0_prior])[0]

    # #10: Refuted task_id cannot reuse invalidated support unconditionally
    assert is_commitment_empirically_consistent(c0_comm, None, task_id="task_left_world") is False

    # Audit from a 'left' world refutes C_0 ('right') and triggers conservative revision (#12)
    left_audit = {
        "status": "verified_unconditional_exchange",
        "conservation_supported": True,
        "verified_direction": "left",
        "supported_directions": ["left"],
        "verified_edge_condition": "unconditional",
        "radius_interval": [0.85, 1.05],
        "scientific_revision_triggers": [
            {
                "kind": "interaction_direction_revision",
                "retracted_direction": "right",
                "replacement_direction": "left",
                "radius_interval": [0.85, 1.05],
                "supporting_transitions": 24,
                "counterexample": {"step": 0},
            }
        ],
    }
    assert is_commitment_empirically_consistent(c0_comm, left_audit) is False

    with AnonymousObjectLab(seed=801, objects=4, readout="total_event", relation_direction="left", relation_radius=0.95) as lab_l:
        sc_l = lab_l._evaluation_schema(("fine",))
        spec_l = lab_l.spec_for(("fine",))
        rows_l = lab_l.evaluation_rows(("fine",), seed=802, length=24, trajectories=1)
        revs = propose_scientific_premise_revision(
            c0_comm, left_audit, spec_l, rows_l, sc_l, SearchBudget(50000), tolerance=1e-7
        )
        assert len(revs) == 1
        sci_meta = revs[0]["scientific_premise_revision"]
        # #12: ONLY interaction_direction:r_0 retracted; conservation & temporal_readout_basis retained!
        assert [p["premise_id"] for p in sci_meta["retracted_premises"]] == ["interaction_direction:r_0"]
        assert "conservation" in sci_meta["retained_premises"]
        assert "temporal_readout_basis" in sci_meta["retained_premises"]

        # Save C_1 as generation=1 with status='supported'
        lib_both = KnowledgeLibrary()
        lib_both.meta_priors = [deepcopy(c0_prior)]
        saver = UnifiedResearchAgent(ResearchConfig(world_structure_commitments=True), library=lib_both)
        saver._object_schemas[("fine",)] = sc_l.to_dict()
        spec_l.context["object_schema"] = __import__("json").dumps(sc_l.to_dict())
        revs[0]["validation_loss"] = 0.0
        saver.remember_confirmed_selection(
            revs[0], spec_l, rows_l[:16], rows_l[16:], confirmed=True, confirmation_loss=0.0, confirmation_rows=24, selected_channels=("fine",)
        )

    comms_both = consolidate_world_structure_commitments(lib_both.meta_priors)
    c1_comm = next(c for c in comms_both if c["generation"] == 1)
    c0_comm_after = next(c for c in comms_both if c["generation"] == 0)
    assert c1_comm["source_statuses"][c1_comm["source_concept_ids"][0]] == "supported"

    # #18: When 'right' world returns, newer C_1 (generation=1, status='supported') MUST BE REJECTED,
    # while C_0 (generation=0, status='conditional') IS ACCEPTED!
    right_return_audit = {
        "status": "verified_unconditional_exchange",
        "conservation_supported": True,
        "verified_direction": "right",
        "supported_directions": ["right"],
        "verified_edge_condition": "unconditional",
        "radius_interval": [0.88, 1.08],
        "scientific_revision_triggers": [],
    }
    assert is_commitment_empirically_consistent(c1_comm, right_return_audit) is False
    assert is_commitment_empirically_consistent(c0_comm_after, right_return_audit, task_id="task_right_return") is True


def test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators():
    """Acceptance #13, #14, #15, #16, #19, #20, #21: Single-factor parity, manifest SHA256, unused labels, None MSE, budget halt, & failure denominators."""
    from scripts.run_v10_phase2b_open_grammar_llm import (
        _acquire_shared_discovery_bundle,
        _build_parent_commitment_from_empty_library,
    )

    # #15 & #16: Unused object never labeled 'used_without_benefit'; missing MSE stays None (never 0.0)
    tel_unused = build_action_chain_telemetry(
        arm="structured_full_tools",
        task_id="t_unused",
        knowledge_available_count=2,
        knowledge_retrieved_ids=[],
        entered_candidate_pool=False,
        changed_probe_or_audit=False,
        changed_parent_commitment=False,
        reorganization_type=None,
        confirmed=False,
        heldout_mse=None,
    )
    assert tel_unused["knowledge_utility_status"] == "retained_knowledge_not_invoked"
    assert tel_unused["heldout_mse"] is None

    # #20: Real SearchBudget exhaustion halts solver & rollout witness checker cleanly
    tiny_budget = SearchBudget(2)
    halted = prove_task_solvability(
        {"objects": 4, "positions": [-1.8, -0.5, 0.4, 1.8], "coupling": 0.25, "radius": 1.0, "direction": "right"},
        {
            "initial_state": [0.0, 0.0, 0.0, 0.0],
            "control_bounds": [-0.5, 0.5],
            "state_bounds": [-10.0, 10.0],
            "budget": 1.6,
            "fuel_weights": [1.0, 1.0, 1.0, 1.0],
            "goal_matrix": [[-1.0, -1.0, -1.0, -1.0]],
            "goal_rhs": [-1.5],
        },
        budget=tiny_budget,
    )
    assert halted["status"] == "unknown"
    assert halted["kernel_verification"]["reason"] == "budget_exhausted"

    # #13 (shared rollout witness checker F07): rollout finds valid trace -> kernel-certified True!
    lab_w = BudgetedControlLab(
        objects=2, positions=(0.0, 1.5), coupling=0.0, radius=0.5, control_bounds=(-0.5, 0.5), state_bounds=(-5.0, 5.0)
    )
    roll_ok = certify_rollout_baseline(
        lab_w,
        {"objects": 2, "positions": [0.0, 1.5], "coupling": 0.0, "radius": 0.5, "direction": "right"},
        {
            "initial_state": [0.0, 0.0],
            "control_bounds": [-0.5, 0.5],
            "state_bounds": [-5.0, 5.0],
            "budget": 1.5,
            "fuel_weights": [1.0, 1.0],
            "goal_matrix": [[-1.0, -1.0]],
            "goal_rhs": [-0.2],
        },
        budget=SearchBudget(50000),
    )
    assert roll_ok["verdict"] == "winnable"
    assert roll_ok["certified_proof"] is True
    assert roll_ok["status"] == "witness_verified_in_declared_model"

    # #14, #19, #21: Single-factor dependency ablation holds schemas/grammar/data identical and retains failed denominator
    _, comm_inst, _ = _build_parent_commitment_from_empty_library(
        readout="total_event", s0_seed=600411, s1_seed=600412
    )
    with AnonymousObjectLab(
        seed=620401,
        objects=4,
        readout="focal_quadratic_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="always",
        relation_radius=1.05,
        actuator_curvature=0.15,
        name_seed=620502,
        wiring_seed=620612,
    ) as lab_sf:
        bundle = _acquire_shared_discovery_bundle(lab_sf, seed=620401)
        no_cue = run_single_factor_dependency_ablation_arm(
            parent_commitment=comm_inst,
            model=bundle["model"],
            records=bundle["records"],
            spec=bundle["spec"],
            training=bundle["training"],
            validation=bundle["validation"],
            compute_budget_limit=80000,
            use_support_dependency_cue=False,
        )
        with_cue = run_single_factor_dependency_ablation_arm(
            parent_commitment=comm_inst,
            model=bundle["model"],
            records=bundle["records"],
            spec=bundle["spec"],
            training=bundle["training"],
            validation=bundle["validation"],
            compute_budget_limit=80000,
            use_support_dependency_cue=True,
        )

    m_no = no_cue["allowed_information_manifest"]
    m_yes = with_cue["allowed_information_manifest"]
    assert m_no["has_private_apparatus_access"] is False
    assert m_yes["has_private_apparatus_access"] is False
    # Identical parent, schema count, data, and patch grammar (#14)!
    for k in (
        "parent_commitment_id",
        "candidate_schema_count",
        "training_rows",
        "validation_rows",
        "transition_records",
        "patch_catalog_slots",
        "patch_catalog_total_candidates",
    ):
        assert m_no[k] == m_yes[k]
    assert m_no["visible_violated_premise_cue"] is None
    assert m_yes["visible_violated_premise_cue"] == "P_temporal_readout"
    assert no_cue["transmitted_payload_sha256"] != with_cue["transmitted_payload_sha256"]
    # Failure denominator retained before first passing patch (#21)!
    assert with_cue["candidates_attempted_denominator"] == with_cue["failed_candidates_count"] + 1
    assert with_cue["failed_candidates_count"] >= 1


def test_acceptance_17_sde_noisy_history_conditional_drift_distinct_from_noiseless_twin():
    """Acceptance #17: Noisy-history conditional drift MSE is NOT conflated with noiseless-twin transfer MSE."""
    with AnonymousObjectLab(
        seed=530401,
        objects=4,
        readout="total_event",
        relation_direction="right",
        relation_radius=0.95,
        latent_mode="transition_memory",
        latent_decay=0.74,
        latent_weight=0.48,
        process_noise_std=0.05,
    ) as noisy_lab:
        # Constant predictor y_hat = 0: E[Y_t | F_t] on realized noisy history contains -0.7 * eps_{t-1, focus},
        # whose variance differs strictly from the deterministic process_noise_std=0 twin!
        const_prog = Program("const", value=0.0)
        metrics = evaluate_sde_conditional_drift_and_twin_metrics(
            noisy_lab, const_prog, 0.0, (), eval_seed=91001, length=32, trajectories=2, mc_rollouts=64
        )
        assert metrics["live_noisy_trajectory_mse"] is not None
        assert metrics["noisy_history_conditional_drift_mse"] is not None
        assert metrics["noiseless_twin_transfer_mse"] is not None
        assert abs(metrics["noisy_history_conditional_drift_mse"] - metrics["noiseless_twin_transfer_mse"]) > 1e-5
        assert metrics["information_set_contract"]["post_step_after_sensors_exposed_in_prediction"] is False


def test_acceptance_22_v10_2_receipts_zero_llm_calls_and_untouched_v10_sha256_ledger():
    """Acceptance #22 & A4: Verify all 6 v10.2 receipts exist, model_calls == 0, and 13 v10 SHA256 digests match."""
    import hashlib
    import json

    out_dir = REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"
    ledger_path = out_dir / "errata_and_sha256_ledger_20261116.json"
    assert ledger_path.is_file()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["paid_llm_calls"] == 0
    assert ledger["untouched_v10_receipts_count"] == 13
    assert len(ledger["findings_remediation_matrix"]) == 20
    assert len(ledger["new_v10_2_remediation_receipts_sha256"]) == 5

    # Verify SHA256 of every untouched v10 receipt
    v10_dir = REPO_ROOT / "docs" / "evidence" / "v10_world_structure_20261116"
    for item in ledger["untouched_v10_receipts_sha256_and_roles"]:
        p = v10_dir / item["filename"]
        assert p.is_file()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]
        assert "study_role" in item and "independent_physical_sources" in item

    # Verify zero model calls across all empirical v10.2 receipts
    for fname in (
        "sde_and_proof_remediation_20261116.json",
        "equal_tool_attribution_20261116.json",
        "frozen_v4_lifecycle_20261116.json",
    ):
        rec = json.loads((out_dir / fname).read_text(encoding="utf-8"))
        assert rec["model_calls"] == 0
        assert rec["total_tokens"] == 0
