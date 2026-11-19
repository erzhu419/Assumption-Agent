"""M0 Unit & Integration Controls for Hegel V16.6 Contracts, Math Counterexamples, and Wormhole Governance."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from hegel_machine.cognition.family_plugins import (
    BottleneckSaturationPlugin,
    OrderedBoundaryPlugin,
    RelaxationFamilyPlugin,
)
from hegel_machine.cognition.growth_methods import ResearchMethodBook
from hegel_machine.cognition.measurement_lab import MeasurementChannel
from hegel_machine.cognition.models import Candidate, Observation, Probe, TaskSpec
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent, open_assigned_question
from hegel_machine.cognition.question_semantics import (
    adopt_question_semantics,
    evaluate_decision_region,
    validate_answer_type,
)
from hegel_machine.cognition.workflow_execution import WorkflowEpisode
from hegel_machine.cognition.wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    TwoStageWormholeGate,
    authorize_abstraction_use,
    compose_wormhole_chain,
    fidelity_compress_nodes,
)


def test_m0_question_identity_wires_all_contracts_and_invalidates_stale_answers(tmp_path) -> None:
    """Directly verifies fix for GPT-6's verify_question_identity_integration.py control."""
    agent = InquiryLLMAgent(None, tmp_path, arm="hegel")
    q_init = {
        "id": "q_contract_check",
        "question": "Does the target satisfy the scoped saturation contract?",
        "kind": "mechanism",
        "answer_type": "scoped_outcome",
        "answer_resolution_ref": {"kind": "fixed"},
        "query_semantics": {"mode": "model_query", "query_expression": {"op": "HypothesisTest"}},
        "protocol_spec": {"reset_requirement": "cold"},
        "readout_spec": {"raw_channel": "I"},
        "scale_spec": {"coarse_graining_op": "identity"},
        "task_view": {"goal_kind": "steady_mean_throughput", "tolerance": 0.02},
        "observation_map": {"raw_channel": "throughput", "unit_scale": 1.0, "sign_convention": 1},
        "scale_map": {"coarse_graining_op": "window_average", "closure_scope": "steady_state_only"},
        "claim_contract": {"claim_id": "rat_sat", "tolerance": 0.02},
    }
    live = open_assigned_question(agent, q_init)
    live["semantic_answer"] = {"status": "answered", "value": {"outcome_status": "SUPPORTED_IN_SCOPE"}, "version": 1}
    agent.research_input_bindings = {"consumer.py": {"slot": "q_contract_check"}}
    agent.research_candidates = {"consumer.py": {"status": "evaluated", "current_evaluation": {"mse": 0.0}}}

    # Cosmetic label edit must NOT bump meaning_version or invalidate cache
    open_assigned_question(agent, {
        "id": "q_contract_check",
        "task_view": {"goal_kind": "steady_mean_throughput", "tolerance": 0.02, "display_label": "Mean Tput"},
    })
    assert live["meaning_version"] == 1
    assert "semantic_answer" in live
    assert agent.research_candidates["consumer.py"]["status"] == "evaluated"

    # Operational edits to protocol_spec / scale_map / observation_map / task_view MUST bump meaning_version!
    expected_ver = 1
    for field, new_val in [
        ("protocol_spec", {"reset_requirement": "warm"}),
        ("readout_spec", {"raw_channel": "flux"}),
        ("scale_spec", {"coarse_graining_op": "window_average"}),
        ("task_view", {"goal_kind": "burst_p99_tail_latency", "tolerance": 0.02}),
        ("observation_map", {"raw_channel": "throughput", "unit_scale": 1.0, "sign_convention": -1}),
        ("scale_map", {"coarse_graining_op": "identity", "closure_scope": "exact"}),
    ]:
        live["semantic_answer"] = {"status": "answered", "value": {"outcome_status": "SUPPORTED_IN_SCOPE"}, "version": 1}
        agent.research_candidates["consumer.py"] = {"status": "evaluated", "current_evaluation": {"mse": 0.0}}
        open_assigned_question(agent, {"id": "q_contract_check", field: new_val})
        expected_ver += 1
        assert live["meaning_version"] == expected_ver
        assert "semantic_answer" not in live
        assert agent.research_candidates["consumer.py"]["status"] == "needs_rebuild"
        assert "current_evaluation" not in agent.research_candidates["consumer.py"]

    assert validate_answer_type({"outcome_status": "REFUTED_IN_SCOPE"}, "scoped_outcome")
    assert not validate_answer_type({"outcome_status": "MADE_UP_STATE"}, "scoped_outcome")


def test_m0_decision_region_requires_action_ref_and_rejects_empty_candidate_set() -> None:
    """HEC decision region must bind concrete action_ref and route empty set to MODEL_MISMATCH."""
    with pytest.raises(ValueError, match="action_ref"):
        evaluate_decision_region({"action_ref": "L3", "output_contract": {"tol": 0.01}}, ["h1"])

    region = {
        "action_ref": "fit_rational_saturation_original_scale",
        "output_contract": {"metric": "rmse_original_scale", "tol": 0.02},
        "tolerated_loss": 0.02,
        "compatible_candidate_refs": ["amdahl_norm", "mm_norm"],
        "independent_confirmation_ref": "heldout_amplitude_probe",
    }
    empty_res = evaluate_decision_region(region, [])
    assert empty_res["status"] == "MODEL_MISMATCH"
    assert empty_res["stop_discrimination"] is False

    eq_res = evaluate_decision_region(region, ["amdahl_norm", "mm_norm"])
    assert eq_res["status"] == "DECISION_EQUIVALENT_IN_SCOPE"
    assert eq_res["stop_discrimination"] is True
    assert eq_res["requires_independent_confirmation"] is True


def test_m0_nine_mathematical_and_physical_counterexamples() -> None:
    """Verifies all 6 counterexamples from verify_multiscale_contract.py + 3 V16 M0 counterexamples."""
    # 1. Epsilon closeness non-transitivity (0.0, 0.75, 1.5 with eps=1.0)
    e1 = {"wormhole_id": "wh_01", "hop_error_budget": 0.75, "lipschitz_factor": 1.0, "permission_tier": "local_solver_guidance"}
    e2 = {"wormhole_id": "wh_12", "hop_error_budget": 0.75, "lipschitz_factor": 1.0, "permission_tier": "local_solver_guidance"}
    chain = compose_wormhole_chain([e1, e2], target_tolerance=1.0)
    assert chain["cumulative_error_budget"] == pytest.approx(1.5)
    assert chain["within_tolerance"] is False
    assert chain["effective_permission_tier"] == "bounded_trial"

    # 2. Non-Markov state aggregation: x1'=-x1, x2'=-2x2, z=x1+x2
    z_t1_a, z_t1_b = math.exp(-1.0), math.exp(-2.0)
    assert abs(z_t1_a - z_t1_b) > 0.23
    scale_agg = ScaleMap(
        coarse_graining_op="aggregate_sum",
        preserved_queries=["steady_state_equilibrium"],
        closure_scope="steady_state_only",
        proposed_discarded_variables=["mode_split_x1_minus_x2"],
    ).to_dict()
    claim = ClaimContract("agg_decay", "relaxation", [], {"action": "predict"}, {}, 0.05, "SUPPORTED_IN_SCOPE").to_dict()
    denied_transient = authorize_abstraction_use(
        requested_tier="local_solver_guidance",
        task_view={"goal_kind": "transient_trajectory", "tolerance": 0.05},
        target_features={},
        claim_contract=claim,
        scale_map=scale_agg,
        heldout_confirmed=True,
    )
    assert denied_transient["granted"] is False
    assert denied_transient["reason"] == "scale_map_does_not_preserve_task_view"

    # 3. Globally strictly convex nonlinear gradient system with finite-step enhancement
    def fz(u: float, z: float) -> float:
        return z + math.exp(u + z) - 0.1 * math.exp(2 * u - z) - 0.9

    def fu(u: float, z: float) -> float:
        return u + math.exp(u + z) + 0.2 * math.exp(2 * u - z)

    u_step = math.log(6.0)
    lo, hi = -10.0, 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if fz(u_step, mid) < 0:
            lo = mid
        else:
            hi = mid
    z_star = 0.5 * (lo + hi)
    p0, p_fr, p_rel = fu(0.0, 0.0), fu(u_step, 0.0), fu(u_step, z_star)
    p_dot_initial = -(math.exp(u_step) - 0.2 * math.exp(2 * u_step)) * fz(u_step, 0.0)
    assert p_dot_initial == pytest.approx(1.8)
    assert (p_rel - p0) / (p_fr - p0) == pytest.approx(1.02237635188368, rel=1e-9)
    a_eq, b_eq = math.exp(u_step + z_star), 0.1 * math.exp(2 * u_step - z_star)
    schur = RelaxationFamilyPlugin.local_schur_stiffness_bound(1 + a_eq + 4 * b_eq, a_eq - 2 * b_eq, 1 + a_eq + b_eq)
    assert schur["valid"] and schur["attenuated_le_frozen"] and schur["nonnegative_relaxed"]
    assert schur["k_relaxed"] < schur["k_frozen"]

    # 4. Zero-RL fixed initial linkage & exact sign-corrected flux recovery psi = u - L*p
    l_ind, psi0, u_end = 2.0, 3.0, 5.0
    p_minus_i = (u_end - psi0) / l_ind
    assert RelaxationFamilyPlugin.recover_total_flux_linkage(u_end, l_ind, p_minus_i) == pytest.approx(psi0)

    # 5. SLS topology parameter distinction
    kelvin = RelaxationFamilyPlugin.sls_parameters_by_topology(2.0, 3.0, 5.0, topology="kelvin_series")
    maxwell = RelaxationFamilyPlugin.sls_parameters_by_topology(2.0, 3.0, 5.0, topology="maxwell_parallel")
    assert kelvin["tau"] == pytest.approx(1.0) and kelvin["frozen_stiffness"] == pytest.approx(2.0)
    assert maxwell["tau"] == pytest.approx(2.5) and maxwell["frozen_stiffness"] == pytest.approx(5.0)

    # 6. Bimodal Maxwell complete relaxation (not single-mode)
    g = lambda t: 2.0 * math.exp(-t) + 3.0 * math.exp(-t / 4.0)
    r1 = g(1.0) / g(0.0)
    r2 = g(2.0) / g(1.0)
    assert r1 != pytest.approx(r2) and g(100.0) < 1e-10

    # 7 & 8. RL unreset history sign reversal (0 < t < ln 2) & smooth pulse ratio theta/3 > 1
    # L*dI/dt + R*I = -du/dt with L=R=1, I(0)=-1, u(t)=t => I(t) = -1 + (1 - 0)^... -> I(t) = -1 (wait: dI/dt + I = -1 => I(t)=-1! For I(0)=+1, u(t)=t: dI/dt+I=-1 => I(t)=-1+2*exp(-t), so p=-I = 1-2*exp(-t) < 0 for t < ln 2!)
    t_test = 0.3  # < ln(2) ~= 0.693
    p_unreset = 1.0 - 2.0 * math.exp(-t_test)
    assert p_unreset < 0.0  # Instantaneous u_dot * p < 0 due to unreset initial current!
    # Smooth pulse u(theta) = theta^2 * exp(-theta): response ratio exceeds 1 for theta > 3
    theta = 4.0
    # Exact ODE d(psi)/dtheta + (psi - u) = 0 with psi(0)=0 gives p = u - psi = (theta^2 - 2*theta + 2)*exp(-theta) - 2*exp(-theta)...
    # Specifically for u(theta)=theta^2*e^{-theta}, psi(theta)=e^{-theta}*int_0^theta s^2 ds = (theta^3/3)*e^{-theta}!
    u_pulse = (theta ** 2) * math.exp(-theta)
    psi_pulse = (theta ** 3 / 3.0) * math.exp(-theta)
    assert psi_pulse / u_pulse == pytest.approx(theta / 3.0) and (psi_pulse / u_pulse) > 1.0

    # 9. Nonlinear sensor y_i = c_i^2 raw sum drift under exact mass conservation c1 + c2 = 1.0
    c_init, c_late = (0.9, 0.1), (0.5, 0.5)
    assert sum(c_init) == pytest.approx(sum(c_late))
    y_init, y_late = sum(x * x for x in c_init), sum(x * x for x in c_late)
    assert y_init != pytest.approx(y_late)


def test_m0_conditional_refutation_witness_compiles_and_branches_in_replay() -> None:
    """growth_methods + workflow_execution route on outcome_status and isolate execution_error."""
    bounds = {"u": (0.0, 10.0)}
    spec = TaskSpec("branch_task", "Conditional workflow test", ("u",), (), (),
                    feature_bounds=bounds, context={"control_roles": {"u": "focus"}})
    channels = (MeasurementChannel("ch_main", 1, "sensor"),)
    seq_pilot = [{"u": 1.0}, {"u": 3.0}, {"u": 5.0}, {"u": 9.0}]
    seq_extra = [{"u": 2.0}, {"u": 7.0}]
    seq_conf = [{"u": 4.0}, {"u": 8.0}]

    trace = [
        {"event_id": "e_pilot", "action": "measure", "produces": ["rows_pilot"],
         "recipe": {"sequence": seq_pilot, "channels": ["ch_main"]}},
        {"event_id": "e_crash", "action": "measure", "produces": ["bad_rows"],
         "execution_error": True, "outcome_status": "EXECUTION_ERROR",
         "recipe": {"sequence": seq_pilot, "channels": ["ch_main"]}},
        {"event_id": "e_check", "action": "scientific_refutation_witness",
         "consumes": ["rows_pilot"], "produces": ["witness_out"],
         "outcome_status": "REFUTED_IN_SCOPE",
         "recipe": {"predicate_spec": {"kind": "diagnostic_predicate",
                                       "field": "monotone_flag", "op": "==", "value": True,
                                       "on_false": "REFUTED_IN_SCOPE"}}},
        {"event_id": "e_refuted_extra", "action": "measure",
         "consumes": ["rows_pilot"], "produces": ["rows_extra"],
         "when_outcome": {"from_event": "e_check", "outcomes": ["REFUTED_IN_SCOPE"]},
         "recipe": {"sequence": seq_extra, "channels": ["ch_main"]}},
        {"event_id": "e_fit", "action": "fit",
         "consumes": ["rows_pilot"], "produces": ["fitted_model"],
         "recipe": {"estimator": "residual_expression_search", "channels": ["ch_main"]}},
        {"event_id": "e_confirm", "action": "confirm", "consumes": ["fitted_model"],
         "recipe": {"sequence": seq_conf, "channels": ["ch_main"]}},
    ]
    book = ResearchMethodBook()
    diag = {"feature_bounds": bounds, "control_roles": {"u": "focus"},
            "channel_names": ["ch_main"], "feature_count": 1, "alias_error": 0.1}
    method = book.learn(trace, diag, success=True, task_id="src_branch")
    actions_compiled = [n["action"] for n in method["nodes"]]
    assert "scientific_refutation_witness" in actions_compiled
    guarded = [n for n in method["nodes"] if "when_outcome" in n]
    assert len(guarded) == 1 and guarded[0]["when_outcome"]["outcomes"] == ["REFUTED_IN_SCOPE"]

    # Replay 1: monotone_flag=True -> condition yields SUPPORTED_IN_SCOPE -> extra probe is SKIPPED!
    ep_sup = WorkflowEpisode(book, spec, channels, context={"control_roles": {"u": "focus"}})
    log_sup: list[dict] = []
    acquired_sup = []

    def acquire_ok(ch, purpose, probe):
        acquired_sup.append(len(probe.sequence))
        return [Observation({"u": r["u"]}, 2.0 * r["u"]) for r in probe.sequence]

    def fit_stub(subset, train, valid):
        cand = Candidate("lin", Program("mul", children=(Program("const", value=2.0), Program("feature", name="u"))), "fit")
        return [{"candidate": cand, "training_loss": 0.0, "validation_loss": 0.0}]

    ep_sup.diagnostics["monotone_flag"] = True
    res_sup = ep_sup.replay(
        acquire=acquire_ok,
        project=lambda obs, src, sub: obs,
        fit=fit_stub,
        legal_views=[("ch_main",)],
        remaining=lambda: {"observation_remaining": 50, "measurement_remaining": 50, "compute_remaining": 200},
        confirmation_length=2,
        threshold=1e-6,
        diagnostics={"alias_error": 0.1},
        actions=log_sup,
    )
    assert res_sup["adequate"] is True
    assert res_sup["node_outcomes"]["step_1"] == "SUPPORTED_IN_SCOPE"
    assert len(res_sup["skipped_steps"]) == 1
    assert acquired_sup == [4]  # Extra 2-row refuted branch was NOT purchased!

    # Replay 2: monotone_flag=False -> condition yields REFUTED_IN_SCOPE -> extra probe IS executed!
    ep_ref = WorkflowEpisode(book, spec, channels, context={"control_roles": {"u": "focus"}})
    acquired_ref = []
    ep_ref.diagnostics["monotone_flag"] = False
    res_ref = ep_ref.replay(
        acquire=lambda ch, p, probe: (acquired_ref.append(len(probe.sequence)) or [Observation({"u": r["u"]}, 2.0 * r["u"]) for r in probe.sequence]),
        project=lambda obs, src, sub: obs,
        fit=fit_stub,
        legal_views=[("ch_main",)],
        remaining=lambda: {"observation_remaining": 50, "measurement_remaining": 50, "compute_remaining": 200},
        confirmation_length=2,
        threshold=1e-6,
        diagnostics={"alias_error": 0.1},
        actions=[],
    )
    assert res_ref["node_outcomes"]["step_1"] == "REFUTED_IN_SCOPE"
    assert len(res_ref["skipped_steps"]) == 0
    assert acquired_ref == [4, 2]

    # Replay 3: Tool execution_error must NOT record a false scientific failure in book
    examples_before = len(book.methods[0]["examples"])
    ep_err = WorkflowEpisode(book, spec, channels, context={"control_roles": {"u": "focus"}})
    res_err = ep_err.replay(
        acquire=lambda ch, p, probe: {"execution_error": True, "reason": "simulator_timeout"},
        project=lambda obs, src, sub: obs,
        fit=fit_stub,
        legal_views=[("ch_main",)],
        remaining=lambda: {"observation_remaining": 50, "measurement_remaining": 50, "compute_remaining": 200},
        confirmation_length=2,
        threshold=1e-6,
        diagnostics={"alias_error": 0.1},
        actions=[],
    )
    assert res_err["execution_error"] is True
    assert len(book.methods[0]["examples"]) == examples_before


def test_m0_forest_horizontal_wormholes_and_or_support_and_fidelity_compression() -> None:
    """Verifies mid-layer wormholes, self-exemption block, FidelityCompression, and clean removal."""
    forest = PhilosophicalForest()
    forest.record_raw_episode({"episode_id": "raw_cpu_1", "points": [(1, 0.5), (2, 0.67)]})
    forest.record_raw_episode({"episode_id": "raw_bio_1", "points": [(1, 0.4), (4, 0.8)]})

    tv = TaskView("resource_sizing", "rmse_original_scale", 0.03)
    n_cpu_mid = forest.register_node(
        node_id="cpu_mid_rat", tree_id="tree_cpu", layer_role="mid",
        task_view=tv,
        observation_map=ObservationMap("speedup_norm", parameter_semantics={"alpha": "serial_fraction"}),
        claim_contract=ClaimContract("rat_family", "bottleneck", [], {"family": "rational_saturation"}, {}, 0.03, "SUPPORTED_IN_SCOPE"),
        probe_signatures=["sig:diminishing_marginal", "sig:reciprocal_linear_candidate"],
        source_cluster_ids=["cluster_cpu_01"],
    )
    n_bio_mid = forest.register_node(
        node_id="bio_mid_rat", tree_id="tree_bio", layer_role="mid",
        task_view=tv,
        observation_map=ObservationMap("rate_norm", parameter_semantics={"alpha": "inv_vmax"}),
        claim_contract=ClaimContract("rat_family_bio", "bottleneck", [], {"family": "rational_saturation"}, {}, 0.03, "SUPPORTED_IN_SCOPE"),
        probe_signatures=["sig:diminishing_marginal", "sig:reciprocal_linear_candidate"],
        source_cluster_ids=["cluster_bio_01"],
    )
    # Exclusive child depending ONLY on cpu_mid_rat vs shared derived node with OR-paths from cpu OR bio
    forest.register_node(
        node_id="cpu_leaf_amdahl", tree_id="tree_cpu", layer_role="leaf",
        task_view=tv,
        observation_map=ObservationMap("speedup_norm", parameter_semantics={"alpha": "1_minus_parallel_f"}),
        claim_contract=ClaimContract("amdahl_exact", "bottleneck", [{"field": "lock_free", "op": "==", "value": False}], {"family": "amdahl"}, {}, 0.01, "SUPPORTED_IN_SCOPE"),
        support_paths=[[{"node_id": "cpu_mid_rat", "version": n_cpu_mid["version"]}]],
    )
    forest.register_node(
        node_id="shared_rat_diagnostic", tree_id="tree_shared", layer_role="mid",
        task_view=tv,
        observation_map=ObservationMap("norm_out"),
        claim_contract=ClaimContract("shared_diag", "bottleneck", [], {"probe": "3pt_rational_check"}, {}, 0.03, "SUPPORTED_IN_SCOPE"),
        support_paths=[
            [{"node_id": "cpu_mid_rat", "version": n_cpu_mid["version"]}],
            [{"node_id": "bio_mid_rat", "version": n_bio_mid["version"]}],
        ],
    )

    wh = forest.add_horizontal_wormhole(
        wormhole_id="wh_cpu_bio_mid",
        node_a_id="cpu_mid_rat",
        node_b_id="bio_mid_rat",
        probe_signature_refs=["sig:reciprocal_linear_candidate"],
        role_and_observation_map={"u_map": "workers->substrate", "leaf_semantics_isolated": True},
        hop_error_budget=0.01,
    )
    # Near-miss ALIGN exclusion refinement (unbounded sqrt concave rejected!)
    wh_rev = forest.refine_wormhole_with_near_miss(
        "wh_cpu_bio_mid",
        near_miss_episode_id="ep_sqrt_nearmiss",
        exclusion_predicate={"field": "unbounded_growth_ratio", "op": ">", "value": 1.35},
        reason="Unbounded concave growth violates finite saturation asymptote",
    )
    assert wh_rev["version"] == 2
    denied = authorize_abstraction_use(
        requested_tier="bounded_trial",
        task_view=tv.to_dict(),
        target_features={"unbounded_growth_ratio": 1.42},
        claim_contract=n_bio_mid["claim_contract"],
        wormhole_edge=wh_rev,
    )
    assert denied["granted"] is False and denied["reason"] == "align_near_miss_exclusion_triggered"

    # Clean versioned removal: exclusive cpu_leaf_amdahl & wh_cpu_bio_mid are removed,
    # whereas OR-supported shared_rat_diagnostic AND raw_episode_archive survive!
    pruned = forest.without_node_version("cpu_mid_rat", 1)
    assert "cpu_mid_rat" not in pruned.nodes
    assert "cpu_leaf_amdahl" not in pruned.nodes
    assert "wh_cpu_bio_mid" not in pruned.horizontal_wormhole_edges
    assert "bio_mid_rat" in pruned.nodes
    assert "shared_rat_diagnostic" in pruned.nodes
    assert len(pruned.raw_episode_archive) == 2

    # FidelityCompression must reject over-restrictive merge that fabricates false rejections
    node_pos_a = deepcopy(n_cpu_mid)
    node_pos_a["claim_contract"]["preconditions"] = [{"field": "domain_tag", "op": "==", "value": "A"}]
    node_pos_b = deepcopy(n_bio_mid)
    node_pos_b["claim_contract"]["preconditions"] = [{"field": "domain_tag", "op": "==", "value": "B"}]

    def eval_rule(node, q):
        # Out-of-scope precondition evaluates to UNK unless explicitly violated on valid input;
        # conjoining incompatible domain preconditions turns valid ACC cases into lost_accepts / false_new_rejects!
        if q.get("empirical_bad"):
            return "REJ"
        matches = all(q.get(r["field"]) == r["value"] for r in node["claim_contract"]["preconditions"])
        if len(node["claim_contract"]["preconditions"]) > 1 and not matches:
            return "REJ"  # Over-restrictive merged rule falsely rejects!
        return "ACC" if matches else "UNK"

    comp = fidelity_compress_nodes(
        node_pos_a, node_pos_b,
        [{"domain_tag": "A"}, {"domain_tag": "B"}, {"domain_tag": "C", "empirical_bad": True}],
        eval_rule,
    )
    assert comp["accepted"] is False
    assert comp["lost_accept_indices"] == [0, 1]
    assert comp["false_new_reject_indices"] == [0, 1]
