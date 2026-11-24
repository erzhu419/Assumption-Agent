"""Tests for Hegel M8.0 Persistent Mechanism Graph & Variational Action Planner.

Covers:
  1. All 13 mathematical/architectural counterexamples from `Hegel_M7_7_M8_Audit/theory/theory_counterexamples.json`.
  2. Unified Stage-C 5-scenario sequential stream on ONE persistent `PersistentMechanismGraph`
     (`AncestorReweight` + unseen-task forward transfer + zero unrelated drift, `LocalRefit` collateral guard,
     `LocalSplit` localized regime exception, `ConceptLift` nonlinear synergy, `ChangeMemory` non-Markovian history,
     and `NoChange` heteroscedastic noise safety).
"""

from __future__ import annotations

import numpy as np
import pytest

from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    ResearchDecisionRecord,
    SequentialGaussianBelief,
    TypedMechanism,
    compile_safe_feature_conduit,
    compute_deduplicated_model_posterior,
    compute_likelihood_curvature,
    compute_out_of_support_inflation,
    diagnose_residual_signals,
    estimate_conduit_conditioning_and_domain,
    evaluate_gated_vs_mixture_log_likelihood,
    fit_nonlinear_power_parameter_escape_zero_jacobian,
    fit_scale_covariant_nig_linear_model,
    select_discriminating_observation_probe,
)


def test_m8_all_13_gpt6_theory_counterexamples() -> None:
    """Verify all 13 mathematical counterexamples from `theory_counterexamples.json` are cleanly satisfied."""
    rng = np.random.default_rng(20261125)

    # 1. T01: `mixture_prediction_not_mixture_likelihood`
    y0 = np.zeros(6)
    branch_preds = np.tile([[3.0, -3.0]], (6, 1))
    gates = np.tile([[0.5, 0.5]], (6, 1))
    ll_cmp = evaluate_gated_vs_mixture_log_likelihood(y0, branch_preds, gates, sigma=0.5)
    assert ll_cmp["gated_predictor_loglik"] > ll_cmp["mixture_experts_loglik"] + 40.0

    # 2. T02: `gaussian_factor_two` (exact residual 2nd derivative is 1/sigma^2, NOT 2/sigma^2)
    curv_g = compute_likelihood_curvature(np.array([1.2, -0.7]), np.array([[1.0], [2.0]]), likelihood="gaussian", sigma=2.0)
    assert np.allclose(curv_g["residual_second_derivative"], 0.25)
    assert np.allclose(curv_g["gauss_newton_hessian"], [[1.25]])
    assert curv_g["observed_is_psd"] is True

    # 3. T03: `student_t_non_psd` (tail residual |r| > sqrt(nu)*sigma has negative observed curvature, PSD Expected Fisher!)
    curv_t = compute_likelihood_curvature(np.array([4.0]), np.array([[1.0]]), likelihood="student_t", sigma=1.0, student_df=3.0)
    assert curv_t["residual_second_derivative"][0] < -0.10
    assert curv_t["observed_is_psd"] is False
    assert curv_t["expected_fisher_is_psd"] is True
    assert curv_t["expected_fisher"][0, 0] > 0.5

    # 4. T04: `zero_residual_profile_noise_collapse` (exact noiseless fit stays finite under Inverse-Gamma prior!)
    x_clean = np.linspace(-2.0, 2.0, 30)
    Phi_clean = np.column_stack([np.ones_like(x_clean), x_clean])
    nig_zero = fit_scale_covariant_nig_linear_model(Phi_clean, 2.5 * x_clean - 1.0)
    assert np.isfinite(nig_zero["log_marginal_likelihood"])
    assert nig_zero["posterior_sigma2"] > 0.0

    # 5. T05: `candidate_pool_normalization_changes_objective` (fixed grammar prior + semantic deduplication)
    post_a = compute_deduplicated_model_posterior([("m1", "sem_A", 12.0), ("m2", "sem_B", 14.0)])
    post_dup = compute_deduplicated_model_posterior(
        [("m1", "sem_A", 12.0), ("m1_clone1", "sem_A", 12.0), ("m1_clone2", "sem_A", 12.0), ("m2", "sem_B", 14.0)]
    )
    assert abs(post_a["unique_semantic_models"]["m1"]["posterior_prob"] - post_dup["unique_semantic_models"]["m1"]["posterior_prob"]) < 1e-12
    assert len(post_dup["duplicates_collapsed"]) == 2

    # 6 & 7. T08: `coordinate_rescaling_flips_fixed_ridge_r2` (exact invariance under x1 -> 1000*x1, x2 -> 1e-3*x2)
    x1 = rng.normal(0.0, 1.0, 80)
    x2 = rng.normal(0.0, 1.0, 80)
    y_sc = 1.7 * x1 - 0.9 * x2 + rng.normal(0.0, 0.2, 80)
    fit_u = fit_scale_covariant_nig_linear_model(np.column_stack([np.ones(80), x1, x2]), y_sc)
    fit_s = fit_scale_covariant_nig_linear_model(np.column_stack([np.ones(80), 1000.0 * x1, 1e-3 * x2]), y_sc)
    assert abs(fit_u["log_marginal_likelihood"] - fit_s["log_marginal_likelihood"]) < 1e-9
    assert abs(fit_u["train_r2"] - fit_s["train_r2"]) < 1e-9

    diag_u = diagnose_residual_signals(y_sc, {"x": {"f1": x1, "f2": x2}}, ("f1", "f2"))
    diag_s = diagnose_residual_signals(y_sc, {"x": {"f1": 1000.0 * x1, "f2": 1e-3 * x2}}, ("f1", "f2"))
    assert abs(diag_u["linear_shift_r2"] - diag_s["linear_shift_r2"]) < 1e-9

    # 8. T06: `zero_jacobian_parameter_trap` (f(x; w) = w^2 * x initialized at w = 0 where J(0) == 0)
    z_trap = fit_nonlinear_power_parameter_escape_zero_jacobian(x_clean, 4.0 * x_clean, w_init=0.0)
    assert z_trap["jacobian_norm_at_init"] == 0.0
    assert z_trap["solved_by_parameter_refit_without_concept_lift"] is True
    assert abs(z_trap["fitted_w_squared"] - 4.0) < 1e-4

    # 9. T08: `heteroscedastic_noise_not_white_noise` (r = x * eps has zero conditional mean!)
    xh = rng.uniform(-3.0, 3.0, 220)
    rh = (0.1 + 1.2 * np.abs(xh)) * rng.normal(0.0, 1.0, 220)
    diag_h = diagnose_residual_signals(rh, {"x": {"x0": xh}}, ("x0",))
    assert diag_h["linear_shift_r2"] < 0.08
    assert diag_h["heteroscedastic_abs_r2"] > 0.15
    assert diag_h["status"] == "HETEROSCEDASTIC_OR_UNRESOLVED_NOISE"

    # 10. T07: `double_counting_prior_and_data` (sequential update == joint refit; duplicate batch raises ValueError!)
    bel = SequentialGaussianBelief(prior_mean=np.zeros(2), prior_precision=np.eye(2) * 0.5, sigma2=0.25)
    Xb1, yb1 = rng.normal(size=(25, 2)), rng.normal(size=25)
    Xb2, yb2 = rng.normal(size=(30, 2)), rng.normal(size=30)
    bel.sequential_update("batch_1", Xb1, yb1)
    m_seq, P_seq = bel.sequential_update("batch_2", Xb2, yb2)
    m_jnt, P_jnt = bel.joint_refit_from_prior(["batch_1", "batch_2"])
    assert np.allclose(m_seq, m_jnt, atol=1e-11)
    assert np.allclose(P_seq, P_jnt, atol=1e-11)
    with pytest.raises(ValueError, match="double-count"):
        bel.sequential_update("batch_1", Xb1, yb1)

    # 11. T09: `posthoc_label_leakage` (forbidden retrospective keys rejected from pre_action_state!)
    bad_rec = ResearchDecisionRecord(
        decision_id="d1",
        step_index=1,
        task_id="t1",
        pre_action_state={"chosen_was_best": True},
        available_actions=["NoChange"],
        selected_action="NoChange",
        selected_candidate_id="c0",
        execution_status="EXECUTED_OK",
        actual_cost=0.0,
        public_val_log_predictive_delta=0.0,
        collateral_min_delta=0.0,
        rationale="test",
    )
    with pytest.raises(ValueError, match="Post-hoc leakage"):
        bad_rec.validate_prefix_clean()

    # 12. T10: `lipschitz_false_safety_and_overrejection` + `AcquireObservation` active probe utility
    steep_fn = compile_safe_feature_conduit("np.tanh(25.0 * x['f0'])", ("f0",))
    cond_steep = estimate_conduit_conditioning_and_domain(steep_fn, {"f0": np.linspace(-1.0, 1.0, 81)})
    assert cond_steep["finite_on_support"] is True
    assert cond_steep["empirical_lipschitz"] > 10.0
    inf_ood = compute_out_of_support_inflation({"f0": np.array([0.0, 3.5])}, cond_steep["support_bounds"], lipschitz_scale=cond_steep["empirical_lipschitz"])
    assert inf_ood[0] == 0.0 and inf_ood[1] > 0.5

    probe_pick = select_discriminating_observation_probe(
        [
            ("m_prod", compile_safe_feature_conduit("x['f0'] * x['f1']", ("f0", "f1"))),
            ("m_diff", compile_safe_feature_conduit("x['f0'] - x['f1']", ("f0", "f1"))),
        ],
        [{"f0": 0.0, "f1": 0.0}, {"f0": 2.0, "f1": 2.0}],
    )
    assert probe_pick["selected_index"] == 1 and probe_pick["max_utility"] > 3.0

    # 13. T11: `execution_failure_contaminates_mechanism_belief`
    g_err = PersistentMechanismGraph()
    g_err.register_mechanism(
        TypedMechanism(mechanism_id="shared_err", parent_id=None, role="shared_upstream_premise", feature_names=("f0",), shared_weights={"f0": 1.0})
    )
    pl_err = M8VariationalPlanner(g_err)
    x_e = rng.normal(0.0, 1.0, 60)
    pl_err.register_task(
        task_id="t_exec_fault",
        shared_upstream_id="shared_err",
        feature_names=("f0",),
        train_data={"x": {"f0": x_e[:40]}, "y": x_e[:40] + rng.normal(0.0, 0.1, 40)},
        val_data={"x": {"f0": x_e[40:]}, "y": x_e[40:] + rng.normal(0.0, 0.1, 20)},
    )
    res_err = pl_err.evaluate_and_revise_task("t_exec_fault", injected_broken_expr="x['f0'] / 0.0")
    broken_cands = [c for c in res_err["evaluated_candidates"] if c["execution_status"] == "EXECUTION_ERROR"]
    assert broken_cands and broken_cands[0]["epistemic_outcome"] == "UNRESOLVED_EXECUTION_FAILURE"
    assert res_err["selected_candidate"]["action_type"] == "NoChange"
    assert g_err.mechanisms["shared_err"].epistemic_status == "SUPPORTED_IN_SCOPE"


def test_m8_stage_c_five_scenarios_on_single_persistent_graph() -> None:
    """Execute all 5 continual scenarios sequentially on ONE `PersistentMechanismGraph`, ONE planner, and ONE ledger."""
    rng = np.random.default_rng(778899)
    graph = PersistentMechanismGraph()
    planner = M8VariationalPlanner(graph)

    # Register TWO independent shared upstream premises in the SAME persistent graph:
    #   A. `shared_aesthetic_contrast` (initially dislikes high `skin_tone`: -1.6)
    #   B. `shared_unrelated_acoustics` (completely independent physical domain: +2.2*freq - 0.9*damping)
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_aesthetic_contrast",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("skin_tone", "eye_contrast"),
            shared_weights={"skin_tone": -1.6, "eye_contrast": 0.8},
        ),
        tree_id="tree_aesthetic",
    )
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_unrelated_acoustics",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("freq_khz", "damping"),
            shared_weights={"freq_khz": 2.2, "damping": -0.9},
        ),
        tree_id="tree_acoustics",
    )

    def _split_ds(x_dict: dict[str, np.ndarray], y_arr: np.ndarray, n_tr: int, extra_tr: dict | None = None, extra_va: dict | None = None):
        tr = {"x": {k: v[:n_tr] for k, v in x_dict.items()}, "y": y_arr[:n_tr], **(extra_tr or {})}
        va = {"x": {k: v[n_tr:] for k, v in x_dict.items()}, "y": y_arr[n_tr:], **(extra_va or {})}
        return tr, va

    # Register stationary unrelated task `task_0_unrelated_acoustics`
    fq0 = rng.normal(0.0, 1.0, 120)
    dm0 = rng.normal(0.0, 1.0, 120)
    y0 = 2.2 * fq0 - 0.9 * dm0 + rng.normal(0.0, 0.12, 120)
    tr0, va0 = _split_ds({"freq_khz": fq0, "damping": dm0}, y0, 80)
    planner.register_task(
        task_id="task_0_unrelated_acoustics",
        shared_upstream_id="shared_unrelated_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr0,
        val_data=va0,
    )
    unrelated_rmse_before = planner.evaluate_task_zero_shot_from_shared("task_0_unrelated_acoustics")["val_rmse"]

    # -------------------------------------------------------------------------
    # Scenario 1: Global shared upstream preference/parameter shift -> `AncestorReweight`
    # ("以前不喜欢黑皮 `-1.6`，现在喜欢黑皮 `+1.75`") across related tasks `task_1a` & `task_1b`,
    # transferring zero-shot gain to unseen future task `task_1c` with 0.0 drift on `task_0`!
    # -------------------------------------------------------------------------
    for tid in ("task_1a_portrait_eval", "task_1b_studio_eval"):
        sk = rng.normal(0.0, 1.0, 120)
        ey = rng.normal(0.0, 1.0, 120)
        y_shift = 1.75 * sk + 0.8 * ey + rng.normal(0.0, 0.12, 120)
        tr_i, va_i = _split_ds({"skin_tone": sk, "eye_contrast": ey}, y_shift, 80)
        planner.register_task(
            task_id=tid,
            shared_upstream_id="shared_aesthetic_contrast",
            feature_names=("skin_tone", "eye_contrast"),
            train_data=tr_i,
            val_data=va_i,
        )

    # Measure unseen future task `task_1c` zero-shot error BEFORE `AncestorReweight`
    sk_c = rng.normal(0.0, 1.0, 100)
    ey_c = rng.normal(0.0, 1.0, 100)
    yc = 1.75 * sk_c + 0.8 * ey_c + rng.normal(0.0, 0.12, 100)
    tr_c, va_c = _split_ds({"skin_tone": sk_c, "eye_contrast": ey_c}, yc, 60)
    w_before = dict(graph.mechanisms["shared_aesthetic_contrast"].shared_weights)
    pre_transfer_rmse = float(np.sqrt(np.mean((va_c["y"] - (w_before["skin_tone"] * va_c["x"]["skin_tone"] + w_before["eye_contrast"] * va_c["x"]["eye_contrast"])) ** 2)))

    out_s1 = planner.evaluate_and_revise_task("task_1a_portrait_eval")
    assert out_s1["selected_candidate"]["action_type"] == "AncestorReweight"
    assert out_s1["selected_candidate"]["unrelated_max_abs_drift"] == 0.0

    # Register unseen future related task `task_1c_future_transfer` AFTER `AncestorReweight`
    planner.register_task(
        task_id="task_1c_future_transfer",
        shared_upstream_id="shared_aesthetic_contrast",
        feature_names=("skin_tone", "eye_contrast"),
        train_data=tr_c,
        val_data=va_c,
    )
    post_transfer_rmse = planner.evaluate_task_zero_shot_from_shared("task_1c_future_transfer")["val_rmse"]
    assert pre_transfer_rmse > 2.5 and post_transfer_rmse < 0.22

    # Converse collateral check: an idiosyncratic local quirk on ONE task MUST pick `LocalRefit` (not `AncestorReweight`!)
    sk_q = rng.normal(0.0, 1.0, 120)
    ey_q = rng.normal(0.0, 1.0, 120)
    yq = -1.3 * sk_q + 0.8 * ey_q + rng.normal(0.0, 0.12, 120)
    tr_q, va_q = _split_ds({"skin_tone": sk_q, "eye_contrast": ey_q}, yq, 80)
    planner.register_task(
        task_id="task_1d_idiosyncratic_local_quirk",
        shared_upstream_id="shared_aesthetic_contrast",
        feature_names=("skin_tone", "eye_contrast"),
        train_data=tr_q,
        val_data=va_q,
    )
    out_s1_local = planner.evaluate_and_revise_task("task_1d_idiosyncratic_local_quirk")
    assert out_s1_local["selected_candidate"]["action_type"] == "LocalRefit"

    # -------------------------------------------------------------------------
    # Scenario 2: Localized sub-manifold exception -> `LocalSplit`
    # ("不喜欢胖子，但个别骨相极佳 `bone_structure > 1.15` 能看出瘦时很好看")
    # -------------------------------------------------------------------------
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_physique_prior",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("body_mass", "bone_structure"),
            shared_weights={"body_mass": -1.4, "bone_structure": 0.3},
        ),
        tree_id="tree_physique",
    )
    bm = rng.normal(0.0, 1.0, 220)
    bs = rng.normal(0.0, 1.0, 220)
    exc_mask = (bs > 1.15).astype(float)
    y_s2 = -1.4 * bm + 0.3 * bs + exc_mask * (3.4 + 1.8 * bm) + rng.normal(0.0, 0.12, 220)
    tr_s2, va_s2 = _split_ds({"body_mass": bm, "bone_structure": bs}, y_s2, 150)
    planner.register_task(
        task_id="task_2_bone_structure_exception",
        shared_upstream_id="shared_physique_prior",
        feature_names=("body_mass", "bone_structure"),
        train_data=tr_s2,
        val_data=va_s2,
    )
    out_s2 = planner.evaluate_and_revise_task("task_2_bone_structure_exception")
    assert out_s2["selected_candidate"]["action_type"] == "LocalSplit"
    assert out_s2["selected_candidate"]["metadata"]["scope_guard"]["field"] == "bone_structure"

    # -------------------------------------------------------------------------
    # Scenario 3: Missing nonlinear feature interaction -> `ConceptLift`
    # ("深肤色衬托眼睛更大更有异域风情": `skin_tone * eye_contrast` multiplicative conduit)
    # -------------------------------------------------------------------------
    sk3 = rng.normal(0.0, 1.0, 180)
    ey3 = rng.normal(0.0, 1.0, 180)
    w_cur = graph.mechanisms["shared_aesthetic_contrast"].shared_weights
    y_s3 = w_cur["skin_tone"] * sk3 + w_cur["eye_contrast"] * ey3 + 2.35 * (sk3 * ey3) + rng.normal(0.0, 0.12, 180)
    tr_s3, va_s3 = _split_ds({"skin_tone": sk3, "eye_contrast": ey3}, y_s3, 120)
    planner.register_task(
        task_id="task_3_exotic_eye_synergy",
        shared_upstream_id="shared_aesthetic_contrast",
        feature_names=("skin_tone", "eye_contrast"),
        train_data=tr_s3,
        val_data=va_s3,
    )
    out_s3 = planner.evaluate_and_revise_task("task_3_exotic_eye_synergy")
    assert out_s3["selected_candidate"]["action_type"] == "ConceptLift"
    assert "skin_tone" in out_s3["selected_candidate"]["metadata"]["expression"]
    assert "eye_contrast" in out_s3["selected_candidate"]["metadata"]["expression"]

    # -------------------------------------------------------------------------
    # Scenario 4: Same present, different past -> `ChangeMemory` (`Pi_fold` memory kernel)
    # -------------------------------------------------------------------------
    n4 = 180
    hist4 = rng.normal(0.0, 1.5, size=(n4, 8))
    decay_true = np.exp(-1.0 / 2.0)
    z_true = np.zeros(n4)
    for k in range(8):
        z_true = decay_true * z_true + (1.0 - decay_true) * hist4[:, k]
    # Present features are decorrelated from past history (`same present, different past -> different future`!)
    fq4 = rng.normal(0.0, 1.0, n4)
    dm4 = rng.normal(0.0, 1.0, n4)
    y_s4 = 2.2 * fq4 - 0.9 * dm4 + 2.6 * z_true + rng.normal(0.0, 0.12, n4)
    tr_s4, va_s4 = _split_ds(
        {"freq_khz": fq4, "damping": dm4},
        y_s4,
        120,
        extra_tr={"past_driver_history": hist4[:120], "history_dt": 1.0},
        extra_va={"past_driver_history": hist4[120:], "history_dt": 1.0},
    )
    planner.register_task(
        task_id="task_4_nonmarkovian_memory",
        shared_upstream_id="shared_unrelated_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr_s4,
        val_data=va_s4,
    )
    out_s4 = planner.evaluate_and_revise_task("task_4_nonmarkovian_memory")
    assert out_s4["selected_candidate"]["action_type"] == "ChangeMemory"

    # -------------------------------------------------------------------------
    # Scenario 5: Heteroscedastic zero-mean noise -> `NoChange` (`UNRESOLVED` safety!)
    # -------------------------------------------------------------------------
    fq5 = rng.normal(0.0, 1.0, 180)
    dm5 = rng.normal(0.0, 1.0, 180)
    het_noise = (0.12 + 0.55 * np.abs(fq5)) * rng.normal(0.0, 1.0, 180)
    y_s5 = 2.2 * fq5 - 0.9 * dm5 + het_noise
    tr_s5, va_s5 = _split_ds({"freq_khz": fq5, "damping": dm5}, y_s5, 120)
    planner.register_task(
        task_id="task_5_heteroscedastic_noise",
        shared_upstream_id="shared_unrelated_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr_s5,
        val_data=va_s5,
    )
    out_s5 = planner.evaluate_and_revise_task("task_5_heteroscedastic_noise")
    assert out_s5["selected_candidate"]["action_type"] == "NoChange"

    # Verify persistent graph integrity, zero unrelated collateral drift, and prefix-clean ledger!
    unrelated_rmse_after = planner.evaluate_task_zero_shot_from_shared("task_0_unrelated_acoustics")["val_rmse"]
    assert abs(unrelated_rmse_after - unrelated_rmse_before) < 1e-12
    assert len(graph.decision_ledger) == 6
    for drec in graph.decision_ledger:
        drec.validate_prefix_clean()
    assert len(graph.forest.nodes) >= 12
