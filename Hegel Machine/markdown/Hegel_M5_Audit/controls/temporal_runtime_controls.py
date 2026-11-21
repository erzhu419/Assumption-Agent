"""Read-only, narrowly targeted controls for the pinned M5 review."""
import copy
import json
import random
from unittest.mock import patch

import hegel_machine.cognition.m5_temporal_multiscale_and_base_subtree as m5
from hegel_machine.cognition.hipporag_wormhole_index import (
    FlatEqualMetadataIndex,
    HippoRAGWormholeIndex,
)


def run_pipeline(ds, forced_selection=None, disable_temporal_helpers=False):
    forest, priors = m5._build_seed_forest_and_priors()
    observed_fit_inputs = []
    original_fit = m5._fit_candidate_model_class

    def recording_fit(model_class, train_rows, **kwargs):
        observed_fit_inputs.append({
            "model_class": model_class,
            "train_rows": copy.deepcopy(train_rows),
            "tau_scale": kwargs["tau_scale"],
        })
        return original_fit(model_class, train_rows, **kwargs)

    def forbidden_helper(*args, **kwargs):
        raise AssertionError("Temporal observation/folding helper was consumed")

    kwargs = dict(
        forest=forest, priors_bundle=priors,
        hippo_idx=HippoRAGWormholeIndex(forest),
        flat_idx=FlatEqualMetadataIndex(forest), rng=random.Random(733),
    )
    with patch.object(m5, "_fit_candidate_model_class", recording_fit):
        if forced_selection is not None:
            with patch.object(m5, "select_m5_discriminating_probe", return_value=forced_selection):
                receipt = m5._run_genuine_solver_pipeline(ds, "Hegel_M5_Full_TemporalSubtree_Forest", **kwargs)
        elif disable_temporal_helpers:
            with patch.object(m5, "extract_causal_temporal_regimes", forbidden_helper), patch.object(m5, "predict_via_pi_fold_mode", forbidden_helper):
                receipt = m5._run_genuine_solver_pipeline(ds, "Hegel_M5_Full_TemporalSubtree_Forest", **kwargs)
        else:
            receipt = m5._run_genuine_solver_pipeline(ds, "Hegel_M5_Full_TemporalSubtree_Forest", **kwargs)
    return {"receipt": receipt, "fits": observed_fit_inputs}


results = {}
c_ds = m5._build_disjoint_dataset(
    "audit_subtree_probe", "audit", "C_two_layer_conjunctive_subtree", False,
    ["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"], False,
    {"vmax": 6.8, "rho": 0.70, "local_branch_coeff": 0.40, "tau_scale": 1.0},
    random.Random(104),
)
c_runs = {}
for selected, action in (
    ("open_branch_discriminating_probe", "acquire_probe"),
    ("nuisance_high_var_aux", "acquire_probe"),
    (None, "hec_decision_region_stop"),
):
    c_runs[str(selected)] = run_pipeline(c_ds, {
        "selected_probe_id": selected, "action": action,
        "compute_cost": 2.0,
    })
results["forced_subtree_probe"] = {
    "receipts": {k: v["receipt"] for k, v in c_runs.items()},
    "fit_inputs_identical": c_runs["open_branch_discriminating_probe"]["fits"] == c_runs["nuisance_high_var_aux"]["fits"] == c_runs["None"]["fits"],
    "first_fit_train_count": len(c_runs["None"]["fits"][0]["train_rows"]),
    "open_probe_consumed_even_after_hec_stop": c_ds.probe_pool["open_branch_discriminating_probe"]["row"] in c_runs["None"]["fits"][0]["train_rows"],
}

b_ds = m5._build_disjoint_dataset(
    "audit_reversal_probe", "audit", "B_short_gain_long_reversal", False,
    ["sig:short_window_positive_gain", "sig:endogenous_adaptation_candidate"], False,
    {"amp_fast": 1.45, "tau_fast": 0.35, "amp_slow": 2.4, "tau_slow": 12.0, "horizon_H": 20.0},
    random.Random(104),
)
b_runs = {}
for selected in ("slow_horizon_discriminating_probe", "nuisance_high_var_aux"):
    b_runs[selected] = run_pipeline(b_ds, {
        "selected_probe_id": selected, "action": "acquire_probe",
        "compute_cost": 2.0,
    })
results["forced_reversal_probe"] = {
    "receipts": {k: v["receipt"] for k, v in b_runs.items()},
    "fit_inputs_identical": b_runs["slow_horizon_discriminating_probe"]["fits"] == b_runs["nuisance_high_var_aux"]["fits"],
    "first_fit_train_count": len(b_runs["nuisance_high_var_aux"]["fits"][0]["train_rows"]),
    "slow_probe_consumed_after_nuisance_selection": b_ds.probe_pool["slow_horizon_discriminating_probe"]["row"] in b_runs["nuisance_high_var_aux"]["fits"][0]["train_rows"],
}

mem_ds = m5._build_disjoint_dataset(
    "audit_memory_helper_consumption", "audit", "B_memory_kernel_fold", False,
    ["sig:slow_horizon_query", "sig:coupled_transient_lag"], True,
    {"a_slow": 1.85, "b_couple": 1.25, "c_drive": 1.4, "tau_fast": 0.65, "g_feedback": 0.52, "horizon_H": 0.44},
    random.Random(104),
)
results["memory_task_with_both_temporal_helpers_disabled"] = run_pipeline(mem_ds, disable_temporal_helpers=True)["receipt"]

params = {
    "a_slow": 1.0, "b_couple": 0.0, "c_drive": 0.0,
    "tau_fast": 1.0, "g_feedback": 0.0, "d_fast_drive": 0.0,
    "k_x": 0.0, "k_z": 1.0,
}
trajectory = m5.simulate_coupled_fast_slow_trajectory(
    **params, x0=0.0, z0=1.0, u_fn=lambda _: 0.0,
    timestamps=[0.0, 0.1, 0.2, 0.5], rk4_dt=0.001,
)
origin_preds = {}
for shift in (0.0, 100.0):
    shifted = [{**row, "t": row["t"] + shift} for row in trajectory[:3]]
    regimes = m5.extract_causal_temporal_regimes(
        shifted, t0=0.2 + shift, extraction_window_w=0.2,
        intrinsic_taus=(1.0,), forecast_horizon_H=0.3,
    )
    origin_preds[str(shift)] = m5.predict_via_pi_fold_mode(
        "memory_kernel", params=params, causal_regimes=regimes,
        u_future_const=0.0, horizon_H=0.3, latent_z0_estimate=1.0,
    )
results["memory_time_origin_invariance"] = {
    "predictions_by_timestamp_shift": origin_preds,
    "true_future_y": trajectory[-1]["y"],
    "same_observed_values_and_time_intervals": True,
}

print(json.dumps(results, indent=2, sort_keys=True))
