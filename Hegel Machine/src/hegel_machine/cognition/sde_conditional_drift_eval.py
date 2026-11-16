"""Post-hoc benchmark evaluation harness for noisy-history SDE conditional drift vs. noiseless twin (F08).

IMPORTANT: This module is strictly a post-hoc evaluation harness run AFTER the cognitive agent has
terminated and frozen its delivered ``Program``. No cognitive/research module may import or call this
evaluator during learning.
"""

from __future__ import annotations

from math import cos
from random import Random

from .anonymous_object_lab import AnonymousObjectLab
from .evaluation import loss, predict
from .models import Observation


def evaluate_sde_conditional_drift_and_twin_metrics(
    noisy_lab,
    program,
    fallback: float,
    selected_channels: tuple[str, ...] | list[str] = (),
    *,
    eval_seed: int = 1009,
    length: int = 48,
    trajectories: int = 2,
    mc_rollouts: int = 64,
) -> dict:
    """Trusted evaluator computing BOTH noisy-history conditional drift MSE and noiseless-twin transfer MSE (F08).

    Strictly separates three mathematically distinct quantities (never exposing simulator truth to the learner):
      1. ``live_noisy_trajectory_mse``:
         ``E_{F_t, eps_t ~ E_noisy} [ (m_hat(F_t) - Y_t)^2 ]`` on realized noisy trajectories.
      2. ``noisy_history_conditional_drift_mse``:
         ``E_{F_t ~ E_noisy} [ (m_hat(F_t) - E[Y_t | F_t])^2 ]`` conditioned on the ACTUAL realized noisy
         state/history prefix ``F_t`` drawn from ``E_noisy`` (computed both analytically with ``eps_t = 0``
         and cross-checked via ``mc_rollouts`` independent step-``t`` noise draws from the realized prefix).
      3. ``noiseless_twin_transfer_mse``:
         MSE on a separate ``process_noise_std = 0.0`` twin apparatus (a distribution-shifted trajectory
         transfer metric, NOT equal to ``noisy_history_conditional_drift_mse`` whenever past noise enters ``F_t``).
    """
    if program is None:
        return {
            "live_noisy_trajectory_mse": None,
            "noisy_history_conditional_drift_mse": None,
            "noisy_history_mc_conditional_drift_mse": None,
            "noiseless_twin_transfer_mse": None,
            "information_set_contract": None,
        }

    channels = noisy_lab._selected(tuple(selected_channels))
    settings_noisy = dict(noisy_lab._settings)
    sigma_p = float(settings_noisy.get("process_noise_std") or 0.0)
    mc_rng = Random(eval_seed + 999983)

    def _compute_target_from_after(
        s_lab,
        after_vec: list[float],
        quantity: list[float],
        position: list[float],
        drive: list[float],
        focus: int,
        prev_flows: list[float],
        prev_combined_events: list[float],
        prev_filtered_total: float,
        next_latent_state: list[float],
    ) -> tuple[float, float, float | None, float]:
        flow_t = after_vec[focus] - quantity[focus] - drive[focus]
        event_t = flow_t - prev_flows[-1] if prev_flows else 0.0
        total_t = s_lab._total_weight * sum(after_vec)
        rd = s_lab._readout
        comb_ev_t = None
        filt_tot_t = prev_filtered_total
        if rd == "total":
            tgt = total_t
        elif rd == "position_focus":
            tgt = after_vec[focus] + 0.23 * position[focus]
        elif rd == "window":
            win_flows = [*(prev_flows[-(s_lab._window - 1) :] if s_lab._window > 1 else []), flow_t]
            tgt = sum(win_flows)
        elif rd == "event":
            tgt = event_t
        elif rd == "total_event":
            tgt = total_t + 0.7 * event_t
        elif rd == "focus_window":
            win_flows = [*(prev_flows[-(s_lab._window - 1) :] if s_lab._window > 1 else []), flow_t]
            tgt = after_vec[focus] + 0.6 * sum(win_flows)
        elif rd == "filtered_total":
            filt_tot_t = 0.8 * prev_filtered_total + total_t
            tgt = filt_tot_t
        elif rd == "window_total_event":
            comb_ev_t = total_t + 0.7 * event_t
            win_ce = [
                *(prev_combined_events[-(s_lab._window - 1) :] if s_lab._window > 1 else []),
                comb_ev_t,
            ]
            tgt = sum(win_ce)
        elif rd == "focal_quadratic_total_event":
            tgt = total_t + 0.7 * event_t + 0.35 * (quantity[focus] ** 2)
        else:
            tgt = after_vec[focus]

        if s_lab._latent_mode in {
            "focal_memory",
            "oscillatory_focal_memory",
            "quadratic_energy_focal_memory",
        }:
            tgt += s_lab._latent_weight * next_latent_state[focus]
        elif s_lab._latent_mode == "coupled_focal_memory":
            focal_flux = s_lab._spatial_latent_flux(position, quantity, next_latent_state)
            tgt += s_lab._latent_weight * next_latent_state[focus] + focal_flux[focus]
        return float(tgt), float(flow_t), comb_ev_t, float(filt_tot_t)

    sq_noisy = 0.0
    sq_cond_exact = 0.0
    sq_cond_mc = 0.0
    total_rows = 0

    with AnonymousObjectLab(**settings_noisy) as eval_lab:
        u_lab = eval_lab._lab
        s_lab = u_lab._lab
        scale = len(s_lab.object_ids) if channels == ("coarse",) else 1
        s_schema = s_lab.schema_for(objects=None if channels else (), scale=scale)
        anon_map = eval_lab._measurement_names[channels] if channels else eval_lab._control_names
        anon_order = eval_lab._measurement_order[channels] if channels else eval_lab._control_order

        def _rename_row_features(raw_pt: dict, s_bef: dict) -> dict:
            u_feat = (
                {u_lab._prefix + k: v for k, v in s_bef.items()} if channels else dict(raw_pt)
            )
            mapped = {anon_map[k]: v for k, v in u_feat.items()}
            return {k: mapped[k] for k in anon_order}

        for index in range(trajectories):
            anon_probe = eval_lab.random_probe(
                eval_seed + 10007 * index, length=length, name=f"evaluation_{index}"
            )
            phys_seq = [
                {eval_lab._physical_controls[name]: val for name, val in pt.items()}
                for pt in anon_probe.sequence
            ]
            flows: list[float] = []
            combined_events: list[float] = []
            filtered_total = 0.0
            latent_state = [0.0] * len(s_lab.object_ids)
            latent_prev = [0.0] * len(s_lab.object_ids)
            ar2_a1 = 2.0 * s_lab._latent_decay * cos(s_lab._latent_freq)
            ar2_a2 = -(s_lab._latent_decay ** 2)
            obs_history: list[dict] = []

            for pt in phys_seq:
                quantity, position, drive, focus, det_after = s_lab._step(
                    pt, latent_state=latent_state
                )
                if s_lab._latent_mode in {"transition_memory", "coupled_transition_memory"}:
                    next_latent = [
                        s_lab._latent_decay * latent_state[i] + drive[i]
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_prev
                elif s_lab._latent_mode == "oscillatory_transition_memory":
                    next_latent = [
                        ar2_a1 * latent_state[i] + ar2_a2 * latent_prev[i] + drive[i]
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_state
                elif s_lab._latent_mode == "van_der_pol_transition_memory":
                    next_latent = [
                        max(
                            -6.0,
                            min(
                                6.0,
                                ar2_a1 * latent_state[i]
                                + ar2_a2 * latent_prev[i]
                                - s_lab._latent_nonlinearity * (latent_state[i] ** 3)
                                + drive[i],
                            ),
                        )
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_state
                elif s_lab._latent_mode == "bilinear_work_transition_memory":
                    next_latent = [
                        s_lab._latent_decay * latent_state[i] + quantity[i] * drive[i]
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_prev
                elif s_lab._latent_mode in {"focal_memory", "coupled_focal_memory"}:
                    next_latent = [
                        s_lab._latent_decay * latent_state[i] + quantity[i] + 0.5 * drive[i]
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_prev
                elif s_lab._latent_mode == "oscillatory_focal_memory":
                    next_latent = [
                        ar2_a1 * latent_state[i]
                        + ar2_a2 * latent_prev[i]
                        + quantity[i]
                        + 0.5 * drive[i]
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_state
                elif s_lab._latent_mode == "quadratic_energy_focal_memory":
                    next_latent = [
                        s_lab._latent_decay * latent_state[i] + (quantity[i] ** 2)
                        for i in range(len(quantity))
                    ]
                    next_prev = latent_prev
                else:
                    next_latent = latent_state
                    next_prev = latent_prev

                cond_mean_exact, _, _, _ = _compute_target_from_after(
                    s_lab,
                    det_after,
                    quantity,
                    position,
                    drive,
                    focus,
                    flows,
                    combined_events,
                    filtered_total,
                    next_latent,
                )

                if sigma_p > 0.0 and mc_rollouts > 0:
                    mc_sum = 0.0
                    for _ in range(mc_rollouts):
                        mc_after = [v + mc_rng.gauss(0.0, sigma_p) for v in det_after]
                        mc_y, _, _, _ = _compute_target_from_after(
                            s_lab,
                            mc_after,
                            quantity,
                            position,
                            drive,
                            focus,
                            flows,
                            combined_events,
                            filtered_total,
                            next_latent,
                        )
                        mc_sum += mc_y
                    cond_mean_mc = mc_sum / mc_rollouts
                else:
                    cond_mean_mc = cond_mean_exact

                if sigma_p > 0.0:
                    realized_after = [
                        v + s_lab._noise_rng.gauss(0.0, sigma_p) for v in det_after
                    ]
                else:
                    realized_after = list(det_after)
                realized_y, flow_t, comb_ev_t, filtered_total = _compute_target_from_after(
                    s_lab,
                    realized_after,
                    quantity,
                    position,
                    drive,
                    focus,
                    flows,
                    combined_events,
                    filtered_total,
                    next_latent,
                )
                flows.append(flow_t)
                if comb_ev_t is not None:
                    combined_events.append(comb_ev_t)
                latent_prev, latent_state = next_prev, next_latent

                bef_raw = s_lab._view(s_schema, quantity, position, drive, focus)
                row_feats = _rename_row_features(pt, bef_raw)
                obs_row = Observation(row_feats, realized_y, tuple(obs_history))
                y_hat = float(predict(program, obs_row, fallback))

                sq_noisy += (y_hat - realized_y) ** 2
                sq_cond_exact += (y_hat - cond_mean_exact) ** 2
                sq_cond_mc += (y_hat - cond_mean_mc) ** 2
                total_rows += 1
                obs_history.append({**row_feats, "__target__": realized_y})

    twin_settings = {**settings_noisy, "process_noise_std": 0.0}
    with AnonymousObjectLab(**twin_settings) as twin_lab:
        twin_rows = twin_lab.evaluation_rows(
            channels, seed=eval_seed, length=length, trajectories=trajectories
        )
        noiseless_twin_mse = float(loss(program, twin_rows, fallback, 0.5))

    obs_vars = sorted({name for name, _ in program.observed_variables()})
    uses_past_realized_target = "__target__" in obs_vars
    return {
        "live_noisy_trajectory_mse": sq_noisy / max(1, total_rows),
        "noisy_history_conditional_drift_mse": sq_cond_exact / max(1, total_rows),
        "noisy_history_mc_conditional_drift_mse": sq_cond_mc / max(1, total_rows),
        "noiseless_twin_transfer_mse": noiseless_twin_mse,
        "evaluated_rows": total_rows,
        "mc_rollouts_per_step": mc_rollouts,
        "information_set_contract": {
            "selected_channels": list(channels),
            "current_step_features_timing": "pre_step_before_process_noise",
            "post_step_after_sensors_exposed_in_prediction": False,
            "uses_lagged_realized_targets_in_history": uses_past_realized_target,
            "observed_variables": obs_vars,
        },
    }
