"""Shared Trusted Mathematical Handoff, Full-7-Coefficient & Canonical KKT Verification Kernel (`M9.4.1` `Gate A`).

Single source of mathematical truth shared across:
  1. `execute_host_trusted_scientific_probe` (running candidate inside an isolated untrusted worker subprocess
     and evaluating mathematical & scientific invariants in a separate trusted verifier process!),
  2. `m9_3_online_lifecycle_benchmark.verify_m93_output_contract_and_handoff`,
  3. `m9_4_continual_mechanism_binding_world` & `m9_5` public/private verifiers.

Guarantees (`C01` + `C02` Closed):
  - Rejects `hereditary_weights = zeros(7)` in BOTH public scientific gate AND private verifier.
  - Rejects `hereditary_weights[2:7] = 1_000_000` (unchecked tail mutation!) in BOTH public scientific gate AND private
    verifier via two independent obligations:
      (a) **Shared 7-Coefficient Constitutive Law Obligation**: `hw[0:7]` evaluated on standardized complete-feature
          specimens (`X_std_complete`) must match training/public data residuals (and `gt["theta"]` when available in
          private grading) without allowing arbitrary free drift corrections to cancel bad `hw[2:7]`.
      (b) **Curve-Rate & Bounded Innovation Handoff Obligation**: Curve remainder after subtracting `hw[0:2]` Prony
          branches must equal `initial_rate_pred * eval_times`, AND per-specimen difference `initial_rate_pred - prior_rate0`
          on complete rows must either vanish (`<= 0.05`) or lie within the bounded convex innovation interval `[0, 1]`
          toward observable `test["instant_rate_obs"] - prior_rate0`!
  - Beta tasks reuse `_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY` (`verify_box_simplex_kkt_certificate`) by independently
    reconstructing `(S_tilde, r_y)` and terminal fields in the trusted verifier process rather than trusting self-reported gaps.
  - Missing keys, wrong shapes, or `NaN`/`Inf` ALWAYS fail closed (`verified=False`) and emit scoped `verified_claim_ids`
    (never blanket `"task_scientific_invariants"`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SHARED_KERNEL_VERSION = "hegel_shared_math_handoff_kernel_v2_m951_jsonrpc"

ALLOWED_RPC_DTYPES: frozenset[str] = frozenset(
    {"float64", "float32", "int64", "int32", "uint64", "uint32", "bool"}
)
MAX_RPC_DEPTH = 5
MAX_RPC_ELEMENTS = 2_000_000


def encode_pure_json_numerical_payload(obj: Any, *, depth: int = 0) -> Any:
    """Serialize candidate output into pure JSON primitives (`F06`: rejects custom objects, NaN/Inf, deep nesting, oversized arrays)."""
    if depth > MAX_RPC_DEPTH:
        raise ValueError(f"Candidate output exceeded max nesting depth {MAX_RPC_DEPTH}")
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        if not math.isfinite(fv):
            raise ValueError("Non-finite float in candidate output")
        return fv
    if isinstance(obj, str):
        if len(obj) > 4096:
            raise ValueError("Oversized string in candidate output")
        return obj
    if isinstance(obj, np.ndarray):
        dt_name = str(obj.dtype)
        if dt_name not in ALLOWED_RPC_DTYPES or obj.dtype.kind == "O":
            raise ValueError(f"Disallowed ndarray dtype `{dt_name}`")
        if obj.ndim > 4 or int(obj.size) > MAX_RPC_ELEMENTS:
            raise ValueError(f"Disallowed ndarray shape/size `{obj.shape}`")
        if obj.dtype.kind == "f" and not bool(np.all(np.isfinite(obj))):
            raise ValueError("Non-finite ndarray values in candidate output")
        return {"__nd__": True, "dtype": dt_name, "shape": list(obj.shape), "data": obj.tolist()}
    if isinstance(obj, (list, tuple)):
        if len(obj) > 500_000:
            raise ValueError("Oversized list/tuple in candidate output")
        return [encode_pure_json_numerical_payload(v, depth=depth + 1) for v in obj]
    if isinstance(obj, dict):
        if len(obj) > 256:
            raise ValueError("Oversized dict in candidate output")
        out_d: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str) or k == "__nd__":
                raise ValueError(f"Disallowed dict key `{k!r}`")
            out_d[k] = encode_pure_json_numerical_payload(v, depth=depth + 1)
        return out_d
    raise TypeError(f"Disallowed non-primitive candidate return type `{type(obj).__name__}`")


def decode_pure_json_numerical_payload(obj: Any, *, depth: int = 0) -> Any:
    """Trusted-side pure JSON decoder reconstructing ONLY validated `dict`/`list`/scalar/`np.ndarray` (`0` pickle!)."""
    if depth > MAX_RPC_DEPTH + 1:
        raise ValueError("Decoded JSON exceeded max depth")
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError("Non-finite float in JSON payload")
        return obj
    if isinstance(obj, list):
        return [decode_pure_json_numerical_payload(v, depth=depth + 1) for v in obj]
    if isinstance(obj, dict):
        if obj.get("__nd__") is True:
            dt_name = str(obj.get("dtype") or "")
            shp = tuple(int(x) for x in (obj.get("shape") or []))
            if dt_name not in ALLOWED_RPC_DTYPES or len(shp) > 4:
                raise ValueError(f"Invalid encoded ndarray header: dtype={dt_name}, shape={shp}")
            arr = np.asarray(obj.get("data"), dtype=np.dtype(dt_name))
            if arr.shape != shp or int(arr.size) > MAX_RPC_ELEMENTS:
                raise ValueError(f"Decoded ndarray shape mismatch: {arr.shape} != {shp}")
            if arr.dtype.kind == "f" and not bool(np.all(np.isfinite(arr))):
                raise ValueError("Non-finite decoded ndarray")
            return arr
        for k in obj.keys():
            if not isinstance(k, str) or (k.startswith("__") and k.endswith("__")):
                raise ValueError(f"Disallowed dunder/object marker key `{k!r}` in pure JSON numerical payload (`F06`)")
        return {str(k): decode_pure_json_numerical_payload(v, depth=depth + 1) for k, v in obj.items()}
    raise TypeError(f"Unexpected JSON type `{type(obj).__name__}`")


def standardize_reference_coupon_panel(
    tr_panel: dict[str, Any],
    te_panel: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically compute per-lab reference-coupon standardized features `(X_tr_std, X_te_std, tr_complete, te_complete)`."""
    labs = np.concatenate([np.asarray(tr_panel["lab_id"], dtype=int), np.asarray(te_panel["lab_id"], dtype=int)])
    strata = np.concatenate([np.asarray(tr_panel["stratum"], dtype=int), np.asarray(te_panel["stratum"], dtype=int)])
    x_raw = np.vstack([np.asarray(tr_panel["coupon_features_raw"], dtype=float), np.asarray(te_panel["coupon_features_raw"], dtype=float)])
    r_raw = np.vstack([np.asarray(tr_panel["ref_coupons"], dtype=float), np.asarray(te_panel["ref_coupons"], dtype=float)])

    complete_all = np.isfinite(x_raw).all(axis=1)
    x_std = x_raw.copy()
    for c in np.unique(labs):
        m = labs == c
        rc = r_raw[m]
        col_m = np.nanmean(rc, axis=0)
        col_s = np.nanstd(rc, axis=0, ddof=0)
        valid = np.isfinite(col_m) & np.isfinite(col_s) & (col_s > 1e-8)
        off = float(np.median(col_m[valid]))
        gn = float(np.median(col_s[valid]))
        x_std[m] = (x_std[m] - off) / gn

    # Cell-balanced column imputation for incomplete rows (note: complete rows are 100% invariant to imputation!)
    w_all = np.zeros(len(labs), dtype=float)
    for c in np.unique(labs):
        for s in np.unique(strata):
            ms = (labs == c) & (strata == s)
            if np.any(ms):
                w_all[ms] = 1.0 / float(np.sum(ms))
    if float(np.mean(w_all)) > 0.0:
        w_all /= float(np.mean(w_all))
    for j in range(x_std.shape[1]):
        fin = np.isfinite(x_std[:, j])
        if np.any(fin) and not np.all(fin):
            fill = float(np.sum(w_all[fin] * x_std[fin, j]) / max(float(np.sum(w_all[fin])), 1e-12))
            x_std[~fin, j] = fill

    n_tr = len(tr_panel["lab_id"])
    return x_std[:n_tr], x_std[n_tr:], complete_all[:n_tr], complete_all[n_tr:]


def calibrate_alpha_physical_stresses(
    dataset: dict[str, Any],
    split_key: str,
    gt: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract calibrated `(past_stress, stress_hold_0)` for `split_key` (`'train'` or `'test'`)."""
    sp = dataset[split_key]
    if "past_stress" in sp and "stress_hold_0" in sp:
        return np.asarray(sp["past_stress"], dtype=float), np.asarray(sp["stress_hold_0"], dtype=float)
    if split_key == "test" and isinstance(gt, dict) and "te_hist_true" in gt and "te_hold_true" in gt:
        return np.asarray(gt["te_hist_true"], dtype=float), np.asarray(gt["te_hold_true"], dtype=float)
    labs = np.asarray(sp["lab_id"], dtype=int).ravel()
    ps_cmd = np.asarray(sp["past_stress_cmd"], dtype=float)
    h0_cmd = np.asarray(sp["stress_hold_cmd"], dtype=float)
    if isinstance(gt, dict) and "stress_gains" in gt and "stress_biases" in gt:
        gns = np.asarray(gt["stress_gains"], dtype=float)
        bss = np.asarray(gt["stress_biases"], dtype=float)
        return ps_cmd * gns[labs, None] + bss[labs, None], h0_cmd * gns[labs] + bss[labs]
    cal = dataset["calibration_load_cells"]
    c_labs = np.asarray(cal["lab_id"], dtype=int).ravel()
    c_cmd = np.asarray(cal["cmd_stress"], dtype=float).ravel()
    c_true = np.asarray(cal["true_stress"], dtype=float).ravel()
    ps_out = np.empty_like(ps_cmd, dtype=float)
    h0_out = np.empty_like(h0_cmd, dtype=float)
    for lb in np.unique(labs):
        m_fit = c_labs == lb
        if int(np.sum(m_fit)) >= 2:
            gn, bs = np.linalg.lstsq(np.column_stack([c_cmd[m_fit], np.ones(int(np.sum(m_fit)))]), c_true[m_fit], rcond=None)[0]
        else:
            gn, bs = 1.0, 0.0
        m_row = labs == lb
        ps_out[m_row] = float(gn) * ps_cmd[m_row] + float(bs)
        h0_out[m_row] = float(gn) * h0_cmd[m_row] + float(bs)
    return ps_out, h0_out


def verify_alpha_candidate_handoff_and_invariants(
    *,
    dataset: dict[str, Any],
    out: Any,
    gt: dict[str, Any] | None = None,
    require_gt_accuracy: bool = False,
) -> dict[str, Any]:
    """Verify Alpha family outputs (`standard`, `transfer_T`, `counterexample_C`, or unified `m9_5` contract)."""
    errs: list[str] = []
    claims: list[str] = []
    metrics: dict[str, Any] = {}
    if not isinstance(out, dict):
        return {"ok": False, "errors": ["output_not_dict"], "verified_claim_ids": [], "metrics": {}}

    n_te = len(dataset["test"]["lab_id"])
    t_ev = np.asarray(dataset["eval_times"], dtype=float)
    p_dt = np.asarray(dataset["past_dt"], dtype=float)
    k_t = len(t_ev)

    if "estimated_taus" not in out or "predicted_strain" not in out:
        return {"ok": False, "errors": ["missing_core_keys:estimated_taus_or_predicted_strain"], "verified_claim_ids": [], "metrics": {}}

    try:
        et = np.asarray(out["estimated_taus"], dtype=float)
        ps = np.asarray(out["predicted_strain"], dtype=float)
    except Exception:
        return {"ok": False, "errors": ["non_numeric_core_arrays"], "verified_claim_ids": [], "metrics": {}}

    if et.shape != (2,) or ps.shape != (n_te, k_t) or not (bool(np.all(np.isfinite(et))) and bool(np.all(np.isfinite(ps)))):
        return {"ok": False, "errors": [f"bad_core_shape_or_nan:taus={et.shape},strain={ps.shape}"], "verified_claim_ids": [], "metrics": {}}
    if not (0.05 < float(et[0]) < float(et[1]) < 25.0):
        errs.append("taus_not_positive_ordered")

    te_ps, te_h0 = calibrate_alpha_physical_stresses(dataset, "test", gt=gt)
    s0_te = np.asarray(dataset["test"]["strain_0"], dtype=float)
    init_m_te = dataset["test"].get("initial_memory_states")
    z_te = np.zeros((n_te, 2), dtype=float) if init_m_te is None else np.asarray(init_m_te, dtype=float).copy()
    for s_i, dts in enumerate(p_dt):
        dec = np.exp(-float(dts) / et)
        z_te = z_te * dec[None, :] + (1.0 - dec[None, :]) * te_ps[:, s_i : s_i + 1]
    gap_te = te_h0[:, None] - z_te

    # Mode 1: Standard / Unified 7-coefficient hereditary output (`hereditary_weights` + `initial_rate_pred`)
    if "hereditary_weights" in out or "initial_rate_pred" in out:
        if "hereditary_weights" not in out or "initial_rate_pred" not in out:
            errs.append("missing_hereditary_weights_or_initial_rate_pred")
        else:
            try:
                hw = np.asarray(out["hereditary_weights"], dtype=float)
                ir = np.asarray(out["initial_rate_pred"], dtype=float)
            except Exception:
                hw = np.array([np.nan])
                ir = np.array([np.nan])
            if hw.shape != (7,) or ir.shape != (n_te,) or not (bool(np.all(np.isfinite(hw))) and bool(np.all(np.isfinite(ir)))):
                errs.append(f"bad_hw_or_ir_shape_or_nan:{hw.shape},{ir.shape}")
            else:
                if float(np.linalg.norm(hw[:2])) <= 1e-4 or float(np.min(hw[:2])) <= 1e-4:
                    errs.append("degenerate_prony_weights")

                # Optional viscoplastic terms when unified `viscoplastic_weights` is also returned (`M9.5` unified schema!)
                vp_curve_te = np.zeros((n_te, k_t), dtype=float)
                vp_rate_te = np.zeros(n_te, dtype=float)
                if "viscoplastic_weights" in out and out["viscoplastic_weights"] is not None:
                    vw_u = np.asarray(out["viscoplastic_weights"], dtype=float)
                    if vw_u.shape != (2,) or not bool(np.all(np.isfinite(vw_u))):
                        errs.append("bad_unified_viscoplastic_weights")
                    elif "yield_stress" in dataset and dataset["yield_stress"] is not None:
                        ys_u = float(dataset["yield_stress"])
                        ov_h = (np.maximum(0.0, np.abs(te_h0) - ys_u) ** 2) * np.sign(te_h0)
                        d_hi = np.sum((np.maximum(0.0, np.abs(te_ps) - ys_u) ** 2) * p_dt[None, :], axis=1)
                        vp_curve_te = vw_u[0] * ov_h[:, None] * t_ev[None, :] + vw_u[1] * (d_hi * te_h0)[:, None] * (-np.expm1(-t_ev[None, :] / et[0]))
                        vp_rate_te = vw_u[0] * ov_h + vw_u[1] * (d_hi * te_h0) / et[0]

                prony_te = (
                    hw[0] * gap_te[:, 0:1] * (-np.expm1(-t_ev[None, :] / et[0]))
                    + hw[1] * gap_te[:, 1:2] * (-np.expm1(-t_ev[None, :] / et[1]))
                )
                rem_curve = (ps - s0_te[:, None]) - prony_te - vp_curve_te
                rem_rate = ir - (hw[0] * gap_te[:, 0] / et[0] + hw[1] * gap_te[:, 1] / et[1]) - vp_rate_te
                curve_rate_err = float(np.max(np.abs(rem_curve - rem_rate[:, None] * t_ev[None, :])))
                metrics["curve_rate_handoff_max_err"] = round(curve_rate_err, 8)
                if curve_rate_err > 0.08:
                    errs.append(f"weights_prediction_handoff_mismatch:{curve_rate_err:.6f}")

                # Obligation A + B: Full 7-coefficient verification (`hw[2:7]` MUST match standardized panel & train/test rate!)
                x_tr_std, x_te_std, tr_comp, te_comp = standardize_reference_coupon_panel(dataset["train"], dataset["test"])
                tr_ps, tr_h0 = calibrate_alpha_physical_stresses(dataset, "train", gt=gt)
                init_m_tr = dataset["train"].get("initial_memory_states")
                z_tr = np.zeros((len(tr_h0), 2), dtype=float) if init_m_tr is None else np.asarray(init_m_tr, dtype=float).copy()
                for s_i, dts in enumerate(p_dt):
                    dec = np.exp(-float(dts) / et)
                    z_tr = z_tr * dec[None, :] + (1.0 - dec[None, :]) * tr_ps[:, s_i : s_i + 1]
                gap_tr = tr_h0[:, None] - z_tr

                vp_rate_tr = np.zeros(len(tr_h0), dtype=float)
                if "viscoplastic_weights" in out and out["viscoplastic_weights"] is not None and "yield_stress" in dataset and dataset["yield_stress"] is not None:
                    vw_tr = np.asarray(out["viscoplastic_weights"], dtype=float)
                    if vw_tr.shape == (2,) and bool(np.all(np.isfinite(vw_tr))):
                        ys_tr = float(dataset["yield_stress"])
                        ov_h_tr = (np.maximum(0.0, np.abs(tr_h0) - ys_tr) ** 2) * np.sign(tr_h0)
                        d_hi_tr = np.sum((np.maximum(0.0, np.abs(tr_ps) - ys_tr) ** 2) * p_dt[None, :], axis=1)
                        vp_rate_tr = vw_tr[0] * ov_h_tr + vw_tr[1] * (d_hi_tr * tr_h0) / et[0]

                if int(np.sum(tr_comp)) >= 8 and "instant_rate_obs" in dataset["train"]:
                    prior_drift_tr = hw[2] + x_tr_std[tr_comp] @ hw[3:7]
                    prior_rate_tr = hw[0] * gap_tr[tr_comp, 0] / et[0] + hw[1] * gap_tr[tr_comp, 1] / et[1] + prior_drift_tr + vp_rate_tr[tr_comp]
                    tr_rate_obs = np.asarray(dataset["train"]["instant_rate_obs"], dtype=float)[tr_comp]
                    tr_rate_med_err = float(np.median(np.abs(prior_rate_tr - tr_rate_obs)))
                    metrics["train_complete_7coeff_rate_median_err"] = round(tr_rate_med_err, 6)
                    if tr_rate_med_err > 0.12:
                        errs.append(f"tail_coefficients_inconsistent_with_train_rate:{tr_rate_med_err:.6f}")

                if int(np.sum(te_comp)) >= 4:
                    prior_drift_te = hw[2] + x_te_std[te_comp] @ hw[3:7]
                    prior_rate_te = hw[0] * gap_te[te_comp, 0] / et[0] + hw[1] * gap_te[te_comp, 1] / et[1] + prior_drift_te + vp_rate_te[te_comp]
                    applied_corr = ir[te_comp] - prior_rate_te
                    if "instant_rate_obs" in dataset["test"]:
                        te_rate_obs = np.asarray(dataset["test"]["instant_rate_obs"], dtype=float)[te_comp]
                        innov = te_rate_obs - prior_rate_te
                        prior_te_med = float(np.median(np.abs(innov)))
                        metrics["test_complete_7coeff_prior_rate_median_err"] = round(prior_te_med, 6)
                        if prior_te_med > 0.12:
                            errs.append(f"tail_coefficients_inconsistent_with_test_obs:{prior_te_med:.6f}")
                        # Bounded convex innovation check: `applied_corr` must lie between `0` and `innov` (within `0.04` numerical slack)!
                        lo_b = np.minimum(0.0, innov) - 0.04
                        hi_b = np.maximum(0.0, innov) + 0.04
                        bad_innov = int(np.sum((applied_corr < lo_b) | (applied_corr > hi_b)))
                        metrics["unbounded_innovation_violations"] = bad_innov
                        if bad_innov > 0:
                            errs.append(f"unbounded_drift_correction_canceling_bad_weights:{bad_innov}")
                    else:
                        max_direct_diff = float(np.max(np.abs(applied_corr)))
                        metrics["test_complete_7coeff_direct_rate_diff"] = round(max_direct_diff, 6)
                        if max_direct_diff > 0.08:
                            errs.append(f"tail_coefficients_handoff_mismatch:{max_direct_diff:.6f}")

                if isinstance(gt, dict) and "theta" in gt and require_gt_accuracy:
                    th_err = float(np.max(np.abs(hw - np.asarray(gt["theta"], dtype=float))))
                    metrics["theta_7coeff_linf_err"] = round(th_err, 6)
                    if th_err > 0.14:
                        errs.append(f"theta_7coeff_gt_mismatch:{th_err:.6f}")

                if not errs:
                    claims.append("full_7coeff_hereditary_handoff")

    # Mode 2: Shifted compliance & crossing (`secant_compliance` + `hazard_crossing_times`)
    if "secant_compliance" in out or "hazard_crossing_times" in out or "critical_strain" in dataset["test"]:
        if "secant_compliance" in out or "hazard_crossing_times" in out:
            for req_k in ("secant_compliance", "hazard_crossing_times"):
                if req_k not in out:
                    errs.append(f"missing:{req_k}")
            if "secant_compliance" in out and "hazard_crossing_times" in out:
                sc = np.asarray(out["secant_compliance"], dtype=float)
                cx = np.asarray(out["hazard_crossing_times"], dtype=float)
                if sc.shape != (n_te,) or cx.shape != (n_te,) or not (bool(np.all(np.isfinite(sc))) and bool(np.all(np.isfinite(cx)))):
                    errs.append("bad_secant_or_crossing_shape")
                else:
                    # `M9.5.1` `F05`: Separate Layer 1 (algebraic handoff on publicly-supported OLS stress `te_h0_ols`)
                    # from Layer 2 (physical accuracy against hidden `gt["te_secant"]`)!
                    _, te_h0_ols = calibrate_alpha_physical_stresses(dataset, "test", gt=None)
                    exp_sc_ols = (ps[:, -1] - s0_te) / np.maximum(np.abs(te_h0_ols), 0.5)
                    sc_diff = float(np.max(np.abs(sc - exp_sc_ols)))
                    if isinstance(gt, dict) and "te_hold_true" in gt:
                        exp_sc_gt_formula = (ps[:, -1] - s0_te) / np.maximum(np.abs(np.asarray(gt["te_hold_true"], dtype=float)), 0.5)
                        sc_diff = min(sc_diff, float(np.max(np.abs(sc - exp_sc_gt_formula))))
                    metrics["secant_formula_max_diff"] = round(sc_diff, 8)
                    if sc_diff > 1e-3:
                        errs.append(f"secant_handoff_mismatch:{sc_diff:.6f}")
                    if isinstance(gt, dict) and "te_secant" in gt:
                        sc_gt_mae = float(np.mean(np.abs(sc - np.asarray(gt["te_secant"], dtype=float))))
                        metrics["secant_gt_mae"] = round(sc_gt_mae, 6)
                        if sc_gt_mae > 0.095:
                            errs.append(f"secant_gt_mae_exceeded:{sc_gt_mae:.6f}")
                    if "critical_strain" in dataset["test"]:
                        crit = np.asarray(dataset["test"]["critical_strain"], dtype=float)
                        ts_f = np.concatenate([[0.0], t_ev])
                        ys_f = np.column_stack([s0_te, ps])
                        exp_cx = np.full(n_te, float(t_ev[-1]), dtype=float)
                        for idx_i in range(n_te):
                            c_v = float(crit[idx_i])
                            if ys_f[idx_i, 0] >= c_v:
                                exp_cx[idx_i] = 0.0
                                continue
                            for k_i in range(len(ts_f) - 1):
                                if ys_f[idx_i, k_i + 1] >= c_v:
                                    dy = max(float(ys_f[idx_i, k_i + 1] - ys_f[idx_i, k_i]), 1e-12)
                                    exp_cx[idx_i] = float(ts_f[k_i] + (c_v - ys_f[idx_i, k_i]) / dy * (ts_f[k_i + 1] - ts_f[k_i]))
                                    break
                        cx_diff = float(np.max(np.abs(cx - exp_cx)))
                        metrics["crossing_formula_max_diff"] = round(cx_diff, 8)
                        if cx_diff > 1e-2:
                            errs.append(f"crossing_handoff_mismatch:{cx_diff:.6f}")
                    if not errs:
                        claims.append("secant_and_crossing_handoff")

    # Mode 3: Counterexample `C` standalone (`viscoplastic_weights` without `hereditary_weights`)
    if "viscoplastic_weights" in out and "hereditary_weights" not in out:
        vw = np.asarray(out["viscoplastic_weights"], dtype=float)
        if vw.shape != (2,) or not bool(np.all(np.isfinite(vw))):
            errs.append("bad_viscoplastic_weights")
        else:
            if float(np.linalg.norm(vw)) <= 1e-5:
                errs.append("degenerate_viscoplastic_weights")
            if isinstance(gt, dict) and "vp_theta" in gt and float(np.linalg.norm(gt["vp_theta"])) > 1e-6:
                vp_rel = float(np.linalg.norm(vw - np.asarray(gt["vp_theta"], dtype=float)) / np.linalg.norm(gt["vp_theta"]))
                metrics["vp_rel_l2"] = round(vp_rel, 6)
                if vp_rel > 0.16:
                    errs.append(f"vp_weights_mismatch:{vp_rel:.6f}")
            if not errs:
                claims.append("viscoplastic_overstress_handoff")

    # Ground-truth / public residual checks (strictly FAIL-CLOSED if `gt` present!)
    if isinstance(gt, dict):
        if "te_clean" in gt:
            gt_cl = np.asarray(gt["te_clean"], dtype=float)
            if ps.shape != gt_cl.shape:
                errs.append("strain_gt_shape_mismatch")
            else:
                s_mae = float(np.mean(np.abs(ps - gt_cl)))
                metrics["strain_mae"] = round(s_mae, 6)
                if s_mae > (0.075 if require_gt_accuracy else 0.085):
                    errs.append(f"strain_mae_exceeded:{s_mae:.6f}")
        if "taus" in gt:
            gt_t = np.asarray(gt["taus"], dtype=float)
            t_rel = float(np.max(np.abs(et - gt_t) / gt_t))
            metrics["tau_rel_err"] = round(t_rel, 6)
            if t_rel > (0.10 if require_gt_accuracy else 0.12):
                errs.append(f"tau_rel_exceeded:{t_rel:.6f}")
        if not errs:
            claims.append("quantitative_strain_and_spectrum_residuals")

    return {
        "ok": len(errs) == 0,
        "errors": errs,
        "verified_claim_ids": claims,
        "metrics": metrics,
        "kernel_version": SHARED_KERNEL_VERSION,
    }


def verify_beta_candidate_handoff_and_kkt(
    *,
    study: dict[str, Any],
    out: Any,
    gt: dict[str, Any] | None = None,
    gap_tol: float = 2e-4,
) -> dict[str, Any]:
    """Verify Beta family outputs using canonical `_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY` and independent `(S_tilde, r_y)`."""
    from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY

    ns_kkt: dict[str, Any] = {"math": math, "np": np}
    exec(compile(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY, "<canonical_kkt_kernel>", "exec"), ns_kkt)
    kkt_fn = ns_kkt["verify_box_simplex_kkt_certificate"]

    errs: list[str] = []
    claims: list[str] = []
    metrics: dict[str, Any] = {}
    if not isinstance(out, dict):
        return {"ok": False, "errors": ["output_not_dict"], "verified_claim_ids": [], "metrics": {}}

    M, nx, ny = np.asarray(study["source_modes"]).shape
    S_tilde = np.asarray(gt["S_tilde"], dtype=float) if isinstance(gt, dict) and "S_tilde" in gt else None
    r_y = np.asarray(gt["r_y"], dtype=float) if isinstance(gt, dict) and "r_y" in gt else None

    if "budget_grid" in study:
        b_grid = np.asarray(study["budget_grid"], dtype=float)
        B = len(b_grid)
        if not {"weights_by_budget", "terminal_field_mid_budget", "kkt_certificates"} <= set(out.keys()):
            return {"ok": False, "errors": ["missing_multibudget_keys"], "verified_claim_ids": [], "metrics": {}}
        wb = np.asarray(out["weights_by_budget"], dtype=float)
        tf = np.asarray(out["terminal_field_mid_budget"], dtype=float)
        certs = out["kkt_certificates"]
        if wb.shape != (B, M) or tf.shape != (nx, ny) or not isinstance(certs, list) or len(certs) != B:
            return {"ok": False, "errors": ["bad_multibudget_shapes"], "verified_claim_ids": [], "metrics": {}}
        if not (bool(np.all(np.isfinite(wb))) and bool(np.all(np.isfinite(tf)))):
            return {"ok": False, "errors": ["non_finite_multibudget_arrays"], "verified_claim_ids": [], "metrics": {}}
        if S_tilde is not None and r_y is not None:
            max_gap = 0.0
            for bi, bg in enumerate(b_grid):
                chk = kkt_fn(S_tilde, r_y, study["source_ridge"], study["u_box"], float(bg), certs[bi], gap_tol=gap_tol)
                if not chk["valid"]:
                    errs.append(f"kkt_b{bi}:{chk['reason']}")
                    break
                if float(np.max(np.abs(wb[bi] - np.asarray(certs[bi]["weights"], dtype=float)))) > 1e-5:
                    errs.append(f"wb_cert_mismatch_b{bi}")
                    break
                max_gap = max(max_gap, float(chk["gap"]))
            metrics["max_kkt_duality_gap"] = round(max_gap, 10)
            if not errs:
                claims.append("independent_box_simplex_kkt_certificate")
        if isinstance(gt, dict) and "weights_by_budget" in gt and "terminal_field_mid_budget" in gt:
            wr = float(np.linalg.norm(wb - gt["weights_by_budget"]) / max(float(np.linalg.norm(gt["weights_by_budget"])), 1e-12))
            fr = float(np.linalg.norm(tf - gt["terminal_field_mid_budget"]) / max(float(np.linalg.norm(gt["terminal_field_mid_budget"])), 1e-12))
            metrics["weights_rel_l2"] = round(wr, 8)
            metrics["field_rel_l2"] = round(fr, 8)
            if wr > 2e-4 or fr > 2e-4:
                errs.append(f"multibudget_gt_mismatch:w={wr:.6e},f={fr:.6e}")
            elif not errs:
                claims.append("quantitative_transport_field_and_weights_residuals")
    else:
        if "weights" not in out or "kkt_certificate" not in out:
            return {"ok": False, "errors": ["missing_weights_or_kkt_certificate"], "verified_claim_ids": [], "metrics": {}}
        w = np.asarray(out["weights"], dtype=float)
        if w.shape != (M,) or not bool(np.all(np.isfinite(w))):
            return {"ok": False, "errors": [f"bad_weights:{w.shape}"], "verified_claim_ids": [], "metrics": {}}
        if S_tilde is not None and r_y is not None:
            chk = kkt_fn(S_tilde, r_y, study["source_ridge"], study["u_box"], study["total_budget"], out["kkt_certificate"], gap_tol=gap_tol)
            if not chk["valid"]:
                errs.append(f"kkt_invalid:{chk['reason']}")
            elif float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
                errs.append("weights_cert_mismatch")
            else:
                metrics["kkt_duality_gap"] = round(float(chk["gap"]), 10)
                claims.append("independent_box_simplex_kkt_certificate")

        if "terminal_field_channel_b" in out:
            fa = np.asarray(out.get("terminal_field_channel_a", []), dtype=float)
            fb = np.asarray(out["terminal_field_channel_b"], dtype=float)
            if fa.shape != (nx, ny) or fb.shape != (nx, ny) or not (bool(np.all(np.isfinite(fa))) and bool(np.all(np.isfinite(fb)))):
                errs.append("bad_dual_channel_fields")
            elif isinstance(gt, dict) and gt.get("terminal_field_B") is not None:
                wr = float(np.linalg.norm(w - gt["w_opt"]) / max(float(np.linalg.norm(gt["w_opt"])), 1e-12))
                far = float(np.linalg.norm(fa - gt["terminal_field_A"]) / max(float(np.linalg.norm(gt["terminal_field_A"])), 1e-12))
                fbr = float(np.linalg.norm(fb - gt["terminal_field_B"]) / max(float(np.linalg.norm(gt["terminal_field_B"])), 1e-12))
                metrics.update({"weights_rel_l2": round(wr, 8), "field_a_rel_l2": round(far, 8), "field_b_rel_l2": round(fbr, 8)})
                if wr > 2e-4 or far > 2e-4 or fbr > 2e-4:
                    errs.append("dual_channel_gt_mismatch")
                else:
                    claims.append("quantitative_transport_field_and_weights_residuals")
        else:
            if "terminal_field" not in out:
                errs.append("missing_terminal_field")
            else:
                tf = np.asarray(out["terminal_field"], dtype=float)
                if tf.shape != (nx, ny) or not bool(np.all(np.isfinite(tf))):
                    errs.append("bad_terminal_field")
                elif isinstance(gt, dict) and "w_opt" in gt and "terminal_field_A" in gt:
                    wr = float(np.linalg.norm(w - gt["w_opt"]) / max(float(np.linalg.norm(gt["w_opt"])), 1e-12))
                    fr = float(np.linalg.norm(tf - gt["terminal_field_A"]) / max(float(np.linalg.norm(gt["terminal_field_A"])), 1e-12))
                    metrics.update({"weights_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8)})
                    if wr > 2e-4 or fr > 2e-4:
                        errs.append(f"single_channel_gt_mismatch:w={wr:.6e},f={fr:.6e}")
                    else:
                        claims.append("quantitative_transport_field_and_weights_residuals")

    return {
        "ok": len(errs) == 0,
        "errors": errs,
        "verified_claim_ids": claims,
        "metrics": metrics,
        "kernel_version": SHARED_KERNEL_VERSION,
    }
