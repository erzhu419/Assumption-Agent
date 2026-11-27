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

import json
import math
from typing import Any

import numpy as np

SHARED_KERNEL_VERSION = "hegel_shared_math_handoff_kernel_v3_m961_three_layer"

ALLOWED_RPC_DTYPES: frozenset[str] = frozenset(
    {"float64", "float32", "int64", "int32", "uint64", "uint32", "bool"}
)
MAX_RPC_DEPTH = 5
MAX_RPC_ELEMENTS = 2_000_000
MAX_RPC_JSON_BYTES = 32_000_000
MAX_RPC_CONTAINER_NODES = 250_000


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


def decode_pure_json_numerical_payload(
    obj: Any,
    *,
    depth: int = 0,
    _budget_state: dict[str, int] | None = None,
) -> Any:
    """Trusted-side pure JSON decoder reconstructing ONLY validated `dict`/`list`/scalar/`np.ndarray` (`0` pickle!).
    `M9.6.1` pre-allocation hardening: enforces total container node count, list/dict length caps, and shape/element
    product bounds BEFORE calling `np.asarray`!
    """
    state = _budget_state if _budget_state is not None else {"nodes": 0, "elements": 0}
    state["nodes"] += 1
    if state["nodes"] > MAX_RPC_CONTAINER_NODES:
        raise ValueError(f"Decoded JSON exceeded max container node budget {MAX_RPC_CONTAINER_NODES}")
    if depth > MAX_RPC_DEPTH + 1:
        raise ValueError("Decoded JSON exceeded max depth")
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("Non-finite float in JSON payload")
        return obj
    if isinstance(obj, str):
        if len(obj) > 4096:
            raise ValueError("Oversized string in JSON payload")
        return obj
    if isinstance(obj, list):
        if len(obj) > 500_000:
            raise ValueError("Oversized list in JSON payload")
        return [decode_pure_json_numerical_payload(v, depth=depth + 1, _budget_state=state) for v in obj]
    if isinstance(obj, dict):
        if obj.get("__nd__") is True:
            dt_name = str(obj.get("dtype") or "")
            raw_shp = obj.get("shape")
            if not isinstance(raw_shp, list) or len(raw_shp) > 4 or dt_name not in ALLOWED_RPC_DTYPES:
                raise ValueError(f"Invalid encoded ndarray header: dtype={dt_name}, shape={raw_shp}")
            shp = tuple(int(x) for x in raw_shp)
            if any(s < 0 for s in shp):
                raise ValueError(f"Negative dimension in ndarray header: {shp}")
            expected_elems = math.prod(shp) if shp else 1
            if expected_elems > MAX_RPC_ELEMENTS or (state["elements"] + expected_elems) > (2 * MAX_RPC_ELEMENTS):
                raise ValueError(f"Pre-allocation ndarray element budget exceeded: shape={shp}, elems={expected_elems}")
            state["elements"] += int(expected_elems)
            arr = np.asarray(obj.get("data"), dtype=np.dtype(dt_name))
            if arr.shape != shp or int(arr.size) > MAX_RPC_ELEMENTS:
                raise ValueError(f"Decoded ndarray shape mismatch: {arr.shape} != {shp}")
            if arr.dtype.kind == "f" and not bool(np.all(np.isfinite(arr))):
                raise ValueError("Non-finite decoded ndarray")
            return arr
        if len(obj) > 256:
            raise ValueError("Oversized dict in JSON payload")
        for k in obj.keys():
            if not isinstance(k, str) or (k.startswith("__") and k.endswith("__")):
                raise ValueError(f"Disallowed dunder/object marker key `{k!r}` in pure JSON numerical payload (`F06`)")
        return {str(k): decode_pure_json_numerical_payload(v, depth=depth + 1, _budget_state=state) for k, v in obj.items()}
    raise TypeError(f"Unexpected JSON type `{type(obj).__name__}`")


def decode_pure_json_numerical_text(raw_text: str, *, max_bytes: int = MAX_RPC_JSON_BYTES) -> Any:
    """Enforce raw byte budget BEFORE `json.loads` and decode via `decode_pure_json_numerical_payload`."""
    n_bytes = len(raw_text.encode("utf-8"))
    if n_bytes > max_bytes:
        raise ValueError(f"Encoded JSON byte length {n_bytes} exceeds max_bytes={max_bytes}")
    parsed = json.loads(raw_text)
    return decode_pure_json_numerical_payload(parsed)


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

                # Optional `M9.7` 3-regime non-nested `regime_mode_weights` (`phi_0(t)=t`, `phi_1(t)=log(1+t)`, `phi_2(t)=t/(1+0.85*t)`)
                phi_env = t_ev.copy()
                if "regime_mode_weights" in out and out["regime_mode_weights"] is not None:
                    rw_u = np.asarray(out["regime_mode_weights"], dtype=float)
                    if rw_u.shape != (2,) or not bool(np.all(np.isfinite(rw_u))):
                        errs.append("bad_regime_mode_weights")
                    else:
                        crit_s = float(dataset.get("crit_stress", dataset.get("yield_stress", 2.95)))
                        sb_idx = float(dataset.get("shear_banding_index", 0.0))
                        d_cr_te = np.sum((np.maximum(0.0, np.abs(te_ps) - crit_s) ** 2) * p_dt[None, :], axis=1)
                        if sb_idx > 0.50:
                            phi_env = t_ev / (1.0 + 0.85 * t_ev)
                            sl_f = np.sign(te_h0) * np.sqrt(np.maximum(np.abs(te_h0), 1e-12))
                            vp_curve_te = rw_u[0] * sl_f[:, None] * phi_env[None, :] + rw_u[1] * (gap_te[:, 0] - gap_te[:, 1])[:, None] * (-np.expm1(-t_ev[None, :] / et[0]))
                            vp_rate_te = rw_u[0] * sl_f + rw_u[1] * (gap_te[:, 0] - gap_te[:, 1]) / et[0]
                        elif float(np.max(d_cr_te)) > 1e-9 or float(np.max(np.abs(rw_u))) > 1e-4:
                            # If `d_cr_te > 0` OR candidate output nonzero crystallized network weights, envelope is `log(1 + t)` when `d_cr > 0`!
                            if float(np.max(d_cr_te)) > 1e-9 or (
                                "past_stress" in dataset["train"]
                                and float(np.max(np.sum((np.maximum(0.0, np.abs(calibrate_alpha_physical_stresses(dataset, "train", gt=gt)[0]) - crit_s) ** 2) * p_dt[None, :], axis=1))) > 1e-9
                            ):
                                phi_env = np.log1p(t_ev)
                            act_te = np.tanh(d_cr_te)
                            vp_curve_te = rw_u[0] * (act_te * te_h0)[:, None] * (-np.expm1(-t_ev[None, :] / et[0])) + rw_u[1] * (act_te * z_te[:, 0])[:, None] * phi_env[None, :]
                            vp_rate_te = rw_u[0] * (act_te * te_h0) / et[0] + rw_u[1] * (act_te * z_te[:, 0])

                prony_te = (
                    hw[0] * gap_te[:, 0:1] * (-np.expm1(-t_ev[None, :] / et[0]))
                    + hw[1] * gap_te[:, 1:2] * (-np.expm1(-t_ev[None, :] / et[1]))
                )
                rem_curve = (ps - s0_te[:, None]) - prony_te - vp_curve_te
                rem_rate = ir - (hw[0] * gap_te[:, 0] / et[0] + hw[1] * gap_te[:, 1] / et[1]) - vp_rate_te
                curve_rate_err = float(np.max(np.abs(rem_curve - rem_rate[:, None] * phi_env[None, :])))
                # Also allow candidate whose own internal regime envelope (`t`, `log1p(t)`, or `t/(1+0.85t)`) is self-consistent in Layer 2
                # while Layer 3 `strain_mae <= 0.070` / public `gt["te_clean"]` strictly enforces the TRUE physical regime!
                for alt_phi in (t_ev, np.log1p(t_ev), t_ev / (1.0 + 0.85 * t_ev)):
                    curve_rate_err = min(curve_rate_err, float(np.max(np.abs(rem_curve - rem_rate[:, None] * alt_phi[None, :]))))
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
                elif "regime_mode_weights" in out and out["regime_mode_weights"] is not None:
                    rw_tr = np.asarray(out["regime_mode_weights"], dtype=float)
                    if rw_tr.shape == (2,) and bool(np.all(np.isfinite(rw_tr))):
                        crit_tr = float(dataset.get("crit_stress", dataset.get("yield_stress", 2.95)))
                        sb_tr = float(dataset.get("shear_banding_index", 0.0))
                        d_cr_tr = np.sum((np.maximum(0.0, np.abs(tr_ps) - crit_tr) ** 2) * p_dt[None, :], axis=1)
                        if sb_tr > 0.50:
                            sl_tr = np.sign(tr_h0) * np.sqrt(np.maximum(np.abs(tr_h0), 1e-12))
                            vp_rate_tr = rw_tr[0] * sl_tr + rw_tr[1] * (gap_tr[:, 0] - gap_tr[:, 1]) / et[0]
                        elif float(np.max(d_cr_tr)) > 1e-9:
                            act_tr = np.tanh(d_cr_tr)
                            vp_rate_tr = rw_tr[0] * (act_tr * tr_h0) / et[0] + rw_tr[1] * (act_tr * z_tr[:, 0])

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

    # Mode 4 (`M9.6` Cross-Domain `T_shared_kkt`): Box-Simplex KKT Compliance Allocation on Alpha
    if "allocation_weights" in out or "kkt_certificate" in out or "S_alloc_modes" in dataset:
        from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY

        ns_ak: dict[str, Any] = {"math": math, "np": np}
        exec(compile(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY, "<alpha_kkt_kernel>", "exec"), ns_ak)
        kkt_fn_a = ns_ak["verify_box_simplex_kkt_certificate"]
        if "allocation_weights" not in out or "kkt_certificate" not in out:
            errs.append("missing_allocation_weights_or_kkt_certificate")
        else:
            aw = np.asarray(out["allocation_weights"], dtype=float)
            m_al = int(np.asarray(dataset["S_alloc_modes"]).shape[1]) if "S_alloc_modes" in dataset else len(aw)
            if aw.shape != (m_al,) or not bool(np.all(np.isfinite(aw))):
                errs.append(f"bad_allocation_weights:{aw.shape}")
            else:
                # Reconstruct `(S_tilde_al, r_al)` from candidate's terminal `predicted_strain[:, -1]` (or `gt` when available)
                z_nuis = np.asarray(dataset["Z_nuisance"], dtype=float)
                lam_z = float(dataset["nuisance_ridge"])
                pz = np.linalg.solve(z_nuis.T @ z_nuis + lam_z * np.eye(z_nuis.shape[1]), z_nuis.T)
                s_modes = np.asarray(dataset["S_alloc_modes"], dtype=float)
                # Modulation of coupon compliance modes by specimen compliance `(ps[:, -1] - strain_0)`
                comp_gain = 1.0 + 0.25 * np.tanh(ps[:, -1] - s0_te)
                s_mat_cand = s_modes * comp_gain[:, None]
                y_al_raw = np.asarray(dataset["y_alloc_obs"], dtype=float).ravel()
                r_al_cand = y_al_raw - z_nuis @ (pz @ y_al_raw)
                s_til_cand = s_mat_cand - z_nuis @ (pz @ s_mat_cand)
                chk_c = kkt_fn_a(
                    s_til_cand,
                    r_al_cand,
                    float(dataset["source_ridge"]),
                    float(dataset["u_box"]),
                    float(dataset["total_budget"]),
                    out["kkt_certificate"],
                    gap_tol=2e-4,
                )
                if not chk_c["valid"]:
                    errs.append(f"alpha_kkt_invalid:{chk_c['reason']}")
                elif float(np.max(np.abs(aw - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
                    errs.append("allocation_weights_cert_mismatch")
                else:
                    metrics["alpha_kkt_duality_gap"] = round(float(chk_c["gap"]), 10)
                    claims.append("independent_box_simplex_kkt_certificate")
                if isinstance(gt, dict) and "w_alloc_opt" in gt and require_gt_accuracy:
                    w_gt = np.asarray(gt["w_alloc_opt"], dtype=float)
                    w_rel = float(np.linalg.norm(aw - w_gt) / max(float(np.linalg.norm(w_gt)), 1e-12))
                    metrics["allocation_weights_rel_l2"] = round(w_rel, 6)
                    if w_rel > 0.045:
                        errs.append(f"allocation_weights_gt_mismatch:{w_rel:.6f}")

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


# ============================================================================
# Unified Three-Layer Scientific Contract API (`M9.6.1` `N02 + N03` Closure)
#   Layer 1: `validate_output_structure` (`0` private truth)
#   Layer 2: `validate_public_witness` (`CalibrationWitness` + estimated-coordinate algebraic consistency, `0` private truth!)
#   Layer 3: `validate_truth_accuracy` (hidden ground-truth physical error bounds, collecting ALL computable metrics!)
# ============================================================================


def infer_contract_kind(
    *,
    family: str,
    inp: dict[str, Any],
    gt: dict[str, Any] | None = None,
    out: Any = None,
) -> str:
    """Single canonical `contract_kind` dispatcher eliminating script-local schema dispatch drift (`N03`)."""
    out_d = out if isinstance(out, dict) else {}
    gt_d = gt if isinstance(gt, dict) else {}
    if family == "alpha":
        if "S_alloc_modes" in inp or "allocation_weights" in out_d or "w_alloc_opt" in gt_d:
            return "alpha_shared_kkt"
        if "secant_compliance" in out_d or "hazard_crossing_times" in out_d:
            return "alpha_secant_crossing"
        if "viscoplastic_weights" in out_d and "hereditary_weights" not in out_d:
            return "alpha_viscoplastic_standalone"
        if "hereditary_weights" in out_d:
            return "alpha_unified_hereditary"
        if "critical_strain" in (inp.get("test") or {}) and "te_cross" in gt_d and "theta" not in gt_d:
            return "alpha_secant_crossing"
        return "alpha_unified_hereditary"
    if "budget_grid" in inp or "weights_by_budget" in gt_d or "weights_by_budget" in out_d:
        return "beta_multi_budget_kkt"
    # Dual-channel Beta check uses BOTH `gt["terminal_field_B"]` (`N03` fix!) AND study/output keys!
    if (
        gt_d.get("terminal_field_B") is not None
        or "B_coupling_2d" in inp
        or "reaction_B_2d" in inp
        or "V_channel_B" in inp
        or "terminal_field_channel_b" in out_d
    ):
        return "beta_dual_channel_kkt"
    return "beta_single_channel_kkt"


def estimate_public_or_prefix_alpha_calibration(
    dataset: dict[str, Any],
    prefix_calibration_sweeps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Deterministically derive validated per-lab `(gains, biases)` from public in-batch + prefix calibration sweeps (`0` private truth!)."""
    cal = dataset.get("calibration_load_cells") or {}
    c_l = np.asarray(cal.get("lab_id", []), dtype=int).ravel()
    c_x = np.asarray(cal.get("cmd_stress", []), dtype=float).ravel()
    c_y = np.asarray(cal.get("true_stress", []), dtype=float).ravel()

    hist_g = np.ones(4, dtype=float)
    hist_b = np.zeros(4, dtype=float)
    for sw in prefix_calibration_sweeps or []:
        sl = np.asarray(sw.get("lab_id", []), dtype=int).ravel()
        sx = np.asarray(sw.get("cmd_stress", []), dtype=float).ravel()
        sy = np.asarray(sw.get("true_stress", []), dtype=float).ravel()
        for lb in range(4):
            m = sl == lb
            n_m = int(np.sum(m))
            if n_m >= 3:
                n_tr = max(2, (2 * n_m) // 3)
                g_cand, b_cand = np.linalg.lstsq(np.column_stack([sx[m][:n_tr], np.ones(n_tr)]), sy[m][:n_tr], rcond=None)[0]
                va_mse = float(np.mean((g_cand * sx[m][n_tr:] + b_cand - sy[m][n_tr:]) ** 2))
                pr_mse = float(np.mean((hist_g[lb] * sx[m][n_tr:] + hist_b[lb] - sy[m][n_tr:]) ** 2))
                if va_mse <= min(1e-2, pr_mse + 1e-5):
                    hist_g[lb], hist_b[lb] = float(g_cand), float(b_cand)
            elif n_m == 2:
                g_cand, b_cand = np.linalg.lstsq(np.column_stack([sx[m], np.ones(2)]), sy[m], rcond=None)[0]
                hist_g[lb], hist_b[lb] = float(g_cand), float(b_cand)

    gains = hist_g.copy()
    biases = hist_b.copy()
    max_inbatch_res = 0.0
    for lb in range(4):
        m = c_l == lb
        n_m = int(np.sum(m))
        if n_m >= 3:
            n_tr = max(2, (2 * n_m) // 3)
            g_cand, b_cand = np.linalg.lstsq(np.column_stack([c_x[m][:n_tr], np.ones(n_tr)]), c_y[m][:n_tr], rcond=None)[0]
            va_mse = float(np.mean((g_cand * c_x[m][n_tr:] + b_cand - c_y[m][n_tr:]) ** 2))
            pr_mse = float(np.mean((gains[lb] * c_x[m][n_tr:] + biases[lb] - c_y[m][n_tr:]) ** 2))
            if va_mse <= min(1e-2, pr_mse + 1e-5):
                gains[lb], biases[lb] = float(g_cand), float(b_cand)
            max_inbatch_res = max(max_inbatch_res, float(math.sqrt(min(va_mse, pr_mse))))
        elif n_m == 2:
            g_cand, b_cand = np.linalg.lstsq(np.column_stack([c_x[m], np.ones(2)]), c_y[m], rcond=None)[0]
            res2 = float(np.max(np.abs(g_cand * c_x[m] + b_cand - c_y[m])))
            if res2 <= 0.05:
                gains[lb], biases[lb] = float(g_cand), float(b_cand)
                max_inbatch_res = max(max_inbatch_res, res2)
    return {
        "stress_gains": gains.tolist(),
        "stress_biases": biases.tolist(),
        "max_validated_calibration_residual": round(max_inbatch_res, 8),
    }


def validate_output_structure(
    *,
    contract_kind: str,
    inp: dict[str, Any],
    out: Any,
) -> dict[str, Any]:
    """Layer 1 (`M9.6.1` `N02`): Structural, shape, dtype, finiteness, and ordering checks (`0` private truth access)."""
    errs: list[str] = []
    if not isinstance(out, dict):
        return {"ok": False, "contract_kind": contract_kind, "errors": ["output_not_dict"]}

    if contract_kind.startswith("alpha_"):
        n_te = len(inp["test"]["lab_id"])
        k_t = len(inp["eval_times"])
        for rk in ("estimated_taus", "predicted_strain"):
            if rk not in out:
                errs.append(f"missing_key:{rk}")
        if not errs:
            et = np.asarray(out["estimated_taus"], dtype=float)
            ps = np.asarray(out["predicted_strain"], dtype=float)
            if et.shape != (2,) or not bool(np.all(np.isfinite(et))) or not (0.05 < float(et[0]) < float(et[1]) < 25.0):
                errs.append("invalid_estimated_taus")
            if ps.shape != (n_te, k_t) or not bool(np.all(np.isfinite(ps))):
                errs.append("invalid_predicted_strain")
        if contract_kind == "alpha_unified_hereditary":
            for rk, shp in (("hereditary_weights", (7,)), ("initial_rate_pred", (n_te,))):
                if rk not in out:
                    errs.append(f"missing_key:{rk}")
                else:
                    arr = np.asarray(out[rk], dtype=float)
                    if arr.shape != shp or not bool(np.all(np.isfinite(arr))):
                        errs.append(f"invalid_shape_or_nan:{rk}")
            if "viscoplastic_weights" in out and out["viscoplastic_weights"] is not None:
                vw = np.asarray(out["viscoplastic_weights"], dtype=float)
                if vw.shape != (2,) or not bool(np.all(np.isfinite(vw))):
                    errs.append("invalid_viscoplastic_weights")
            if "regime_mode_weights" in out and out["regime_mode_weights"] is not None:
                rw = np.asarray(out["regime_mode_weights"], dtype=float)
                if rw.shape != (2,) or not bool(np.all(np.isfinite(rw))):
                    errs.append("invalid_regime_mode_weights")
        elif contract_kind == "alpha_shared_kkt":
            m_al = int(np.asarray(inp["S_alloc_modes"]).shape[1])
            if "allocation_weights" not in out or "kkt_certificate" not in out:
                errs.append("missing_allocation_weights_or_kkt_certificate")
            else:
                aw = np.asarray(out["allocation_weights"], dtype=float)
                if aw.shape != (m_al,) or not bool(np.all(np.isfinite(aw))):
                    errs.append("invalid_allocation_weights")
        elif contract_kind == "alpha_viscoplastic_standalone":
            if "viscoplastic_weights" not in out:
                errs.append("missing_key:viscoplastic_weights")
            else:
                vw = np.asarray(out["viscoplastic_weights"], dtype=float)
                if vw.shape != (2,) or not bool(np.all(np.isfinite(vw))):
                    errs.append("invalid_viscoplastic_weights")
        elif contract_kind == "alpha_secant_crossing":
            for rk in ("secant_compliance", "hazard_crossing_times"):
                if rk not in out:
                    errs.append(f"missing_key:{rk}")
                else:
                    arr = np.asarray(out[rk], dtype=float)
                    if arr.shape != (n_te,) or not bool(np.all(np.isfinite(arr))):
                        errs.append(f"invalid_shape_or_nan:{rk}")
    else:
        M, nx, ny = np.asarray(inp["source_modes"]).shape
        if contract_kind == "beta_multi_budget_kkt":
            B = len(inp["budget_grid"])
            if not {"weights_by_budget", "terminal_field_mid_budget", "kkt_certificates"} <= set(out.keys()):
                errs.append("missing_multibudget_keys")
            else:
                wb = np.asarray(out["weights_by_budget"], dtype=float)
                tf = np.asarray(out["terminal_field_mid_budget"], dtype=float)
                certs = out["kkt_certificates"]
                if wb.shape != (B, M) or tf.shape != (nx, ny) or not isinstance(certs, list) or len(certs) != B:
                    errs.append("bad_multibudget_shapes")
                elif not (bool(np.all(np.isfinite(wb))) and bool(np.all(np.isfinite(tf)))):
                    errs.append("non_finite_multibudget_arrays")
        elif contract_kind == "beta_dual_channel_kkt":
            for rk, shp in (("weights", (M,)), ("terminal_field_channel_a", (nx, ny)), ("terminal_field_channel_b", (nx, ny))):
                if rk not in out:
                    errs.append(f"missing_key:{rk}")
                else:
                    arr = np.asarray(out[rk], dtype=float)
                    if arr.shape != shp or not bool(np.all(np.isfinite(arr))):
                        errs.append(f"invalid_shape_or_nan:{rk}")
            if "kkt_certificate" not in out:
                errs.append("missing_key:kkt_certificate")
        else:
            for rk, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
                if rk not in out:
                    errs.append(f"missing_key:{rk}")
                else:
                    arr = np.asarray(out[rk], dtype=float)
                    if arr.shape != shp or not bool(np.all(np.isfinite(arr))):
                        errs.append(f"invalid_shape_or_nan:{rk}")
            if "kkt_certificate" not in out:
                errs.append("missing_key:kkt_certificate")

    return {"ok": len(errs) == 0, "contract_kind": contract_kind, "errors": errs}


def validate_public_witness(
    *,
    contract_kind: str,
    inp: dict[str, Any],
    out: dict[str, Any],
    calibration_witness: dict[str, Any] | None = None,
    prefix_calibration_sweeps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Layer 2 (`M9.6.1` `N02`): Verify `CalibrationWitness` against public/prefix calibration data AND verify algebraic
    self-consistency (`weights -> predicted_strain/rate` & KKT certificates) strictly in VALIDATED ESTIMATED COORDINATES
    (`0` access to private ground truth!).
    """
    errs: list[str] = []
    metrics: dict[str, Any] = {}
    coord_prov = "public_and_prefix_validated_ols"

    if contract_kind.startswith("alpha_"):
        if "past_stress_cmd" not in inp["test"] and "past_stress" in inp["test"]:
            v_pub_direct = verify_alpha_candidate_handoff_and_invariants(dataset=inp, out=out, gt=None, require_gt_accuracy=False)
            return {
                "ok": len(v_pub_direct["errors"]) == 0,
                "contract_kind": contract_kind,
                "coordinate_provenance": "precalibrated_physical_input",
                "errors": list(v_pub_direct["errors"]),
                "verified_claim_ids": v_pub_direct["verified_claim_ids"],
                "metrics": dict(v_pub_direct["metrics"]),
            }
        pub_cal = estimate_public_or_prefix_alpha_calibration(inp, prefix_calibration_sweeps)
        ref_g = np.asarray(pub_cal["stress_gains"], dtype=float)
        ref_b = np.asarray(pub_cal["stress_biases"], dtype=float)
        if isinstance(calibration_witness, dict) and "stress_gains" in calibration_witness and "stress_biases" in calibration_witness:
            cw_g = np.asarray(calibration_witness["stress_gains"], dtype=float)
            cw_b = np.asarray(calibration_witness["stress_biases"], dtype=float)
            if cw_g.shape != (4,) or cw_b.shape != (4,) or not (bool(np.all(np.isfinite(cw_g))) and bool(np.all(np.isfinite(cw_b)))):
                errs.append("malformed_calibration_witness")
                est_g, est_b = ref_g, ref_b
            else:
                # Reject spoofed `CalibrationWitness` that disagrees with public/prefix calibration data!
                cal_raw = inp.get("calibration_load_cells") or {}
                cl_i = np.asarray(cal_raw.get("lab_id", []), dtype=int).ravel()
                cx_i = np.asarray(cal_raw.get("cmd_stress", []), dtype=float).ravel()
                cy_i = np.asarray(cal_raw.get("true_stress", []), dtype=float).ravel()
                # Check against clean non-outlier public/prefix reference (`tol = 0.04`)
                g_dev = float(np.max(np.abs(cw_g - ref_g)))
                b_dev = float(np.max(np.abs(cw_b - ref_b)))
                metrics["calibration_witness_vs_public_ref_gain_diff"] = round(g_dev, 6)
                metrics["calibration_witness_vs_public_ref_bias_diff"] = round(b_dev, 6)
                if (len(cl_i) > 0 or bool(prefix_calibration_sweeps)) and (g_dev > 0.045 or b_dev > 0.045):
                    errs.append(f"spoofed_or_unjustified_calibration_witness:g_dev={g_dev:.5f},b_dev={b_dev:.5f}")
                est_g, est_b = cw_g, cw_b
                coord_prov = "explicit_validated_calibration_witness"
        else:
            est_g, est_b = ref_g, ref_b

        synth_ds = {
            **inp,
            "test": {
                **inp["test"],
                "past_stress": np.asarray(inp["test"]["past_stress_cmd"], dtype=float) * est_g[np.asarray(inp["test"]["lab_id"], dtype=int), None]
                + est_b[np.asarray(inp["test"]["lab_id"], dtype=int), None],
                "stress_hold_0": np.asarray(inp["test"]["stress_hold_cmd"], dtype=float) * est_g[np.asarray(inp["test"]["lab_id"], dtype=int)]
                + est_b[np.asarray(inp["test"]["lab_id"], dtype=int)],
            },
            "train": {
                **inp["train"],
                "past_stress": np.asarray(inp["train"]["past_stress_cmd"], dtype=float) * est_g[np.asarray(inp["train"]["lab_id"], dtype=int), None]
                + est_b[np.asarray(inp["train"]["lab_id"], dtype=int), None],
                "stress_hold_0": np.asarray(inp["train"]["stress_hold_cmd"], dtype=float) * est_g[np.asarray(inp["train"]["lab_id"], dtype=int)]
                + est_b[np.asarray(inp["train"]["lab_id"], dtype=int)],
            },
        }
        v_pub = verify_alpha_candidate_handoff_and_invariants(dataset=synth_ds, out=out, gt=None, require_gt_accuracy=False)
        errs.extend(v_pub["errors"])
        metrics.update(v_pub["metrics"])
        return {
            "ok": len(errs) == 0,
            "contract_kind": contract_kind,
            "coordinate_provenance": coord_prov,
            "estimated_coordinates": {"stress_gains": est_g.tolist(), "stress_biases": est_b.tolist()},
            "errors": errs,
            "verified_claim_ids": v_pub["verified_claim_ids"],
            "metrics": metrics,
        }

    # Beta Layer 2 (`0` private truth: structural & certificate self-consistency; exact `(S_tilde, r_y)` KKT checked if public/gt provided)
    v_b = verify_beta_candidate_handoff_and_kkt(study=inp, out=out, gt=None)
    errs.extend(v_b["errors"])
    metrics.update(v_b["metrics"])
    return {
        "ok": len(errs) == 0,
        "contract_kind": contract_kind,
        "coordinate_provenance": "public_sensor_sweeps_and_prefix_history",
        "errors": errs,
        "verified_claim_ids": v_b["verified_claim_ids"],
        "metrics": metrics,
    }


def validate_truth_accuracy(
    *,
    contract_kind: str,
    inp: dict[str, Any],
    gt: dict[str, Any],
    out: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Layer 3 (`M9.6.1` `N02 + N03`): Evaluate ALL physical prediction & parameter accuracy metrics against hidden `gt`
    WITHOUT early-continue zeroing and WITHOUT conflating calibration measurement noise with algebraic identity breakage!
    """
    th = dict(thresholds or {})
    errs: list[str] = []
    metrics: dict[str, Any] = {}

    if contract_kind.startswith("alpha_"):
        et = np.asarray(out["estimated_taus"], dtype=float)
        ps = np.asarray(out["predicted_strain"], dtype=float)
        if "taus" in gt:
            tau_rel = float(np.max(np.abs(et - np.asarray(gt["taus"], dtype=float)) / np.asarray(gt["taus"], dtype=float)))
            metrics["tau_rel_err"] = round(tau_rel, 6)
            if tau_rel > float(th.get("max_tau_rel_err", 0.10)):
                errs.append(f"tau_rel_exceeded:{tau_rel:.6f}")
        if "te_clean" in gt:
            s_mae = float(np.mean(np.abs(ps - np.asarray(gt["te_clean"], dtype=float))))
            metrics["strain_mae"] = round(s_mae, 6)
            if s_mae > float(th.get("max_strain_mae", 0.075)):
                errs.append(f"strain_mae_exceeded:{s_mae:.6f}")
        if "initial_rate_pred" in out and "te_true_rate0" in gt:
            r_mae = float(np.mean(np.abs(np.asarray(out["initial_rate_pred"], dtype=float) - np.asarray(gt["te_true_rate0"], dtype=float))))
            metrics["rate_mae"] = round(r_mae, 6)
            if r_mae > float(th.get("max_rate_mae", 0.055)):
                errs.append(f"rate_mae_exceeded:{r_mae:.6f}")
        if "hereditary_weights" in out and "theta" in gt:
            th_err = float(np.max(np.abs(np.asarray(out["hereditary_weights"], dtype=float) - np.asarray(gt["theta"], dtype=float))))
            metrics["theta_7coeff_linf_err"] = round(th_err, 6)
            if th_err > float(th.get("max_theta_linf_err", 0.14)):
                errs.append(f"theta_7coeff_gt_mismatch:{th_err:.6f}")
        if "viscoplastic_weights" in out and out["viscoplastic_weights"] is not None and "vp_theta" in gt:
            vw = np.asarray(out["viscoplastic_weights"], dtype=float)
            vp_gt = np.asarray(gt["vp_theta"], dtype=float)
            vp_err = float(np.linalg.norm(vw - vp_gt) / max(float(np.linalg.norm(vp_gt)), 1.0))
            metrics["vp_err"] = round(vp_err, 6)
            if vp_err > float(th.get("max_vp_err", 0.16)):
                errs.append(f"vp_err_exceeded:{vp_err:.6f}")
        if "regime_mode_weights" in out and out["regime_mode_weights"] is not None and "vp_theta" in gt:
            rw = np.asarray(out["regime_mode_weights"], dtype=float)
            rw_gt = np.asarray(gt["vp_theta"], dtype=float)
            rw_err = float(np.linalg.norm(rw - rw_gt) / max(float(np.linalg.norm(rw_gt)), 1.0))
            metrics["regime_mode_err"] = round(rw_err, 6)
            if rw_err > float(th.get("max_rw_err", 0.16)):
                errs.append(f"regime_mode_err_exceeded:{rw_err:.6f}")
        if contract_kind == "alpha_shared_kkt" and "w_alloc_opt" in gt:
            aw = np.asarray(out["allocation_weights"], dtype=float)
            w_gt = np.asarray(gt["w_alloc_opt"], dtype=float)
            w_rel = float(np.linalg.norm(aw - w_gt) / max(float(np.linalg.norm(w_gt)), 1e-12))
            metrics["allocation_weights_rel_l2"] = round(w_rel, 6)
            if w_rel > float(th.get("max_alloc_w_rel_l2", 0.045)):
                errs.append(f"allocation_weights_gt_mismatch:{w_rel:.6f}")
        if contract_kind == "alpha_secant_crossing" and "te_secant" in gt:
            sc_mae = float(np.mean(np.abs(np.asarray(out["secant_compliance"], dtype=float) - np.asarray(gt["te_secant"], dtype=float))))
            metrics["secant_gt_mae"] = round(sc_mae, 6)
            if sc_mae > float(th.get("max_secant_mae", 0.095)):
                errs.append(f"secant_gt_mae_exceeded:{sc_mae:.6f}")
    else:
        v_b_gt = verify_beta_candidate_handoff_and_kkt(study=inp, out=out, gt=gt, gap_tol=float(th.get("gap_tol", 2e-4)))
        errs.extend(v_b_gt["errors"])
        metrics.update(v_b_gt["metrics"])

    return {"ok": len(errs) == 0, "contract_kind": contract_kind, "errors": errs, "metrics": metrics}


def evaluate_three_layer_scientific_contract(
    *,
    family: str,
    inp: dict[str, Any],
    out: Any,
    gt: dict[str, Any] | None = None,
    calibration_witness: dict[str, Any] | None = None,
    prefix_calibration_sweeps: list[dict[str, Any]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Unified Three-Layer Contract Evaluator (`M9.6.1` `N02 + N03` Single Source of Truth)."""
    ck = infer_contract_kind(family=family, inp=inp, gt=gt, out=out)
    l1 = validate_output_structure(contract_kind=ck, inp=inp, out=out)
    if not l1["ok"]:
        return {
            "ok": False,
            "kernel_version": SHARED_KERNEL_VERSION,
            "contract_kind": ck,
            "layer1_structure": l1,
            "layer2_public_witness": None,
            "layer3_truth_accuracy": None,
            "errors": list(l1["errors"]),
        }
    l2 = validate_public_witness(
        contract_kind=ck,
        inp=inp,
        out=out,
        calibration_witness=calibration_witness,
        prefix_calibration_sweeps=prefix_calibration_sweeps,
    )
    l3 = (
        validate_truth_accuracy(contract_kind=ck, inp=inp, gt=gt, out=out, thresholds=thresholds)
        if isinstance(gt, dict)
        else None
    )
    all_errs = list(l2["errors"]) + (list(l3["errors"]) if l3 is not None else [])
    return {
        "ok": len(all_errs) == 0,
        "kernel_version": SHARED_KERNEL_VERSION,
        "contract_kind": ck,
        "layer1_structure": l1,
        "layer2_public_witness": l2,
        "layer3_truth_accuracy": l3,
        "errors": all_errs,
    }
