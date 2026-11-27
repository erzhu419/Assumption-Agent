"""Calibrated two-branch hereditary response with linear terminal flow."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = "/tmp/hegel_continual_archive/archive_calibration_history.json"


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _calibrate(cal):
    coefficients = np.empty((4, 2))
    history = None
    # Measured clean sweeps have residuals below 2e-4 stress units.
    noise = 2e-4

    def archived_line(lab):
        nonlocal history
        if history is None:
            with open(_ARCHIVE) as f:
                history = json.load(f)
        sweep = next(row for row in reversed(history) if lab in row['lab_id'])
        keep = np.asarray(sweep['lab_id']) == lab
        return np.polyfit(np.asarray(sweep['cmd_stress'])[keep],
                          np.asarray(sweep['true_stress'])[keep], 1)

    labs = np.asarray(cal['lab_id'])
    for lab in range(4):
        x, y = (np.asarray(cal[key])[labs == lab]
                for key in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 2:
            coefficients[lab] = archived_line(lab)
            continue
        design = np.column_stack((x, np.ones(len(x))))
        coefficients[lab] = np.linalg.lstsq(design, y, rcond=None)[0]
        candidates = []
        inconsistent = False
        possible_emi = False
        for j in range(len(x)) if len(x) >= 3 else ():
            keep = np.arange(len(x)) != j
            if np.ptp(x[keep]) == 0:
                continue
            inverse = np.linalg.pinv(design[keep])
            line = inverse @ y[keep]
            predict = design[j] @ inverse
            deleted = abs(y[j] - predict @ y[keep])
            uncertainty = noise * (1 + np.abs(predict).sum())
            # Check line consistency separately from possible spike amplitude:
            # the highest-leverage reading can have the least precise prediction.
            # Three-point sweeps need a tighter consistency screen to
            # distinguish a borderline spike from a clean near-repeat.
            consistency_limit = uncertainty * (0.625 if len(x) == 3 else 1.0)
            inconsistent |= deleted > consistency_limit
            possible_emi |= deleted + uncertainty > 3.0
            error = np.max(np.abs(y[keep] - design[keep] @ line))
            if deleted + uncertainty > 3.0:
                candidates.append((error, keep, line))
        if not (inconsistent and possible_emi):
            continue
        prior = archived_line(lab)
        archive_error = np.abs(y - design @ prior)
        keep = archive_error <= 2 * noise
        if keep.sum() == len(x) - 1 and archive_error[~keep][0] > 3.0 - 2 * noise:
            # The archive identifies the spike independently of extrapolation.
            coefficients[lab] = (np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
                                 if np.ptp(x[keep]) >= 0.1 else prior)
            continue
        best_error = min(item[0] for item in candidates)
        plausible = [item for item in candidates if item[0] <= best_error + noise]
        _, keep, line = min(plausible, key=lambda item: np.max(
            np.abs(y[item[1]] - design[item[1]] @ prior)))
        # Do not estimate a new slope from two effectively repeated commands
        # when their clean readings already agree with the archived line.
        if np.ptp(x[keep]) < 0.1 and np.max(np.abs(y[keep] - design[keep] @ prior)) <= 2 * noise:
            line = prior
        coefficients[lab] = line
    return coefficients[:, 0], coefficients[:, 1]


def _design(split, aligned, gain, bias, dt, t):
    lab = split['lab_id']
    past = gain[lab, None] * split['past_stress_cmd'] + bias[lab, None]
    hold = gain[lab] * split['stress_hold_cmd'] + bias[lab]
    features = np.column_stack((np.ones(len(lab)), aligned))
    remaining = np.cumsum(dt[::-1])[::-1] - dt

    def evaluate(log_tau):
        taus = np.exp(log_tau)
        kernel = np.exp(-remaining[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
        gaps = hold[:, None] - past @ kernel
        curve = np.concatenate((
            gaps[:, None, :] * -np.expm1(-t[:, None] / taus),
            t[None, :, None] * features[:, None, :],
        ), axis=2)
        return curve, np.column_stack((gaps / taus, features))

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, t):
    cm, rm = np.isfinite(curve_y), np.isfinite(rate_y)
    y = np.r_[curve_y[cm], rate_y[rm]]
    groups = np.r_[np.broadcast_to(np.arange(len(t)), cm.shape)[cm], np.full(rm.sum(), len(t))]
    masks = [groups == j for j in np.unique(groups)]
    lo = np.log(min(dt.min(), t[t > 0].min()) / 20)
    hi = np.log(20 * max(dt.sum(), t.max()))

    def matrix(log_tau):
        curve, rate = evaluate(log_tau)
        return np.vstack((curve[cm], rate[rm]))

    def project(m, weights):
        return np.linalg.lstsq(m * weights[:, None], y * weights, rcond=None)[0]

    def scales(residual):
        result = np.empty_like(residual)
        for mask in masks:
            r = residual[mask]
            result[mask] = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        return result

    def profile(log_tau, weights):
        m = matrix(log_tau)
        return (m @ project(m, weights) - y) * weights

    rough = np.clip(np.log([0.45, 3.5]), lo, hi)
    base = matrix(rough)
    weights = np.ones_like(y)
    for _ in range(7):
        residual = base @ project(base, weights) - y
        scale = scales(residual)
        weights = 1 / (scale * (1 + (residual / (3 * scale))**2)**0.25)
    starts = [rough, np.clip(np.log([0.18, 1.6]), lo, hi), np.clip(np.log([1.2, 8.0]), lo, hi)]
    fits = [least_squares(profile, s, args=(weights,), bounds=(lo, hi), max_nfev=60) for s in starts]
    tau = min(fits, key=lambda fit: fit.cost).x
    for _ in range(9):
        m = matrix(tau)
        residual = m @ project(m, weights) - y
        scale = scales(residual)
        weights = 1 / np.hypot(scale, residual / 2.5)
        tau = least_squares(profile, tau, args=(weights,), bounds=(lo, hi), max_nfev=40).x
    beta = project(matrix(tau), weights)
    order = np.argsort(tau)
    return tau[order], np.r_[beta[:2][order], beta[2:]]


def solve_alpha_m97_c03_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    xtr, xte = _align(train, test)
    train_design = _design(train, xtr, gain, bias, dt, t)
    # Fit on measured features; use aligned cell means when predicting missing ones.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    curve_y = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate_y = train['instant_rate_obs']
    log_tau, beta = _fit(train_design, np.where(complete[:, None], curve_y, np.nan),
                         np.where(complete, rate_y, np.nan), dt, t)
    curve, rate = _design(test, xte, gain, bias, dt, t)(log_tau)
    return dict(estimated_taus=np.exp(log_tau), hereditary_weights=beta,
                regime_mode_weights=np.zeros(2),
                predicted_strain=test['strain_0'][:, None] + curve @ beta,
                initial_rate_pred=rate @ beta)
