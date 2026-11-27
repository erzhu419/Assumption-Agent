"""Calibrated two-branch rheology with saturating terminal flow and slip."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _prefix_calibration():
    with open(_ARCHIVE) as handle:
        sweeps = json.load(handle)
    fits = [[] for _ in range(4)]
    for sweep in sweeps:
        lab_ids = np.asarray(sweep['lab_id'])
        cmds = np.asarray(sweep['cmd_stress'])
        measured = np.asarray(sweep['true_stress'])
        for lab in range(4):
            rows = (lab_ids == lab) & np.isfinite(cmds) & np.isfinite(measured)
            if rows.sum() >= 2:
                fits[lab].append(_line(cmds[rows], measured[rows]))
    return np.asarray([np.median(v, axis=0) for v in fits])


def _calibrate(cal):
    params = np.empty((4, 2))
    archive = None
    for lab in range(4):
        rows = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][rows], cal['true_stress'][rows]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            dx = x[j] - x[i]
            use = dx != 0
            if use.any():
                slope = np.median((y[j] - y[i])[use] / dx[use])
                offset = np.median(y - slope*x)
                clean = np.abs(y - slope*x - offset) <= 3.0
        # With at most four rows, a spike can tilt even the median-slope
        # fit. Screen for a gross departure from the nominal unit-gain
        # response, then use the prefix line to apply the actual EMI cutoff.
        if 2 <= len(x) <= 4 and np.any(np.abs(y - x) > 1.5):
            if archive is None:
                archive = _prefix_calibration()
            gain, bias = archive[lab]
            clean &= np.abs(y - (gain*x + bias)) <= 3.0
        if clean.sum() < 2 or not clean.all():
            if archive is None:
                archive = _prefix_calibration()
        if clean.sum() < 2 or (not clean.all() and clean.sum() < 4):
            params[lab] = archive[lab]
        else:
            params[lab] = _line(x[clean], y[clean])
    return params


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate((train[k], test[k])) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty(raw.shape, dtype=float)
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (raw[mask] - center) / scale
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i in range(len(cells)):
        block = x[ids == i]
        n = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), n, out=means[i], where=n > 0)
    for i, (lab, _) in enumerate(cells):
        for k in np.flatnonzero(~np.isfinite(means[i])):
            pool = means[cells[:, 0] == lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            means[i, k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
    x = np.where(np.isfinite(x), x, means[ids])
    return np.split(x, [len(train['lab_id'])])


def _design(split, x, params, dt, t, jammed):
    g, b = params[split['lab_id']].T
    hist = g[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = g*split['stress_hold_cmd'] + b
    age = np.cumsum(dt[::-1])[::-1] - dt
    # Use the exact impulse-response sum of the zero-initialized recurrence.
    flow = np.column_stack((np.ones(len(x)), x))
    phi = t/(1.0 + 0.85*t) if jammed else t
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    curve = np.zeros((len(x), len(t), 9))
    rate = np.zeros((len(x), 9))
    curve[:, :, 2:7] = flow[:, None, :]*phi[None, :, None]
    rate[:, 2:7] = flow
    if jammed:
        curve[:, :, 7] = slip[:, None]*phi
        rate[:, 7] = slip

    def evaluate(tau):
        kernel = np.exp(-age[:, None]/tau)*(-np.expm1(-dt[:, None]/tau))
        gap = hold[:, None] - hist @ kernel
        exp_term = -np.expm1(-t[:, None]/tau)
        curve[:, :, :2] = gap[:, None, :]*exp_term
        rate[:, :2] = gap/tau
        if jammed:
            gap_delta = gap[:, 0] - gap[:, 1]
            curve[:, :, 8] = gap_delta[:, None]*exp_term[:, 0]
            rate[:, 8] = gap_delta/tau[0]
        return curve, rate

    return evaluate


def _fit(evaluate, curve_y, rate_y, dt, t, rate_weight=1.0, row_mask=None):
    if row_mask is None:
        row_mask = np.ones(len(rate_y), dtype=bool)
    cm = np.isfinite(curve_y) & row_mask[:, None]
    rm = np.isfinite(rate_y) & row_mask
    y = np.r_[curve_y[cm], rate_weight*rate_y[rm]]

    def project(log_tau, coefficients=False):
        tau = np.sort(np.exp(log_tau))
        curve, rate = evaluate(tau)
        design = np.vstack((curve[cm], rate_weight*rate[rm]))
        w = np.linalg.lstsq(design, y, rcond=None)[0]
        return (tau, w) if coefficients else design @ w - y

    t0 = max(np.min(t[t > 0]), np.min(dt))
    horizon = max(np.sum(dt), np.max(t))
    bounds = np.log([t0/20.0, horizon*30.0])
    starts = [(t0, horizon/3), (2*t0, horizon/2), (t0/3, horizon),
              (horizon/5, horizon*2), (t0/2, 2*t0)]
    best = min((least_squares(project, np.log(start), bounds=tuple(bounds),
                              max_nfev=90, ftol=1e-9, gtol=1e-9) for start in starts),
               key=lambda fit: fit.cost)
    return project(best.x, coefficients=True)



def _fit_robust(evaluate, curve_y, rate_y, dt, t, row_mask=None):
    if row_mask is None:
        row_mask = np.ones(len(rate_y), dtype=bool)
    tau, w = _fit(evaluate, curve_y, rate_y, dt, t, row_mask=row_mask)
    cm = np.isfinite(curve_y) & row_mask[:, None]
    rm = np.isfinite(rate_y) & row_mask
    p = np.r_[np.log(tau), w]
    lo = np.r_[[np.log(max(min(t[t > 0]), min(dt))/20)]*2, [-np.inf]*len(w)]
    hi = np.r_[[np.log(max(sum(dt), max(t))*30)]*2, [np.inf]*len(w)]

    def errors(p):
        curve, rate = evaluate(np.sort(np.exp(p[:2])))
        return (curve @ p[2:] - curve_y)[cm], (rate @ p[2:] - rate_y)[rm]

    def scale(a):
        return max(1.4826*np.median(np.abs(a - np.median(a))), 1e-6)

    # Stabilize the fit before applying a loss that strongly downweights spikes.
    for loss, cutoff in [('soft_l1', 1.5)]*3 + [('cauchy', 2.5)]:
        if loss == 'soft_l1':
            sc, sr = map(scale, errors(p))

        def residual(p):
            ec, er = errors(p)
            return np.r_[ec/sc, er/sr]

        p = least_squares(residual, p, bounds=(lo, hi), loss=loss,
                          f_scale=cutoff, x_scale='jac', max_nfev=160).x
    return np.sort(np.exp(p[:2])), p[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    params = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    jammed = dataset['shear_banding_index'] > 0.50
    design = _design(train, x_train, params, dt, t, jammed)
    tau, w = _fit_robust(design, train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'], dt, t)
    curve, rate = _design(test, x_test, params, dt, t, jammed)(tau)
    return {'estimated_taus': tau, 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
