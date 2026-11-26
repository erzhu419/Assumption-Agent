"""Calibrated two-branch hereditary creep with quadratic yield activation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _checks(table, lab):
    x = np.asarray(table['cmd_stress'], dtype=float)
    y = np.asarray(table['true_stress'], dtype=float)
    mask = (np.asarray(table['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _sweep(x, y):
    if len(x) < 4 or np.ptp(x) < .5:
        return None
    i, j = np.triu_indices(len(x), 1)
    dx = x[i] - x[j]
    good = np.abs(dx) > 1e-6
    gain = np.median((y[i] - y[j])[good] / dx[good])
    bias = np.median(y - gain*x)
    r = y - (gain*x + bias)
    mad = 1.4826*np.median(np.abs(r - np.median(r)))
    clean = np.abs(r) <= np.clip(4*mad, .002, .02)
    if clean.sum() < max(4, len(x)//2 + 1) or np.ptp(x[clean]) < .5:
        return None
    return np.polyfit(x[clean], y[clean], 1)


def _calibrate(table, labs):
    checks = {lab: _checks(table, lab) for lab in np.unique(labs)}
    fits = {lab: _sweep(x, y) for lab, (x, y) in checks.items()}
    if all(fit is not None for fit in fits.values()):
        return fits

    history = json.loads(_HISTORY.read_text()) if _HISTORY.exists() else []
    sweeps = [{lab: _sweep(*_checks(old, lab)) for lab in checks} for old in reversed(history)]

    def evidence(lab, prior):
        if prior is None:
            return 0., 0.
        direct = fits[lab]
        if direct is not None:
            err = np.max(np.abs(direct-prior))
            return (1. if err <= .02 else 0.), min(1., err/.02)
        x, y = checks[lab]
        if not len(x):
            return 0., 0.
        r = np.sort(np.abs(y - np.polyval(prior, x)))
        return np.mean(r <= .02), min(1., r[(len(r)-1)//2]/.02)

    def rank(lab, sweep):
        own_support, own_loss = evidence(lab, sweep[lab])
        other = [evidence(other_lab, prior) for other_lab, prior in sweep.items() if other_lab != lab]
        return (own_support >= .5, sum(q[0] for q in other), own_support, -own_loss, -sum(q[1] for q in other))

    selected = {}
    for lab, (x, y) in checks.items():
        fit = fits[lab]
        if fit is None:
            candidates = [sweep for sweep in sweeps if sweep[lab] is not None]
            if not candidates:
                fit = np.polyfit(x, y, 1) if len(x) >= 2 and np.ptp(x) >= .5 else np.array([1., np.median(y-x) if len(x) else 0.])
            else:
                fit = max(candidates, key=lambda sweep: rank(lab, sweep))[lab].copy()
                r = y - np.polyval(fit, x)
                close = np.abs(r) <= .02
                if close.sum() >= 2 and np.ptp(x[close]) >= .5:
                    fit = np.polyfit(x[close], y[close], 1)
                elif close.any():
                    fit[1] += np.median(r[close])
                elif len(r) >= 2 and np.ptp(r) <= .004:
                    fit[1] += np.median(r)
        selected[lab] = fit
    return selected


def _prep(d):
    def join(k):
        return np.concatenate((d['train'][k], d['test'][k]))
    labs, strata = join('lab_id'), join('stratum')
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    h, p = join('stress_hold_cmd').astype(float), join('past_stress_cmd').astype(float)
    for lab, (gain, bias) in _calibrate(d['calibration_load_cells'], labs).items():
        m = labs == lab
        h[m], p[m] = gain*h[m]+bias, gain*p[m]+bias
        controls = ref[m]
        controls = controls[:, np.isfinite(controls).sum(axis=0) >= 2]
        if controls.size:
            center = np.median(np.nanmean(controls, axis=0))
            scale = np.median(np.nanstd(controls, axis=0, ddof=0))
            x[m] = (x[m] - center) / (scale if scale > 0 else 1.)
    cells = [x[(labs == a) & (strata == b)] for a, b in np.unique(np.c_[labs, strata], axis=0)]
    counts = np.array([np.isfinite(cell).sum(0) for cell in cells])
    sums = np.array([np.nansum(cell, axis=0) for cell in cells])
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(0) / np.maximum(1, (counts > 0).sum(0))
    x = np.where(np.isfinite(x), x, fill)
    return np.column_stack((np.ones(len(h)), x)), h, p


def _design(tau, x, h, p, dt, t, yield_stress):
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    weights = np.exp(-after[:, None]/tau) * -np.expm1(-dt[:, None]/tau)
    gap = h[:, None] - p @ weights
    shape = -np.expm1(-t[:, None]/tau)
    ov_hold = np.maximum(np.abs(h)-yield_stress, 0.)**2 * np.sign(h)
    work_hold = (np.maximum(np.abs(p)-yield_stress, 0.)**2 @ dt) * h
    curves = np.concatenate((gap[:, None, :]*shape,
                             x[:, None, :]*t[None, :, None],
                             (ov_hold[:, None]*t)[:, :, None],
                             (work_hold[:, None]*shape[:, 0])[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x, ov_hold, work_hold/tau[0]))
    return curves, rates


def _fit(x, h, p, dt, t, sy, curves, rates):
    n, k = curves.shape
    obs = np.r_[curves.ravel(), rates]
    def design(tau):
        c, r = _design(tau, x, h, p, dt, t, sy)
        return np.vstack((c.reshape(-1, 9), r))

    def profile(w, start):
        def fit(v, return_all=False):
            tau = np.array([v[0], v[0] + (24.99-v[0])*v[1]])
            a = design(tau)
            coef = np.linalg.lstsq(a*w[:, None], obs*w, rcond=None)[0]
            residual = np.einsum("ij,j->i", a, coef) - obs
            return (tau, coef, residual) if return_all else residual*w
        result = least_squares(fit, start, bounds=([.0501, 1e-6], [24.98, 1.]),
                               x_scale='jac', max_nfev=90, ftol=1e-10)
        return result, fit(result.x, True)

    def start(lo, hi):
        return [lo, (hi-lo)/(24.99-lo)]

    w = np.r_[np.ones(n*k), np.full(n, np.sqrt(k))]
    optimum, model = min((profile(w, start(lo, hi)) for lo, hi in
                          ((.26, 1.3), (.55, 2.7), (1.1, 5.), (2., 9.))), key=lambda q: q[0].cost)
    for _ in range(5):
        res = model[2]
        s = [max(1e-7, 1.4826*np.median(np.abs(r-np.median(r))))
             for r in (res[:n*k], res[n*k:])]
        scale = np.r_[np.full(n*k, s[0]), np.full(n, s[1])]
        w = 1. / (scale * np.hypot(1., res/(3.*scale)))
        optimum, model = profile(w, optimum.x)
    # Refit the retained measurements after the robust pilot. The cutoff
    # follows the inlier scale, not the magnitude of the isolated spikes.
    e = model[2]
    ec = e[:n*k].reshape(n, k)
    sc = np.maximum(1e-7, 1.4826 * np.median(np.abs(ec - np.median(ec, axis=0)), axis=0))
    er = e[n*k:]
    sr = max(1e-7, 1.4826 * np.median(np.abs(er - np.median(er))))
    scale = np.r_[np.tile(sc, n), np.full(n, sr)]
    keep = np.abs(e) <= 6 * scale
    optimum, model = profile(keep / scale, optimum.x)
    return model[:2]


def solve_alpha_w03_batch_05(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    n = len(tr['lab_id'])
    x, h, p = _prep(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    sy = dataset['yield_stress']
    complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
    i = np.flatnonzero(complete) if complete.sum() >= max(10, .6*n) else np.arange(n)
    tau, coef = _fit(x[i], h[i], p[i], dt, t, sy,
                     tr['strain_curves_obs'][i] - tr['strain_0'][i, None], tr['instant_rate_obs'][i])
    c, r = _design(tau, x[n:], h[n:], p[n:], dt, t, sy)
    curves = te['strain_0'][:, None] + c @ coef
    rates = r @ coef
    observed_rate = np.asarray(te['instant_rate_obs'])
    incomplete = ~np.isfinite(te['coupon_features_raw']).all(axis=1) & np.isfinite(observed_rate)

    # Estimate persistent lab drift from late training measurements without
    # changing the shared hereditary or viscoplastic coefficients.
    late = (t >= np.median(t)) & (t > 0)
    if late.any():
        j = np.flatnonzero(complete)
        train_c, _ = _design(tau, x[j], h[j], p[j], dt, t[late], sy)
        residual = (tr['strain_curves_obs'][j][:, late] - tr['strain_0'][j, None]
                    - train_c @ coef)
        drift = np.median(residual / t[late], axis=1)
        train_labs = tr['lab_id'][j]
        for lab in np.unique(train_labs):
            use = (te['lab_id'] == lab) & ~incomplete
            correction = np.median(drift[train_labs == lab])
            curves[use] += correction * t
            rates[use] += correction

    # Missing covariates affect only linear drift, so the onset-rate
    # innovation contributes a constant rate correction over the hold.
    innovation = observed_rate[incomplete] - rates[incomplete]
    curves[incomplete] += innovation[:, None] * t
    rates[incomplete] += innovation
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], viscoplastic_weights=coef[7:],
                predicted_strain=curves, initial_rate_pred=rates)
