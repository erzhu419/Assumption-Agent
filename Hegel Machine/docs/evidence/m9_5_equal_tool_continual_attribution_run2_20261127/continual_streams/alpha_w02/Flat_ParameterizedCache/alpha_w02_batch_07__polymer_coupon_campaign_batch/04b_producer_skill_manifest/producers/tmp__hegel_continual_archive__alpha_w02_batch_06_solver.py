"""Profile the linear constitutive coefficients at two relaxation times."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _lab_checks(checks, lab):
    x, y = (np.asarray(checks[k], float) for k in ('cmd_stress', 'true_stress'))
    take = (np.asarray(checks['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _line(x, y, tol, prior=None):
    i, j = np.triu_indices(x.size, 1)
    usable = abs(x[j] - x[i]) >= 4 * tol
    i, j = i[usable], j[usable]
    gains = (y[j] - y[i]) / (x[j] - x[i])
    lines = np.c_[gains, y[i] - gains * x[i]]
    if prior is not None:
        lines = np.vstack((lines, prior))
    lines = lines[lines[:, 0] > 0]
    if not lines.size:
        return None, np.zeros(x.size, bool)
    r = abs(y - lines[:, :1] * x - lines[:, 1:])
    cost = np.minimum((r / tol)**2, 1).sum(axis=1)
    if prior is not None:
        cost += .025 * np.minimum(((lines - prior)**2).sum(axis=1) / tol**2, 1)
    mask = r[np.argmin(cost)] < tol
    if mask.sum() >= 2 and np.ptp(x[mask]) >= 4 * tol:
        fit = np.linalg.lstsq(np.c_[x[mask], np.ones(mask.sum())], y[mask], rcond=None)[0]
        return fit, mask
    return None, mask


def _calibrate(checks, sy):
    history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.exists() else []
    fits = []
    tol = .045 * sy
    tight = .004 * sy
    for lab in range(4):
        x, y = _lab_checks(checks, lab)
        # Verified sweeps are precise: a broad EMI rejection window is not
        # evidence that a short sweep identifies its slope.
        fit, mask = _line(x, y, tight)
        broad = fit is not None and mask.sum() >= 3 and np.ptp(x[mask]) >= .55 * sy
        # Four inliers across a useful span establish a current sweep without a prior.
        if broad and mask.sum() >= 4:
            fits.append(fit)
            continue
        priors = []
        for episode in history:
            xx, yy = _lab_checks(episode, lab)
            p, m = _line(xx, yy, tol)
            if p is not None and m.sum() >= 3 and np.ptp(xx[m]) >= .55 * sy:
                priors.append(p)
        if priors:
            priors = np.asarray(priors)
            if x.size:
                delta = y - priors[:, :1] * x - priors[:, 1:]
                # Archived sweeps in the same instrument regime agree much more
                # closely than the broad tolerance used to reject EMI.
                count = (abs(delta) < tight).sum(axis=1)
                cost = np.minimum((delta / tight)**2, 1).sum(axis=1)
                best = np.lexsort((-np.arange(len(priors)), cost, -count))[0]
                prior = priors[best]
                good = abs(delta[best]) < tight
                if good.any() and (not broad or good.sum() >= min(2, x.size)):
                    fit = prior + [0, np.median(delta[best, good])]
                elif not broad:
                    # Without a wide current sweep, do not extrapolate a gain
                    # estimated from close command values or shift on an EMI point.
                    fit = priors[-1]
            else:
                fit = priors[-1]
        fits.append(fit if fit is not None else np.array([1., np.median(y - x) if x.size else 0.]))
    return np.asarray(fits)


def _features(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    feat = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        m = labs == lab
        r = ref[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[m] = (feat[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    _, cell = np.unique(np.c_[labs, strata], axis=0, return_inverse=True)
    cell = cell.ravel()
    finite = np.isfinite(feat)
    means = []
    for k in np.unique(cell):
        rows = cell == k
        count = finite[rows].sum(axis=0)
        total = np.where(finite[rows], feat[rows], 0.).sum(axis=0)
        means.append(np.divide(total, count, out=np.full(feat.shape[1], np.nan), where=count > 0))
    means = np.array(means)
    count = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), count, out=np.zeros(feat.shape[1]), where=count > 0)
    return np.c_[np.ones(len(feat)), np.where(finite, feat, fill)]


def _inputs(split, feat, cal, dt, sy):
    gain, bias = cal[np.asarray(split['lab_id'], int)].T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    ov = np.maximum(abs(hold) - sy, 0)**2 * np.sign(hold)
    work = np.sum(np.maximum(abs(past) - sy, 0)**2 * dt, axis=1) * hold
    return past, hold, np.c_[feat, ov], work


def _design(inputs, dt, t, tau):
    past, hold, drift, work = inputs
    end = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    mem = np.exp(-end[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    delta = hold[:, None] - past @ mem
    response = -np.expm1(-t[:, None] / tau)
    curve = np.concatenate((delta[:, None, :] * response,
                            drift[:, None, :] * t[None, :, None],
                            (work[:, None] * response[:, 0])[:, :, None]), axis=2)
    rate = np.c_[delta / tau, drift, work / tau[0]]
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _mad(x):
    return 1.4826 * np.nanmedian(abs(x - np.nanmedian(x)))


def _fit(inputs, dt, t, y):
    valid = np.isfinite(y)
    raw = y[valid]
    active = np.ones(9, bool)
    active[7] = np.any(inputs[2][:, -1])
    active[8] = np.any(inputs[3])
    weights = np.broadcast_to(np.r_[np.ones(len(t)), 1 / 1.5], y.shape)[valid]
    b = raw * weights

    def profile(z):
        tau = np.sort(np.exp(z))
        a = _design(inputs, dt, t, tau)[valid][:, active] * weights[:, None]
        coef = np.linalg.lstsq(a, b, rcond=None)[0]
        return coef, (a * coef).sum(axis=1) - b

    limits = np.log(([.051, .051], [24.99, 24.99]))
    starts = ((.26, 1.6), (.64, 3.8), (1.5, 9.))
    pilot = min((least_squares(lambda z: profile(z)[1], np.log(start), bounds=limits)
                 for start in starts), key=lambda fit: fit.cost)
    r = np.full(y.shape, np.nan)
    r[valid] = profile(pilot.x)[1] / weights
    curve_scale = max(_mad(r[:, :-1]), 1e-9)
    rate_scale = _mad(r[:, -1]) if valid[:, -1].any() else curve_scale
    scale = np.r_[np.full(len(t), curve_scale), max(rate_scale, curve_scale / 4)]
    base_weights = np.broadcast_to(1 / scale, y.shape)[valid]
    z = pilot.x
    for _ in range(5):
        resid = profile(z)[1] / weights
        weights = base_weights / np.sqrt(1 + (base_weights * resid / 3)**2)
        b = raw * weights
        z = least_squares(lambda v: profile(v)[1], z, bounds=limits).x
    coef = np.zeros(9)
    coef[active] = profile(z)[0]
    return np.sort(np.exp(z)), coef


def _rate_gains(inputs, dt, t, y, lab, complete):
    """Estimate how much rate innovation persists as linear curve drift."""
    gains = np.zeros(4)
    for j in np.unique(lab):
        fit = (lab != j) & complete
        valid = (lab == j) & complete & np.isfinite(y[:, -1])
        if fit.sum() < 9 or valid.sum() < 9:
            continue
        # A held-out lab cannot estimate a plastic term absent in the other labs.
        if any(np.any(v[valid]) and not np.any(v[fit]) for v in (inputs[2][:, -1], inputs[3])):
            continue
        tau, coef = _fit(tuple(v[fit] for v in inputs), dt, t, y[fit])
        err = y[valid] - _design(tuple(v[valid] for v in inputs), dt, t, tau) @ coef
        positive = t > 0
        if not positive.any():
            continue
        curves = err[:, :-1][:, positive] / t[positive]
        rows = np.isfinite(curves).any(axis=1)
        rate = err[rows, -1]
        drift = np.nanmedian(curves[rows], axis=1)
        if len(drift) < 9 or not np.any(rate):
            continue
        scale = max(_mad(drift - rate), 1e-9)
        gains[j] = least_squares(lambda k: k[0] * rate - drift, [.25],
                                 bounds=(0., 1.), loss='soft_l1', f_scale=scale).x[0]
    return gains


def _apply_rate(pred, split, t, gains):
    if 'instant_rate_obs' not in split:
        return pred
    observed = np.asarray(split['instant_rate_obs'])
    complete = np.isfinite(split['coupon_features_raw']).all(axis=1)
    gain = np.where(complete, gains[np.asarray(split['lab_id'], int)], 1.)
    drift = gain * np.where(np.isfinite(observed), observed - pred[:, -1], 0.)
    return pred + drift[:, None] * np.r_[t, 1.]


def solve_alpha_w02_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    sy = float(dataset['yield_stress'])
    cal = _calibrate(dataset['calibration_load_cells'], sy)
    feat = _features(train, test)
    n = len(train['lab_id'])
    y = np.c_[train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None], train['instant_rate_obs']]
    inputs = _inputs(train, feat[:n], cal, dt, sy)
    # Mean imputation is for prediction, not a measured material covariate.
    # Keep the full data if complete coupons do not cover an active plastic term.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1) & np.isfinite(y).any(axis=1)
    gains = (_rate_gains(inputs, dt, t, y, np.asarray(train['lab_id'], int), complete)
             if 'instant_rate_obs' in test else np.zeros(4))
    if (complete.sum() >= 9 and
            all(not np.any(v) or np.any(v[complete]) for v in (inputs[2][:, -1], inputs[3]))):
        inputs, y = tuple(v[complete] for v in inputs), y[complete]
    tau, coef = _fit(inputs, dt, t, y)
    pred = _design(_inputs(test, feat[n:], cal, dt, sy), dt, t, tau) @ coef
    pred = _apply_rate(pred, test, t, gains)
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], viscoplastic_weights=coef[7:],
                predicted_strain=pred[:, :-1] + np.asarray(test['strain_0'])[:, None], initial_rate_pred=pred[:, -1])
