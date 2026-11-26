"""Fit the two-branch creep law and reconcile sparse load-cell checks."""
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


def _calibration_seed(checks, sy):
    history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.exists() else []
    tol = .004 * sy
    seed, fixed = [], []
    for lab in range(4):
        prior = np.array([1., 0.])
        archived = False
        for episode in reversed(history):
            x, y = _lab_checks(episode, lab)
            line, inliers = _line(x, y, tol)
            if line is not None and inliers.sum() >= 4 and np.ptp(x[inliers]) >= .55 * sy:
                prior, archived = line, True
                break
        x, y = _lab_checks(checks, lab)
        line, inliers = _line(x, y, tol, prior if archived else None)
        agrees = archived and x.size >= 2 and np.sum(abs(y - (prior[0] * x + prior[1])) < tol) >= 2
        trusted = (line is not None and np.ptp(x[inliers]) >= .55 * sy
                   and (inliers.sum() >= 3 or agrees))
        seed.append(line if trusted else prior)
        fixed.append(trusted)
    fixed = np.array(fixed, dtype=bool)
    if not fixed.any():
        # Without an absolute check, the archived lab-0 line defines the scale.
        fixed[0] = True
    return np.asarray(seed), fixed


def _regress(train, feat, dt, t, sy, seed, fixed):
    y = np.c_[train['strain_curves_obs'] - np.asarray(train['strain_0'])[:, None],
              train['instant_rate_obs']]
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    labs = np.asarray(train['lab_id'])
    for lab in np.unique(labs):
        if np.sum(complete & (labs == lab)) < 5:
            complete[labs == lab] = True
    base_inputs = _inputs(train, feat, seed, dt, sy)
    for signal in (base_inputs[2][:, -1], base_inputs[3]):
        if signal.any() and not signal[complete].any():
            complete[:] = True
    a = {k: np.asarray(train[k])[complete] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd')}
    f, y = feat[complete], y[complete]
    valid = np.isfinite(y)
    observed = y[valid]
    # Isolated curve spikes are much larger than ordinary rate and curve noise.
    # Reweight individual observations rather than dropping an entire coupon.
    weights = np.broadcast_to(np.r_[np.ones(len(t)), 2.], y.shape)[valid].copy()
    base = weights.copy()
    free = ~fixed
    lo = np.r_[np.log([.05, .45]), np.tile([.25, -1.1 * sy], free.sum())]
    hi = np.r_[np.log([4., 25.]), np.tile([3., 1.1 * sy], free.sum())]

    def unpack(z):
        cal = seed.copy()
        cal[free] = z[2:].reshape(-1, 2)
        return np.sort(np.exp(z[:2])), cal

    def profile(z):
        tau, cal = unpack(z)
        x = _design(_inputs(a, f, cal, dt, sy), dt, t, tau)[valid]
        coef = np.linalg.lstsq(x * weights[:, None], observed * weights, rcond=1e-10)[0]
        return coef, np.einsum('ij,j->i', x, coef) - observed

    def optimize(z):
        return least_squares(lambda p: weights * profile(p)[1], z, bounds=(lo, hi),
                             max_nfev=75, ftol=2e-7, xtol=2e-7, gtol=2e-7)

    # A rate-only pilot avoids pulling the affine calibration towards curve EMI.
    is_rate = np.broadcast_to(np.arange(len(t) + 1) == len(t), y.shape)[valid]
    if is_rate.sum() >= 2 * (9 + len(lo)):
        weights = is_rate.astype(float)
    starts = ([.28, 1.8], [.55, 3.9], [1.1, 7.])
    z = min((optimize(np.clip(np.r_[np.log(start), seed[free].ravel()], lo + 1e-6, hi - 1e-6))
             for start in starts), key=lambda run: run.cost).x
    for _ in range(6):
        r = profile(z)[1]
        scale = np.empty(2)
        for i, mask in enumerate((~is_rate, is_rate)):
            v = r[mask]
            scale[i] = max(.002, 1.4826 * np.median(abs(v - np.median(v)))) if v.size else .01
        scaled = r / (3 * scale[is_rate.astype(int)])
        weights = base / np.sqrt(1 + scaled**2)
        z = optimize(z).x
    tau, cal = unpack(z)
    return tau, profile(z)[0], cal


def solve_alpha_w02_batch_08(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    sy = float(dataset['yield_stress'])
    n = len(train['lab_id'])
    features = _features(train, test)
    seed, fixed = _calibration_seed(dataset['calibration_load_cells'], sy)
    tau, coef, cal = _regress(train, features[:n], dt, t, sy, seed, fixed)
    pred = _design(_inputs(test, features[n:], cal, dt, sy), dt, t, tau) @ coef
    # The feature term is linear in time. When it is partly imputed,
    # the measured onset rate identifies its unresolved slope without
    # changing the fitted hereditary or plastic coefficients.
    if 'instant_rate_obs' in test:
        measured = np.asarray(test['instant_rate_obs'])
        missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
        use = missing & np.isfinite(measured)
        delta = measured[use] - pred[use, -1]
        pred[use, :-1] += delta[:, None] * t
        pred[use, -1] = measured[use]
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], viscoplastic_weights=coef[7:],
                predicted_strain=pred[:, :-1] + np.asarray(test['strain_0'])[:, None],
                initial_rate_pred=pred[:, -1])
