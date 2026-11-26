"""Load-cell calibration and yield-gated hereditary creep for batch 04."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _checks(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    keep = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _supported_line(x, y):
    """Identify a positive affine map from a majority of well-spaced checks."""
    if x.size < 3 or np.ptp(x) < .1:
        return None
    i, j = np.triu_indices(x.size, 1)
    keep = np.abs(x[j] - x[i]) >= .1
    i, j = i[keep], j[keep]
    gain = (y[j] - y[i]) / (x[j] - x[i])
    bias = y[i] - gain * x[i]
    error = np.abs(y - gain[:, None] * x - bias[:, None])
    inliers = error <= .001
    support = inliers.sum(axis=1)
    eligible = (gain > 0) & (support >= max(3, x.size // 2 + 1))
    if not np.any(eligible):
        return None
    score = np.sum(np.where(inliers, error**2, 0.), axis=1)
    best = np.lexsort((score, -support, ~eligible))[0]
    use = inliers[best]
    return np.polyfit(x[use], y[use], 1) if np.ptp(x[use]) >= .1 else None


def _calibrate(panel):
    rows = [_checks(panel, lab) for lab in range(4)]
    maps = [_supported_line(*row) for row in rows]
    history = json.loads(_ARCHIVE.read_text()) if any(m is None for m in maps) and _ARCHIVE.is_file() else []
    for lab, (x, y) in enumerate(rows):
        if maps[lab] is not None:
            continue
        prior = np.array([1., 0.])
        for sweep in reversed(history):
            line = _supported_line(*_checks(sweep, lab))
            if line is not None:
                prior = line
                break
        delta = y - np.polyval(prior, x)
        close = np.abs(delta) <= .02
        if close.sum() >= 2 and np.ptp(x[close]) >= .5:
            maps[lab] = np.polyfit(x[close], y[close], 1)
        else:
            if np.any(close):
                prior[1] += np.median(delta[close])
            maps[lab] = prior
    return np.array(maps)


def _alignment(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for k in np.unique(lab):
        mask = lab == k
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        if panel.shape[1]:
            center = np.median(np.nanmean(panel, axis=0))
            scale = np.median(np.nanstd(panel, axis=0, ddof=0))
            raw[mask] = (raw[mask] - center) / (scale if scale > 0 else 1.)
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    finite = np.isfinite(raw)
    sums = np.zeros((cell.max() + 1, raw.shape[1]))
    count = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(finite, raw, 0.))
    np.add.at(count, cell, finite)
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    fill = means.sum(axis=0) / np.maximum(1, (count > 0).sum(axis=0))
    n = len(train['lab_id'])
    return np.where(finite, raw, fill), finite.all(axis=1), n


def _inputs(block, x, maps, dt, sy):
    gain, bias = maps[np.asarray(block['lab_id'], dtype=int)].T
    hist = np.asarray(block['past_stress_cmd']) * gain[:, None] + bias[:, None]
    hold = np.asarray(block['stress_hold_cmd']) * gain + bias
    ov = np.maximum(0., np.abs(hold) - sy)**2 * np.sign(hold)
    work = (np.maximum(0., np.abs(hist) - sy)**2 @ dt) * hold
    return hist, hold, np.column_stack((np.ones(len(x)), x)), ov, work


def _design(taus, inputs, dt, time):
    hist, hold, cov, ov, work = inputs
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-tail[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    jump = hold[:, None] - hist @ memory
    relax = -np.expm1(-time[:, None] / taus)
    a = np.concatenate((jump[:, None, :] * relax, cov[:, None, :] * time[None, :, None],
                        (ov[:, None] * time)[..., None], (work[:, None] * relax[:, 0])[..., None]), axis=2)
    b = np.column_stack((jump / taus, cov, ov, work / taus[0]))
    return a, b


def _fit(train, inputs, dt, time, plastic=True):
    eps = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    rate = np.asarray(train['instant_rate_obs'])
    y = np.r_[eps.ravel(), rate]
    valid = np.isfinite(y)
    y = y[valid]
    active = np.r_[np.ones(7, dtype=bool), [plastic and np.any(v) for v in inputs[-2:]]]
    channels = np.r_[np.repeat(np.arange(len(time)), len(eps)).reshape(len(time), -1).T.ravel(),
                     np.full(len(rate), len(time))][valid]
    unit = np.where(channels == len(time), np.sqrt(len(time)), 1.)

    def project(z, weight):
        t1, frac = z
        taus = np.array([t1, t1 + (24.99 - t1) * frac])
        a, b = _design(taus, inputs, dt, time)
        A = np.concatenate((a.reshape(-1, 9), b))[valid][:, active]
        w = np.linalg.lstsq(A * weight[:, None], y * weight, rcond=None)[0]
        return (A @ w - y), taus, w

    bounds = ([.0501, .0001], [24., .9999])

    def optimize(z, weight):
        return least_squares(lambda q: project(q, weight)[0] * weight, z,
                             bounds=bounds, max_nfev=75)

    starts = [(a, (b-a)/(24.99-a)) for a, b in ((.18, 1.4), (.45, 3.), (.9, 6.), (1.7, 12.))]
    best = min((optimize(z, unit) for z in starts), key=lambda fit: fit.cost)
    z = best.x
    weight = unit
    for _ in range(5):
        e = project(z, weight)[0]
        scale = np.empty_like(e)
        for k in np.unique(channels):
            use = channels == k
            r = e[use]
            scale[use] = max(1e-5, 1.4826 * np.median(np.abs(r - np.median(r))))
        weight = 1. / np.hypot(scale, e / 3.)
        z = optimize(z, weight).x
    _, taus, coeff = project(z, weight)
    w = np.zeros(9)
    w[active] = coeff
    return taus, w


def solve_alpha_w01_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    maps = _calibrate(dataset['calibration_load_cells'])
    x, complete, n = _alignment(train, test)
    keep = complete[:n] if complete[:n].sum() >= 12 else np.ones(n, dtype=bool)
    fit_train = {k: np.asarray(v)[keep] for k, v in train.items()}
    sy = dataset['yield_stress']
    taus, w = _fit(fit_train, _inputs(fit_train, x[:n][keep], maps, dt, sy), dt, time)
    a, b = _design(taus, _inputs(test, x[n:], maps, dt, sy), dt, time)
    rate = b @ w
    observed = np.asarray(test['instant_rate_obs'])
    correction = np.where(~complete[n:] & np.isfinite(observed), observed - rate, 0.)
    return {'estimated_taus': taus, 'hereditary_weights': w[:7], 'viscoplastic_weights': w[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + a @ w + correction[:, None] * time,
            'initial_rate_pred': rate + correction}
