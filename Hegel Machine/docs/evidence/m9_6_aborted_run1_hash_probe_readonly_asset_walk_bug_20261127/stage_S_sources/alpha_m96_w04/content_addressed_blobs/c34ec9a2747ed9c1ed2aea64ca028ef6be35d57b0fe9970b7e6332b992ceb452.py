import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sigma(r):
    return max(1e-8, 1.4826 * np.median(np.abs(r - np.median(r))))


def _archive_prior():
    gains, biases = np.ones(4), np.zeros(4)
    if not _ARCHIVE.is_file():
        return gains, biases
    with _ARCHIVE.open() as f:
        history = json.load(f)
    pairs = [[] for _ in range(4)]

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            g = value.get('stress_gains', value.get('stress_gain'))
            b = value.get('stress_biases', value.get('stress_bias'))
            if g is not None and b is not None:
                if 'lab_id' in value and np.ndim(g) == np.ndim(b) == 0:
                    c = int(value['lab_id'])
                    if 0 <= c < 4:
                        pairs[c].append((g, b))
                elif np.shape(g) == np.shape(b) == (4,):
                    for c in range(4):
                        pairs[c].append((g[c], b[c]))
            else:
                for item in value.values():
                    visit(item)

    visit(history)
    for c, entries in enumerate(pairs):
        if entries:
            gains[c], biases[c] = entries[-1]
    return gains, biases


def _calibration(checks):
    lab = np.asarray(checks['lab_id'])
    cmd, actual = (np.asarray(checks[k], float) for k in ('cmd_stress', 'true_stress'))
    gains, biases = _archive_prior()
    for c in range(4):
        keep = (lab == c) & np.isfinite(cmd) & np.isfinite(actual)
        x, y = cmd[keep], actual[keep]
        if not x.size:
            continue
        if x.size == 2 and np.ptp(x) >= 1e-6:
            gains[c] = (y[1] - y[0]) / (x[1] - x[0])
            biases[c] = np.mean(y - gains[c] * x)
            continue
        if x.size < 3 or np.ptp(x) < 1e-6:
            # A short set does not independently constrain an affine correction.
            delta = np.median(y - (gains[c] * x + biases[c]))
            if abs(delta) < 0.01:
                biases[c] += delta
            continue
        i, j = np.triu_indices(x.size, 1)
        dx = x[j] - x[i]
        spaced = np.abs(dx) >= 0.08 * np.ptp(x)
        i, j, dx = i[spaced], j[spaced], dx[spaced]
        slopes = (y[j] - y[i]) / dx
        intercepts = y[i] - slopes * x[i]
        slopes = np.r_[gains[c], slopes]
        intercepts = np.r_[biases[c], intercepts]
        res = np.abs(slopes[:, None] * x + intercepts[:, None] - y)
        h = x.size // 2 + 1
        quality = np.partition(res, h - 1, axis=1)[:, h - 1]
        best = np.argmin(quality)
        core = res[best] <= max(4 * quality[best], 3e-4)
        A = np.column_stack((x, np.ones(x.size)))
        if np.ptp(x[core]) >= 0.08 * np.ptp(x):
            gains[c], biases[c] = np.linalg.lstsq(A[core], y[core], rcond=None)[0]
        else:
            gains[c], biases[c] = slopes[best], intercepts[best]
    return gains, biases


def _aligned(train, test):
    lab, strata, x, ref = (np.concatenate((train[k], test[k]), axis=0) for k in
                            ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = x.astype(float, copy=True)
    for c in np.unique(lab):
        idx = lab == c
        panel = ref[idx]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        location = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[idx] = (x[idx] - location) / scale
    cells = np.unique(np.column_stack((lab, strata)), axis=0)
    means = []
    for c, s in cells:
        block = x[(lab == c) & (strata == s)]
        n = np.isfinite(block).sum(axis=0)
        means.append(np.divide(np.nansum(block, axis=0), n,
                               out=np.full(x.shape[1], np.nan), where=n > 0))
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isnan(x), fill, x)
    return np.split(x, [len(train['lab_id'])])


def _stress(split, gains, biases):
    g, b = gains[split['lab_id']], biases[split['lab_id']]
    return g[:, None] * split['past_stress_cmd'] + b[:, None], g * split['stress_hold_cmd'] + b


def _design(taus, hist, hold, drift, dt, times):
    # Sum the exact history recurrence backwards from the end of loading.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    inv_tau = 1 / np.asarray(taus)
    memory = -np.expm1(-dt[:, None] * inv_tau) * np.exp(-remaining[:, None] * inv_tau)
    gaps = hold[:, None] - hist @ memory
    relax = -np.expm1(-times[:, None] * inv_tau)
    curve = np.concatenate((gaps[:, None, :] * relax[None, :, :],
                            drift[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gaps * inv_tau, drift))
    return curve, rate


def _fit(hist, hold, drift, dt, times, curve_y, rate_y):
    # The first two coordinates encode tau1 and the fraction of the remaining
    # interval occupied by tau2. The seven weights enter linearly.
    def taus(p):
        return np.array([p[0], p[0] + (8 - p[0]) * p[1]])

    def matrix(p):
        c, r = _design(taus(p), hist, hold, drift, dt, times)
        return np.vstack((c.reshape(-1, 7), r))

    y = np.r_[curve_y.ravel(), rate_y]
    nc = curve_y.size
    bounds = (np.r_[0.2, 1e-6, [-np.inf] * 7], np.r_[7.999, 1, [np.inf] * 7])
    sig = np.ones_like(y)

    def residual(p):
        return (matrix(p[:2]) @ p[2:] - y) / sig

    starts = []
    for a, b in ((0.25, 1.2), (0.45, 3), (0.8, 6), (1.5, 4), (2.8, 7.5)):
        t = np.array([a, (b - a) / (8 - a)])
        A = matrix(t)
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        starts.append(np.r_[t, w])
    warm = min((least_squares(residual, p, bounds=bounds, loss='soft_l1',
                              f_scale=0.08, max_nfev=75) for p in starts), key=lambda r: r.cost)
    err = residual(warm.x)
    sig[:nc] = 2.0 * _sigma(err[:nc])
    sig[nc:] = 2.5 * _sigma(err[nc:])
    fit = least_squares(residual, warm.x, bounds=bounds, loss='cauchy', max_nfev=100)
    return taus(fit.x), fit.x[2:]


def solve_alpha_m96_w04_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    gains, biases = _calibration(dataset['calibration_load_cells'])
    xtr, xte = _aligned(train, test)
    htr, str_ = _stress(train, gains, biases)
    hte, ste = _stress(test, gains, biases)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    drift = lambda x: np.column_stack((np.ones(len(x)), x))
    taus, weights = _fit(htr, str_, drift(xtr), dt, times,
                         train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    curves, rates = _design(taus, hte, ste, drift(xte), dt, times)
    strain = test['strain_0'][:, None] + curves @ weights
    rate = rates @ weights
    # On incomplete coupons, use the measured onset to recover the total
    # drift that the filled features cannot identify individually.
    onset = np.asarray(test['instant_rate_obs'])
    use_onset = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(onset)
    strain[use_onset] += (onset[use_onset] - rate[use_onset])[:, None] * times
    rate[use_onset] = onset[use_onset]
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': strain, 'initial_rate_pred': rate}
