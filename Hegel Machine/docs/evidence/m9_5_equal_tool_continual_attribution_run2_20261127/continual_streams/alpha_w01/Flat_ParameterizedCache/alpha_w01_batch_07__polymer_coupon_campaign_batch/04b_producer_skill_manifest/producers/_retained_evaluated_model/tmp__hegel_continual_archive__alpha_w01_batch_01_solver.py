import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _lab_checks(checks, lab):
    labs = np.asarray(checks['lab_id'])
    x = np.asarray(checks['cmd_stress'], dtype=float)
    y = np.asarray(checks['true_stress'], dtype=float)
    good = (labs == lab) & np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _affine(x, y, prior):
    """Return the best local affine fit and whether the checks identify it."""
    if len(x) < 2 or np.ptp(x) < .05:
        return np.array([prior[0], np.median(y - prior[0]*x) if len(x) else prior[1]]), False
    a, b = np.triu_indices(len(x), k=1)
    sep = np.abs(x[a] - x[b]) >= max(.05, .05 * np.ptp(x))
    a, b = a[sep], b[sep]
    gain = (y[b] - y[a]) / (x[b] - x[a])
    bias = y[a] - gain * x[a]
    residual = np.abs(y - gain[:, None] * x - bias[:, None])
    # Trim gross EMI. If only two points agree, prefer the line nearest the prior.
    h = max(2, len(x) // 2 + 1)
    score = (np.sort(residual, axis=1)[:, :h]**2).sum(axis=1)
    support = (residual <= .001).sum(axis=1)
    if support.max() < h or h == 2:
        candidates = np.flatnonzero(support == support.max())
        delta = (gain[candidates] - prior[0]) * max(1., np.ptp(x))
        best = candidates[np.argmin(delta**2 + (bias[candidates] - prior[1])**2)]
    else:
        best = np.argmin(score)
    err = residual[best]
    clean = err <= max(.001, min(.005, 5 * np.median(err)))
    A = np.column_stack((x[clean], np.ones(clean.sum())))
    fit = np.linalg.lstsq(A, y[clean], rcond=None)[0]
    identified = clean.sum() >= max(h, 3) and np.ptp(x[clean]) >= .1 and fit[0] > 0
    return fit, identified


def _load_cells(checks):
    nominal = np.array([1., 0.])
    data = [_lab_checks(checks, lab) for lab in range(4)]
    local = [_affine(x, y, nominal) for x, y in data]
    if all(ok for _, ok in local):
        return np.array([fit for fit, _ in local])
    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = json.loads(path.read_text()) if path.is_file() else []
    for lab, (fit, ok) in enumerate(local):
        if ok:
            continue
        x, y = data[lab]
        for panel in reversed(history):
            prior, identified = _affine(*_lab_checks(panel, lab), nominal)
            if identified:
                offset = y - (prior[0] * x + prior[1])
                # Two separated current checks can identify a changed affine map.
                # Keep the archive when only one check disagrees: that may be EMI.
                if len(x) == 2 and np.ptp(x) >= .1 and fit[0] > 0 and np.all(np.abs(offset) > .001):
                    local[lab] = (fit, True)
                    break
                close = offset[np.abs(offset) <= .02]
                if close.size:
                    prior[1] += np.median(close)
                local[lab] = (prior, True)
                break
        else:
            local[lab] = (fit if len(x) == 2 else nominal.copy(), False)
    return np.array([fit for fit, _ in local])


def _alignment(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw']))
    for lab in np.unique(labs):
        m = labs == lab
        panel = refs[m]
        cols = np.sum(np.isfinite(panel), axis=0) >= 2
        if np.any(cols):
            center = np.median(np.nanmean(panel[:, cols], axis=0))
            scale = np.median(np.nanstd(panel[:, cols], axis=0))
            raw[m] = (raw[m] - center) / (scale if scale > 0 else 1.)
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    finite = np.isfinite(raw)
    counts = np.zeros((int(cell.max()) + 1, raw.shape[1]))
    sums = np.zeros_like(counts)
    np.add.at(counts, cell, finite)
    np.add.at(sums, cell, np.where(finite, raw, 0.))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    available = (counts > 0).sum(axis=0)
    fill = means.sum(axis=0) / np.maximum(1, available)
    x = np.where(finite, raw, fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _stress(block, maps):
    gain, bias = maps[np.asarray(block['lab_id'])].T
    return (block['past_stress_cmd'] * gain[:, None] + bias[:, None],
            block['stress_hold_cmd'] * gain + bias)


def _jump(taus, hist, hold, initial, dt):
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    weights = np.exp(-tail[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    state = initial * np.exp(-dt.sum() / taus) + hist @ weights
    return hold[:, None] - state


def _fit(train, x, maps, dt, time):
    hist, hold = _stress(train, maps)
    initial = train['initial_memory_states']
    cov = np.column_stack((np.ones(len(x)), x))
    eps = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    y = np.r_[eps.ravel(), rate]
    drift = (time[None, :, None] * cov[:, None, :]).reshape(-1, 5)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]

    def evaluate(p, jac=False):
        t = p[:2]
        end = tail + dt
        early, late = np.exp(-tail[:, None]/t), np.exp(-end[:, None]/t)
        seed = initial * np.exp(-dt.sum()/t)
        jump = hold[:, None] - hist @ (early - late) - seed
        decay = np.exp(-time[:, None]/t)
        phi = -np.expm1(-time[:, None]/t)
        A = np.column_stack((np.vstack(((jump[:, None, :] * phi).reshape(-1, 2), jump/t)), np.vstack((drift, cov))))
        if not jac:
            return A
        dj = -(hist @ (tail[:, None]*early - end[:, None]*late) + dt.sum()*seed) / t**2
        dphi = -time[:, None]*decay/t**2
        J = np.vstack(((dj[:, None, :]*phi + jump[:, None, :]*dphi).reshape(-1, 2), dj/t - jump/t**2)) * p[2:4]
        return np.column_stack((J, A))

    lo = np.r_[.051, .051, np.full(7, -np.inf)]
    hi = np.r_[24.99, 24.99, np.full(7, np.inf)]
    def residual(p):
        return np.einsum('ij,j->i', evaluate(p), p[2:]) - y

    fits = []
    for start in ((.45, 3.2), (.9, 6.), (1.6, 11.)):
        p = np.r_[start, np.zeros(7)]
        p[2:] = np.linalg.lstsq(evaluate(p), y, rcond=None)[0]
        fits.append(least_squares(residual, p, jac=lambda p: evaluate(p, True), bounds=(lo, hi),
                                  loss='soft_l1', f_scale=.035, max_nfev=70))
    p = min(fits, key=lambda f: f.cost).x
    err = residual(p)
    scales = [max(1e-5, 3 * 1.4826 * np.median(np.abs(e - np.median(e)))) for e in (err[:eps.size], err[eps.size:])]
    scale = np.r_[np.full(eps.size, scales[0]), np.full(rate.size, scales[1])]
    p = least_squares(lambda p: residual(p)/scale, p, jac=lambda p: evaluate(p, True)/scale[:, None],
                      bounds=(lo, hi), loss='cauchy', max_nfev=80).x
    order = np.argsort(p[:2])
    return p[:2][order], p[2:4][order], p[4:]


def _crossings(strain, initial, critical, time):
    # The hazard is the first crossing of the sampled curve, anchored at t=0.
    values = np.column_stack((initial, strain))
    times = np.r_[0., time]
    hits = values >= critical[:, None]
    first = hits.argmax(axis=1)
    out = np.full(len(initial), time[-1])
    out[hits[:, 0]] = 0.
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    below, above = values[rows, j - 1], values[rows, j]
    out[rows] = times[j - 1] + (times[j] - times[j - 1]) * (critical[rows] - below) / (above - below)
    return out


def solve_alpha_w01_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    maps = _load_cells(dataset['calibration_load_cells'])
    x, xt = _alignment(train, test)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    if complete.sum() >= 12:
        train = {key: value[complete] for key, value in train.items()}
        x = x[complete]
    taus, w, drift_coef = _fit(train, x, maps, dt, time)
    hist, hold = _stress(test, maps)
    amplitude = _jump(taus, hist, hold, test['initial_memory_states'], dt)*w
    drift = np.column_stack((np.ones(len(xt)), xt)) @ drift_coef
    # After subtracting the hereditary rate, onset measures the linear drift.
    rate = np.asarray(test['instant_rate_obs'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    informed = missing & np.isfinite(rate)
    drift[informed] = rate[informed] - amplitude[informed] @ (1 / taus)
    initial = test['strain_0']
    strain = initial[:, None] + amplitude @ (-np.expm1(-time[:, None]/taus)).T + drift[:, None]*time
    return {'estimated_taus': taus, 'predicted_strain': strain,
            'secant_compliance': (strain[:, -1] - initial) / np.maximum(np.abs(hold), .5),
            'hazard_crossing_times': _crossings(strain, initial, test['critical_strain'], time)}
