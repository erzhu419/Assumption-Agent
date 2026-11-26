import json
import pickle
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


def _check_panels(value):
    if isinstance(value, dict):
        if {'lab_id', 'cmd_stress', 'true_stress'} <= value.keys():
            yield value
        for key in ('dataset', 'calibration_load_cells'):
            if key in value:
                yield from _check_panels(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _check_panels(item)


def _archived_maps(labs, root):
    """Find the latest identifiable historical calibration per requested lab."""
    files = [p for p in Path(root).rglob('*')
             if p.suffix.lower() in ('.json', '.pkl', '.pickle', '.npz') and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    found = {}
    for p in files:
        if len(found) == len(labs):
            break
        try:
            ext = p.suffix.lower()
            if ext == '.json':
                value = json.loads(p.read_text())
            elif ext == '.npz':
                with np.load(p, allow_pickle=False) as arrays:
                    if not {'lab_id', 'cmd_stress', 'true_stress'} <= set(arrays.files):
                        continue
                    value = {k: arrays[k] for k in ('lab_id', 'cmd_stress', 'true_stress')}
            else:
                with p.open('rb') as handle:
                    value = pickle.load(handle)
            for panel in reversed(list(_check_panels(value))):
                for lab in labs - found.keys():
                    fit, reliable = _affine(*_lab_checks(panel, lab), (1., 0.))
                    if reliable:
                        found[lab] = fit
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue
    return found


def _load_cells(checks, archive_root='/tmp/hegel_continual_archive'):
    nominal = np.array([1., 0.])
    data = [_lab_checks(checks, lab) for lab in range(4)]
    local = [_affine(x, y, nominal) for x, y in data]
    sparse = {lab for lab, (_, identified) in enumerate(local) if not identified}
    archive = _archived_maps(sparse, archive_root) if sparse else {}
    for lab, prior in archive.items():
        x, y = data[lab]
        # A spot check can update the offset but cannot identify a gain.
        offset = y - (prior[0] * x + prior[1])
        close = offset[np.abs(offset) <= .02]
        gain, bias = prior
        if len(close):
            bias += np.median(close)
        local[lab] = (np.array([gain, bias]), True)
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


def _design(taus, hist, hold, cov, dt, time, initial):
    # Integrate each constant-stress interval, then age its memory to hold onset.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    q = dt[:, None] / taus
    memory = np.exp(-tail[:, None] / taus) * -np.expm1(-q)
    jump = hold[:, None] - hist @ memory - initial * np.exp(-np.sum(dt) / taus)
    drift = cov[:, None, :] * time[None, :, None]
    relax = jump[:, None, :] * -np.expm1(-time[:, None] / taus)
    return np.concatenate((relax, drift), axis=2), np.column_stack((jump / taus, cov))


def _mad(e):
    return max(1e-8, 1.4826 * np.median(np.abs(e - np.median(e))))


def _fit(train, x, maps, dt, time):
    hist, hold = _stress(train, maps)
    cov = np.column_stack((np.ones(len(x)), x))
    eps = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    y = np.r_[eps.ravel(), rate]
    ncurve = eps.size

    def design(taus):
        a, b = _design(taus, hist, hold, cov, dt, time, train['initial_memory_states'])
        return np.vstack((a.reshape(-1, 7), b))

    def unpack(p):
        return np.array([p[0], p[0] + (8. - p[0]) * p[1]])

    def residual(p):
        return design(unpack(p)) @ p[2:] - y

    bounds = (np.r_[.2, 1e-5, np.full(7, -np.inf)],
              np.r_[7.999, 1., np.full(7, np.inf)])
    starts = []
    for t1, t2 in ((.27, 1.2), (.43, 3.), (.65, 5.5), (1., 2.), (1.3, 6.), (2.5, 7.5)):
        A = design(np.array([t1, t2]))
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        starts.append(np.r_[t1, (t2 - t1) / (8. - t1), w])
    prelim = [least_squares(residual, p, bounds=bounds, loss='soft_l1',
                             f_scale=.05, max_nfev=85) for p in starts]
    best = min(prelim, key=lambda opt: opt.cost)
    p = best.x
    for _ in range(2):
        e = residual(p)
        scale = np.r_[np.full(ncurve, 2.5 * _mad(e[:ncurve])),
                      np.full(len(rate), 2.5 * _mad(e[ncurve:]))]
        opt = least_squares(lambda q: residual(q) / scale, p, bounds=bounds,
                            loss='cauchy', max_nfev=100)
        p = opt.x
    return unpack(p), p[2:]


