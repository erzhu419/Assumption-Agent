"""Fit the calibrated two-branch hereditary creep law with yield activation."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _rows(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    use = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _consensus(x, y):
    i, j = np.triu_indices(len(x), 1)
    spread = x[j] - x[i]
    use = np.abs(spread) >= 0.12
    i, j, spread = i[use], j[use], spread[use]
    if not len(i):
        return None
    gain = (y[j] - y[i]) / spread
    bias = y[i] - gain * x[i]
    error = np.abs(y - gain[:, None] * x - bias[:, None])
    gate = 0.0015
    count = (error <= gate).sum(axis=1)
    loss = np.minimum((error / gate)**2, 1).sum(axis=1)
    keep = error[np.lexsort((loss, -count))[0]] <= gate
    if keep.sum() >= max(3, (len(x) + 1)//2) and np.ptp(x[keep]) >= 0.30:
        return _line(x[keep], y[keep])
    return None


def _historical_fits():
    if not _ARCHIVE.exists():
        return []
    archive = json.loads(_ARCHIVE.read_text())
    if isinstance(archive, dict):
        archive = next(v for v in archive.values() if isinstance(v, list))
    return [[_consensus(*_rows(p, lab)) for lab in range(4)] for p in reversed(archive)]


def _calibrate(panel):
    rows = [_rows(panel, lab) for lab in range(4)]
    fits = [_consensus(*r) for r in rows]
    if all(f is not None for f in fits):
        return np.array(fits)
    history = _historical_fits()
    for lab, fit in enumerate(fits.copy()):
        if fit is not None:
            continue
        x, y = rows[lab]
        peer = lab ^ 1
        xp, yp = rows[peer]
        options = []
        for old in history:
            if old[lab] is None:
                continue
            err = np.abs(y - old[lab][0]*x - old[lab][1])
            close = err < 0.004
            peer_err = (np.abs(yp - old[peer][0]*xp - old[peer][1])
                        if old[peer] is not None else np.full(len(xp), 0.004))
            peer_matches = int((peer_err < 0.004).sum())
            loss = float(np.minimum(err, 0.004).sum() + np.minimum(peer_err, 0.004).sum())
            options.append(((-int(close.sum()), -peer_matches, loss), old[lab], close))
        if options:
            score, prior, close = min(options, key=lambda item: item[0])
        else:
            score, prior, close = (0, 0, 0), np.array([1., 0.]), np.zeros(len(x), bool)
        if len(x) == 2 and np.ptp(x) >= 0.30 and not close.any():
            fits[lab] = _line(x, y)
        else:
            fits[lab] = prior
    return np.array(fits)
def _align(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.concatenate([train['ref_coupons'], test['ref_coupons']])
    raw = np.concatenate([train['coupon_features_raw'], test['coupon_features_raw']])
    aligned = np.empty_like(raw, dtype=float)
    for c in np.unique(labs):
        lab = labs == c
        r = refs[lab]
        r = r[:, np.isfinite(r).any(axis=0)]
        center = np.median(np.nanmean(r, axis=0))
        spread = np.median(np.nanstd(r, axis=0))
        x = (raw[lab] - center) / spread
        z = strata[lab]
        cells = np.unique(z)
        means = np.full((len(cells), x.shape[1]), np.nan)
        for i, s in enumerate(cells):
            block = x[z == s]
            count = np.isfinite(block).sum(axis=0)
            np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
        counts = np.isfinite(means).sum(axis=0)
        fallback = np.divide(np.nansum(means, axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0)
        for i, s in enumerate(cells):
            take = z == s
            x[take] = np.where(np.isfinite(x[take]), x[take], np.where(np.isfinite(means[i]), means[i], fallback))
        aligned[lab] = x
    return aligned[:len(train['lab_id'])], aligned[len(train['lab_id']):]


def _stress(split, affine):
    gain, bias = affine[np.asarray(split['lab_id'])].T
    return gain[:, None] * split['past_stress_cmd'] + bias[:, None], gain * split['stress_hold_cmd'] + bias


def _inputs(split, affine, aligned, dt, yield_stress):
    hist, hold = _stress(split, affine)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    excess = np.maximum(np.abs(hold) - yield_stress, 0.)
    ov_hold = np.sign(hold) * excess**2
    work = np.maximum(np.abs(hist) - yield_stress, 0.)**2 @ dt
    return hist, hold, drift, ov_hold, work*hold


def _design(taus, dt, times, data, active):
    hist, hold, drift, ov, work_hold = data
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    delta = hold[:, None] - hist @ memory
    relaxation = -np.expm1(-times[:, None] / taus)
    a = [delta[:, None, :]*relaxation, drift[:, None, :]*times[None, :, None]]
    b = [delta / taus, drift]
    if active[0]:
        a.append((ov[:, None]*times)[:, :, None])
        b.append(ov[:, None])
    if active[1]:
        a.append((work_hold[:, None]*relaxation[:, 0])[:, :, None])
        b.append((work_hold / taus[0])[:, None])
    return np.concatenate(a, axis=2), np.concatenate(b, axis=1)


def _fit(dt, times, data, curves, rate, active):
    n, k = curves.shape
    cols = 7 + sum(active)
    target = np.r_[curves.ravel(), rate]

    def unpack(p):
        return np.array([p[0], p[0] + p[1]*(24.99 - p[0])])

    def build(t):
        a, b = _design(t, dt, times, data, active)
        return np.vstack((a.reshape(-1, cols), b))

    bounds = ([0.05001, 1e-6], [24.98, 1.])
    def profile(p, scale):
        m = build(unpack(p))/scale[:, None]
        y = target/scale
        w = np.linalg.lstsq(m, y, rcond=None)[0]
        return np.einsum('ij,j->i', m, w) - y, w

    scale = np.ones_like(target)
    pilots = [least_squares(lambda p: profile(p, scale)[0], [a, (b-a)/(24.99-a)],
                           bounds=bounds, max_nfev=100)
              for a, b in ((0.35, 1.5), (0.65, 3.0), (1.4, 6.0))]
    pilot = min(pilots, key=lambda fit: fit.cost)
    def mad(r):
        return max(1e-7, 1.4826*np.median(np.abs(r-np.median(r))))
    scale = 2.385*np.r_[np.full(n*k, mad(pilot.fun[:n*k])), np.full(n, mad(pilot.fun[n*k:]))]
    _, w = profile(pilot.x, scale)
    def fun(p):
        return (np.einsum('ij,j->i', build(unpack(p)), p[2:]) - target)/scale
    fit = least_squares(fun, np.r_[pilot.x, w],
                        bounds=(np.r_[bounds[0], np.full(cols, -np.inf)],
                                np.r_[bounds[1], np.full(cols, np.inf)]),
                        x_scale='jac', loss='cauchy', max_nfev=120)
    return unpack(fit.x), fit.x[2:]


def solve_alpha_w04_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    affine = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    a = _inputs(train, affine, x_train, dt, dataset['yield_stress'])
    b = _inputs(test, affine, x_test, dt, dataset['yield_stress'])
    active = tuple(bool(np.any(c)) for c in a[-2:])
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(dt, times, a, curves, train['instant_rate_obs'], active)
    vp = np.zeros(2)
    vp[list(active)] = weights[7:]
    curves, rate = _design(taus, dt, times, b, active)
    strain = test['strain_0'][:, None] + curves @ weights
    initial_rate = rate @ weights
    # Cell means keep the aligned feature design defined, but cannot recover
    # a missing coupon's individual drift. Its measured rate identifies that
    # additive term once the fitted instantaneous mechanical response is fixed.
    observed_rate = np.asarray(test['instant_rate_obs'])
    use_rate = (np.isnan(test['coupon_features_raw']).any(axis=1)
                & np.isfinite(observed_rate))
    drift_residual = observed_rate[use_rate] - initial_rate[use_rate]
    strain[use_rate] += drift_residual[:, None] * times
    initial_rate[use_rate] += drift_residual
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'viscoplastic_weights': vp, 'predicted_strain': strain,
            'initial_rate_pred': initial_rate}
