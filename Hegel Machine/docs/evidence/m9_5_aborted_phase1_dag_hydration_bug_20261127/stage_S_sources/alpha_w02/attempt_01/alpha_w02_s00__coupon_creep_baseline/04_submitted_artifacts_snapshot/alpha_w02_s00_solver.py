"""Robust calibration and two-branch sub-yield creep identification."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _affine(x, y, prior):
    """Fit an affine correction using consensus over pairs of spot checks."""
    if not len(x):
        return prior
    # Fit corrections rather than the dominant identity component.
    dy = y - x
    a, b = prior[0] - 1, prior[1]
    if len(x) == 1:
        return a + 1, dy[0] - a*x[0]
    i, j = np.triu_indices(len(x), 1)
    keep = np.abs(x[i] - x[j]) >= max(0.04, 0.12*np.ptp(x))
    i, j = i[keep], j[keep]
    if not len(i):
        return a + 1, np.median(dy - a*x)
    slopes = np.r_[a, (dy[j] - dy[i])/(x[j] - x[i])]
    offsets = np.r_[np.median(dy - a*x), dy[i] - slopes[1:]*x[i]]
    err = abs(dy - slopes[:, None]*x - offsets[:, None])
    consensus = err < 0.0007
    score = consensus.sum(axis=1)
    loss = np.minimum(err/0.0007, 1).sum(axis=1)
    win = np.lexsort((np.abs(slopes-a), loss, -score))[0]
    m = consensus[win]
    if m.sum() < 2 or np.ptp(x[m]) < 0.04:
        return a + 1, np.median(dy - a*x)
    if len(x) <= 3 and m.sum() < len(x):
        # Too few checks to identify a new slope and reject a glitch independently.
        return a + 1, np.median(dy - a*x)
    u = x[m] - x[m].mean()
    slope = np.dot(u, dy[m])/np.dot(u, u)
    return 1 + slope, dy[m].mean() - slope*x[m].mean()


def _checks(table, lab):
    ids, x, y = map(np.asarray, (table['lab_id'], table['cmd_stress'], table['true_stress']))
    m = (ids == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _reliable(x, y, line):
    m = abs(y - np.polyval(line, x)) < 0.0007
    return m.sum() >= max(3, len(x)//2 + 1) and np.ptp(x[m]) >= 0.12


def _archive_calibration(labs):
    """Recover per-lab priors from archived JSON load-cell tables, if mounted."""
    prior = {}
    for path in sorted(Path('/tmp/hegel_continual_archive').rglob('*.json')):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        entry = entry.get('dataset', entry)
        if not isinstance(entry, dict):
            continue
        table = entry.get('calibration_load_cells', entry)
        if not isinstance(table, dict) or not {'lab_id', 'cmd_stress', 'true_stress'} <= table.keys():
            continue
        for l in labs:
            x, y = _checks(table, l)
            fit = _affine(x, y, (1., 0.))
            if _reliable(x, y, fit):
                prior.setdefault(l, []).append(fit)
    return {l: np.median(fits, axis=0) for l, fits in prior.items()}


def _calibration(d):
    c = d['calibration_load_cells']
    return np.array([_affine(*_checks(c, l), (1., 0.)) for l in range(4)])


def _align(d):
    tr, te = d['train'], d['test']
    features, ref, lab, stratum = [np.concatenate((tr[k], te[k])) for k in
        ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    x = features.astype(float, copy=True)
    observed = np.isfinite(x)
    for l in np.unique(lab):
        m = lab == l
        r = ref[m]
        center = np.nanmedian(np.nanmean(r, axis=0))
        scale = np.nanmedian(np.nanstd(r, axis=0, ddof=0))
        x[m] = (x[m] - center)/scale
    _, cells = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    counts = np.bincount(cells)
    # Average available values within each cell, then weight cells equally.
    cell_means = np.full((len(counts), x.shape[1]), np.nan)
    for cell in range(len(counts)):
        v = x[cells == cell]
        n = np.isfinite(v).sum(axis=0)
        np.divide(np.nansum(v, axis=0), n, out=cell_means[cell], where=n>0)
    x = np.where(observed, x, np.nanmean(cell_means, axis=0))
    n = len(tr['lab_id'])
    return x[:n], x[n:], observed[:n].all(axis=1)


def _state_inputs(split, calibration):
    a, b = calibration[np.asarray(split['lab_id'])].T
    hist = a[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = a*split['stress_hold_cmd'] + b
    return hist, hold


def _basis(tau, hist, hold, dt, times, x):
    dt = np.asarray(dt)
    after = np.cumsum(dt[::-1])[::-1] - dt
    coeff = -np.expm1(-dt[:, None]/tau)*np.exp(-after[:, None]/tau)
    diff = hold[:, None] - hist @ coeff
    t = np.asarray(times)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((diff[:, None, :] * (-np.expm1(-t[:, None]/tau))[None, :, :],
                           drift[:, None, :]*t[None, :, None]), axis=2)
    rate = np.column_stack((diff/tau, drift))
    return curve, rate


def _noise(r):
    return max(1e-9, 1.4826*np.median(np.abs(r - np.median(r))))


def _fit(hist, hold, dt, t, x, curve, rate):
    k = len(t)
    yc = curve.ravel()
    mc, mr = np.isfinite(yc), np.isfinite(rate)
    yc, yr = yc[mc], rate[mr]
    y = np.r_[yc, yr]
    n = len(yc)

    def design(tau):
        bc, br = _basis(tau, hist, hold, dt, t, x)
        return np.vstack((bc.reshape(-1, 7)[mc], br[mr]))

    def search(scale, starts):
        def residual(p):
            return (design(p[:2]) @ p[2:] - y)/scale
        best = None
        lo = np.r_[0.2, 0.2, np.full(7, -np.inf)]
        hi = np.r_[8., 8., np.full(7, np.inf)]
        for p in starts:
            fit = least_squares(residual, p, bounds=(lo, hi), loss='cauchy',
                                x_scale='jac', max_nfev=110, ftol=2e-9)
            if best is None or fit.cost < best.cost:
                best = fit
        return best.x

    starts = []
    for tau in ([0.35, 2.4], [0.7, 4.0], [1.3, 6.5]):
        b = design(np.asarray(tau))
        w = np.linalg.lstsq(b, y, rcond=None)[0]
        starts.append(np.r_[tau, w])
    r0 = design(starts[1][:2]) @ starts[1][2:] - y
    def scales(r, multiple):
        sc = _noise(r[:n])
        sr = _noise(r[n:]) if len(yr) else sc
        return multiple*np.r_[np.full(n, sc), np.full(len(yr), sr)]
    p = search(scales(r0, 1.7), starts)
    r = design(p[:2]) @ p[2:] - y
    p = search(scales(r, 2.0), [p])
    if p[0] > p[1]:
        p[:4] = p[[1, 0, 3, 2]]
    return p[:2], p[2:]


def _sparse_calibration(d, cal, x, complete):
    tr = d['train']
    labs, dt, t = tr['lab_id'], d['past_dt'], d['eval_times']
    checks = [_checks(d['calibration_load_cells'], l) for l in range(4)]
    # Three apparent inliers can still be a bad line when two commands
    # cluster together and a corrupted third check supplies the leverage.
    trusted = np.array([len(q) >= 4 and _reliable(q, v, c)
                        for (q, v), c in zip(checks, cal)])
    if not trusted.any():
        # If the entire batch is sparse, retain identifiable three-point fits.
        trusted = np.array([_reliable(q, v, c) for (q, v), c in zip(checks, cal)])
    sparse = np.flatnonzero(~trusted)
    if not len(sparse):
        return cal
    priors = _archive_calibration(sparse)
    use = trusted[labs] & complete
    y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
    r = tr['instant_rate_obs']
    if use.sum() >= 12:
        h, p = _state_inputs(tr, cal)
        tau, w = _fit(h[use], p[use], dt, t, x[use], y[use], r[use])
        bc, br = _basis(tau, h[use], p[use], dt, t, x[use])
        sc = _noise((bc @ w - y[use])[np.isfinite(y[use])])
        sr = _noise((br @ w - r[use])[np.isfinite(r[use])])
    for l in sparse:
        q, v = checks[l]
        gain, bias = priors.get(l, (1., 0.))
        j = (labs == l) & complete
        if l in priors:
            # A lone conflicting check cannot distinguish EMI from calibration drift.
            agrees = np.abs(v - (gain*q + bias)) < 0.003
            if agrees.sum() >= 2 and np.ptp(q[agrees]) >= 0.12:
                gain, bias = _affine(q[agrees], v[agrees], (gain, bias))
            elif agrees.any():
                bias = np.median(v[agrees] - gain*q[agrees])
            cal[l] = gain, bias
            continue
        if len(q) == 2 and np.ptp(q) >= 0.12 and (use.sum() < 12 or j.sum() < 8):
            # With no independent gain estimate, retain the identified two-point line.
            cal[l] = np.polyfit(q, v, 1)
            continue
        if use.sum() >= 12 and j.sum() >= 8 and len(q):
            h, p = tr['past_stress_cmd'][j], tr['stress_hold_cmd'][j]
            bc, br = _basis(tau, h, p, dt, t, x[j])
            ec, er = _basis(tau, np.ones_like(h), np.ones_like(p), dt, t, x[j])
            drift = w[2] + x[j] @ w[3:]
            ac, ar = bc[:, :, :2] @ w[:2], br[:, :2] @ w[:2]
            ec, er = ec[:, :, :2] @ w[:2], er[:, :2] @ w[:2]
            # A constant stress offset contributes the same hereditary curve
            # to every coupon in this lab. Profile it separately from gain.
            C = np.column_stack((ac.ravel(), ec.ravel(), np.tile(t, j.sum()),
                                 (drift[:, None]*t).ravel()))
            R = np.column_stack((ar, er, np.ones(j.sum()), drift))
            A = np.vstack((C/sc, R/sr))
            target = np.r_[((y[j]-drift[:, None]*t)/sc).ravel(), (r[j]-drift)/sr]
            m = np.isfinite(target)
            fit = least_squares(lambda z: A[m] @ z-target[m], [gain, bias, 0., 0.],
                                loss='cauchy', f_scale=2.)
            gain = fit.x[0]
        if len(q) >= 2:
            i, k = np.triu_indices(len(q), 1)
            span = q[k] - q[i]
            # Confirm a measured slope with the independent gain estimate.
            # For nearly coincident checks the response estimate is better conditioned.
            wide = abs(span) >= 0.12
            i, k, span = i[wide], k[wide], span[wide]
            if len(span):
                slopes = (v[k]-v[i])/span
                agreement = abs(slopes-gain)
                best = np.argmin(agreement)
                if agreement[best] < 0.015:
                    a, b = i[best], k[best]
                    cal[l] = slopes[best], v[a]-slopes[best]*q[a]
                    continue
        if len(q):
            # For three checks this preserves a clean majority even when
            # no pair spans enough command range to estimate a precise slope.
            # For two incompatible checks, the midpoint bounds the ambiguity.
            bias = np.median(v-gain*q)
        cal[l] = gain, bias
    return cal


def solve_alpha_w02_s00(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    cal = _calibration(dataset)
    x, xt, complete = _align(dataset)
    cal = _sparse_calibration(dataset, cal, x, complete)
    hist, hold = _state_inputs(tr, cal)
    hist_t, hold_t = _state_inputs(te, cal)
    # Do not fit material parameters to proxies for missing coupon covariates.
    taus, weights = _fit(hist[complete], hold[complete], dt, t, x[complete],
                        (tr['strain_curves_obs'] - tr['strain_0'][:, None])[complete],
                        tr['instant_rate_obs'][complete])
    b, v = _basis(taus, hist_t, hold_t, dt, t, xt)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': te['strain_0'][:, None] + b @ weights,
            'initial_rate_pred': v @ weights}
