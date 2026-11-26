import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _checks(checks, lab):
    labs, x, y = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    m = (labs == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _consensus(x, y):
    # An EMI spike cannot define the line when three or more checks agree.
    i, j = np.triu_indices(len(x), 1)
    good = np.abs(x[i]-x[j]) > 1e-8
    i, j = i[good], j[good]
    if not len(i):
        return []
    g = (y[j]-y[i])/(x[j]-x[i])
    r = (y[None, :]-y[i, None])-g[:, None]*(x[None, :]-x[i, None])
    keep = np.abs(r) <= .001
    masks = np.unique(keep[keep.sum(axis=1) == keep.sum(axis=1).max()], axis=0)
    fits = []
    for m in masks:
        fit = np.linalg.lstsq(np.c_[x[m], np.ones(m.sum())], y[m], rcond=None)[0]
        fits.append((fit, m.sum(), np.ptp(x[m])))
    return fits


def _archive_checks(root):
    # Archive records may be standalone check tables or prior batch datasets.
    def tables(record):
        if isinstance(record, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= record.keys():
                yield record
            elif 'calibration_load_cells' in record:
                yield record['calibration_load_cells']
            elif 'dataset' in record:
                yield from tables(record['dataset'])
        elif isinstance(record, (list, tuple)):
            for item in reversed(record):
                yield from tables(item)

    for path in sorted(root.rglob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        try:
            if path.suffix == '.json':
                with path.open() as f:
                    yield from tables(json.load(f))
            elif path.suffix in ('.pkl', '.pickle'):
                with path.open('rb') as f:
                    yield from tables(pickle.load(f))
            elif path.suffix == '.npz':
                with np.load(path) as data:
                    if {'lab_id', 'cmd_stress', 'true_stress'} <= set(data.files):
                        yield {k: data[k] for k in ('lab_id', 'cmd_stress', 'true_stress')}
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue


def _with_prior(x, y, fits, prior):
    gain, bias = prior
    if not len(x):
        return prior
    # Fix the slope when a narrow or contaminated spot-check panel cannot
    # estimate it. A shared offset across checks identifies bias drift.
    offsets = y - gain*x - bias
    center = np.median(offsets)
    agreeing = np.abs(offsets-center) <= .004
    if agreeing.sum() > len(x)//2 and (len(x) > 1 or abs(center) < .10):
        bias += np.median(offsets[agreeing])
    else:
        i = np.argmin(np.abs(offsets))
        if abs(offsets[i]) < .10:
            bias += offsets[i]
    for fit, count, span in fits:
        if (count >= 3 and span >= .4 and abs(fit[0]-gain) <= .01
                and abs(fit[1]-bias) <= .01):
            return fit
    return gain, bias


def _calibrate(checks, root=Path('/tmp/hegel_continual_archive')):
    result = np.tile([1., 0.], (4, 1))
    unresolved = {}
    for lab in range(4):
        x, y = _checks(checks, lab)
        fits = _consensus(x, y)
        if len(x) >= 4 and len(fits) == 1 and fits[0][1] >= 3 and fits[0][2] >= .1:
            result[lab] = fits[0][0]
        else:
            unresolved[lab] = x, y, fits
    if unresolved:
        for table in _archive_checks(root):
            for lab in list(unresolved):
                xa, ya = _checks(table, lab)
                prior = _consensus(xa, ya)
                if len(prior) != 1 or prior[0][1] < 3 or prior[0][2] < .1:
                    continue
                x, y, fits = unresolved.pop(lab)
                result[lab] = _with_prior(x, y, fits, prior[0][0])
            if not unresolved:
                break
    for lab, (x, y, fits) in unresolved.items():
        supported = [f for f in fits if f[2] >= .1]
        if supported:
            result[lab] = min(supported, key=lambda f: np.linalg.norm((f[0]-[1., 0.])*[2., 1.]))[0]
        elif len(x):
            result[lab, 1] = np.median(y-x)
    return result


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


def _design(calibrated, feat, dt, t, logtaus, derivative=False):
    past, hold = calibrated
    tau = np.exp(logtaus)
    remaining = np.cumsum(dt[::-1])[::-1]
    end = np.r_[remaining[1:], 0.]
    a = end[:, None]/tau
    b = dt[:, None]/tau
    ea, eb = np.exp(-a), np.exp(-b)
    mix = ea * -np.expm1(-b)
    d = hold[:, None] - past @ mix
    u = t[:, None]/tau
    e = np.exp(-u)
    f = -np.expm1(-u)
    n = len(past)
    curve = np.concatenate((d[:, None, :]*f[None, :, :], np.broadcast_to(feat[:, None, :]*t[None, :, None], (n, len(t), 5))), axis=2)
    rate = np.c_[d/tau, feat]
    A = np.concatenate((curve, rate[:, None, :]), axis=1)
    if not derivative:
        return A
    dd = -past @ (a*mix - ea*b*eb)
    dq = dd[:, None, :]*f[None, :, :] - d[:, None, :]*(u*e)[None, :, :]
    return A, np.concatenate((dq, ((dd-d)/tau)[:, None, :]), axis=1)


def _fit(calibrated, feat, dt, t, y):
    valid = np.isfinite(y)
    yy = y[valid]
    # Keep the pilot well scaled before separate noise estimates are available.
    init_scale = np.ones(y.shape)
    init_scale[:, -1] = 1.5
    s = init_scale[valid]

    def linear(A, scale, rounds):
        B, z = A/scale[:, None], yy/scale
        w = np.ones(len(z))
        for _ in range(rounds):
            coef = np.linalg.lstsq(B*w[:, None], z*w, rcond=None)[0]
            r = B@coef-z
            robust = max(0.001, 1.4826*np.median(np.abs(r-np.median(r))))
            w = 1./np.sqrt(1.+(r/(2*robust))**2)
        return coef, r

    def evaluate(p, jac=False):
        if jac:
            A, G = _design(calibrated, feat, dt, t, p[:2], True)
            return np.c_[G[valid]*p[2:4], A[valid]] / s[:, None]
        A = _design(calibrated, feat, dt, t, p[:2])
        return (A[valid]@p[2:] - yy)/s

    bounds = (np.r_[np.full(2, np.log(.20)), np.full(7, -np.inf)],
              np.r_[np.full(2, np.log(8.)), np.full(7, np.inf)])
    guesses = []
    grid = np.geomspace(.22, 7.8, 9)
    for i, a in enumerate(grid[:-1]):
        for b in grid[i+1:]:
            logs = np.log([a, b])
            A = _design(calibrated, feat, dt, t, logs)[valid]
            coef, r = linear(A, s, 4)
            guesses.append((np.mean(np.log1p((r/.04)**2)), np.r_[logs, coef]))
    guesses.sort(key=lambda v: v[0])
    def optimize(start, fscale):
        return least_squares(evaluate, start, jac=lambda p: evaluate(p, True), bounds=bounds,
                             loss='cauchy', f_scale=fscale, x_scale='jac', max_nfev=160)

    pilots = [optimize(start, .04) for _, start in guesses[:4]]
    p = min(pilots, key=lambda f: f.cost).x
    residual = _design(calibrated, feat, dt, t, p[:2])@p[2:] - y
    scales = []
    for v in (residual[:, :-1], residual[:, -1]):
        center = np.nanmedian(v)
        scales.append(max(1e-5, 1.4826*np.nanmedian(np.abs(v-center))))
    final_scale = np.broadcast_to(np.r_[np.full(len(t), scales[0]), scales[1]], y.shape)
    s = final_scale[valid]
    fits = [optimize(start, 2.) for start in [p]+[f.x for f in pilots if f.x is not p]]
    p = min(fits, key=lambda f: f.cost).x
    order = np.argsort(p[:2])
    taus = np.exp(p[:2][order])
    coef = np.r_[p[2:4][order], p[4:]]
    return taus, coef


def solve_alpha_w02_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    calib = _calibrate(dataset['calibration_load_cells'])
    X = _features(train, test)
    n = len(train['lab_id'])

    def stress(split):
        gain, bias = calib[split['lab_id']].T
        return split['past_stress_cmd']*gain[:, None]+bias[:, None], split['stress_hold_cmd']*gain+bias

    y = np.c_[train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']]
    taus, coef = _fit(stress(train), X[:n], dt, time, y)
    pred = _design(stress(test), X[n:], dt, time, np.log(taus)) @ coef
    return {'estimated_taus': taus, 'hereditary_weights': coef, 'viscoplastic_weights': np.zeros(2),
            'predicted_strain': pred[:, :-1]+test['strain_0'][:, None], 'initial_rate_pred': pred[:, -1]}
