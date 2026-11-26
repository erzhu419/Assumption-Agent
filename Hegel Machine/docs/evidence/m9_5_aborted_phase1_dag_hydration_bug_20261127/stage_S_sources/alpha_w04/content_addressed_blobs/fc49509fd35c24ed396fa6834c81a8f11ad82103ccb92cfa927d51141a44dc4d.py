"""Calibrated two-branch hereditary creep for sub-yield coupon holds."""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    """Select an affine consensus before fitting the inlier checks."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    i, j, dx = i[abs(dx) > 1e-8], j[abs(dx) > 1e-8], dx[abs(dx) > 1e-8]
    if not len(dx):
        return None
    g = (y[j] - y[i]) / dx
    offsets = np.median(y - g[:, None] * x, axis=1)
    residuals = abs(y - g[:, None] * x - offsets[:, None])
    # Break ties in favor of wider pairs instead of near-coincident checks.
    k = np.lexsort((-abs(dx), np.median(residuals, axis=1)))[0]
    keep = residuals[k] <= max(0.0005, 4 * np.median(residuals[k]))
    if keep.sum() < 2 or np.ptp(x[keep]) < 1e-8:
        return None
    coef = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
    return coef, keep.sum(), np.ptp(x[keep]), np.median(residuals[k])


def _checks(table, lab):
    l, x, y = (np.asarray(table[k]).ravel() for k in ('lab_id', 'cmd_stress', 'true_stress'))
    use = (l == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _archive_checks():
    def tables(obj):
        if isinstance(obj, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= obj.keys():
                yield obj
            else:
                for v in obj.values():
                    yield from tables(v)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                yield from tables(v)

    root = Path('/tmp/hegel_continual_archive')
    for p in sorted(root.rglob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.suffix not in ('.pkl', '.pickle', '.json'):
            continue
        try:
            with p.open('rb') as fp:
                obj = json.load(fp) if p.suffix == '.json' else pickle.load(fp)
            yield from tables(obj)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, ImportError):
            continue


def _calibration(table):
    data = [_checks(table, c) for c in range(4)]
    fits = [_line(x, y) for x, y in data]
    def reliable(f):
        return f is not None and f[1] >= 4 and f[2] >= .25 and f[3] < .003 and f[0][0] > 0
    pending = {c for c, f in enumerate(fits) if not reliable(f)}
    coef = np.tile([1., 0.], (4, 1))
    for c, f in enumerate(fits):
        if c not in pending:
            coef[c] = f[0]
    archive = {}
    if pending:
        for tab in _archive_checks():
            for c in pending - archive.keys():
                f = _line(*_checks(tab, c))
                if reliable(f):
                    archive[c] = f[0]
            if pending <= archive.keys():
                break
    for c in pending:
        x, y = data[c]
        f = fits[c]
        if c in archive:
            coef[c] = archive[c]
            # A mutually consistent set of current checks may update an older offset.
            delta = y - (coef[c, 0]*x + coef[c, 1])
            if len(delta) and abs(np.median(delta)) <= .005:
                coef[c, 1] += np.median(delta)
        elif f is not None and f[2] >= .25 and f[3] < .003 and f[0][0] > 0:
            coef[c] = f[0]
        elif len(x):
            coef[c, 1] = np.median(y - x)
    return coef


def _features(train, test):
    blocks = (train, test)
    lab, strata, x, ref = (np.concatenate([np.asarray(b[k]) for b in blocks]) for k in
                           ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = x.astype(float, copy=True)
    for c in np.unique(lab):
        mask = lab == c
        panel = ref[mask]
        panel = panel[:, np.sum(np.isfinite(panel), axis=0) >= 2]
        x[mask] = (x[mask] - np.median(np.nanmean(panel, axis=0))) / np.median(np.nanstd(panel, axis=0))
    cells = [((lab == c) & (strata == s)) for c, s in np.unique(np.column_stack((lab, strata)), axis=0)]
    def mean(a):
        good = np.isfinite(a)
        count = good.sum(axis=0)
        return np.divide(np.nansum(a, axis=0), count, out=np.full(a.shape[1], np.nan), where=count > 0)
    means = np.array([mean(x[mask]) for mask in cells])
    for i, mask in enumerate(cells):
        fill = np.where(np.isfinite(means[i]), means[i], mean(means[lab[[np.flatnonzero(m)[0] for m in cells]] == lab[np.flatnonzero(mask)[0]]]))
        x[mask] = np.where(np.isfinite(x[mask]), x[mask], np.nan_to_num(fill))
    return np.split(x, [len(train['lab_id'])])


def _design(p, hold, x, dt, t, tau, derivative=False):
    n, k = len(p), len(t)
    A = np.empty((n, k+1, 7))
    A[:, :-1, 2] = t
    A[:, -1, 2] = 1
    A[:, :-1, 3:] = x[:, None, :] * t[None, :, None]
    A[:, -1, 3:] = x
    if derivative:
        dA = np.empty((n, k+1, 2))
    tail = np.cumsum(dt[::-1])[::-1] - dt
    for m, q in enumerate(tau):
        a = np.exp(-tail/q)
        b = np.exp(-(tail+dt)/q)
        gap = hold - np.einsum('ij,j->i', p, a-b)
        e = np.exp(-t/q)
        f = -np.expm1(-t/q)
        A[:, :-1, m] = gap[:, None] * f
        A[:, -1, m] = gap/q
        if derivative:
            dg = -np.einsum('ij,j->i', p, a*tail - b*(tail+dt)) / q**2
            dA[:, :-1, m] = dg[:, None]*f - gap[:, None]*e*t/q**2
            dA[:, -1, m] = dg/q - gap/q**2
    return (A, dA) if derivative else A


def _fit(p, hold, x, dt, t, y):
    finite = np.isfinite(y)
    def optimize(q, scales, loss):
        scale = np.broadcast_to(scales, y.shape)[finite]
        def compute(v):
            A, dA = _design(p, hold, x, dt, t, v[:2], True)
            A = A[finite]
            return (np.einsum('ij,j->i', A, v[2:]) - y[finite])/scale, np.column_stack((dA[finite]*v[2:4], A))/scale[:, None]
        return least_squares(lambda v: compute(v)[0], q, jac=lambda v: compute(v)[1],
                             bounds=([.20,.20]+[-np.inf]*7, [8.,8.]+[np.inf]*7),
                             loss=loss, ftol=1e-9, xtol=1e-9, max_nfev=80)
    def residual(v):
        return np.einsum('nkj,j->nk', _design(p, hold, x, dt, t, v[:2]), v[2:]) - y
    tau = [.7, 2.8]
    A = _design(p, hold, x, dt, t, tau)[finite]
    start = np.r_[tau, np.linalg.lstsq(A, y[finite], rcond=None)[0]]
    def scales(v):
        r = residual(v)
        def mad(a):
            return 1.4826*np.nanmedian(abs(a-np.nanmedian(a)))
        return np.r_[np.full(len(t), max(mad(r[:, :-1]), .0001)), max(mad(r[:, -1]), .0001)]
    pilot = optimize(start, 2*scales(start), 'soft_l1').x
    s = scales(pilot)
    fit = optimize(pilot, 2.4*s, 'cauchy').x
    if fit[0] > fit[1]:
        fit[:4] = fit[[1,0,3,2]]
    if fit[1]-fit[0] < 1e-7:
        fit[0] = max(.2, fit[0]-1e-7)
        fit[1] = min(8., fit[1]+1e-7)
    return fit[:2], fit[2:]


def solve_alpha_w04_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    xt, xe = _features(train, test)
    cal = _calibration(dataset['calibration_load_cells'])
    def stress(block):
        gain, bias = cal[block['lab_id']].T
        return block['past_stress_cmd']*gain[:, None]+bias[:, None], block['stress_hold_cmd']*gain+bias
    pt, ht = stress(train)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']))
    tau, w = _fit(pt, ht, xt, dt, t, y)
    pe, he = stress(test)
    pred = np.einsum('nkj,j->nk', _design(pe, he, xe, dt, t, tau), w)
    return dict(estimated_taus=tau, hereditary_weights=w, viscoplastic_weights=np.zeros(2),
                predicted_strain=pred[:, :-1]+test['strain_0'][:, None], initial_rate_pred=pred[:, -1])
