"""Calibrate the load cells and fit the sub-yield two-branch creep law."""
from pathlib import Path
import json
import pickle

import numpy as np
from scipy.optimize import least_squares


def _rows(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    take = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _line(x, y, prior):
    """Find an affine consensus, then refit on the inliers."""
    if len(x) == 0:
        return prior, np.zeros(0, dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    keep = np.abs(x[j] - x[i]) >= 0.12
    i, j = i[keep], j[keep]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    pairs = np.column_stack((slopes, y[i] - slopes * x[i]))
    candidates = np.vstack((pairs, prior))
    residual = np.abs(y - candidates[:, :1] * x - candidates[:, 1:])
    gate = 0.003
    inliers = residual <= gate
    counts = inliers.sum(axis=1)
    costs = np.minimum((residual / gate)**2, 1).sum(axis=1)
    proximity = (candidates[:, 0] - prior[0])**2 + (candidates[:, 1] - prior[1])**2
    # For a two-point tie the well-supported prior is more informative
    # than zero residual on the points that defined the candidate.
    winner = np.lexsort((costs, proximity * (counts < 3), -counts))[0]
    m = inliers[winner]
    if m.sum() >= 2 and np.ptp(x[m]) >= 0.12:
        return np.linalg.lstsq(np.column_stack((x[m], np.ones(m.sum()))), y[m], rcond=None)[0], m
    return prior.copy(), m


def _archive_panels():
    """Read only archived load-cell panels, newest batch first."""
    root = Path('/tmp/hegel_continual_archive')
    def panels(record):
        if isinstance(record, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= record.keys():
                yield {k: np.atleast_1d(record[k]) for k in ('lab_id', 'cmd_stress', 'true_stress')}
            else:
                for key in ('dataset', 'calibration_load_cells', 'batches', 'records'):
                    if key in record:
                        yield from panels(record[key])
        elif isinstance(record, (list, tuple)):
            for entry in reversed(record):
                yield from panels(entry)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.pkl', '.pickle', '.json')]
    for path in sorted(files, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True):
        try:
            with path.open('rb') as stream:
                record = json.load(stream) if path.suffix == '.json' else pickle.load(stream)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError, ImportError):
            continue
        yield from panels(record)


def _calibrate(panel):
    identity = np.array([1., 0.])
    rows = [_rows(panel, c) for c in range(4)]
    fits = np.tile(identity, (4, 1))
    weak = set()
    for c, (x, y) in enumerate(rows):
        fit, m = _line(x, y, identity)
        if m.sum() >= max(3, (len(x) + 1)//2) and np.ptp(x[m]) >= 0.35:
            fits[c] = fit
        else:
            weak.add(c)
    archived = {}
    if weak:
        for panel_old in _archive_panels():
            for c in weak - archived.keys():
                x, y = _rows(panel_old, c)
                prior, m = _line(x, y, identity)
                if m.sum() >= 3 and np.ptp(x[m]) >= 0.35:
                    archived[c] = prior
            if weak <= archived.keys():
                break
    for c in sorted(weak):
        peer = c ^ 1
        prior = archived.get(c, fits[peer] if peer not in weak else archived.get(peer, identity))
        x, y = rows[c]
        # Two separated checks determine this lab's line if no historical
        # calibration can distinguish that line from a contaminated pair.
        if c not in archived and len(x) == 2 and np.ptp(x) >= 0.35:
            fits[c], _ = _line(x, y, prior)
            continue
        if len(x):
            shift = y - prior[0] * x - prior[1]
            med = np.median(shift)
            agree = np.abs(shift - med) <= 0.003
            # Trust a lone spot without a lab-specific archive, but do not
            # let it overwrite a well-supported historical calibration.
            if agree.sum() > len(x)//2 and (agree.sum() >= 2 or c not in archived or abs(med) <= 0.003):
                if agree.sum() >= 2 and np.ptp(x[agree]) >= 0.35 and c not in archived:
                    fits[c], _ = _line(x[agree], y[agree], prior)
                else:
                    fits[c] = prior + [0., np.mean(shift[agree])]
                continue
        fits[c] = prior
    return fits


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


def _design(taus, dt, times, hist, hold, aligned, initial):
    after = np.cumsum(dt[::-1])[::-1] - dt
    kernel = np.exp(-after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    memory = hist @ kernel + initial * np.exp(-np.sum(dt) / taus)
    delta = hold[:, None] - memory
    drift = np.column_stack((np.ones(len(hold)), aligned))
    rate = np.column_stack((delta / taus, drift))
    curves = np.concatenate((delta[:, None, :] * (-np.expm1(-times[:, None] / taus)),
                            drift[:, None, :] * times[None, :, None]), axis=2)
    return curves, rate

def _fit(dt, times, hist, hold, aligned, curves, rate, initial):
    n, k = curves.shape
    target = np.r_[curves.ravel(), rate]
    def build(taus):
        a, b = _design(taus, dt, times, hist, hold, aligned, initial)
        return np.vstack((a.reshape(-1, 7), b))
    def unpack(p):
        return np.array([p[0], p[0] + p[1] * (8. - p[0])]), p[2:]
    lo = np.r_[0.20, 1e-5, np.full(7, -np.inf)]
    hi = np.r_[8. - 1e-5, 1., np.full(7, np.inf)]
    def optimize(p, scales, loss):
        def fun(q):
            t, w = unpack(q)
            return (np.einsum('ij,j->i', build(t), w) - target) / scales
        return least_squares(fun, p, bounds=(lo, hi), loss=loss, x_scale='jac', max_nfev=160)
    pilots = []
    for a, b in ((0.35, 1.3), (0.60, 3.0), (1.15, 5.4)):
        m = build(np.array([a, b]))
        w = np.linalg.lstsq(m, target, rcond=None)[0]
        pilots.append(optimize(np.r_[a, (b-a)/(8.-a), w], 0.05, 'soft_l1'))
    pilot = min(pilots, key=lambda o: o.cost)
    raw = pilot.fun * 0.05
    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r))))
    scales = np.r_[np.full(n*k, mad(raw[:n*k])), np.full(n, mad(raw[n*k:]))]
    fit = optimize(pilot.x, 2.385 * scales, 'cauchy')
    return unpack(fit.x)


