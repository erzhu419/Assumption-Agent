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
    memory = np.exp(-after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    z = initial * np.exp(-dt.sum() / taus) + hist @ memory
    delta = hold[:, None] - z
    drift = np.column_stack((np.ones(len(hold)), aligned))
    rate = np.column_stack((delta / taus, drift))
    curves = np.concatenate((delta[:, None, :] * (-np.expm1(-times[:, None] / taus)),
                             drift[:, None, :] * times[None, :, None]), axis=2)
    return curves, rate


def _fit_seed(dt, times, hist, hold, aligned, initial, curves, rate):
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


def _fit(dt,T,h,p,x,z,y,r):
    tau,w=_fit_seed(dt,T,h,p,x,z,y,r)
    # Pairwise differences of time-normalized strain cancel linear drift;
    # robustly profile coupon drift in the original observation coordinates instead.
    D=np.r_[T,1.]; target=np.column_stack((y,r))
    q0=np.r_[tau[0], (tau[1]-tau[0])/(8.-tau[0]), w[:2]]
    def ordered(q):
        return np.array([q[0], q[0] + q[1]*(8.-q[0])])
    b,o=_design(tau,dt,T,h,p,x,z)
    resid=np.column_stack((y-b@w,r-o@w))
    scale=np.maximum(1e-6,1.4826*np.median(np.abs(resid-np.median(resid,axis=0)),axis=0)); scale*=2.385
    A=D/scale
    def fun(q, profiled=True):
        b,o=_design(ordered(q),dt,T,h,p,x,z)
        u=(target-np.column_stack((b[:,:,:2]@q[2:],o[:,:2]@q[2:])))/scale
        off=np.median(u/A,axis=1)
        for _ in range(8):
            wt=1/(1+(u-off[:,None]*A)**2)
            off=((wt*u)@A)/(wt@(A*A))
        return (u-off[:,None]*A).ravel() if profiled else off
    q=least_squares(fun,q0,bounds=([.2,1e-5,-np.inf,-np.inf],[8.-1e-5,1.,np.inf,np.inf]),loss='cauchy',max_nfev=80).x
    drift=fun(q,False)
    X=np.column_stack((np.ones(len(x)),x)); error=drift-X@w[2:]; spread=max(.003,1.4826*np.median(abs(error-np.median(error))))
    v=least_squares(lambda b:(drift-X@b)/spread,w[2:],loss='soft_l1').x
    return ordered(q),np.r_[q[2:],v]

def _fit_split(dataset, train, test):
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    affine = _calibrate(dataset['calibration_load_cells'])
    x, xt = _align(train, test)
    hist, hold = _stress(train, affine)
    init = train['initial_memory_states']
    rate = train['instant_rate_obs']
    increments = train['strain_curves_obs'] - train['strain_0'][:, None]
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    taus, theta = _fit(dt, times, hist[complete], hold[complete], x[complete],
                       init[complete], increments[complete], rate[complete])
    # Learn the residual affine drift in each lab from complete coupons.
    # A median across the longer horizons limits the influence of bad curve samples.
    train_basis, train_onset = _design(taus, dt, times, hist, hold, x, init)
    late = times >= 1.
    relaxation = train_basis[:, late, :2] @ theta[:2]
    drift_obs = np.median((increments[:, late] - relaxation) / times[late], axis=1)
    drift_prior = train_onset[:, 2:] @ theta[2:]

    hist_test, hold_test = _stress(test, affine)
    basis, onset = _design(taus, dt, times, hist_test, hold_test, xt,
                           test['initial_memory_states'])
    drift_test = onset[:, 2:] @ theta[2:]
    adjust = np.zeros(len(hold_test))
    for lab in np.unique(train['lab_id'][complete]):
        use = complete & (train['lab_id'] == lab)
        center = np.median(drift_prior[use])
        x_lab = drift_prior[use] - center
        residual = drift_obs[use] - drift_prior[use]
        scale = max(0.003, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        offset, slope = least_squares(
            lambda par: (residual - par[0] - par[1] * x_lab) / scale,
            [0., 0.], loss='soft_l1').x
        rows = test['lab_id'] == lab
        adjust[rows] = offset + slope * (drift_test[rows] - center)

    # Missing features only affect drift. Subtract relaxation from the
    # independently observed onset to recover that coupon's drift instead.
    rate_fit = onset @ theta
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    adjust[missing] = (test['instant_rate_obs'] - rate_fit)[missing]
    strain = test['strain_0'][:, None] + basis @ theta + adjust[:, None] * times
    return taus, strain, hold_test, rate_fit + adjust


def _crossing(times, start, strain, critical):
    grid = np.r_[0., times]
    curve = np.column_stack((start, strain))
    above = curve >= np.asarray(critical)[:, None]
    first = above.argmax(axis=1)
    cross = np.full(len(start), times[-1], dtype=float)
    cross[above[:, 0]] = 0.
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    left, right = curve[rows, j-1], curve[rows, j]
    cross[rows] = grid[j-1] + (grid[j] - grid[j-1]) * (critical[rows] - left) / (right - left)
    return cross


def solve_alpha_w04_batch_01(dataset: dict) -> dict:
    test = dataset['test']
    taus, strain, hold, _ = _fit_split(dataset, dataset['train'], test)
    return {'estimated_taus': taus, 'predicted_strain': strain,
            'secant_compliance': (strain[:, -1] - test['strain_0']) / np.maximum(np.abs(hold), 0.5),
            'hazard_crossing_times': _crossing(dataset['eval_times'], test['strain_0'], strain,
                                               test['critical_strain'])}
