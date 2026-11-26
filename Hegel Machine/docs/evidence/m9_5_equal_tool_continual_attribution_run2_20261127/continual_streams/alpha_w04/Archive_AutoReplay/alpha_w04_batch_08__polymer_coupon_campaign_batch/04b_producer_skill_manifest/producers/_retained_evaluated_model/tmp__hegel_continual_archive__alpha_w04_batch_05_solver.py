"""Estimate a shared two-branch creep law using calibrated coupon stresses."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _rows(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    take = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _consensus(x, y):
    """Fit a line on the largest set of mutually consistent checks."""
    i, j = np.triu_indices(len(x), 1)
    wide = np.abs(x[j] - x[i]) >= 0.12
    i, j = i[wide], j[wide]
    if not len(i):
        return None, np.zeros(len(x), dtype=bool)
    gain = (y[j] - y[i]) / (x[j] - x[i])
    bias = y[i] - gain * x[i]
    r = np.abs(y - gain[:, None] * x - bias[:, None]) / 0.003
    count = (r <= 1).sum(axis=1)
    score = np.minimum(r**2, 1).sum(axis=1)
    keep = r[np.lexsort((score, -count))[0]] <= 1
    return np.linalg.lstsq(np.c_[x[keep], np.ones(keep.sum())], y[keep], rcond=None)[0], keep


def _calibrate(panel):
    rows = [_rows(panel, lab) for lab in range(4)]
    fits = np.full((4, 2), np.nan)
    weak = []
    for lab, (x, y) in enumerate(rows):
        line, keep = _consensus(x, y)
        if keep.sum() >= max(3, len(x)//2 + 1) and np.ptp(x[keep]) >= 0.35:
            fits[lab] = line
        else:
            weak.append(lab)
    if not weak:
        return fits

    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    panels = json.loads(path.read_text()) if path.exists() else []
    priors = []
    for record in reversed(panels):
        candidate = np.full((4, 2), np.nan)
        for lab in range(4):
            x, y = _rows(record, lab)
            line, keep = _consensus(x, y)
            if keep.sum() >= max(3, len(x)//2 + 1) and np.ptp(x[keep]) >= 0.35:
                candidate[lab] = line
        priors.append(candidate)

    def agreement(candidate, labs):
        residuals = [np.abs(y - (candidate[c, 0]*x + candidate[c, 1]))
                     for c in labs for x, y in [rows[c]] if len(x) and np.isfinite(candidate[c, 0])]
        if not residuals:
            return (0, 0)
        res = np.concatenate(residuals)
        return ((res <= 0.003).sum(), -np.minimum(res, 0.003).sum())

    strong = [c for c in range(4) if c not in weak]
    for lab in weak:
        x, y = rows[lab]
        available = [p for p in priors if np.isfinite(p[lab, 0])]
        if available:
            # Independent labs identify the campaign even if this lab has no reliable checks.
            prior_panel = max(available, key=lambda p: (agreement(p, strong), agreement(p, [lab]), agreement(p, weak)))
            prior = prior_panel[lab]
        else:
            prior = np.array([1., 0.])
        residual = y - (prior[0]*x + prior[1])
        agree = np.abs(residual) <= 0.003
        line, keep = _consensus(x, y)
        if agree.sum() >= 2 and np.ptp(x[agree]) >= 0.35:
            fits[lab] = np.linalg.lstsq(np.c_[x[agree], np.ones(agree.sum())], y[agree], rcond=None)[0]
        elif agree.any():
            fits[lab] = prior + [0., np.median(residual[agree])]
        elif keep.sum() >= (3 if available else 2) and np.ptp(x[keep]) >= 0.35:
            fits[lab] = line
        elif len(x) and not available:
            fits[lab] = prior + [0., np.median(residual)]
        else:
            fits[lab] = prior
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


def _design(taus, dt, times, hist, hold, aligned, yield_stress):
    after = np.cumsum(dt[::-1])[::-1] - dt
    memory = np.exp(-after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    delta = hold[:, None] - np.einsum('ns,sm->nm', hist, memory)
    relax = -np.expm1(-times[:, None] / taus)
    ov_hold = np.sign(hold) * np.maximum(np.abs(hold) - yield_stress, 0.)**2
    work = np.einsum('ns,s->n', np.maximum(np.abs(hist) - yield_stress, 0.)**2, dt)
    work_hold = work * hold
    drift = np.column_stack((np.ones(len(hold)), aligned))
    curves = np.concatenate((delta[:, None, :] * relax,
                             drift[:, None, :] * times[None, :, None],
                             (ov_hold[:, None] * times)[:, :, None],
                             (work_hold[:, None] * relax[:, 0])[:, :, None]), axis=2)
    rates = np.column_stack((delta / taus, drift, ov_hold, work_hold / taus[0]))
    return curves, rates


def _fit(dt, times, hist, hold, aligned, curves, rate, yield_stress):
    n, k = curves.shape
    target = np.r_[curves.ravel(), rate]
    def unpack(p):
        return np.array([p[0], p[0] + p[1] * (8. - p[0])])
    def build(taus):
        a, b = _design(taus, dt, times, hist, hold, aligned, yield_stress)
        return np.vstack((a.reshape(-1, 9), b))
    active = np.any(build(np.array([.6, 3.])) != 0., axis=0)
    lo = np.r_[0.20, 1e-5, np.full(active.sum(), -np.inf)]
    hi = np.r_[8. - 1e-5, 1., np.full(active.sum(), np.inf)]
    def optimize(p, scales, loss):
        def fun(q):
            pred = np.einsum('ij,j->i', build(unpack(q))[:, active], q[2:])
            return (pred - target) / scales
        return least_squares(fun, p, bounds=(lo, hi), loss=loss, x_scale='jac', max_nfev=150)
    pilots = []
    for a, b in ((.35, 1.3), (.60, 3.), (1.15, 5.4)):
        m = build(np.array([a, b]))[:, active]
        w = np.linalg.lstsq(m, target, rcond=None)[0]
        pilots.append(optimize(np.r_[a, (b-a)/(8.-a), w], .05, 'soft_l1'))
    pilot = min(pilots, key=lambda p: p.cost)
    raw = pilot.fun * .05
    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r))))
    scales = np.r_[np.full(n*k, mad(raw[:n*k])), np.full(n, mad(raw[n*k:]))]
    fit = optimize(pilot.x, 2.385 * scales, 'cauchy')
    weights = np.zeros(9)
    weights[active] = fit.x[2:]
    return unpack(fit.x), weights


def _missing_drift(train, test, aligned_train, aligned_test, drift_weights, train_rate, test_rate):
    """Update only the uncertain drift from missing features using the noisy rate."""
    tr_finite = np.isfinite(train['coupon_features_raw'])
    te_finite = np.isfinite(test['coupon_features_raw'])
    rows = np.flatnonzero(~te_finite.all(axis=1) & np.isfinite(test['instant_rate_obs']))
    correction = np.zeros(len(test_rate))
    if not len(rows):
        return correction
    residual = (train['instant_rate_obs'] - train_rate)[tr_finite.all(axis=1)]
    if not len(residual):
        residual = train['instant_rate_obs'] - train_rate
    noise = max(1e-6, 1.4826 * np.median(np.abs(residual - np.median(residual))))
    finite = np.vstack((tr_finite, te_finite))
    aligned = np.vstack((aligned_train, aligned_test))
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    for i in rows:
        missing = ~te_finite[i]
        matched = finite[:, missing].all(axis=1)
        same_lab = matched & (labs == test['lab_id'][i])
        peers = same_lab & (strata == test['stratum'][i])
        if peers.sum() < 2:
            peers = same_lab if same_lab.sum() >= 2 else matched
        if peers.sum() >= 2:
            variance = np.var(aligned[peers][:, missing] @ drift_weights[missing])
        else:
            variance = np.sum(drift_weights[missing]**2)
        innovation = test['instant_rate_obs'][i] - test_rate[i]
        correction[i] = variance / (variance + noise**2) * innovation
    return correction


def _alignment_drift(train, test, x_train, x_test, strain_residual,
                     rate_residual, rate_innovation, times):
    """Estimate residual coupon drift without changing the Prony or plastic weights."""
    correction = np.zeros(len(x_test))
    late = times >= times[len(times) // 4]
    late &= times > 0
    if not late.any():
        return correction
    slope = np.median(strain_residual[:, late] / times[late], axis=1)
    complete_train = np.isfinite(train['coupon_features_raw']).all(axis=1)
    complete_test = np.isfinite(test['coupon_features_raw']).all(axis=1)
    rate_scale = .012
    for lab in np.unique(test['lab_id']):
        fit = (train['lab_id'] == lab) & complete_train & np.isfinite(slope) & np.isfinite(rate_residual)
        pred = (test['lab_id'] == lab) & complete_test
        if fit.sum() < 8 or not pred.any():
            continue
        innovation = rate_residual[fit]
        design = np.column_stack((np.ones(fit.sum()), x_train[fit], innovation / rate_scale))
        def residual(q):
            return np.r_[design @ q - slope[fit], np.sqrt(2.) * q[1:5]]
        q = least_squares(residual, np.r_[np.median(slope[fit]), np.zeros(5)],
                          bounds=(np.r_[np.full(5, -np.inf), 0.],
                                  np.r_[np.full(5, np.inf), rate_scale]),
                          loss='soft_l1', f_scale=.003).x
        r = np.where(np.isfinite(rate_innovation[pred]), rate_innovation[pred], 0.)
        r = np.clip(r, *np.quantile(innovation, [.02, .98]))
        correction[pred] = q[0] + x_test[pred] @ q[1:5] + q[5] * r / rate_scale
    return correction


def solve_alpha_w04_batch_05(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    yield_stress = dataset['yield_stress']
    affine = _calibrate(dataset['calibration_load_cells'])
    aligned_train, aligned_test = _align(train, test)
    taus, theta = _fit(dt, times, *_stress(train, affine), aligned_train,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'], yield_stress)
    a, b = _design(taus, dt, times, *_stress(test, affine), aligned_test, yield_stress)
    a_train, b_train = _design(taus, dt, times, *_stress(train, affine), aligned_train, yield_stress)
    rate = b @ theta
    drift = _missing_drift(train, test, aligned_train, aligned_test, theta[3:7], b_train @ theta, rate)
    drift += _alignment_drift(
        train, test, aligned_train, aligned_test,
        train['strain_curves_obs'] - train['strain_0'][:, None] - a_train @ theta,
        train['instant_rate_obs'] - b_train @ theta, test['instant_rate_obs'] - rate, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta[:7],
            'viscoplastic_weights': theta[7:],
            'predicted_strain': test['strain_0'][:, None] + a @ theta + drift[:, None] * times,
            'initial_rate_pred': rate + drift}
