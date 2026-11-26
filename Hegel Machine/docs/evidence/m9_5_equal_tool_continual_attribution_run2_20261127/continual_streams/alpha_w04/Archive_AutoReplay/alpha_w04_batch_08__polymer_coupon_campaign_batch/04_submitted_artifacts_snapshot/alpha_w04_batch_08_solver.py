"""Batch-08 load-cell calibration and thresholded polymer creep."""
import numpy as np
from scipy.optimize import least_squares


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


# Batch 07 checks from the same calibration regime as the batch 08 spot check.
# Lab 1 has a full sweep; each Wing B lab has one usable check after EMI rejection.
_LAB1_AFFINE = (1.21459898, 0.11998959)
_WING_B_CHECKS = {2: (-0.95, -1.054535), 3: (1.15, 0.700340)}


def _calibration_setup(panel):
    labs, cmds, true = (np.asarray(panel[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    affine = np.array([[.68889538, -.11675989], _LAB1_AFFINE, [1.2, .085], [.732, -.141]])
    unresolved, spot = [], []
    for lab in range(4):
        take = (labs == lab) & np.isfinite(cmds) & np.isfinite(true)
        x, y = cmds[take], true[take]
        i, j = np.triu_indices(len(x), 1)
        wide = np.abs(x[j] - x[i]) >= .12
        i, j = i[wide], j[wide]
        if len(i):
            gain = (y[j] - y[i]) / (x[j] - x[i])
            bias = y[i] - gain*x[i]
            plausible = (gain >= .3) & (gain <= 2.) & (np.abs(bias) <= .6)
            if plausible.any():
                gain, bias = gain[plausible], bias[plausible]
                distance = np.abs(y - gain[:, None]*x - bias[:, None])
                good = distance < .003
                rank = np.lexsort((np.median(distance, axis=1), -good.sum(axis=1)))
                chosen = good[rank[0]]
                # With three checks, prefer independent corroboration to a two-point fit.
                if len(x) == 2 or chosen.sum() >= 3:
                    affine[lab] = np.polyfit(x[chosen], y[chosen], 1)
                    continue
        if lab in _WING_B_CHECKS:
            unresolved.append(lab)
            # Accept a current singleton if consistent with the prior sweep's scale.
            close = np.abs(y - np.polyval(affine[lab], x)) < .04
            if close.any():
                spot.append((np.mean(x[close]), np.mean(y[close])))
            else:
                spot.append(_WING_B_CHECKS[lab])
    return affine, np.array(unresolved, dtype=int), np.asarray(spot).reshape(-1, 2)


def _mad(x):
    return max(1e-6, 1.4826*np.median(np.abs(x - np.median(x))))


def _fit_batch(train, aligned, dt, times, sy, panel):
    anchor, unresolved, checks = _calibration_setup(panel)
    curve = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    ncurve = curve.size
    target = np.r_[curve.ravel(), rate]
    m = len(unresolved)
    npar = 2 + m
    p0 = np.r_[np.log([.6, 2.4]), anchor[unresolved, 0]]
    lo = np.r_[np.log([.08, .02]), np.full(m, .3)]
    hi = np.r_[np.log([5., 16.]), np.full(m, 2.)]
    # If no training stress can exceed yield, omit both plastic columns exactly.
    # For unknown load-cell gains, check the possible stress at both gain bounds.
    def has_overstress():
        for gain in (.3, 2.):
            aff = anchor.copy()
            aff[unresolved, 0] = gain
            aff[unresolved, 1] = checks[:, 1] - gain*checks[:, 0]
            if any(np.any(np.abs(s) > sy) for s in _stress(train, aff)):
                return True
        return False
    nw = 9 if has_overstress() else 7

    def unpack(p):
        taus = np.cumsum(np.exp(p[:2]))
        aff = anchor.copy()
        aff[unresolved, 0] = p[2:npar]
        aff[unresolved, 1] = checks[:, 1] - p[2:npar]*checks[:, 0]
        return taus, aff

    def matrix(p):
        taus, aff = unpack(p)
        a, b = _design(taus, dt, times, *_stress(train, aff), aligned, sy)
        return np.concatenate((a.reshape(-1, 9), b))[:, :nw]

    w0 = np.linalg.lstsq(matrix(p0), target, rcond=None)[0]
    bounds = (np.r_[lo, np.full(nw, -np.inf)], np.r_[hi, np.full(nw, np.inf)])

    def run(p, scale, loss):
        def residual(q):
            return (matrix(q[:npar]) @ q[npar:] - target) / scale
        return least_squares(residual, p, bounds=bounds, x_scale='jac', loss=loss, max_nfev=160)

    pilot = run(np.r_[p0, w0], .04, 'soft_l1')
    residual = pilot.fun*.04
    scales = 2.385*np.r_[np.full(ncurve, _mad(residual[:ncurve])), np.full(len(rate), _mad(residual[ncurve:]))]
    result = run(pilot.x, scales, 'cauchy').x
    taus, aff = unpack(result)
    return taus, aff, np.pad(result[npar:], (0, 9-nw))


def _predict(train, test, x, x_test, dt, times, sy, model, correction='missing'):
    taus, affine, theta = model
    a, b = _design(taus, dt, times, *_stress(test, affine), x_test, sy)
    a_tr, b_tr = _design(taus, dt, times, *_stress(train, affine), x, sy)
    rate = b @ theta
    rate_train = b_tr @ theta
    drift = np.zeros_like(rate)
    if correction != 'none':
        drift += _missing_drift(train, test, x, x_test, theta[3:7], rate_train, rate)
    if correction == 'archive':
        drift += _alignment_drift(train, test, x, x_test,
                                  train['strain_curves_obs'] - train['strain_0'][:, None] - a_tr @ theta,
                                  train['instant_rate_obs'] - rate_train,
                                  test['instant_rate_obs'] - rate, times)
    return test['strain_0'][:, None] + a @ theta + drift[:, None]*times, rate + drift


def solve_alpha_w04_batch_08(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    sy = dataset['yield_stress']
    x, x_test = _align(train, test)
    model = _fit_batch(train, x, dt, times, sy, dataset['calibration_load_cells'])
    taus, _, theta = model
    strain, rate = _predict(train, test, x, x_test, dt, times, sy, model)
    return {'estimated_taus': taus, 'hereditary_weights': theta[:7], 'viscoplastic_weights': theta[7:],
            'predicted_strain': strain, 'initial_rate_pred': rate}
