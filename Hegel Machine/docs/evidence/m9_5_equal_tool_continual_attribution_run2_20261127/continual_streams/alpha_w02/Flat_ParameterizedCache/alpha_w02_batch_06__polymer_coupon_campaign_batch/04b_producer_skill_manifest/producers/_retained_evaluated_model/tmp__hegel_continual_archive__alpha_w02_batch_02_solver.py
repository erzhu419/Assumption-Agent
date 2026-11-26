import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y, tol):
    if len(x) < 2 or np.ptp(x) == 0:
        return None, np.zeros(len(x), dtype=bool)
    best = None
    for i, j in zip(*np.triu_indices(len(x), 1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        bias = np.median(y - gain*x)
        r = abs(y - gain*x - bias)
        good = r <= tol
        score = (good.sum(), -np.minimum(r, tol).sum())
        if best is None or score > best[0]:
            best = (score, good)
    good = best[1]
    if good.sum() < 2 or np.ptp(x[good]) == 0:
        return None, good
    return np.linalg.lstsq(np.c_[x[good], np.ones(good.sum())], y[good], rcond=None)[0], good


def _calibration(checks, yield_stress):
    tol = 0.001*yield_stress
    history = None
    result = []
    for lab in range(4):
        k = np.asarray(checks['lab_id']) == lab
        x, y = (np.asarray(checks[key])[k] for key in ('cmd_stress', 'true_stress'))
        estimate, good = _line(x, y, tol)
        if estimate is not None and good.sum() >= max(3, len(x)//2 + 1) and np.ptp(x[good]) >= 0.1*yield_stress:
            result.append(estimate)
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text()) if _HISTORY.exists() else []
        prior = []
        for sweep in history:
            m = np.asarray(sweep['lab_id']) == lab
            a, b = (np.asarray(sweep[key])[m] for key in ('cmd_stress', 'true_stress'))
            fit, inliers = _line(a, b, tol)
            if fit is not None and inliers.sum() >= max(3, len(a)//2 + 1) and np.ptp(a[inliers]) >= 0.1*yield_stress:
                prior.append(fit)
        if prior:
            gain, bias = np.median(prior, axis=0)
            offsets = y-gain*x
            agree = abs(offsets-bias) <= tol
            if agree.any():
                bias = np.median(np.r_[offsets[agree], bias])
            result.append((gain, bias))
        elif estimate is not None:
            result.append(estimate)
        else:
            result.append((1., np.median(y-x) if len(x) else 0.))
    return np.asarray(result)


def _inputs(dataset):
    train, test = dataset['train'], dataset['test']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    table = _calibration(dataset['calibration_load_cells'], dataset['yield_stress'])
    gain, bias = table[lab].T
    past = np.vstack([train['past_stress_cmd'], test['past_stress_cmd']])*gain[:, None]+bias[:, None]
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']]*gain+bias
    mem = np.vstack([train['initial_memory_states'], test['initial_memory_states']])
    ref = np.vstack([train['ref_coupons'], test['ref_coupons']])
    feat = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).astype(float)
    for c in np.unique(lab):
        rows = lab == c
        r = ref[rows]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[rows] = (feat[rows]-np.median(np.nanmean(r, axis=0)))/np.median(np.nanstd(r, axis=0, ddof=0))
    cells = np.unique(np.c_[lab, stratum], axis=0)
    means = []
    for c, s in cells:
        a = feat[(lab == c) & (stratum == s)]
        count = np.isfinite(a).sum(axis=0)
        means.append(np.divide(np.nansum(a, axis=0), count, out=np.full(feat.shape[1], np.nan), where=count > 0))
    feat = np.where(np.isfinite(feat), feat, np.nanmean(means, axis=0))
    return past, hold, mem, feat


def _components(inputs, dt, tau, variant=0):
    past, hold, mem, feat = inputs
    remaining = np.cumsum(dt[::-1])[::-1]
    end = np.r_[remaining[1:], 0.]
    weight = np.exp(-end[:, None]/tau)*(-np.expm1(-dt[:, None]/tau))
    delta = hold[:, None] - past@weight - mem*np.exp(-remaining[0]/tau)
    z = np.c_[np.ones(len(hold)), feat]
    if variant in (1, 3):
        z = np.c_[z, hold[:, None]*z]
    if variant in (2, 3):
        i, j = np.triu_indices(feat.shape[1]); z = np.c_[z, feat[:, i]*feat[:, j]]
    return delta, z


def _design(inputs, dt, times, tau, variant):
    delta, z = _components(inputs, dt, tau, variant)
    curve = np.concatenate((delta[:, None, :]*(-np.expm1(-times[:, None]/tau)), z[:, None, :]*times[:, None]), axis=2)
    rate = np.c_[delta/tau, z]
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _fit(inputs, dt, times, obs, variant=0):
    valid = np.isfinite(obs)
    y = obs[valid]
    scale = np.ones_like(obs)
    scale[:, -1] = 1.5
    w = (1/scale)[valid]

    def fit_at(logtau, robust):
        design = _design(inputs, dt, times, np.exp(logtau), variant)[valid]
        a = design*w[:, None]
        b = y*w
        coef = np.linalg.lstsq(a, b, rcond=None)[0]
        if robust:
            for _ in range(9):
                r = a@coef-b
                u = (1+(r/2.5)**2)**(-0.5)
                coef = np.linalg.lstsq(a*u[:, None], b*u, rcond=None)[0]
        return coef, a@coef-b

    limits = np.log(([0.055, 0.055], [24.9, 24.9]))
    starts = [(0.25, 1.8), (0.65, 5.), (1.7, 10.)]
    fits = [least_squares(lambda lt: fit_at(lt, False)[1], np.log(start), bounds=limits) for start in starts]
    best = min(fits, key=lambda f: f.cost)
    r = np.full(obs.shape, np.nan)
    r[valid] = fit_at(best.x, False)[1]/w
    mad = lambda a: 1.4826*np.nanmedian(abs(a-np.nanmedian(a)))
    curve_scale = max(mad(r[:, :-1]), 1e-8)
    rate_scale = max(mad(r[:, -1]), curve_scale/3)
    scale = np.broadcast_to(np.r_[np.full(len(times), curve_scale), rate_scale], obs.shape)
    w = (1/scale)[valid]
    opt = least_squares(lambda lt: fit_at(lt, True)[1], best.x, bounds=limits, loss='cauchy', f_scale=2.5)
    tau = np.exp(opt.x)
    coef = fit_at(opt.x, True)[0]
    order = np.argsort(tau)
    return tau[order], np.r_[coef[:2][order], coef[2:]]


def _hazard(initial, threshold, times, tau, amplitudes, drift):
    crossing = np.full(len(initial), times[-1])
    for i, (start, critical, amp, slope) in enumerate(zip(initial, threshold, amplitudes, drift)):
        def margin(t):
            return start + amp@(-np.expm1(-t/tau)) + slope*t - critical
        def rate(t):
            return (amp/tau)@np.exp(-t/tau)+slope
        if margin(0.) >= 0:
            crossing[i] = 0.
            continue
        # The second derivative is a sum of two exponentials and has at most one zero.
        knots = [0., float(times[-1])]
        if amp[0]*amp[1] < 0 and tau[0] != tau[1]:
            inflection = np.log(abs(amp[0]/amp[1])*(tau[1]/tau[0])**2)/(1/tau[0]-1/tau[1])
            if 0 < inflection < knots[-1]:
                knots.insert(1, inflection)
        extrema = [brentq(rate, a, b) for a, b in zip(knots[:-1], knots[1:]) if rate(a)*rate(b) < 0]
        a = 0.
        for b in [*extrema, knots[-1]]:
            if margin(b) >= 0:
                crossing[i] = brentq(margin, a, b)
                break
            a = b
    return crossing


def solve_alpha_w02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    inputs = _inputs(dataset)
    obs = np.c_[train['strain_curves_obs']-train['strain_0'][:, None], train['instant_rate_obs']]
    tau, coef = _fit(tuple(a[:n] for a in inputs), dt, times, obs)
    delta, z = _components(tuple(a[n:] for a in inputs), dt, tau)
    amp, drift = delta*coef[:2], z@coef[2:]
    # A missing feature leaves the coupon-specific drift uncertain. The measured
    # initial rate identifies that drift once the two memory rates are removed.
    if 'instant_rate_obs' in test:
        measured = np.asarray(test['instant_rate_obs'])
        use = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(measured)
        drift[use] = measured[use] - amp[use]@(1/tau)
    initial = test['strain_0']
    increment = amp@(-np.expm1(-times[:, None]/tau)).T + drift[:, None]*times
    pred = initial[:, None]+increment
    return {'estimated_taus': tau, 'predicted_strain': pred,
            'secant_compliance': (pred[:, -1]-initial)/np.maximum(abs(inputs[1][n:]), 0.5),
            'hazard_crossing_times': _hazard(initial, test['critical_strain'], times, tau, amp, drift)}
