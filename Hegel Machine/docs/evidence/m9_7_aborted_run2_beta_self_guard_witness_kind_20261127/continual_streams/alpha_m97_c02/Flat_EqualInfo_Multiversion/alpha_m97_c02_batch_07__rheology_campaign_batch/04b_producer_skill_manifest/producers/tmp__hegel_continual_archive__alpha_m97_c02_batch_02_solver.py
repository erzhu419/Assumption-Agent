"""Estimate the amorphous-melt Prony model from batch measurements."""

import json

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(data):
    c = data['calibration_load_cells']
    labs, cmd, true = map(np.asarray, (c['lab_id'], c['cmd_stress'], c['true_stress']))
    coefficients = np.empty((4, 2))
    archive = None
    def prior(lab):
        nonlocal archive
        if archive is None:
            with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as f:
                archive = json.load(f)
        for record in reversed(archive):
            previous = np.asarray(record['lab_id']) == lab
            if previous.sum() >= 2:
                return _line(np.asarray(record['cmd_stress'])[previous],
                             np.asarray(record['true_stress'])[previous])
        raise ValueError(f'No archived calibration sweep for lab {lab}')

    for lab in range(4):
        take = (labs == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        clean = np.ones(len(x), dtype=bool)
        archived = None
        if len(x) >= 4:
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            g = np.nanmedian(np.nanmedian(slopes, axis=1))
            offsets = y - g*x
            clean = np.abs(offsets - np.median(offsets)) <= 3.0
        elif len(x) == 3:
            # A three-point sweep can reveal a bad line without identifying
            # which point was hit. Compare each point to the other two first.
            for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                span = x[j] - x[i]
                if span != 0 and abs((y[k]-y[i])*span - (y[j]-y[i])*(x[k]-x[i])) > 3.0*abs(span):
                    archived = prior(lab)
                    offsets = y - archived[0]*x
                    clean = np.abs(offsets - np.median(offsets)) <= 3.0
                    break
        x, y = x[clean], y[clean]
        if len(x) >= 4 or (len(x) >= 2 and clean.all()):
            coefficients[lab] = _line(x, y)
        else:
            g, b = prior(lab) if archived is None else archived
            if len(x):
                b = np.median(y - g*x)
            coefficients[lab] = g, b
    return coefficients

def _align(data):
    train, test = data['train'], data['test']
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for c in np.unique(lab):
        sel = lab == c
        panel = ref[sel]
        shift = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[sel] = (x[sel] - shift)/gain
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        part = x[(lab == c) & (stratum == s)]
        count = np.isfinite(part).sum(axis=0)
        means.append(np.divide(np.nansum(part, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return split['past_stress_cmd']*gain[:, None] + bias[:, None], split['stress_hold_cmd']*gain + bias


def _design(logtau, past, hold, features, dt, t):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    for stress, duration in zip(past.T, dt):
        z += (-np.expm1(-duration/tau)) * (stress[:, None] - z)
    gap = hold[:, None] - z
    basis = -np.expm1(-t[:, None]/tau)
    curves = np.concatenate((gap[:, None, :]*basis[None, :, :],
                             features[:, None, :]*t[None, :, None]), axis=2)
    rates = np.column_stack((gap/tau, features))
    return curves, rates


def _fit(inputs, curves, rates):
    n, k = curves.shape
    y = np.r_[curves.ravel(), rates]

    def design(logtau):
        a, b = _design(logtau, *inputs)
        return np.vstack((a.reshape(-1, 7), b))

    def mad(a, axis=None):
        med = np.median(a, axis=axis, keepdims=True)
        return np.maximum(1.4826*np.median(np.abs(a-med), axis=axis), 1e-8)

    def scale_vector(a, b):
        return np.r_[np.tile(mad(a, axis=0), n), np.full(n, mad(b))]

    scale = scale_vector(curves, rates)
    bounds = (np.r_[np.log([0.04, 0.04]), [-np.inf]*7],
              np.r_[np.log([50., 50.]), [np.inf]*7])

    def residual(p):
        return (design(p[:2]) @ p[2:] - y)/scale

    def jac(p):
        a = design(p[:2])
        delta = 1e-5*np.eye(2)
        d = [((design(p[:2]+h)-design(p[:2]-h)) @ p[2:])/(2e-5) for h in delta]
        return np.column_stack((*d, a))/scale[:, None]

    def start(tau):
        t = np.log(tau)
        return np.r_[t, np.linalg.lstsq(design(t)/scale[:, None], y/scale, rcond=None)[0]]

    def optimize(p, loss):
        return least_squares(residual, p, jac=jac, bounds=bounds, loss=loss,
                             x_scale='jac', max_nfev=90, ftol=1e-9)

    starts = [(0.3, 1.5), (0.6, 3.), (1.2, 6.)]
    best = min((optimize(start(tau), 'soft_l1') for tau in starts), key=lambda fit: fit.cost)
    for _ in range(2):
        r = residual(best.x)*scale
        scale = scale_vector(r[:n*k].reshape(n, k), r[n*k:])
        best = optimize(best.x, 'cauchy')
    order = np.argsort(best.x[:2])
    return best.x[:2][order], np.r_[best.x[2:4][order], best.x[4:]]


def _refine_short_pairs(data, calibration, features, dt, times):
    """Constrain an unresolved two-point slope using in-batch rheology."""
    cal = data['calibration_load_cells']
    labs, cmd, true = map(np.asarray, (cal['lab_id'], cal['cmd_stress'], cal['true_stress']))
    pairs = {}
    for lab in range(4):
        take = (labs == lab) & np.isfinite(cmd) & np.isfinite(true)
        # At the observed sweep noise (~2e-4), spans below 0.03
        # can give percent-scale gain error by differencing.
        if take.sum() == 2 and np.ptp(cmd[take]) < 0.03:
            pairs[lab] = (cmd[take].mean(), true[take].mean())
    if not pairs:
        return calibration

    train = data['train']
    lab_id = np.asarray(train['lab_id'])
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    rates = train['instant_rate_obs']
    measured = np.isfinite(train['coupon_features_raw']).all(axis=1)
    anchor = ~np.isin(lab_id, list(pairs))
    if not anchor.any():
        return calibration
    past, hold = _physical(train, calibration)
    tau, w = _fit((past[anchor], hold[anchor], features[anchor], dt, times),
                  curves[anchor], rates[anchor])
    a, b = _design(tau, past[anchor], hold[anchor], features[anchor], dt, times)
    resid = np.column_stack((a @ w - curves[anchor], b @ w - rates[anchor]))
    scale = np.maximum(1.4826*np.median(abs(resid - np.median(resid, axis=0)), axis=0), 1e-6)
    calibration = calibration.copy()
    for lab, (xc, yc) in pairs.items():
        use = (lab_id == lab) & measured
        if not use.any():
            continue
        past_cmd = train['past_stress_cmd'][use]
        hold_cmd = train['stress_hold_cmd'][use]
        f = features[use]
        a, b = _design(tau, np.full_like(past_cmd, yc), np.full_like(hold_cmd, yc), f, dt, times)
        offset = np.column_stack((a @ w, b @ w))
        a, b = _design(tau, yc + past_cmd - xc, yc + hold_cmd - xc, f, dt, times)
        signal = np.column_stack((a @ w, b @ w)) - offset
        target = np.column_stack((curves[use], rates[use])) - offset
        gain = least_squares(lambda g: ((g[0]*signal - target)/scale).ravel(),
                             [calibration[lab, 0]], loss='soft_l1').x[0]
        calibration[lab] = gain, yc - gain*xc
    return calibration


def solve_alpha_m97_c02_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    calibration = _calibration(dataset)
    x = _align(dataset)
    n_train = len(train['lab_id'])
    x_train, x_test = x[:n_train], x[n_train:]
    dt = np.asarray(dataset['past_dt'])
    t = np.asarray(dataset['eval_times'])
    calibration = _refine_short_pairs(dataset, calibration,
                                      np.column_stack((np.ones(n_train), x_train)), dt, t)

    def inputs(split, x):
        past, hold = _physical(split, calibration)
        return past, hold, np.column_stack((np.ones(len(x)), x)), dt, t

    logtau, weights = _fit(inputs(train, x_train),
                           train['strain_curves_obs'] - train['strain_0'][:, None],
                           train['instant_rate_obs'])
    curves, rates = _design(logtau, *inputs(test, x_test))
    return {'estimated_taus': np.exp(logtau),
            'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
