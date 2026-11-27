"""Calibrated two-branch creep with a damage-gated crystallization plateau."""
import json

import numpy as np
from scipy.optimize import least_squares


_HISTORY = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _rows(table, channel):
    lab, cmd, true = (np.asarray(table[key]) for key in ('lab_id', 'cmd_stress', 'true_stress'))
    keep = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
    return cmd[keep], true[keep]


def _clean(x, y):
    # An EMI-contaminated sweep has a tight affine consensus and large spikes.
    if len(x) < 3:
        return np.ones(len(x), dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if len(i):
        slope = (y[j] - y[i]) / (x[j] - x[i])
        err = np.abs(y - y[i, None] - slope[:, None] * (x - x[i, None]))
        clean = err <= .01
        legal = np.all(clean | (err > 3.), axis=1)
        if legal.any():
            options = clean[legal]
            return options[np.argmax(options.sum(axis=1))]
    return np.ones(len(x), dtype=bool)


def _calibration(table):
    history = None
    result = []
    for lab in range(4):
        x, y = _rows(table, lab)
        clean = _clean(x, y)
        few = len(x) < 2 or np.ptp(x) == 0
        contaminated = not clean.all()
        if not few and (not contaminated or clean.sum() >= 4):
            result.append(np.polyfit(x[clean], y[clean], 1))
            continue
        if history is None:
            with open(_HISTORY) as file:
                history = json.load(file)
        for sweep in reversed(history):
            u, v = _rows(sweep, lab)
            ok = _clean(u, v)
            if ok.sum() < 4 or np.ptp(u[ok]) == 0:
                continue
            prior = np.polyfit(u[ok], v[ok], 1)
            if np.max(np.abs(v[ok] - np.polyval(prior, u[ok]))) <= .01:
                result.append(prior)
                break
        else:
            raise ValueError(f'No usable load-cell calibration for channel {lab}')
    return np.array(result)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]))

    lab, stratum = join('lab_id'), join('stratum')
    feature, panel = join('coupon_features_raw').astype(float), join('ref_coupons')
    cells = []
    for channel in np.unique(lab):
        rows = lab == channel
        ref = panel[rows]
        ref = ref[:, np.isfinite(ref).sum(axis=0) >= 2]
        feature[rows] = (feature[rows] - np.median(np.nanmean(ref, axis=0))) / np.median(np.nanstd(ref, axis=0, ddof=0))
        for s in np.unique(stratum[rows]):
            block = feature[rows & (stratum == s)]
            count = np.isfinite(block).sum(axis=0)
            cells.append(np.divide(np.nansum(block, axis=0), count,
                                   out=np.full(feature.shape[1], np.nan), where=count > 0))
    features = np.where(np.isfinite(feature), feature, np.nanmean(cells, axis=0))
    gain, bias = _calibration(dataset['calibration_load_cells'])[lab].T
    past = join('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = join('stress_hold_cmd') * gain + bias
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    damage = np.maximum(0., np.abs(past) - dataset['crit_stress'])**2 @ dt
    if dataset['shear_banding_index'] > .50:
        damage[:] = 0.
    return past, hold, features, np.tanh(damage), join('strain_0'), dt, times


def _design(logtau, past, hold, features, act, dt, times):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    decay = np.exp(-dt[:, None] / tau)
    increment = -np.expm1(-dt[:, None] / tau)
    for s, (a, b) in enumerate(zip(decay, increment)):
        z = a*z + b*past[:, s, None]
    gap = hold[:, None] - z
    response = -np.expm1(-times[:, None] / tau)
    # Damage gates the two extra terms, not the shared plateau basis.
    terminal = np.broadcast_to(np.log1p(times), (len(hold), len(times)))
    drift = np.column_stack((np.ones(len(hold)), features))
    h, m = act*hold, act*z[:, 0]
    curves = np.concatenate((gap[:, None, :]*response,
                             drift[:, None, :]*terminal[:, :, None],
                             (h[:, None]*response[:, 0])[:, :, None],
                             (m[:, None]*terminal)[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, drift, h/tau[0], m))
    return curves, rates


def _fit(args, strain, rate):
    use_curve, use_rate = np.isfinite(strain), np.isfinite(rate)
    y = np.r_[strain[use_curve], rate[use_rate]]
    ncurve = use_curve.sum()
    lower, upper = np.log([.035, .035]), np.log([50., 50.])

    def project(p, weight):
        c, r = _design(p, *args)
        a = np.concatenate((c[use_curve], r[use_rate]))
        aw = a*weight[:, None]
        coef = np.linalg.lstsq(aw, y*weight, rcond=1e-10)[0]
        return np.einsum('ij,j->i', a, coef) - y, coef

    # Search in both branch orders: the crystallization term singles out branch 0.
    base = np.r_[np.ones(ncurve), np.full(len(y) - ncurve, .35)]
    def optimize(start, weight):
        return least_squares(lambda p: project(p, weight)[0]*weight, start,
                             bounds=(lower, upper), diff_step=1e-4, max_nfev=100)

    starts = ((.35, 3.), (.9, 7.), (3., .35), (7., .9))
    initial = [optimize(np.log(start), base) for start in starts]
    best = min(initial, key=lambda fit: fit.cost)
    p = best.x
    weight = base.copy()
    for _ in range(5):
        residual, _ = project(p, weight)
        for sl in (slice(None, ncurve), slice(ncurve, None)):
            e = residual[sl]
            if e.size:
                center = np.median(e)
                scale = max(1e-7, 1.4826*np.median(np.abs(e - center)))
                weight[sl] = np.minimum(1., 2.5*scale / np.maximum(1e-12, np.abs(e - center)))**.5 / scale
        p = optimize(p, weight).x
    _, coef = project(p, weight)
    return p, coef


def solve_alpha_m97_c03_batch_07(dataset: dict) -> dict:
    past, hold, features, act, offset, dt, times = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    measured = np.isfinite(train['coupon_features_raw']).all(axis=1)
    args = tuple(a[:n][measured] for a in (past, hold, features, act)) + (dt, times)
    p, w = _fit(args, (train['strain_curves_obs'] - offset[:n, None])[measured],
                train['instant_rate_obs'][measured])
    curve, rate = _design(p, past[n:], hold[n:], features[n:], act[n:], dt, times)
    return {'estimated_taus': np.exp(p), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': offset[n:, None] + curve @ w,
            'initial_rate_pred': rate @ w}
