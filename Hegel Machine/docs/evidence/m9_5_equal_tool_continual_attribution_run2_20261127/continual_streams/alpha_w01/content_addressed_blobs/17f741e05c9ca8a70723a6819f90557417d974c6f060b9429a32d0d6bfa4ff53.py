import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _checks(panel, lab):
    x = np.asarray(panel['cmd_stress'], float)
    y = np.asarray(panel['true_stress'], float)
    mask = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _consensus(x, y, prior):
    i, j = np.triu_indices(len(x), 1)
    good = np.abs(x[j] - x[i]) >= .05
    i, j = i[good], j[good]
    g = (y[j] - y[i]) / (x[j] - x[i])
    b = y[i] - g * x[i]
    lines = np.vstack((prior, np.column_stack((g[g > 0], b[g > 0]))))
    errors = np.abs(lines[:, :1] * x + lines[:, 1:] - y)
    clean = errors < .001
    counts = clean.sum(axis=1)
    cost = np.linalg.norm((lines - prior) * [max(1., np.ptp(x)) if len(x) else 1., 1.], axis=1)
    best = np.lexsort((cost, -counts))[0]
    mask = clean[best]
    if counts[best] >= max(3, len(x)//2 + 1) and np.ptp(x[mask]) >= .1:
        return np.polyfit(x[mask], y[mask], 1), True
    return lines[best], False


def _load_cells(panel):
    nominal = np.array([1., 0.])
    history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.exists() else []
    maps = []
    for lab in range(4):
        x, y = _checks(panel, lab)
        current, identified = _consensus(x, y, nominal)
        if identified:
            maps.append(current)
            continue
        prior = None
        accepted_x, accepted_y = [], []
        for previous in reversed(history):
            hx, hy = _checks(previous, lab)
            fit, reliable = _consensus(hx, hy, nominal)
            if not reliable:
                continue
            if prior is None:
                prior = fit
            elif np.max(np.abs((fit[0] - prior[0])*np.array([-3., 3.]) + fit[1] - prior[1])) > .001:
                break
            good = np.abs(hy - np.polyval(prior, hx)) < .001
            accepted_x.append(hx[good])
            accepted_y.append(hy[good])
        if prior is not None:
            good = np.abs(y - np.polyval(prior, x)) < .001
            # One discordant spot may be EMI; two separated discordant spots
            # support a new map when they are the only available measurements.
            if len(x) == 2 and not good.any() and np.ptp(x) >= .5:
                gain = (y[1] - y[0]) / (x[1] - x[0])
                if gain > 0:
                    maps.append(np.array([gain, y[0] - gain*x[0]]))
                    continue
            current = np.polyfit(np.concatenate(accepted_x + [x[good]]),
                                 np.concatenate(accepted_y + [y[good]]), 1)
        elif len(x) == 1:
            current = np.array([1., y[0] - x[0]])
        maps.append(current)
    return np.array(maps)


def _alignment(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = refs[mask]
        panel = panel[:, (np.isfinite(panel).sum(axis=0) >= 2)]
        if panel.shape[1]:
            center = np.median(np.nanmean(panel, axis=0))
            scale = np.median(np.nanstd(panel, axis=0, ddof=0))
            raw[mask] = (raw[mask] - center) / (scale if scale > 0 else 1.)
    _, cells = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    finite = np.isfinite(raw)
    counts = np.zeros((cells.max()+1, raw.shape[1]))
    sums = np.zeros_like(counts)
    np.add.at(counts, cells, finite)
    np.add.at(sums, cells, np.where(finite, raw, 0.))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / np.maximum(1, (counts > 0).sum(axis=0))
    aligned = np.where(finite, raw, fill)
    return np.split(aligned, [len(train['lab_id'])])


def _inputs(block, aligned, maps, dt, yield_stress):
    gain, bias = maps[np.asarray(block['lab_id'], int)].T
    hist = block['past_stress_cmd']*gain[:, None] + bias[:, None]
    hold = block['stress_hold_cmd']*gain + bias
    ov_hold = np.maximum(np.abs(hold) - yield_stress, 0.)**2 * np.sign(hold)
    work_hold = (np.maximum(np.abs(hist) - yield_stress, 0.)**2 @ dt) * hold
    return hist, hold, np.column_stack((np.ones(len(hold)), aligned)), ov_hold, work_hold


def _design(taus, inputs, dt, time):
    hist, hold, cov, ov_hold, work_hold = inputs
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-tail[:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    jump = hold[:, None] - np.einsum('ij,jk->ik', hist, memory)
    decay = -np.expm1(-time[:, None] / taus)
    curves = np.concatenate((jump[:, None, :] * decay,
                             cov[:, None, :] * time[None, :, None],
                             (ov_hold[:, None] * time)[:, :, None],
                             (work_hold[:, None] * decay[:, 0])[:, :, None]), axis=2)
    rate = np.column_stack((jump / taus, cov, ov_hold, work_hold / taus[0]))
    return curves, rate


def _fit(train, inputs, dt, time):
    eps = train['strain_curves_obs'] - train['strain_0'][:, None]
    y = np.r_[eps.ravel(), train['instant_rate_obs']]
    use = np.r_[np.ones(7, dtype=bool), [np.any(v) for v in inputs[-2:]]]

    def unpack(q):
        t1 = np.exp(q[0])
        return np.array([t1, t1 + (24.99-t1)*q[1]])

    def design(q):
        curve, rate = _design(unpack(q), inputs, dt, time)
        return np.vstack((curve.reshape(-1, 9), rate))[:, use]

    def project(q):
        a = design(q)
        w = lstsq(a, y, lapack_driver='gelsy')[0]
        return (a * w).sum(axis=1) - y

    lo, hi = [np.log(.05001), 1e-5], [np.log(24.98), 1.]
    fits = [least_squares(project, [np.log(a), (b-a)/(24.99-a)], bounds=(lo, hi),
                          loss='soft_l1', f_scale=.05, max_nfev=60)
            for a, b in ((.27, 1.4), (.55, 4.), (1.2, 8.))]
    q = min(fits, key=lambda result: result.cost).x
    p = np.r_[q, lstsq(design(q), y, lapack_driver='gelsy')[0]]
    bounds = (np.r_[lo, np.full(use.sum(), -np.inf)], np.r_[hi, np.full(use.sum(), np.inf)])

    def residual(p):
        return (design(p[:2]) * p[2:]).sum(axis=1) - y

    for _ in range(2):
        e = residual(p)
        a, b = e[:eps.size], e[eps.size:]
        sa, sb = (max(1e-7, 1.4826*np.median(np.abs(v-np.median(v)))) for v in (a, b))
        scale = np.r_[np.full(a.size, 2.5*sa), np.full(b.size, 2.5*sb)]
        p = least_squares(lambda v: residual(v)/scale, p, bounds=bounds,
                          loss='cauchy', max_nfev=80).x
    w = np.zeros(9)
    w[use] = p[2:]
    return unpack(p[:2]), w


def solve_alpha_w01_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    ys = dataset['yield_stress']
    maps = _load_cells(dataset['calibration_load_cells'])
    x, xt = _alignment(train, test)
    inputs = _inputs(train, x, maps, dt, ys)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    if complete.sum() >= 12 and all(not np.any(v) or np.any(v[complete]) for v in inputs[-2:]):
        train = {key: np.asarray(train[key])[complete] for key in ('strain_curves_obs', 'strain_0', 'instant_rate_obs')}
        inputs = tuple(v[complete] for v in inputs)
    taus, w = _fit(train, inputs, dt, time)
    a, b = _design(taus, _inputs(test, xt, maps, dt, ys), dt, time)
    rate = b @ w
    curve = test['strain_0'][:, None] + a @ w
    # Only missing covariates introduce an unknown coupon-specific linear drift.
    # An available onset observation updates that drift, not either memory branch.
    if 'instant_rate_obs' in test:
        missing = ~np.isfinite(test['coupon_features_raw'])
        observed = np.asarray(test['instant_rate_obs'])
        informed = missing.any(axis=1) & np.isfinite(observed)
        if informed.any():
            _, train_rate = _design(taus, inputs, dt, time)
            err = np.asarray(train['instant_rate_obs']) - train_rate @ w
            noise_var = (1.4826 * np.median(np.abs(err - np.median(err))))**2
            latent_drift = x @ (missing[informed] * w[3:7]).T
            drift_var = np.var(latent_drift, axis=0)
            fraction = np.divide(drift_var, drift_var + noise_var,
                                 out=np.zeros_like(drift_var), where=drift_var + noise_var > 0)
            delta = fraction * (observed[informed] - rate[informed])
            curve[informed] += delta[:, None] * time
            rate[informed] += delta
    return {'estimated_taus': taus, 'hereditary_weights': w[:7], 'viscoplastic_weights': w[7:],
            'predicted_strain': curve, 'initial_rate_pred': rate}
