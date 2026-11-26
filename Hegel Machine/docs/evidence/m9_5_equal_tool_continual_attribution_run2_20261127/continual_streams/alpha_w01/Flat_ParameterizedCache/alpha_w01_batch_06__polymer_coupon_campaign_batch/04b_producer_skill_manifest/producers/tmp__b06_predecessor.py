"""Calibrated two-branch creep model with hold and history overstress."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
# Batch 03 load-cell fits, retained for standalone use if the sweep is not mounted.
_PRIOR = np.array([[.75358184, -.13970320], [1.23778990, .09426679],
                   [.70754455, -.11031877], [1.28023218, .13347379]])


def _checks(panel, lab):
    x = np.asarray(panel['cmd_stress'], float)
    y = np.asarray(panel['true_stress'], float)
    ok = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(panel):
    history = json.loads(_ARCHIVE.read_text())[-1] if _ARCHIVE.is_file() else None
    maps = _PRIOR.copy()
    for lab, prior in enumerate(_PRIOR):
        x, y = _checks(panel, lab)
        a, b = np.triu_indices(len(x), 1)
        dx = x[b] - x[a]
        good = np.abs(dx) >= .10
        a, b, dx = a[good], b[good], dx[good]
        gains = (y[b] - y[a]) / dx
        fits = np.vstack((prior, np.column_stack((gains, y[a] - gains*x[a]))))
        er = np.abs(y - fits[:, :1]*x - fits[:, 1:])
        clean = er < .001
        n = clean.sum(1)
        score = np.sum(np.where(clean, er**2, 0), axis=1)
        best = np.lexsort((score, -n))[0]
        keep = clean[best]
        agrees = np.abs(y - prior[0]*x - prior[1]) < .001
        if n[best] >= 3 and np.ptp(x[keep]) >= .10:
            local = _line(x[keep], y[keep])
            # Pool only checks that support the same calibration regime.
            if np.max(np.abs(local - prior)) > .001:
                maps[lab] = local
                continue
            agrees = keep
        if history is not None:
            hx, hy = _checks(history, lab)
            hist_ok = np.abs(hy - prior[0]*hx - prior[1]) < .001
            xx, yy = np.r_[hx[hist_ok], x[agrees]], np.r_[hy[hist_ok], y[agrees]]
            if len(xx) >= 2 and np.ptp(xx) >= .10:
                maps[lab] = _line(xx, yy)
        elif n[best] >= 3 and np.ptp(x[keep]) >= .10:
            maps[lab] = local
        elif np.any(agrees):
            maps[lab, 1] = np.median(y[agrees] - prior[0]*x[agrees])
    return maps


def _align(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    complete = np.isfinite(raw).all(axis=1)
    for lab in np.unique(labs):
        m = labs == lab
        p = refs[m]
        p = p[:, np.isfinite(p).sum(axis=0) >= 2]
        raw[m] = (raw[m] - np.median(np.nanmean(p, axis=0))) / np.median(np.nanstd(p, axis=0))
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    valid = np.isfinite(raw)
    counts = np.zeros((cell.max()+1, raw.shape[1]))
    sums = np.zeros_like(counts)
    np.add.at(counts, cell, valid)
    np.add.at(sums, cell, np.where(valid, raw, 0.))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / np.maximum(1, (counts > 0).sum(axis=0))
    x = np.column_stack((np.ones(len(raw)), np.where(valid, raw, fill)))
    n = len(train['lab_id'])
    return x[:n], x[n:], complete[:n], complete[n:]


def _stress(block, maps):
    gain, bias = maps[np.asarray(block['lab_id'], int)].T
    return block['past_stress_cmd'] * gain[:, None] + bias[:, None], block['stress_hold_cmd'] * gain + bias


def _plastic(hist, hold, dt, yield_stress):
    ov = np.maximum(np.abs(hold) - yield_stress, 0.)**2 * np.sign(hold)
    work_hold = (np.maximum(np.abs(hist) - yield_stress, 0.)**2 @ dt) * hold
    return ov, work_hold


def _design(taus, hist, hold, cov, dt, time, ov, work_hold):
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = np.exp(-tail[:, None]/taus) * -np.expm1(-dt[:, None]/taus)
    jump = hold[:, None] - hist @ memory
    response = -np.expm1(-time[:, None]/taus)
    curve = np.concatenate((jump[:, None]*response, cov[:, None]*time[None, :, None],
                            (ov[:, None]*time)[..., None],
                            (work_hold[:, None]*response[:, 0])[..., None]), axis=2)
    rate = np.column_stack((jump/taus, cov, ov, work_hold/taus[0]))
    return curve, rate


def _mad(e):
    return max(1e-8, 1.4826*np.median(np.abs(e-np.median(e))))


def _fit(train, x, complete, maps, dt, time, sy):
    hist, hold = _stress(train, maps)
    ov, work = _plastic(hist, hold, dt, sy)
    # The onset-rate observation identifies missing linear drift at prediction;
    # estimate shared coefficients from measured covariates when possible.
    if complete.sum() < 12:
        complete = np.ones_like(complete)
    data = (hist[complete], hold[complete], x[complete], dt, time, ov[complete], work[complete])
    dy = (train['strain_curves_obs'] - train['strain_0'][:, None])[complete].ravel()
    rates = train['instant_rate_obs'][complete]
    yfull = np.r_[dy, rates]
    valid = np.isfinite(yfull)
    # Rate observations form a separate channel from each strain horizon.
    channel = np.r_[np.tile(np.arange(len(time)), len(rates)),
                    np.full(len(rates), len(time))][valid]
    y = yfull[valid]
    active = np.r_[np.ones(7, bool), np.any(data[-2]), np.any(data[-1])]
    scale = np.ones_like(y)
    robust = np.ones_like(y)

    def unpack(p):
        fast = np.exp(p[0]); slow = fast + (24.999-fast)*p[1]
        return np.array([fast, slow])

    def profile(p, result=False):
        tau = unpack(p)
        curve, rate = _design(tau, *data)
        A = np.concatenate((curve.reshape(-1, 9), rate), axis=0)[valid][:, active]
        w = robust/scale
        coeff = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
        err = np.einsum('ij,j->i', A, coeff) - y
        if result:
            full = np.zeros(9)
            full[active] = coeff
            return tau, full, err
        return err*w

    bounds = ([np.log(.05001), 1e-5], [np.log(24.), .99999])
    starts = [(a, (b-a)/(24.999-a)) for a, b in ((.2, 1.5), (.6, 4.), (1.5, 9.), (3., 16.))]
    opts = [least_squares(profile, [np.log(a), b], bounds=bounds, max_nfev=60) for a, b in starts]
    p = min(opts, key=lambda fit: fit.cost).x
    e = profile(p, True)[2]
    for j in range(len(time) + 1):
        m = channel == j
        if np.any(m):
            scale[m] = _mad(e[m])
    for _ in range(6):
        robust[:] = 1/np.sqrt(1 + (e/(2.5*scale))**2)
        p = least_squares(profile, p, bounds=bounds, max_nfev=60).x
        taus, theta, e = profile(p, True)
    return taus, theta


def solve_alpha_w01_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], float)
    time = np.asarray(dataset['eval_times'], float)
    sy = float(dataset['yield_stress'])
    maps = _calibrate(dataset['calibration_load_cells'])
    x, xt, complete, complete_test = _align(train, test)
    taus, theta = _fit(train, x, complete, maps, dt, time, sy)
    hist, hold = _stress(test, maps)
    ov, work = _plastic(hist, hold, dt, sy)
    A, B = _design(taus, hist, hold, xt, dt, time, ov, work)
    rate = B @ theta
    observed = test['instant_rate_obs']
    drift_delta = np.where(~complete_test & np.isfinite(observed), observed-rate, 0.)
    return {'estimated_taus': taus, 'hereditary_weights': theta[:7],
            'viscoplastic_weights': theta[7:], 'initial_rate_pred': rate + drift_delta,
            'predicted_strain': test['strain_0'][:, None] + A @ theta + drift_delta[:, None]*time}
