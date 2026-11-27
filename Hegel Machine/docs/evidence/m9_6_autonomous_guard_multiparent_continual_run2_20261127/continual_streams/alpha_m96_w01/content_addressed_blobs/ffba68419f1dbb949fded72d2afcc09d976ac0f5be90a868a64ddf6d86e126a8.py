import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _checks(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    mask = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibration(sweep):
    # Normal sweep noise is about 1e-4 stress units. The inlier band
    # allows for that noise without letting isolated EMI determine a line.
    band = .001
    coeff = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _checks(sweep, lab)
        if len(x) >= 3 and np.ptp(x) >= .25:
            b = _line(x, y)
            if np.max(np.abs(y - (b[0]*x + b[1]))) <= band:
                coeff[lab] = b
                continue
        if history is None:
            history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.is_file() else []
        prior = np.array([1., 0.])
        have_prior = False
        for previous in reversed(history):
            xp, yp = _checks(previous, lab)
            if len(xp) >= 2 and np.ptp(xp) >= .25:
                prior = _line(xp, yp)
                have_prior = True
                break
        proposals = [prior]
        for i, j in zip(*np.triu_indices(len(x), 1)):
            if abs(x[j] - x[i]) >= .25:
                gain = (y[j] - y[i]) / (x[j] - x[i])
                if gain > 0:
                    proposals.append(np.array([gain, y[i] - gain*x[i]]))
        errors = np.abs(y[None, :] - (np.asarray(proposals)[:, :1]*x + np.asarray(proposals)[:, 1:]))
        masks = errors <= band
        counts = masks.sum(axis=1)
        best = int(np.argmax(counts))
        if have_prior:
            # Resolve equal consensus by proximity to the archived affine law.
            delta = np.asarray(proposals) - prior
            proximity = 4. * delta[:, 0]**2 + delta[:, 1]**2
            best = int(np.lexsort((proximity, -counts))[0])
            # Two points cannot corroborate a large drift when only one
            # disagrees with the preceding sweep. Keep the archived gain.
            if counts[best] == 2 and counts[0] == 1 and np.max(errors[0, masks[best]]) > .02:
                best = 0
        m = masks[best]
        if m.sum() >= 2 and np.ptp(x[m]) >= .25:
            coeff[lab] = _line(x[m], y[m])
        else:
            coeff[lab] = prior
            if m.any():
                coeff[lab, 1] = np.mean(y[m] - prior[0]*x[m])
    return coeff


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.empty_like(raw, dtype=float)
    for j in np.unique(lab):
        m = lab == j
        r = ref[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        shift = np.median(np.nanmean(r, axis=0))
        scale = np.median(np.nanstd(r, axis=0, ddof=0))
        x[m] = (raw[m] - shift) / scale
    missing = ~np.isfinite(x)
    _, cells, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    w = 1. / count[cells.ravel()]
    have = (~missing).T @ w
    means = np.divide(np.where(missing, 0., x).T @ w, have,
                      out=np.zeros(x.shape[1]), where=have > 0)
    return np.column_stack((np.ones(len(x)), np.where(missing, means, x))), missing


def _stresses(split, coeff):
    gain, bias = coeff[np.asarray(split['lab_id'])].T
    return (np.asarray(split['past_stress_cmd'])*gain[:, None] + bias[:, None],
            np.asarray(split['stress_hold_cmd'])*gain + bias)


def _basis(past, hold, dt, x, sy):
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    flow = np.maximum(np.abs(hold) - sy, 0.)**2 * np.sign(hold)
    work = (np.maximum(np.abs(past) - sy, 0.)**2 @ dt) * hold
    return past, hold, dt, trailing, x, flow, work


def _design(tau, data, t):
    past, hold, dt, trailing, x, flow, work = data
    kernel = np.exp(-trailing[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    delta = hold[:, None] - past @ kernel
    relax = -np.expm1(-t[:, None] / tau)
    a = np.concatenate((delta[:, None, :] * relax,
                        x[:, None, :] * t[None, :, None],
                        flow[:, None, None] * t[None, :, None],
                        work[:, None, None] * relax[None, :, :1]), axis=2)
    r = np.column_stack((delta / tau, x, flow, work / tau[0]))
    return a, r


def _fit(past, hold, dt, t, x, sy, y, rate):
    data = _basis(past, hold, dt, x, sy)
    cm, rm = np.isfinite(y), np.isfinite(rate)
    obs = np.r_[y[cm], rate[rm]]
    use = np.r_[np.ones(7, dtype=bool), np.any(data[-2]), np.any(data[-1])]
    def matrix(tau):
        a, r = _design(tau, data, t)
        return np.vstack((a[cm], r[rm]))[:, use]

    grid = (.22, .38, .64, 1., 1.65, 2.7, 4.5, 7.5, 12.)
    guesses = []
    for i, u in enumerate(grid[:-1]):
        for v in grid[i+1:]:
            tau = [u, v]
            mat = matrix(tau)
            theta = np.linalg.lstsq(mat, obs, rcond=None)[0]
            errors = obs - mat @ theta
            cap = max(.002, 2.5*np.median(abs(errors)))
            w = np.sqrt(np.minimum(1., cap/np.maximum(abs(errors), 1e-12)))
            theta = np.linalg.lstsq(mat*w[:, None], obs*w, rcond=None)[0]
            errors = obs - mat @ theta
            guesses.append((np.sum(np.log1p((errors/.008)**2)), np.r_[np.log(tau), theta]))

    def residual(q):
        return matrix(np.exp(q[:2])) @ q[2:] - obs

    bounds = (np.r_[np.log([.051, .051]), np.full(use.sum(), -np.inf)],
              np.r_[np.log([24.9, 24.9]), np.full(use.sum(), np.inf)])
    starts = sorted(guesses, key=lambda pair: pair[0])[:3]
    fits = [least_squares(residual, q, bounds=bounds, loss='soft_l1', f_scale=.006,
                          x_scale='jac', max_nfev=100) for _, q in starts]
    fit = min((f for f in fits if f.x[0] < f.x[1]), key=lambda f: f.cost)
    e = residual(fit.x)
    ec, er = e[:cm.sum()], e[cm.sum():]
    def scale(z):
        return max(.001, 2.5*1.4826*np.median(abs(z - np.median(z)))) if z.size else .006
    scales = np.r_[np.full(ec.size, scale(ec)), np.full(er.size, scale(er))]
    refined = least_squares(lambda q: residual(q) / scales, fit.x, bounds=bounds,
                            loss='cauchy', x_scale='jac', max_nfev=100)
    q = refined.x if refined.x[0] < refined.x[1] else fit.x
    theta = np.zeros(9)
    theta[use] = q[2:]
    return np.exp(q[:2]), theta


def solve_alpha_m96_w01_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    coeff = _calibration(dataset['calibration_load_cells'])
    x, missing = _features(train, test)
    dt, t = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    sy = dataset['yield_stress']
    n = len(train['lab_id'])
    keep = ~missing[:n].any(axis=1)
    past, hold = _stresses(train, coeff)
    y = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    rate = np.asarray(train['instant_rate_obs'])
    tau, theta = _fit(past[keep], hold[keep], dt, t, x[:n][keep], sy, y[keep], rate[keep])
    past, hold = _stresses(test, coeff)
    a, r = _design(tau, _basis(past, hold, dt, x[n:], sy), t)
    curve = np.asarray(test['strain_0'])[:, None] + a @ theta
    rate = r @ theta
    # Missing features affect only the linear slope; the measured onset
    # supplies an independent estimate of that slope for those specimens.
    onset = np.asarray(test['instant_rate_obs'])
    update = missing[n:].any(axis=1) & np.isfinite(onset)
    curve[update] += (onset[update] - rate[update])[:, None] * t
    rate[update] = onset[update]
    return dict(estimated_taus=tau, hereditary_weights=theta[:7], viscoplastic_weights=theta[7:],
                predicted_strain=curve, initial_rate_pred=rate)
