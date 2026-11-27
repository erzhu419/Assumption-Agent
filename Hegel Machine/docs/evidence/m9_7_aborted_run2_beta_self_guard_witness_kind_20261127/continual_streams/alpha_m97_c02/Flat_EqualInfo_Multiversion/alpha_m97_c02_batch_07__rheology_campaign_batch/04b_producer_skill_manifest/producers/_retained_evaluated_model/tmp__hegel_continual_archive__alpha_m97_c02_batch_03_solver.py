"""Fit the two-branch jammed-slip rheology law to measured strain and rate."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(data):
    c = data['calibration_load_cells']
    labs, cmd, true = (np.asarray(c[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    history = None

    def prior(lab):
        nonlocal history
        if history is None:
            with open(_ARCHIVE) as f:
                history = json.load(f)
        for sweep in reversed(history):
            keep = np.asarray(sweep['lab_id']) == lab
            if keep.sum() >= 2:
                return _line(np.asarray(sweep['cmd_stress'])[keep], np.asarray(sweep['true_stress'])[keep])
        raise ValueError(f'No archived calibration for lab {lab}')

    for lab in range(4):
        keep = (labs == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[keep], true[keep]
        archived = None
        if len(x) >= 4:
            dx, dy = x[:, None] - x, y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            off = y - gain*x
            if np.any(abs(off - np.median(off)) > 3.0):
                archived = prior(lab)
                keep = abs(off - np.median(off)) <= 3.0
                x, y = x[keep], y[keep]
        elif len(x) == 3:
            # The middle-point residual attenuates endpoint spikes. Screen
            # each point against the other two, then use the archived slope
            # to identify the spike without assuming which point is clean.
            for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
                if x[k] == x[j]:
                    continue
                prediction = y[j] + (y[k]-y[j])*(x[i]-x[j])/(x[k]-x[j])
                if abs(y[i] - prediction) > 3.0:
                    archived = prior(lab)
                    off = y - archived[0]*x
                    keep = abs(off - np.median(off)) <= 3.0
                    x, y = x[keep], y[keep]
                    break
        if len(x) >= 2:
            result[lab] = _line(x, y)
        else:
            archived = prior(lab) if archived is None else archived
            result[lab] = (archived[0], y[0] - archived[0]*x[0]) if len(x) else archived
    return result


def _column_mean(a):
    n = np.isfinite(a).sum(axis=0)
    return np.divide(np.nansum(a, axis=0), n, out=np.full(a.shape[1], np.nan), where=n > 0)


def _aligned(train, test):
    def join(k):
        return np.concatenate((train[k], test[k]), axis=0)
    lab, group, ref = (join(k) for k in ('lab_id', 'stratum', 'ref_coupons'))
    x = join('coupon_features_raw').astype(float)
    for l in np.unique(lab):
        rows = lab == l
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        x[rows] = (x[rows] - np.median(np.nanmean(panel, axis=0))) / np.median(np.nanstd(panel, axis=0, ddof=0))
    cells = {(l, s): _column_mean(x[(lab == l) & (group == s)])
             for l, s in set(zip(lab, group))}
    overall = _column_mean(np.array(list(cells.values())))
    for (l, s), mean in cells.items():
        rows = (lab == l) & (group == s)
        local = _column_mean(np.array([m for (ll, _), m in cells.items() if ll == l]))
        fill = np.where(np.isfinite(mean), mean, np.where(np.isfinite(local), local, overall))
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _inputs(split, x, calibration):
    gain, bias = calibration[split['lab_id']].T
    past = gain[:, None]*split['past_stress_cmd'] + bias[:, None]
    hold = gain*split['stress_hold_cmd'] + bias
    flow = np.column_stack((np.ones(len(x)), x, np.sign(hold)*np.sqrt(abs(hold))))
    return past, hold, flow


def _design(tau, dt, t, past, hold, flow):
    decay = np.exp(-dt[:, None]/tau)
    ent = -np.expm1(-dt[:, None]/tau)
    z = np.zeros((len(hold), 2))
    for stress, a, b in zip(past.T, decay, ent):
        z = a*z + b*stress[:, None]
    gap = hold[:, None] - z
    delta = gap[:, 0] - gap[:, 1]
    growth = -np.expm1(-t[:, None]/tau)
    phi = t/(1.0 + 0.85*t)
    a = np.empty((len(hold), len(t), 9))
    a[:, :, :2] = gap[:, None, :]*growth
    a[:, :, 2:8] = flow[:, None, :]*phi[None, :, None]
    a[:, :, 8] = delta[:, None]*growth[:, 0]
    b = np.column_stack((gap/tau, flow, delta/tau[0]))
    return a, b


def _fit(dt, t, inputs, curves, rates):
    n, k = curves.shape
    y = np.r_[curves.ravel(), rates]

    def design(logtau):
        a, b = _design(np.exp(logtau), dt, t, *inputs)
        return np.vstack((a.reshape(-1, 9), b))

    def scales(curve, rate):
        def mad(a, axis=None):
            return 1.4826*np.median(abs(a-np.median(a, axis=axis, keepdims=True)), axis=axis)
        return np.maximum(np.r_[np.tile(mad(curve, axis=0), n), np.full(n, mad(rate))], 1e-7)

    scale = scales(curves, rates)
    lower = np.r_[np.full(2, np.log(0.025)), np.full(9, -np.inf)]
    upper = np.r_[np.full(2, np.log(100.0)), np.full(9, np.inf)]

    def residual(p):
        return (design(p[:2]) @ p[2:] - y)/scale

    def jac(p):
        deriv = [((design(p[:2]+h)-design(p[:2]-h)) @ p[2:])/(2e-5)
                 for h in 1e-5*np.eye(2)]
        return np.column_stack((*deriv, design(p[:2])))/scale[:, None]

    def optimize(p, loss, f_scale):
        return least_squares(residual, p, jac=jac, bounds=(lower, upper), loss=loss,
                             f_scale=f_scale, x_scale='jac', max_nfev=120, ftol=1e-9)

    starts = []
    for seed in ((0.30, 1.5), (0.60, 3.5), (1.20, 7.0), (3.0, 0.6)):
        lt = np.log(seed)
        w = np.linalg.lstsq(design(lt)/scale[:, None], y/scale, rcond=None)[0]
        starts.append(optimize(np.r_[lt, w], 'soft_l1', 0.06))
    best = min(starts, key=lambda fit: fit.cost)
    for _ in range(2):
        r = residual(best.x)*scale
        scale = scales(r[:n*k].reshape(n, k), r[n*k:])
        best = optimize(best.x, 'cauchy', 2.0)
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    calibration = _calibration(dataset)
    x_train, x_test = _aligned(train, test)
    tau, coef = _fit(dt, t, _inputs(train, x_train, calibration),
                     train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs'])
    a, b = _design(tau, dt, t, *_inputs(test, x_test, calibration))
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:],
            'predicted_strain': test['strain_0'][:, None] + np.einsum('nkj,j->nk', a, coef),
            'initial_rate_pred': np.einsum('nj,j->n', b, coef)}
