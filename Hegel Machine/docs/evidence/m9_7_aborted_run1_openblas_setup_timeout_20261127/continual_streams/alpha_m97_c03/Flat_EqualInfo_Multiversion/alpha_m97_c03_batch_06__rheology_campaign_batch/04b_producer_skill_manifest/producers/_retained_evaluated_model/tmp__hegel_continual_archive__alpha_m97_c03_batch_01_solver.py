"""Calibrated two-branch relaxation and crystallization creep model."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep(cal, lab):
    x, y, c = (np.asarray(cal[k]) for k in ('cmd_stress', 'true_stress', 'lab_id'))
    take = (c == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibration(cal):
    fitted = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _sweep(cal, lab)
        clean = np.ones(len(x), dtype=bool)
        if np.unique(x).size >= 2:
            i, j = np.triu_indices(len(x), 1)
            take = x[i] != x[j]
            slope = np.median((y[j[take]] - y[i[take]]) / (x[j[take]] - x[i[take]]))
            clean = np.abs(y - slope*x - np.median(y - slope*x)) <= 3.0
        # With four readings, the six pair slopes split evenly between
        # clean pairs and pairs touching a single spike. Compare triplets.
        if len(x) == 4:
            best = None
            for omitted in range(4):
                keep = np.arange(4) != omitted
                if np.unique(x[keep]).size < 2:
                    continue
                gain, bias = _line(x[keep], y[keep])
                residual = np.abs(y - (gain*x + bias))
                score = np.linalg.norm(residual[keep])
                if best is None or score < best[0]:
                    best = score, residual, keep
            if best is not None:
                _, residual, keep = best
                clean = keep if residual[~keep].item() > 3.0 and np.all(residual[keep] <= 3.0) else np.ones(4, dtype=bool)
        xc, yc = x[clean], y[clean]
        if np.unique(xc).size >= 2:
            fitted[lab] = _line(xc, yc)
            continue
        if len(x) >= 2 and clean.all():
            raise ValueError(f'Load-cell sweep has no distinct commands for channel {lab}')
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for entry in reversed(history):
            hx, hy = _sweep(entry, lab)
            if np.unique(hx).size >= 2:
                fitted[lab] = _line(hx, hy)
                break
        else:
            raise ValueError(f'No usable load-cell calibration for channel {lab}')
    return fitted.T


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        m = labs == lab
        center = np.nanmedian(np.nanmean(panel[m], axis=0))
        spread = np.nanmedian(np.nanstd(panel[m], axis=0, ddof=0))
        aligned[m] = (raw[m] - center) / spread
    keys, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for i in range(len(keys)):
        values = aligned[ids == i]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(raw.shape[1], np.nan), where=counts > 0))
    means = np.asarray(means)
    for i, (lab, _) in enumerate(keys):
        for j in np.flatnonzero(~np.isfinite(means[i])):
            local = means[keys[:, 0] == lab, j]
            observed = local[np.isfinite(local)]
            if not observed.size:
                observed = means[np.isfinite(means[:, j]), j]
            means[i, j] = observed.mean() if observed.size else 0.
    return np.where(np.isfinite(aligned), aligned, means[ids])


def _inputs(part, aligned, gains, biases, dt, crit, plateau):
    g, b = gains[part['lab_id']], biases[part['lab_id']]
    stress = part['past_stress_cmd']*g[:, None] + b[:, None]
    hold = part['stress_hold_cmd']*g + b
    damage = np.square(np.maximum(0., np.abs(stress) - crit)) @ dt
    active = (damage > 0) & plateau
    act = np.where(active, np.tanh(damage), 0.)
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned)), plateau, act


def _design(logtau, inputs, dt, times):
    stress, hold, drift_inputs, plateau, act = inputs
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    before = after + dt
    n, k = len(hold), len(times)
    a = np.zeros((n, k + 1, 9))
    da = np.zeros((n, k + 1, 2, 9))
    flow = np.log1p(times) if plateau else times
    a[:, :-1, 2:7] = drift_inputs[:, None, :]*flow[None, :, None]
    a[:, -1, 2:7] = drift_inputs
    for j, tau in enumerate(np.exp(logtau)):
        u, v, t = after/tau, before/tau, times/tau
        e, f = np.exp(-u), np.exp(-v)
        z = stress @ (e * -np.expm1(-dt/tau))
        dz = stress @ (u*e - v*f)
        gap = hold - z
        phi = np.r_[-np.expm1(-t), 1/tau]
        dphi = np.r_[-t*np.exp(-t), -1/tau]
        a[:, :, j] = gap[:, None]*phi
        da[:, :, j, j] = gap[:, None]*dphi - dz[:, None]*phi
        if j == 0:
            logtime = np.r_[np.log1p(times), 1.]
            a[:, :, 7] = (act*hold)[:, None]*phi
            a[:, :, 8] = (act*z)[:, None]*logtime
            da[:, :, 0, 7] = (act*hold)[:, None]*dphi
            da[:, :, 0, 8] = (act*dz)[:, None]*logtime
    return a, da


def _fit(inputs, dt, times, y):
    # Two nonlinear time constants; the remaining coefficients enter linearly.
    # Cache the basis so residual and Jacobian calls share it.
    previous = None
    basis = None

    def design(logtau):
        nonlocal previous, basis
        if previous is None or not np.array_equal(logtau, previous):
            previous = logtau.copy()
            basis = _design(logtau, inputs, dt, times)
        return basis

    informative = np.any(inputs[-1] > 0)
    nw = 9 if informative else 7
    def optimize(p, scales, loss):
        def fun(q):
            a, _ = design(q[:2])
            return ((a[:, :, :nw] @ q[2:] - y)/scales).ravel()

        def jac(q):
            a, da = design(q[:2])
            j = np.concatenate((da[:, :, :, :nw] @ q[2:], a[:, :, :nw]), axis=2)
            return (j/scales[:, None]).reshape(-1, nw + 2)

        low = np.r_[np.log(max(0.005, min(dt.min(), times[times > 0].min())/20))*np.ones(2), np.full(nw, -np.inf)]
        high = np.r_[np.log(25*max(dt.sum(), times.max()))*np.ones(2), np.full(nw, np.inf)]
        return least_squares(fun, p, jac=jac, bounds=(low, high), loss=loss,
                             x_scale='jac', max_nfev=100)

    preliminary = np.full(y.shape[1], .06)
    preliminary[-1] = .1
    best = None
    for guess in ([.5, 4.], [1., 8.], [.2, 2.]):
        q = np.log(guess)
        a = design(q)[0][:, :, :nw]/preliminary[:, None]
        w = np.linalg.lstsq(a.reshape(-1, nw), (y/preliminary).ravel(), rcond=None)[0]
        fit = optimize(np.r_[q, w], preliminary, 'soft_l1')
        if (not informative or fit.x[0] < fit.x[1]) and (best is None or fit.cost < best.cost):
            best = fit
        if best is None:
            first = fit
    p = (first if best is None else best).x
    def mad(v):
        return max(1e-6, 1.4826*np.median(np.abs(v - np.median(v))))
    for _ in range(2):
        r = design(p[:2])[0][:, :, :nw] @ p[2:] - y
        scales = 2.385*np.r_[np.full(len(times), mad(r[:, :-1])), mad(r[:, -1])]
        candidate = optimize(p, scales, 'cauchy').x
        if informative and p[0] < p[1] and candidate[0] >= candidate[1]:
            break
        p = candidate
    if not informative and p[0] > p[1]:
        p = p[np.r_[1, 0, 3, 2, np.arange(4, len(p))]]
    return p[:2], np.pad(p[2:], (0, 9 - nw))


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gains, biases = _calibration(dataset['calibration_load_cells'])
    x = _align(tr, te)
    n = len(tr['lab_id'])
    settings = (gains, biases, dt, dataset['crit_stress'], dataset['shear_banding_index'] <= .50)
    y = np.column_stack((tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']))
    logtau, w = _fit(_inputs(tr, x[:n], *settings), dt, times, y)
    a, _ = _design(logtau, _inputs(te, x[n:], *settings), dt, times)
    pred = a @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w[:7],
            'regime_mode_weights': w[7:], 'predicted_strain': pred[:, :-1] + te['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
