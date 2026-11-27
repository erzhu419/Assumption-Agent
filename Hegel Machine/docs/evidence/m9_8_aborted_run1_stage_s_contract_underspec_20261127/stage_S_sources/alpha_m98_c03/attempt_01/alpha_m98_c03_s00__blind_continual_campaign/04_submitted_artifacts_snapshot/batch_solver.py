"""Calibrate axial load and fit a two-branch viscoelastic hold response."""

import numpy as np
from scipy.optimize import least_squares


def _calibrate(c):
    loads, raw = c['true_channel_loads'], c['bridge_raw']
    bridge = np.linalg.lstsq(np.c_[loads, np.ones(len(loads))], raw, rcond=None)[0]
    return np.linalg.solve(bridge[:4], np.ones(4)), bridge[4]


def _panels(*samples):
    lab = np.concatenate([s['lab_id'] for s in samples])
    refs = np.concatenate([s['ref_coupons'] for s in samples])
    return {i: (np.median(refs[lab == i].mean(axis=0)),
                np.median(refs[lab == i].std(axis=0))) for i in np.unique(lab)}


def _standardize(s, panels):
    x = np.asarray(s['coupon_features_raw'], dtype=float).copy()
    for i in np.unique(s['lab_id']):
        loc, scale = panels[i]
        rows = s['lab_id'] == i
        x[rows] = (x[rows] - loc) / (scale if scale > 0 else 1.0)
    return x


def _impute(x, *samples):
    labs = np.concatenate([s['lab_id'] for s in samples])
    strata = np.concatenate([s['stratum'] for s in samples])
    _, il = np.unique(labs, return_inverse=True)
    _, it = np.unique(strata, return_inverse=True)
    _, ix, count = np.unique(np.c_[il, it], axis=0, return_inverse=True, return_counts=True)
    w = 1.0 / count[ix]
    denom = np.isfinite(x).T @ w
    return np.divide(np.nansum(x * w[:, None], axis=0), denom,
                     out=np.zeros(x.shape[1]), where=denom > 0)


def _design(x, fill):
    return np.c_[np.ones(len(x)), np.where(np.isfinite(x), x, fill)]


def _stress(s, calibration):
    total, zero = calibration
    return (s['past_channel_raw'] - zero) @ total, (s['hold_channel_raw'] - zero) @ total


def _kernel(tau, dt, t, u, h):
    age = np.cumsum(dt[::-1])[::-1] - dt
    past = np.exp(-age[:, None] / tau) * (-np.expm1(-dt[:, None] / tau))
    amplitude = h[:, None] - u @ past
    time = -np.expm1(-t[:, None] / tau)
    return np.concatenate((amplitude[:, None, :] * time, (amplitude / tau)[:, None, :]), axis=1)


def _fit_constitutive(s, stress, dt, t, design, noise, prior=None):
    u, h = stress
    y = np.c_[s['strain_curves_obs'] - s['strain_0'][:, None], s['instant_rate_obs']]
    z = np.r_[t, 1.0]
    scale = np.r_[np.full(len(t), noise), noise * .75]
    drift = design[:, None, :] * (z / scale)[None, :, None]
    target = (y / scale).ravel()

    def kernel(log_tau):
        return _kernel(np.exp(log_tau), dt, t, u, h) / scale[None, :, None]

    def residual(p):
        return (kernel(p[:2]) @ p[2:4] + drift @ p[4:]).ravel() - target

    def jacobian(p):
        a = kernel(p[:2])
        step = 1e-5
        da = (kernel(p[:2] + step) - kernel(p[:2] - step)) / (2 * step)
        return np.concatenate((da * p[2:4], a, drift), axis=2).reshape(-1, len(p))

    lower = np.r_[np.log([.050001] * 2), np.zeros(2), np.full(5, -np.inf)]
    upper = np.r_[np.log([24.9999] * 2), np.full(7, np.inf)]
    starts = [] if prior is None else [np.r_[np.log(prior[0]), prior[1], prior[2]]]
    for pair in ((.45, 3.), (.65, 6.), (.2, 12.), (1.5, 17.)):
        log_tau = np.log(pair)
        a = np.concatenate((kernel(log_tau), drift), axis=2).reshape(-1, 7)
        coef = np.linalg.lstsq(a, target, rcond=None)[0]
        coef[:2] = np.maximum(coef[:2], .01)
        starts.append(np.r_[log_tau, coef])
    fits = [least_squares(residual, p, jac=jacobian, bounds=(lower, upper),
                          loss='cauchy', max_nfev=180) for p in starts]
    p = min(fits, key=lambda fit: fit.cost).x
    ix = np.argsort(p[:2])
    return np.exp(p[:2][ix]), p[2:4][ix], p[4:]


def _complete(s, x):
    mask = np.isfinite(x).all(axis=1)
    if np.linalg.matrix_rank(np.c_[np.ones(mask.sum()), x[mask]]) == x.shape[1] + 1:
        return {key: value[mask] for key, value in s.items()}, x[mask]
    return s, x


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    dt = np.asarray(dataset['past_dt'])
    t = np.asarray(dataset['eval_times'])
    cal = _calibrate(dataset['calibration_load_cells'])

    panels = _panels(train, test)
    x_train, x_test = (_standardize(s, panels) for s in (train, test))
    fill = _impute(np.concatenate((x_train, x_test)), train, test)
    fit = _fit_constitutive(train, _stress(train, cal), dt, t, _design(x_train, fill), .012)

    if pilot is not None and len(pilot['strain_0']):
        x = _standardize(pilot, _panels(pilot))
        # Do not let a new panel's poorly determined missing fillers bias the spectrum.
        source, x_complete = _complete(pilot, x)
        design = _design(x_complete, fill)
        if np.linalg.matrix_rank(design) == design.shape[1]:
            fit = _fit_constitutive(source, _stress(source, cal), dt, t, design, .006, fit)

    tau, weights, drift = fit
    transient = _kernel(tau, dt, t, *_stress(test, cal)) @ weights
    rate = _design(x_test, fill) @ drift
    return {'estimated_taus': tau, 'hereditary_weights': np.r_[weights, drift],
            'predicted_strain': test['strain_0'][:, None] + transient[:, :-1] + rate[:, None] * t,
            'initial_rate_pred': transient[:, -1] + rate}
