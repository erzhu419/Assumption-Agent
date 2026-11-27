"""Load-cell calibrated, two-branch jammed-slip creep solver."""
import json

import numpy as np
from scipy.optimize import least_squares


def _design(logtau, past, hold, features, dt, times):
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    decay = np.exp(-dt[:, None] / tau)
    increment = -np.expm1(-dt[:, None] / tau)
    for s, (a, b) in enumerate(zip(decay, increment)):
        z = a*z + b*past[:, s, None]
    gap = hold[:, None] - z
    gap_diff = gap[:, 0] - gap[:, 1]
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    responses = -np.expm1(-times[:, None]/tau)
    terminal = times/(1 + 0.85*times)
    drift_slip = np.column_stack((np.ones(len(hold)), features, slip))
    curve = np.concatenate((gap[:, None, :]*responses[None, :, :],
                            drift_slip[:, None, :]*terminal[None, :, None],
                            (gap_diff[:, None]*responses[None, :, 0])[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, drift_slip, gap_diff/tau[0]))
    return curve, rates


def _fit(past, hold, features, dt, times, strain, rate):
    use_curve = np.isfinite(strain)
    use_rate = np.isfinite(rate)
    y = np.r_[strain[use_curve], rate[use_rate]]
    ncurve = np.count_nonzero(use_curve)
    bounds = np.log([[0.025, 0.025], [60., 60.]])

    def design(p):
        curves, rates = _design(p, past, hold, features, dt, times)
        return np.concatenate((curves[use_curve], rates[use_rate]))

    def project(p, weights, robust=False):
        a = design(p)
        w = weights
        for _ in range(7 if robust else 1):
            coef = np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]
            e = a@coef-y
            if robust:
                w = weights*(1 + (e*weights/2.5)**2)**(-0.25)
        if robust:
            u = e*weights/2.5
            return (e*weights)*np.sqrt(2/(np.hypot(1, u)+1)), coef
        return e, coef

    plain = np.ones_like(y)
    def objective(p):
        return project(p, plain)[0]
    fits = [least_squares(objective, np.log(start), bounds=bounds, diff_step=1e-4)
            for start in ((0.4, 3.), (1., 8.), (0.2, 1.5))]
    best = min((f for f in fits if f.x[0] < f.x[1]), key=lambda f: f.cost)
    residual = objective(best.x)
    def scale(group):
        return max(1e-6, 1.4826*np.median(np.abs(group - np.median(group)))) if len(group) else 1.
    weights = np.r_[np.full(ncurve, 1/scale(residual[:ncurve])),
                    np.full(len(y)-ncurve, 1/scale(residual[ncurve:]))]
    result = least_squares(lambda p: project(p, weights, True)[0], best.x,
                           bounds=bounds, diff_step=1e-4, max_nfev=100)
    p = result.x if result.x[0] < result.x[1] else best.x
    return p, project(p, weights, True)[1]
