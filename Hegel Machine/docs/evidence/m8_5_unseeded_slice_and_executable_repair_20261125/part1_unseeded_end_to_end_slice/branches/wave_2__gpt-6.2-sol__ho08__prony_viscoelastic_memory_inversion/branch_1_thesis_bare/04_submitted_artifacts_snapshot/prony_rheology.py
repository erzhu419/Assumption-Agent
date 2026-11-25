"""Robust identification of a passive Prony relaxation spectrum."""

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter

_TAU_MIN, _TAU_MAX = 0.02, 20.0
_MAD_TO_SIGMA = 1.4826


def _design(t, strain, tau):
    """Unit-modulus memory responses for piecewise linear strain."""
    response = np.zeros((len(t), len(tau)))
    if len(t) > 1 and len(tau):
        dt = np.diff(t)
        increments = np.diff(strain)
        if np.allclose(dt, dt[0], rtol=1e-10, atol=0):
            h = dt[0] / tau
            forcing = -np.expm1(-h) / h
            for j, (decay, gain) in enumerate(zip(np.exp(-h), forcing)):
                response[1:, j] = lfilter([gain], [1., -decay], increments)
        else:
            h = dt[:, None] / tau
            decay = np.exp(-h)
            forcing = -np.expm1(-h) / h * increments[:, None]
            for i in range(len(dt)):
                response[i + 1] = decay[i] * response[i] + forcing[i]
    return np.column_stack((strain, response))


def _residual_scale(residual, floor):
    center = np.median(residual)
    return center, max(floor, _MAD_TO_SIGMA * np.median(np.abs(residual - center)))


def _robust_grid_fit(design, drift, stress, floor):
    """Huber IRLS with the two unconstrained drift terms projected out."""
    weight = np.ones_like(stress)
    for _ in range(20):
        weighted_drift = drift * weight[:, None]
        q, r = np.linalg.qr(weighted_drift, mode='reduced')
        a = design * weight[:, None]
        y = stress * weight
        coef = nnls(a - q @ (q.T @ a), y - q @ (q.T @ y), maxiter=3000)[0]
        b = np.linalg.solve(r, q.T @ (y - a @ coef))
        residual = stress - design @ coef - drift @ b
        _, scale = _residual_scale(residual, floor)
        weight = np.sqrt((1.345 * scale) / np.maximum(1.345 * scale, np.abs(residual)))
    return coef, b, residual


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Return clean ``test_stress`` and the fitted equilibrium modulus ``G_inf``."""
    t, strain, stress, test_t, test_strain = (
        np.asarray(experiment[key], dtype=float) for key in
        ('train_t', 'train_strain', 'train_stress_raw', 'test_t', 'test_strain')
    )
    duration = np.ptp(t)
    drift = np.column_stack((np.ones_like(t), (t - np.mean(t)) / duration))
    tau = np.geomspace(_TAU_MIN, _TAU_MAX, 81)
    floor = max(np.finfo(float).tiny, 1e-12 * np.max(np.abs(stress)))
    coef, b, residual = _robust_grid_fit(_design(t, strain, tau), drift, stress, floor)

    # Exclude isolated spikes before refining adjacent grid atoms as single modes.
    center, scale = _residual_scale(residual, floor)
    keep = np.abs(residual - center) <= 5. * scale
    active = np.flatnonzero(coef[1:] > 1e-8 * np.max(coef))
    if active.size:
        log_tau = np.log(tau[active])
        groups = np.split(active, np.flatnonzero(np.diff(log_tau) > np.log(1.8)) + 1)
        moduli = np.array([coef[1:][g].sum() for g in groups])
        centers = [np.average(np.log(tau[g]), weights=coef[1:][g]) for g in groups]
    else:
        moduli, centers = np.empty(0), []

    m = moduli.size
    time_slice = slice(m + 1, 2 * m + 1)
    initial = np.r_[coef[0], moduli, centers, b]
    lo = np.r_[np.zeros(m + 1), np.full(m, np.log(_TAU_MIN)), -np.inf, -np.inf]
    hi = np.r_[np.full(m + 1, np.inf), np.full(m, np.log(_TAU_MAX)), np.inf, np.inf]
    # Weakly shrink the integrated memory of poorly resolved slow modes. Scale
    # the penalty to the observed noise and modulus, not to the stress units.
    typical_modulus = np.median(moduli) if m else 1.
    penalty = 2.1 * scale / (typical_modulus * (0.30 * duration))

    def objective(params):
        times = np.exp(params[time_slice])
        prediction = _design(t, strain, times) @ params[:m + 1] + drift @ params[-2:]
        return np.r_[(prediction - stress)[keep], penalty * params[1:m + 1] * times]

    fitted = least_squares(objective, np.clip(initial, lo, hi), bounds=(lo, hi),
                           max_nfev=100).x
    coef = fitted[:m + 1]
    times = np.exp(fitted[time_slice])
    return {'G_inf': float(coef[0]), 'test_stress': _design(test_t, test_strain, times) @ coef}
