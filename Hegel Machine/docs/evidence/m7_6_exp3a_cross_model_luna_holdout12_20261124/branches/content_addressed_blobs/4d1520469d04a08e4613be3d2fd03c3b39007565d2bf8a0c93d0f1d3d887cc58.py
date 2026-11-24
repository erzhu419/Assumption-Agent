"""Identify a nonnegative Prony relaxation spectrum from a noisy strain trace."""
import numpy as np
from scipy.optimize import nnls


def _memory_design(t, strain, taus):
    """Sampled hereditary convolution for piecewise-constant strain rate."""
    t = np.asarray(t, dtype=float)
    strain = np.asarray(strain, dtype=float)
    taus = np.asarray(taus, dtype=float)
    if t.ndim != 1 or strain.shape != t.shape or len(t) < 2 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(strain)):
        raise ValueError("time and strain must be finite aligned vectors")
    dt = np.diff(t)
    if np.any(dt <= 0) or np.any(taus <= 0):
        raise ValueError("time and relaxation times must be positive and increasing")
    rate = np.diff(strain) / dt
    out = np.zeros((len(t), len(taus)))
    for i, (h, r) in enumerate(zip(dt, rate), 1):
        out[i] = out[i - 1] * np.exp(-h / taus) + r * taus * (-np.expm1(-h / taus))
    return out


def _fit_coefficients(t, strain, stress, taus):
    physical = np.column_stack((strain, _memory_design(t, strain, taus)))
    drift = np.column_stack((np.ones(len(t)), t))
    # Remove affine sensor drift analytically before enforcing physical positivity.
    project = lambda z: z - drift @ np.linalg.lstsq(drift, z, rcond=None)[0]
    B, z = project(physical), project(stress)
    weights = np.ones(len(t))
    beta = np.zeros(physical.shape[1])
    for _ in range(12):
        sw = np.sqrt(weights)
        norms = np.maximum(np.linalg.norm(B * sw[:, None], axis=0), 1e-12)
        coef, _ = nnls(B * sw[:, None] / norms, z * sw, maxiter=5000)
        beta = coef / norms
        resid = stress - physical @ beta - drift @ np.linalg.lstsq(drift, stress - physical @ beta, rcond=None)[0]
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-10
        new_weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(resid), 1e-12))
        if np.max(np.abs(new_weights - weights)) < 1e-5:
            break
        weights = new_weights
    drift_coef = np.linalg.lstsq(drift, stress - physical @ beta, rcond=None)[0]
    return beta, drift_coef


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Return clean test stress, equilibrium modulus, and fitted Prony spectrum."""
    tt = np.asarray(experiment['train_t'], dtype=float)
    et = np.asarray(experiment['train_strain'], dtype=float)
    yt = np.asarray(experiment['train_stress_raw'], dtype=float)
    tx = np.asarray(experiment['test_t'], dtype=float)
    ex = np.asarray(experiment['test_strain'], dtype=float)
    if tt.shape != et.shape or tt.shape != yt.shape or tx.shape != ex.shape:
        raise ValueError('time, strain and stress arrays must be aligned')
    taus = np.geomspace(0.02, 20.0, 10)
    beta, drift = _fit_coefficients(tt, et, yt, taus)
    pred = np.column_stack((ex, _memory_design(tx, ex, taus))) @ beta
    return {'test_stress': pred, 'G_inf': float(beta[0]), 'relaxation_times': taus,
            'moduli': beta[1:], 'drift': drift}
