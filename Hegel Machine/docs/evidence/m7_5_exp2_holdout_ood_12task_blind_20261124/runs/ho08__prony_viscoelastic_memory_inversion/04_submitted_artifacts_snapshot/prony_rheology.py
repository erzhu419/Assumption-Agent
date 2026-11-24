"""Identify a nonnegative relaxation spectrum and remove affine sensor drift."""

import numpy as np
from scipy.optimize import nnls


def _memory(t, strain, tau):
    """Exponential convolution for the piecewise-linear strain record."""
    h = np.diff(t)[:, None] / tau
    decay = np.exp(-h)
    increment = np.diff(strain)[:, None] * (-np.expm1(-h) / h)
    memory = np.zeros((len(t), len(tau)))
    for i in range(1, len(t)):
        memory[i] = decay[i - 1] * memory[i - 1] + increment[i - 1]
    return memory


def _design(t, strain, tau):
    return np.column_stack((strain, _memory(t, strain, tau)))


def _noise(residual):
    return max(float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
               np.finfo(float).eps)


def _projected_fit(design, drift, y, weight):
    """Solve the unconstrained drift exactly under the current weights."""
    a = design * weight[:, None]
    b = y * weight
    q, r = np.linalg.qr(drift * weight[:, None], mode='reduced')
    qa, qb = q.T @ a, q.T @ b
    ap = a - q @ qa
    scale = np.linalg.norm(ap, axis=0)
    scale[scale == 0] = 1
    moduli = nnls(ap / scale, b - q @ qb, maxiter=20 * a.shape[1])[0] / scale
    beta = np.linalg.solve(r, qb - qa @ moduli)
    return moduli, y - design @ moduli - drift @ beta


def _calibrate(design, drift, y, mode='trim'):
    weight = np.ones_like(y)
    for _ in range(7 if mode != 'ols' else 1):
        moduli, residual = _projected_fit(design, drift, y, weight)
        weight = np.sqrt(np.minimum(1, 1.5 * _noise(residual) /
                                    np.maximum(np.abs(residual), np.finfo(float).eps)))
    if mode == 'trim':
        for _ in range(3):
            inlier = np.abs(residual - np.median(residual)) <= 4.5 * _noise(residual)
            moduli, residual = _projected_fit(design, drift, y, inlier.astype(float))
    return moduli, residual


def _estimate(t, strain, y):
    tau = np.geomspace(0.02, 20.0, 60)
    design = _design(t, strain, tau)
    duration = np.ptp(t)
    drift = np.column_stack((np.ones_like(t), (t - t.mean()) / duration))
    baseline, residual = _calibrate(design, drift, y)
    noise = _noise(residual)
    keep = np.abs(residual - np.median(residual)) <= 4.5 * noise

    q = np.linalg.qr(drift[keep], mode='reduced')[0]
    a = design[keep]
    a = a - q @ (q.T @ a)
    b = y[keep] - q @ (q.T @ y[keep])
    centered_strain = strain[keep] - np.mean(strain[keep])
    strain_norm = np.linalg.norm(centered_strain)
    if strain_norm == 0 or not np.any(baseline):
        return tau, baseline
    modulus_scale = np.linalg.norm(a @ baseline) / strain_norm
    if modulus_scale <= np.finfo(float).eps:
        return tau, baseline

    # Penalize the two poorly resolved tails separately. The slow spectral
    # moment trades off with equilibrium stress and drift; sub-cadence modes
    # contribute little to a smooth ramp but can dominate a new step.
    cadence = np.median(np.diff(t))
    slow = 12.0 * tau / duration
    fast = 1.0 / (1.0 + (tau / (2.0 * cadence))**2)
    penalty = (noise / modulus_scale) * np.column_stack((np.zeros(2), (slow, fast)))
    a = np.vstack((a, penalty))
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1
    moduli = nnls(a / scale, np.r_[b, 0., 0.], maxiter=20 * len(scale))[0] / scale
    return tau, moduli


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Return the equilibrium modulus and the drift-free test stress array."""
    t, strain, y, u, test_strain = (np.asarray(experiment[k], dtype=float) for k in
        ('train_t', 'train_strain', 'train_stress_raw', 'test_t', 'test_strain'))
    tau, moduli = _estimate(t, strain, y)
    return {'G_inf': float(moduli[0]), 'test_stress': _design(u, test_strain, tau) @ moduli}
