"""Robust identification and prediction for a generalized Maxwell/Prony model."""
import numpy as np
from scipy.optimize import nnls


def _memory_column(t, strain, tau):
    """Exact exponential convolution for piecewise-linear strain."""
    t = np.asarray(t, dtype=float).ravel()
    e = np.asarray(strain, dtype=float).ravel()
    if t.size != e.size:
        raise ValueError("time and strain arrays must have equal lengths")
    if t.size == 0:
        return np.empty(0)
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time values must be strictly increasing")
    z = np.zeros_like(e)
    a = np.exp(-dt / float(tau))
    slope = np.diff(e) / dt
    for i in range(1, len(e)):
        z[i] = a[i - 1] * z[i - 1] + slope[i - 1] * tau * (1.0 - a[i - 1])
    return z


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Fit nonnegative Prony strengths with affine drift and spike-robust weights."""
    tt = np.asarray(experiment['train_t'], dtype=float).ravel()
    eps = np.asarray(experiment['train_strain'], dtype=float).ravel()
    y = np.asarray(experiment['train_stress_raw'], dtype=float).ravel()
    if not (tt.size == eps.size == y.size):
        raise ValueError("training arrays must have equal lengths")
    taus = np.geomspace(0.02, 8.0, 40)
    mem = np.column_stack([_memory_column(tt, eps, tau) for tau in taus])
    X = np.column_stack((eps, mem))
    # Treat the unknown affine thermal drift as unbounded nuisance coefficients.
    D = np.column_stack((np.ones_like(tt), tt - np.mean(tt)))
    Q, _ = np.linalg.qr(D, mode='reduced')
    Xp = X - Q @ (Q.T @ X)
    yp = y - Q @ (Q.T @ y)
    weights = np.ones_like(y)
    for _ in range(15):
        sw = np.sqrt(weights)
        coef = nnls(Xp * sw[:, None], yp * sw, maxiter=1000)[0]
        resid = y - X @ coef
        drift = np.linalg.lstsq(D, resid, rcond=None)[0]
        resid -= D @ drift
        center = np.median(resid)
        scale = 1.4826 * np.median(np.abs(resid - center)) + 1e-12
        cutoff = 1.5 * scale
        new_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(resid - center), 1e-12))
        if np.max(np.abs(new_weights - weights)) < 1e-4:
            weights = new_weights
            break
        weights = new_weights
    gi = float(coef[0])
    strengths = coef[1:]
    te = np.asarray(experiment['test_t'], dtype=float).ravel()
    et = np.asarray(experiment['test_strain'], dtype=float).ravel()
    if te.size != et.size:
        raise ValueError("test arrays must have equal lengths")
    mt = np.column_stack([_memory_column(te, et, tau) for tau in taus])
    return {'test_stress': gi * et + mt @ strengths, 'G_inf': gi}
