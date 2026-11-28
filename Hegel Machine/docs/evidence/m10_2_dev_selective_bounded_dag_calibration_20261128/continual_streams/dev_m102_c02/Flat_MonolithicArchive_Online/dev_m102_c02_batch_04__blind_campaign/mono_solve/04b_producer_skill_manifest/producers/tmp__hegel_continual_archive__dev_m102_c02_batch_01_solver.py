"""Calibrate the inputs, then learn the readout and shared relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    bias, gain = np.empty(u.shape[1]), np.empty(u.shape[1])
    for k, (a, b) in enumerate(zip(u.T, v.T)):
        dx = a[j] - a[i]
        ok = abs(dx) > 0.1 * np.ptp(a)
        g = np.median((b[j[ok]] - b[i[ok]]) / dx[ok])
        r = b - g * a
        m = np.median(r)
        keep = abs(r - m) <= max(1e-6, 8 * np.median(abs(r - m)))
        bias[k], gain[k] = np.linalg.lstsq(_affine(a[keep, None]), b[keep], rcond=None)[0]
    return bias, gain


def _warp(x, p, kind):
    if kind == 'powered':
        a, b, c = p
        return np.sign(x) * abs(x)**b / (1 + a * abs(x)**c)
    a, b = p
    return (x**3 + b*x) / np.sqrt(1 + a*abs(x)**3)


def _panel_operator(panel, kind):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    if kind == 'powered':
        start, lo, hi = [0.22, 3, 4], [0, 0.5, 0.1], [5, 7, 16]
    else:
        start, lo, hi = [0.22, 0], [0, -1], [5, 1]

    def fit(p):
        A = _affine(_warp(u, p, kind))
        return A, np.linalg.lstsq(A, v, rcond=None)[0]

    def residual(p):
        A, B = fit(p)
        return (A @ B - v).ravel()

    p = least_squares(residual, start, bounds=(lo, hi), max_nfev=180, gtol=1e-12).x
    _, B = fit(p)
    return lambda x: _affine(_warp(x, p, kind)) @ B


def _relaxation(t, traj):
    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates)[:, None] * t)

    def residual(log_rates):
        F = basis(log_rates)
        return (traj @ np.linalg.pinv(F) @ F - traj).ravel()

    rates = least_squares(residual, np.log([0.45, 1.8]), bounds=(np.log([0.03, 0.8]), np.log([0.79, 15]))).x
    return basis(rates)


def _features(w, form):
    if form == 'linear':
        return _affine(w)
    if form == 'sparse':
        return _affine(np.column_stack((w, w[:, 0]*w[:, 2], w[:, 1]*w[:, 3])))
    i, j = np.triu_indices(w.shape[1])
    return _affine(np.column_stack((w, w[:, i]*w[:, j])))


def _predict(study, kind='powered', y_form='sparse', traj_form='linear'):
    tr = study['train']
    b, g = _calibrate(study['panel_a'])
    operator = _panel_operator(study['panel_b'], kind)
    w = operator((np.asarray(tr['x']) - b) / g)
    wt = operator((np.asarray(study['test']['x']) - b) / g)
    Y = np.asarray(tr['y'])
    T = np.asarray(tr['traj'])
    F = _relaxation(np.asarray(study['eval_times']), T)
    y_hat = _features(wt, y_form) @ np.linalg.lstsq(_features(w, y_form), Y, rcond=None)[0]
    amplitudes = _features(wt, traj_form) @ np.linalg.lstsq(_features(w, traj_form), T @ np.linalg.pinv(F), rcond=None)[0]
    state_pred = amplitudes @ F
    return y_hat, state_pred


def solve_batch(study: dict) -> dict:
    y_hat, state_pred = _predict(study)
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
