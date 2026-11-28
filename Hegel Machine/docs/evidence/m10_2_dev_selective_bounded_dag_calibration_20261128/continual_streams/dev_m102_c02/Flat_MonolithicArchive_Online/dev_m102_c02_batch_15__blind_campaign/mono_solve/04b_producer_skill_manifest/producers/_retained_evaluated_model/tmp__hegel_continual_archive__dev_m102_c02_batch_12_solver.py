"""Panel-calibrated prediction with independent scalar and trajectory readouts."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    dx = u[j] - u[i]
    dv = v[j] - v[i]
    slope = np.array([np.median(dv[ok, k] / dx[ok, k])
                      for k in range(u.shape[1])
                      for ok in [np.abs(dx[:, k]) > 0.1 * np.ptp(u[:, k])]])
    intercept = np.median(v - u * slope, axis=0)
    error = np.linalg.norm(v - u * slope - intercept, axis=1)
    median = np.median(error)
    mad = np.median(np.abs(error - median))
    keep = error <= median + 8 * max(mad, 1e-5)
    u, v = u[keep], v[keep]
    du, dv = u - u.mean(axis=0), v - v.mean(axis=0)
    slope = (du * dv).sum(axis=0) / (du * du).sum(axis=0)
    return v.mean(axis=0) - slope * u.mean(axis=0), slope


def _operator(panel):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))

    def design(x, freq):
        return _affine(np.cos(freq * x))

    def residual(p):
        h = design(u, p[0])
        return (h @ _fit(h, v) - v).ravel()

    freq = least_squares(residual, [2.6], bounds=([0.5], [5.0]),
                         gtol=1e-12).x[0]
    weights = _fit(design(u, freq), v)
    return lambda x: design(x, freq) @ weights


def _scalar(b, q, y):
    # Compare the three disjoint pairings, sharing one coefficient across
    # both products. Select using only the supplied training responses.
    def paired(x, order):
        i, j, k, l = order
        return _affine(np.column_stack((x, x[:, i]*x[:, j] + x[:, k]*x[:, l])))

    candidates = [(None, _affine(b))]
    for pairing in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        candidates.append((pairing, paired(b, pairing)))
    best_error = np.inf
    for pairing, h in candidates:
        w = _fit(h, y)
        leverage = np.sum(h * np.linalg.pinv(h).T, axis=1)
        error = np.mean(((y - h @ w) / np.maximum(1 - leverage, 1e-12))**2)
        if error < best_error:
            best_error = error
            best = pairing, w
    pairing, w = best
    return (_affine(q) if pairing is None else paired(q, pairing)) @ w


def _trajectory(h, q, trajectories, times):
    # First project onto the spatial column space, then fit two shared
    # relaxation rates to the resulting denoised coefficient trajectories.
    u, s, vh = np.linalg.svd(h, full_matrices=False)
    coordinates = u.T @ trajectories

    def modes(p):
        return -np.expm1(-np.exp(p[:, None]) * times)

    def residual(p):
        basis = modes(p)
        return (coordinates @ np.linalg.pinv(basis) @ basis - coordinates).ravel()

    p = least_squares(residual, np.log([0.4, 1.8]),
                      bounds=(np.log([0.04, 0.85]), np.log([0.85, 8.0])),
                      gtol=1e-11).x
    basis = modes(p)
    return q @ (vh.T / s) @ (coordinates @ np.linalg.pinv(basis)) @ basis


def solve_batch(study: dict) -> dict:
    """Predict scalar response, trajectory and their aggregate diagnostics."""
    train = study['train']
    offset, scale = _calibrate(study['panel_a'])
    operator = _operator(study['panel_b'])
    b = operator((np.asarray(train['x']) - offset) / scale)
    q = operator((np.asarray(study['test']['x']) - offset) / scale)
    y_hat = _scalar(b, q, np.asarray(train['y']))
    state = _trajectory(_affine(b), _affine(q), np.asarray(train['traj']),
                        np.asarray(study['eval_times']))
    terminal = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
