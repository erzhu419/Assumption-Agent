"""Robust Prony calibration and exact irregular-grid driver inversion."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _memory_grid(run, taus):
    """Compute the state immediately before the main window and its ZOH weights."""
    past_t = np.asarray(run['past_timestamps'], dtype=float)
    past_u = np.asarray(run['past_driver_history'], dtype=float)
    t = np.asarray(run['timestamps'], dtype=float)
    state = np.full(2, past_u[0])
    for dt, u in zip(np.diff(past_t), past_u[1:]):
        state += -np.expm1(-dt / taus) * (u - state)
    dt = np.diff(t, prepend=past_t[-1])
    z = -dt[:, None] / taus
    return t, dt, state, np.exp(z), -np.expm1(z)


def _calibrate(run, taus):
    t, _, state, a, b = _memory_grid(run, taus)
    u = np.asarray(run['driver_u'], dtype=float)
    y = np.asarray(run['y_obs'], dtype=float)
    memory = np.empty((t.size, 2))
    for k in range(t.size):
        state = a[k] * state + b[k] * u[k]
        memory[k] = state
    x = np.column_stack((u, memory, t - t[0], np.ones(t.size)))
    # Scale the columns, not the data, to keep the drift and memory fit balanced.
    norms = np.linalg.norm(x, axis=0)
    norms[norms == 0] = 1
    x /= norms
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    floor = 16 * np.finfo(float).eps * max(1., np.max(np.abs(y)))
    for cauchy in (False, True):
        for _ in range(60):
            err = y - x @ beta
            scale = max(floor, 1.4826 * np.median(np.abs(err - np.median(err))))
            if cauchy:
                weight = 1. / np.hypot(1., err / (2.385 * scale))
            else:
                weight = np.sqrt(1. / np.maximum(1., np.abs(err) / (1.345 * scale)))
            updated = np.linalg.lstsq(x * weight[:, None], y * weight, rcond=None)[0]
            delta = np.linalg.norm(updated - beta)
            beta = updated
            if delta <= 1e-11 * (1. + np.linalg.norm(beta)):
                break
    # Efficient least squares on the inliers once heavy glitches are isolated.
    err = y - x @ beta
    centered = err - np.median(err)
    scale = max(floor, 1.4826 * np.median(np.abs(centered)))
    keep = np.abs(centered) <= 5. * scale
    if keep.sum() >= x.shape[1]:
        clean, _, rank, _ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
        if rank == x.shape[1]:
            beta = clean
    return (beta / norms)[:3]


def _invert(run, taus, moduli):
    t, dt, initial, a, b = _memory_grid(run, taus)
    n = t.size
    drift = run['drift_offset'] + run['drift_slope'] * (t - t[0])
    target = np.asarray(run['y_obs'], dtype=float) - drift
    lam = run['smoothness_lambda']
    if lam == 0 or n == 1:
        # The unpenalized observation matrix is lower triangular.
        state = initial
        driver = np.empty(n)
        for k in range(n):
            inherited = a[k] * state
            driver[k] = (target[k] - moduli[1:] @ inherited) / (moduli[0] + moduli[1:] @ b[k])
            state = inherited + b[k] * driver[k]
        return driver, target + drift

    # Variables at each sample are (u, m1, m2). Keeping the states makes
    # the normal equations with exact dynamics constraints sparse and local.
    cols = np.arange(3 * n).reshape(n, 3)
    rows = np.repeat(np.arange(2 * n), 2)
    dyn_cols = np.stack((np.broadcast_to(cols[:, :1], (n, 2)), cols[:, 1:]), axis=-1)
    values = np.stack((-b, np.ones_like(b)), axis=-1)
    dynamics = sparse.coo_matrix((np.r_[values.ravel(), -a[1:].ravel()],
                                  (np.r_[rows, np.arange(2, 2*n)],
                                   np.r_[dyn_cols.ravel(), cols[:-1, 1:].ravel()])),
                                 shape=(2*n, 3*n)).tocsc()
    forcing = np.zeros(2*n)
    forcing[:2] = a[0] * initial
    obs = sparse.coo_matrix((np.tile(moduli, n), (np.repeat(np.arange(n), 3), cols.ravel())),
                            shape=(n, 3*n)).tocsc()
    diff = sparse.coo_matrix(
        (np.tile([-1., 1.], n-1), (np.repeat(np.arange(n-1), 2),
                                  np.column_stack((cols[:-1, 0], cols[1:, 0])).ravel())),
        shape=(n-1, 3*n)).tocsc()
    hessian = obs.T @ obs + diff.T @ sparse.diags(lam / dt[1:]) @ diff
    kkt = sparse.bmat([[hessian, dynamics.T], [dynamics, None]], format='csc')
    solution = spsolve(kkt, np.r_[obs.T @ target, forcing])[:3*n]
    return solution[::3].copy(), np.asarray(obs @ solution) + drift


def deconvolve_hereditary_driver(problem: dict) -> dict:
    """Fit shared moduli and minimize the specified blind-run quadratic."""
    taus = np.asarray(problem['taus'], dtype=float)
    moduli = _calibrate(problem['calibration_run'], taus)
    driver, prediction = _invert(problem['blind_run'], taus, moduli)
    return {'moduli': moduli.tolist(), 'recovered_driver': driver,
            'predicted_blind_response': prediction}
