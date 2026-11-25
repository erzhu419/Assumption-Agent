"""Robust Prony calibration and exact irregular-grid quadratic deconvolution."""

import numpy as np


def _prepare(run, taus):
    """Return main-window times, ZOH coefficients and pre-window state."""
    past_t = np.asarray(run['past_timestamps'], dtype=float)
    past_u = np.asarray(run['past_driver_history'], dtype=float)
    t = np.asarray(run['timestamps'], dtype=float)
    m = np.full(2, past_u[0], dtype=float)
    for dt, u in zip(np.diff(past_t), past_u[1:]):
        m += -np.expm1(-dt / taus) * (u - m)
    z = -np.diff(t, prepend=past_t[-1])[:, None] / taus
    return t, np.exp(z), -np.expm1(z), m


def _calibration_design(run, taus):
    t, a, b, m = _prepare(run, taus)
    u = np.asarray(run['driver_u'], dtype=float)
    states = np.empty_like(a)
    for k in range(t.size):
        m = a[k] * m + b[k] * u[k]
        states[k] = m
    elapsed = t - t[0]
    return np.column_stack((u, states, elapsed / elapsed[-1], np.ones(t.size)))


def _calibrate(x, y):
    """Find inliers using Cauchy reweighting, then refit their Gaussian core."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    floor = np.finfo(float).eps * max(1.0, np.max(np.abs(y)))
    for _ in range(30):
        r = y - x @ beta
        scale = max(floor, 1.4826 * np.median(np.abs(r - np.median(r))))
        root_w = 1.0 / np.hypot(1.0, r / (2.385 * scale))
        nxt = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
        change = np.linalg.norm(x @ (nxt - beta), ord=np.inf)
        beta = nxt
        if change <= 1e-5 * scale:
            break
    r = y - x @ beta
    center = np.median(r)
    scale = max(floor, 1.4826 * np.median(np.abs(r - center)))
    inlier = np.abs(r - center) <= 6.0 * scale
    clean, _, rank, _ = np.linalg.lstsq(x[inlier], y[inlier], rcond=None)
    return clean if rank == x.shape[1] else beta


def _invert(run, taus, moduli):
    t, a, b, m = _prepare(run, taus)
    drift = run['drift_slope'] * (t - t[0]) + run['drift_offset']
    y = np.asarray(run['y_obs'], dtype=float) - drift
    w = np.r_[0.0, run['smoothness_lambda'] / np.diff(t)]
    # State at entry to step k: (m1[k-1], m2[k-1], u[k-1]).
    # Eliminate one scalar control per step from the quadratic cost-to-go.
    mode = moduli[1:]
    c = np.column_stack((a * mode, np.zeros(t.size)))
    d = moduli[0] + b @ mode
    p = np.zeros((3, 3))
    h = np.zeros(3)
    feedback = np.empty((t.size, 3))
    feedforward = np.empty(t.size)
    e = np.array([0., 0., 1.])
    for k in range(t.size - 1, -1, -1):
        f = np.r_[a[k], 0.0]  # diagonal of the state transition
        g = np.r_[b[k], 1.0]
        pg = p @ g
        denom = d[k] ** 2 + w[k] + g @ pg
        v = d[k] * c[k] - w[k] * e + f * pg
        j = d[k] * y[k] + g @ h
        gain = v / denom
        step = j / denom
        feedback[k], feedforward[k] = gain, step
        # Closed-loop form of the Schur complement avoids subtracting
        # nearly equal penalty terms when timestamp gaps are very small.
        closed = np.diag(f) - np.outer(g, gain)
        obs = c[k] - d[k] * gain
        delta = e + gain
        p = closed.T @ p @ closed + np.outer(obs, obs) + w[k] * np.outer(delta, delta)
        h = f * h + c[k] * y[k] - v * step
    state = np.r_[m, 0.0]
    u = np.empty(t.size)
    response = np.empty(t.size)
    for k in range(t.size):
        u[k] = feedforward[k] - feedback[k] @ state
        response[k] = c[k] @ state + d[k] * u[k] + drift[k]
        state[:2] = a[k] * state[:2] + b[k] * u[k]
        state[2] = u[k]
    return u, response


def deconvolve_hereditary_driver(problem: dict) -> dict:
    """Estimate the three material moduli and recover the blind driver."""
    taus = np.asarray(problem['taus'], dtype=float)
    calibration = problem['calibration_run']
    x = _calibration_design(calibration, taus)
    moduli = _calibrate(x, np.asarray(calibration['y_obs'], dtype=float))[:3]
    u, y = _invert(problem['blind_run'], taus, moduli)
    return {'moduli': moduli.tolist(), 'recovered_driver': u,
            'predicted_blind_response': y}
