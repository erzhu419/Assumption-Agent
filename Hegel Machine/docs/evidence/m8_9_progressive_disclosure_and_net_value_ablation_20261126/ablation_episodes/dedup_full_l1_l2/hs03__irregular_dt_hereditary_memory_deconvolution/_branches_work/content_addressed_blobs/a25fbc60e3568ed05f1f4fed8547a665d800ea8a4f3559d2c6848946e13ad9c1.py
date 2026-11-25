"""Calibrate a two-mode Prony material, then invert its blind response."""

import numpy as np


def _window(run, taus):
    t = np.asarray(run['timestamps'], dtype=float)
    past = np.asarray(run['past_driver_history'], dtype=float)
    tp = np.asarray(run['past_timestamps'], dtype=float)
    m = np.full(2, past[0], dtype=float)
    for dt, u in zip(np.diff(tp), past[1:]):
        m += -np.expm1(-dt / taus) * (u - m)
    dt = np.diff(t, prepend=tp[-1])
    z = -dt[:, None] / taus
    return t - t[0], dt, np.exp(z), -np.expm1(z), m


def _memory(a, b, m, u):
    states = np.empty_like(a)
    for k in range(len(u)):
        m = a[k] * m + b[k] * u[k]
        states[k] = m
    return states


def _calibrate(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    # Huber iterations move into the inlier basin even if the initial
    # least-squares fit is pulled away by a burst of telemetry glitches.
    for _ in range(40):
        r = y - X @ beta
        sigma = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-10)
        w = np.sqrt(1.0 / np.maximum(1.0, np.abs(r) / (1.345 * sigma)))
        new = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        done = np.max(np.abs(new - beta)) < 1e-9
        beta = new
        if done:
            break
    # Gross glitches are well separated from the ordinary noise. Refit
    # without their leverage and without downweighting ordinary samples.
    for _ in range(4):
        r = y - X @ beta
        med = np.median(r)
        sigma = max(1.4826 * np.median(np.abs(r - med)), 1e-10)
        keep = np.abs(r - med) <= 4.5 * sigma
        new, _, rank, _ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        if rank < X.shape[1]:
            break
        done = np.max(np.abs(new - beta)) < 1e-9
        beta = new
        if done:
            break
    return beta[:3]


def _invert(a, b, initial, dt, moduli, target, lam):
    # State s = (previous memories, previous driver). Each stage chooses
    # one driver value, so eliminating it from the quadratic costs gives
    # an affine feedback rule and requires only a scalar division.
    P = np.zeros((3, 3))
    p = np.zeros(3)
    lag = np.array([0., 0., 1.])
    gains = np.empty((len(dt), 3))
    bias = np.empty(len(dt))
    for k in range(len(dt) - 1, -1, -1):
        f = np.r_[a[k], 0.]
        g = np.r_[b[k], 1.]
        h = np.r_[moduli[1:] * a[k], 0.]
        q = moduli[0] + moduli[1:] @ b[k]
        w = lam / dt[k] if k else 0.
        Pg = P @ g
        curvature = q * q + w + g @ Pg
        if curvature > 0.:
            K = -(q * h - w * lag + f * Pg) / curvature
            c = (q * target[k] - g @ p) / curvature
        else:
            K = np.zeros(3)
            c = 0.
        gains[k], bias[k] = K, c
        # Rebuild the cost under the feedback rule instead of subtracting
        # nearly equal curvature terms in a Schur complement.
        F = np.diag(f) + np.outer(g, K)
        H = h + q * K
        L = K - lag
        p = F.T @ (p + c * Pg) + H * (q * c - target[k]) + (w * c) * L
        P = F.T @ P @ F + np.outer(H, H) + w * np.outer(L, L)
        P = (P + P.T) * 0.5
    s = np.r_[initial, 0.]
    u = np.empty(len(dt))
    for k in range(len(dt)):
        u[k] = gains[k] @ s + bias[k]
        s[:2] = a[k] * s[:2] + b[k] * u[k]
        s[2] = u[k]
    return u


def deconvolve_hereditary_driver(problem: dict) -> dict:
    """Return calibrated moduli, regularized input, and its forward response."""
    taus = np.asarray(problem['taus'], dtype=float)
    cal = problem['calibration_run']
    t, _, a, b, m = _window(cal, taus)
    u = np.asarray(cal['driver_u'], dtype=float)
    X = np.column_stack((u, _memory(a, b, m, u), t, np.ones_like(t)))
    E = _calibrate(X, np.asarray(cal['y_obs'], dtype=float))

    blind = problem['blind_run']
    t, dt, a, b, m = _window(blind, taus)
    drift = blind['drift_slope'] * t + blind['drift_offset']
    target = np.asarray(blind['y_obs'], dtype=float) - drift
    u = _invert(a, b, m, dt, E, target, blind['smoothness_lambda'])
    pred = drift + E[0] * u + _memory(a, b, m, u) @ E[1:]
    return {'moduli': E.tolist(), 'recovered_driver': u,
            'predicted_blind_response': pred}
