"""Irregular-time Prony calibration and quadratic driver reconstruction."""

import numpy as np
from scipy.optimize import least_squares


def _window(run, taus):
    """Return window steps, ZOH coefficients, and memory before the window."""
    pt = np.asarray(run['past_timestamps'], dtype=float)
    pu = np.asarray(run['past_driver_history'], dtype=float)
    t = np.asarray(run['timestamps'], dtype=float)
    m = np.full(taus.size, pu[0])
    for dt, u in zip(np.diff(pt), pu[1:]):
        m += -np.expm1(-dt / taus) * (u - m)
    dt = np.diff(t, prepend=pt[-1])
    exponent = -dt[:, None] / taus
    return t - t[0], dt, np.exp(exponent), -np.expm1(exponent), m


def _mad(residual):
    return 1.4826 * np.median(np.abs(residual - np.median(residual)))


def _calibrate(run, taus):
    t, _, a, b, m = _window(run, taus)
    u = np.asarray(run['driver_u'], dtype=float)
    y = np.asarray(run['y_obs'], dtype=float)
    memory = np.empty_like(a)
    for k, uk in enumerate(u):
        m = a[k] * m + b[k] * uk
        memory[k] = m
    x = np.column_stack((u, memory, t, np.ones_like(t)))

    # Scale the columns for the robust solve, including the drift columns.
    norms = np.linalg.norm(x, axis=0)
    norms[norms == 0] = 1
    x /= norms
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    floor = np.finfo(float).eps * max(1.0, np.max(np.abs(y))) * 32
    for _ in range(8):
        scale = max(_mad(x @ beta - y), floor)
        beta = least_squares(
            lambda v: x @ v - y, beta, jac=lambda _: x.copy(),
            loss='soft_l1', f_scale=scale, gtol=1e-12,
        ).x
        next_scale = max(_mad(x @ beta - y), floor)
        if abs(next_scale - scale) <= 0.05 * scale:
            break

    # Once the large glitches are separated, remove their remaining influence.
    r = x @ beta - y
    keep = np.abs(r - np.median(r)) <= 5 * max(_mad(r), floor)
    refit, _, rank, _ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    if rank == x.shape[1]:
        beta = refit
    return (beta / norms)[:3]


def _invert(run, taus, moduli):
    t, dt, a, b, initial_memory = _window(run, taus)
    drift = run['drift_slope'] * t + run['drift_offset']
    y = np.asarray(run['y_obs'], dtype=float) - drift
    direct = moduli[0] + b @ moduli[1:]
    from_state = np.column_stack((a * moduli[1:], np.zeros_like(t)))
    previous_u = np.array([0.0, 0.0, 1.0])

    # State = [memory_1, memory_2, previous driver]. Eliminate the driver
    # from the quadratic cost-to-go, backwards, one scalar at a time.
    p = np.zeros(3)
    P = np.zeros((3, 3))
    gain = np.empty_like(from_state)
    bias = np.empty_like(y)
    for k in range(len(t) - 1, -1, -1):
        A = np.r_[a[k], 0.0]
        B = np.r_[b[k], 1.0]
        c, d = from_state[k], direct[k]
        w = run['smoothness_lambda'] / dt[k] if k else 0.0
        PB = P @ B
        curvature = d * d + w + B @ PB
        g = -(d * c - w * previous_u + A * PB) / curvature
        gain[k] = g
        bias[k] = (d * y[k] - B @ p) / curvature
        closed = np.diag(A) + np.outer(B, g)
        observation = c + d * g
        change = g - previous_u
        p = closed.T @ p - y[k] * observation
        P = closed.T @ P @ closed + np.outer(observation, observation) + w * np.outer(change, change)

    # Replay the optimal driver through the same physical state updates.
    state = np.r_[initial_memory, 0.0]
    u = np.empty_like(y)
    predicted = np.empty_like(y)
    for k in range(len(t)):
        u[k] = gain[k] @ state + bias[k]
        state[:2] = a[k] * state[:2] + b[k] * u[k]
        state[2] = u[k]
        predicted[k] = drift[k] + moduli[0] * u[k] + moduli[1:] @ state[:2]
    return u, predicted


def deconvolve_hereditary_driver(problem: dict) -> dict:
    """Estimate the shared moduli and recover the regularized blind driver."""
    taus = np.asarray(problem['taus'], dtype=float)
    moduli = _calibrate(problem['calibration_run'], taus)
    u, predicted = _invert(problem['blind_run'], taus, moduli)
    return {
        'moduli': moduli.tolist(),
        'recovered_driver': u,
        'predicted_blind_response': predicted,
    }
