"""Robust Prony calibration and irregular-grid hereditary deconvolution."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _memory_setup(run, taus):
    t = np.asarray(run['timestamps'], dtype=float)
    past_t = np.asarray(run['past_timestamps'], dtype=float)
    past_u = np.asarray(run['past_driver_history'], dtype=float)
    memory = np.full(2, past_u[0])
    for dt, u in zip(np.diff(past_t), past_u[1:]):
        z = -dt / taus
        memory = np.exp(z) * memory - np.expm1(z) * u
    dt = np.diff(t, prepend=past_t[-1])
    z = -dt[:, None] / taus
    return t - t[0], dt, np.exp(z), -np.expm1(z), memory


def _propagate(u, a, b, initial):
    states = np.empty_like(a)
    prev = initial
    for k, value in enumerate(u):
        prev = a[k] * prev + b[k] * value
        states[k] = prev
    return states


def _calibrate(run, taus):
    t, _, a, b, initial = _memory_setup(run, taus)
    u = np.asarray(run['driver_u'], dtype=float)
    y = np.asarray(run['y_obs'], dtype=float)
    design = np.column_stack((u, _propagate(u, a, b, initial), t, np.ones_like(t)))

    # Huber IRLS gets the fit into the inlier basin without relying on a
    # known calibration noise level. Once there, refit to the clean samples.
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    floor = 32 * np.finfo(float).eps * max(1., np.max(np.abs(y)))
    for _ in range(60):
        residual = y - design @ coef
        scale = max(floor, 1.4826 * np.median(np.abs(residual - np.median(residual))))
        weight = np.sqrt((1.345 * scale) / np.maximum(1.345 * scale, np.abs(residual)))
        next_coef = np.linalg.lstsq(design * weight[:, None], y * weight, rcond=None)[0]
        change = np.linalg.norm(design @ (next_coef - coef), ord=np.inf)
        coef = next_coef
        if change <= 1e-10 * (1. + scale):
            break

    residual = y - design @ coef
    centered = residual - np.median(residual)
    scale = max(floor, 1.4826 * np.median(np.abs(centered)))
    keep = np.abs(centered) <= 5. * scale
    clean_coef, _, rank, _ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
    if rank == design.shape[1]:
        coef = clean_coef
    return coef[:3]


def _solve_driver(y, dt, a, b, initial, moduli, regularization):
    n = y.size
    identity = sparse.eye(n, format='csc')
    observation = sparse.hstack([e * identity for e in moduli], format='csc')
    hessian = observation.T @ observation
    if regularization and n > 1:
        w = np.sqrt(regularization / dt[1:])
        difference = sparse.diags((-w, w), (0, 1), shape=(n - 1, n))
        penalty = difference.T @ difference
        hessian += sparse.block_diag((penalty, sparse.csc_matrix((2 * n, 2 * n))))

    # Keep both internal memory states as unknowns. Their recurrences are
    # sparse equalities, so even a long relaxation time needs no dense kernel.
    recurrences = [identity - sparse.diags(a[1:, r], -1, shape=(n, n))
                   for r in range(2)] if n > 1 else [identity, identity]
    constraints = sparse.bmat([
        [-sparse.diags(b[:, 0]), recurrences[0], None],
        [-sparse.diags(b[:, 1]), None, recurrences[1]],
    ], format='csc')
    memory_rhs = np.zeros(2 * n)
    memory_rhs[::n] = a[0] * initial
    system = sparse.bmat([[hessian, constraints.T], [constraints, None]], format='csc')
    rhs = np.concatenate((observation.T @ y, memory_rhs))
    return spsolve(system, rhs)[:n]


def deconvolve_hereditary_driver(problem: dict) -> dict:
    """Fit the material moduli, then solve the regularized blind inverse."""
    taus = np.asarray(problem['taus'], dtype=float)
    moduli = _calibrate(problem['calibration_run'], taus)
    blind = problem['blind_run']
    t, dt, a, b, initial = _memory_setup(blind, taus)
    y = np.asarray(blind['y_obs'], dtype=float)
    drift = blind['drift_slope'] * t + blind['drift_offset']
    u = _solve_driver(y - drift, dt, a, b, initial, moduli, blind['smoothness_lambda'])
    states = _propagate(u, a, b, initial)
    return {
        'moduli': moduli.tolist(),
        'recovered_driver': u,
        'predicted_blind_response': moduli[0] * u + states @ moduli[1:] + drift,
    }
