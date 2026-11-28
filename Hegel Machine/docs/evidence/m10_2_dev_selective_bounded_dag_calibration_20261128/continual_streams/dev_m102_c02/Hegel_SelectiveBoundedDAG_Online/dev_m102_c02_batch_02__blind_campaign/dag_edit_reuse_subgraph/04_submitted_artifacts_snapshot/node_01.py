"""Scalar and trajectory readouts on the frozen calibrated coordinates."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _scalar_inputs(c):
    paired = np.sum(c[:, :2] * c[:, 2:], axis=1)
    return np.column_stack((c, paired))


def _scalar_readout(c, query, y):
    u, uq = _scalar_inputs(c), _scalar_inputs(query)
    x, xq = _affine(u), _affine(uq)

    def design(base, z, w):
        s = z @ w
        return np.column_stack((base, s / (1.0 + s * s)))

    def projected_residual(w):
        d = design(x, u, w)
        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y

    linear = np.linalg.lstsq(x, y, rcond=None)[0][1:]
    if np.linalg.norm(linear) < 1e-10:
        return xq @ np.linalg.lstsq(x, y, rcond=None)[0]
    direction = linear / np.linalg.norm(linear)
    optimum = min((least_squares(projected_residual, scale * direction,
                                 bounds=(-5., 5.), max_nfev=160, gtol=2e-10)
                   for scale in (0.6, 1.2, 2.0)), key=lambda fit: fit.cost)
    d = design(x, u, optimum.x)
    beta = np.linalg.lstsq(d, y, rcond=None)[0]
    return design(xq, uq, optimum.x) @ beta


def _trajectory(c, query, traj, times, intercept=True):
    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)
    beta = np.linalg.lstsq(design, traj, rcond=None)[0]
    fitted = design @ beta
    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    t = times / scale

    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates[:, None]) * t)

    def residual(log_rates):
        modes = basis(log_rates)
        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()

    log_rates = least_squares(residual, np.log([0.6, 2.7]),
                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),
                              max_nfev=100).x
    modes = basis(log_rates)
    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes




def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    coordinates = inputs['node_00']
    c = np.asarray(coordinates['c'], dtype=float)
    query = np.asarray(coordinates['query'], dtype=float)
    train = study['train']
    y_hat = _scalar_readout(c, query, np.asarray(train['y'], dtype=float))
    state_pred = _trajectory(c, query, np.asarray(train['traj'], dtype=float),
                             np.asarray(study['eval_times'], dtype=float))
    final_state = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         final_state.mean(), final_state.std()])}
