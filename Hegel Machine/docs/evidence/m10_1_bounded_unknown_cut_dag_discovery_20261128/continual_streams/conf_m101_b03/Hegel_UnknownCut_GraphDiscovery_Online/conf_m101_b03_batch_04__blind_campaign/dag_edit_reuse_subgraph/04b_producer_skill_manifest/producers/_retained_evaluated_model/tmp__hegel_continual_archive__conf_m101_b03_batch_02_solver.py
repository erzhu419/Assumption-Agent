"""Calibrate the inputs and recover a dual-operator response and relaxation model."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    du, dv = u[i] - u[j], v[i] - v[j]
    slope = np.array([np.median(dv[du[:, k] != 0, k] / du[du[:, k] != 0, k])
                      for k in range(u.shape[1])])
    shift = np.median(v - u * slope, axis=0)
    error = np.linalg.norm(v - shift - u * slope, axis=1)
    keep = error <= 4 * np.median(error)
    design = np.stack((np.ones_like(u[keep]), u[keep]), axis=2)
    shift, slope = np.column_stack([_fit(design[:, k], v[keep, k])
                                    for k in range(u.shape[1])])
    return shift, slope


def _panel_outputs(panel, z):
    u = panel['u']

    def products(x):
        return np.column_stack((np.ones(len(x)), x * np.roll(x, 1, axis=1)))

    a = products(z) @ _fit(products(u), panel['v1'])
    target = panel['v2']

    def harmonic(x, p):
        frequency, weight = p
        return np.column_stack((np.ones(len(x)), x,
                                x*x - 2*weight*np.sin(frequency*x/2)**2))

    def residual(p):
        f = harmonic(u, p)
        return (f @ _fit(f, target) - target).ravel()

    fits = [least_squares(residual, (w, -.2), bounds=((.5, -1.), (5., 1.)))
            for w in (1.5, 2.5, 3.5)]
    p = min(fits, key=lambda fit: fit.cost).x
    c = harmonic(z, p) @ _fit(harmonic(u, p), target)
    return a, c


def _relaxation(times, traj):
    def basis(log_rates):
        return -np.expm1(-times[:, None] * np.exp(log_rates))

    def residual(log_rates):
        h = basis(log_rates)
        return (h @ _fit(h, traj.T) - traj.T).ravel()

    rates = least_squares(residual, np.log([.4, 2.]), bounds=(-5., 3.)).x
    h = basis(np.sort(rates))
    return h, _fit(h, traj.T).T


def _slow_mode(design, design_test, c, c_test, amplitude):
    scale = np.sqrt(np.mean(c*c, axis=0))
    c, c_test = c / scale, c_test / scale

    def kernel(x, y):
        r = np.sqrt(5 / x.shape[1]) * cdist(x, y) / 1.25
        return (1 + r + r*r/3) * np.exp(-r)

    cov = kernel(c, c)
    cov.flat[::len(c)+1] += .003
    factor = cho_factor(cov)
    inv_design = cho_solve(factor, design)
    inv_amp = cho_solve(factor, amplitude)
    trend = _fit(design.T @ inv_design, design.T @ inv_amp)
    return design_test @ trend + kernel(c_test, c) @ (inv_amp - inv_design @ trend)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']))
    shift, scale = _calibration(study['panel_a'])
    a, c = _panel_outputs(study['panel_c'], (x - shift) / scale)
    base = np.column_stack((np.ones(len(x)), a, c))
    paired = a*c
    response = np.column_stack((base, paired.sum(axis=1)))
    y_hat = response[n:] @ _fit(response[:n], train['y'])

    design = np.column_stack((base, paired))
    h, amplitudes = _relaxation(study['eval_times'], train['traj'])
    fast = design[n:] @ _fit(design[:n], amplitudes[:, 1])
    slow = _slow_mode(design[:n], design[n:], c[:n], c[n:], amplitudes[:, 0])
    state_pred = np.column_stack((slow, fast)) @ h.T
    terminal = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
