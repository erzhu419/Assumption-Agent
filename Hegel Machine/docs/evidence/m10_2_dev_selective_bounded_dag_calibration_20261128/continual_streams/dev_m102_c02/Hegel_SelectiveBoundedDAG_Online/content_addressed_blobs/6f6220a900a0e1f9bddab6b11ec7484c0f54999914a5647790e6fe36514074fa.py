"""Invert the coupled sensor map and evaluate the cosine calibration panel."""
import numpy as np
from scipy.optimize import minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _sensor_basis(x):
    w = np.roll(x, -1, axis=-1)
    return np.stack((np.ones_like(x), x, w, x*w, x**3), axis=-1)


def _sensor_coordinates(panel, points):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    basis = _sensor_basis(u)
    rows = np.arange(len(u))
    for _ in range(5):
        coef = np.array([_ls(basis[rows, j], v[rows, j]) for j in range(u.shape[1])])
        err = np.linalg.norm(np.sum(basis * coef, axis=-1) - v, axis=1)
        rows = np.argsort(err)[:len(u)-3]
    coef = np.array([_ls(basis[rows, j], v[rows, j]) for j in range(u.shape[1])])

    def forward(z):
        return np.sum(_sensor_basis(z) * coef, axis=-1)

    calibrated = []
    for x in points:
        guess = (x - coef[:, 0]) / coef[:, 1]
        fit = root(lambda z: forward(z) - x, guess, tol=1e-10)
        calibrated.append(fit.x)
    return np.asarray(calibrated)


def _cosine_coordinates(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, frequency):
        return np.column_stack((np.ones(len(x)), np.cos(frequency*x)))

    def loss(frequency):
        a = design(u, frequency)
        residual = a @ _ls(a, v) - v
        return np.sum(residual**2)

    grid = np.linspace(0.15, 5.0, 61)
    i = int(np.argmin([loss(f) for f in grid]))
    frequency = minimize_scalar(loss, bounds=(grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]),
                                method='bounded', options={'xatol': 1e-10}).x
    return design(z, frequency) @ _ls(design(u, frequency), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    query = np.asarray(study['test']['x'], dtype=float)
    z = _sensor_coordinates(study['panel_a'], np.concatenate((x, query)))
    c = _cosine_coordinates(study['panel_b'], z)
    return {'c': c[:len(x)], 'query': c[len(x):]}
