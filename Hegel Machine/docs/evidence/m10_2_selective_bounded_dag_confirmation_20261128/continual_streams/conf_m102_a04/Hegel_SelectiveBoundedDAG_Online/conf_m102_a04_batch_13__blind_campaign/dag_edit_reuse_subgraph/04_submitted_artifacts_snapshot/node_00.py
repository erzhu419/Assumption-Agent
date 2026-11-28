"""Align study coordinates and evaluate the two calibrated panel responses."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _linear(design, target):
    return np.linalg.lstsq(design, target, rcond=1e-9)[0]


def _with_constant(x):
    return np.column_stack((np.ones(len(x)), x))


def _align(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    offsets, scales = [], []
    i, j = np.triu_indices(len(u), 1)
    for a, b in zip(u.T, v.T):
        delta = a[i] - a[j]
        valid = np.abs(delta) > 1e-8
        slope = np.median((b[i[valid]]-b[j[valid]])/delta[valid])
        intercept = np.median(b-slope*a)
        design = _with_constant(a)
        for _ in range(5):
            error = np.abs(b-intercept-slope*a)
            keep = error <= max(5*np.median(error), .00035)
            intercept, slope = _linear(design[keep], b[keep])
        offsets.append(intercept)
        scales.append(slope)
    return np.array(offsets), np.array(scales)


def _first_basis(x, q):
    return _with_constant(x/(1+q[0]*np.abs(x))+q[1]*x)


def _second_basis(x, q):
    return _with_constant(np.column_stack((x, x*np.roll(x, -1, axis=1), np.arctan(q*x))))


def _panel_maps(panel, z):
    u, v1, v2 = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1', 'v2'))

    def residual(basis, q, y):
        design = basis(u, q)
        return design @ _linear(design, y) - y

    q1 = least_squares(lambda q: residual(_first_basis, q, v1).ravel(),
                       [.85, -.01], bounds=([.02, -1], [10, 1])).x

    def loss(q):
        r = residual(_second_basis, q, v2)
        return np.sum(r*r)

    grid = np.geomspace(.05, 7., 48)
    i = np.argmin([loss(q) for q in grid])
    q2 = minimize_scalar(loss, method='bounded',
                         bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                         options={'xatol': 1e-10}).x
    c = _first_basis(z, q1) @ _linear(_first_basis(u, q1), v1)
    d = _second_basis(z, q2) @ _linear(_second_basis(u, q2), v2)
    return np.column_stack((c, d))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train_x = np.asarray(study['train']['x'], dtype=float)
    test_x = np.asarray(study['test']['x'], dtype=float)
    offset, scale = _align(study['panel_a'])
    z = (np.vstack((train_x, test_x))-offset)/scale
    w = _panel_maps(study['panel_c'], z)
    train_w = w[:len(train_x)]
    return {'features': (w-train_w.mean(axis=0))/np.maximum(train_w.std(axis=0), .1)}
