"""Recover calibrated cosine coordinates and fit the two readouts independently."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _operator(study):
    u, v = (np.asarray(study['panel_a'][key], float) for key in ('u', 'v'))
    distance = np.max(np.abs(v-u), axis=1)
    med = np.median(distance)
    keep = distance < med + 8*max(np.median(abs(distance-med)), 1e-8)
    u, v = u[keep], v[keep]
    um, vm = u.mean(0), v.mean(0)
    gain = np.sum((u-um)*(v-vm), axis=0)/np.sum((u-um)**2, axis=0)
    pu, pv = (np.asarray(study['panel_b'][key], float) for key in ('u', 'v'))
    def design(x, k):
        return _affine(np.cos(k*x))
    def loss(k):
        a = design(pu, k)
        return np.sum((a@_fit(a, pv)-pv)**2)
    grid = np.linspace(0.2, 6., 100)
    j = np.argmin([loss(k) for k in grid])
    k = minimize_scalar(loss, bounds=(grid[max(0,j-1)], grid[min(len(grid)-1,j+1)]), method='bounded').x
    w = _fit(design(pu, k), pv)
    return lambda x: design((x-vm)/gain+um, k)@w


def _quadratic(z):
    return np.column_stack((_affine(z), *[z[:, i]*z[:, j] for i, j in combinations_with_replacement(range(z.shape[1]), 2)]))


def _scalar(z, q, y):
    # Identify the two dominant interactions using training observations only.
    a, b = _quadratic(z), _quadratic(q)
    support = 5 + np.argsort(np.abs(_fit(a, y)[5:]))[-2:]
    d = np.column_stack((a[:, :5], a[:, support].sum(axis=1)))
    e = np.column_stack((b[:, :5], b[:, support].sum(axis=1)))
    return e @ _fit(d, y)


def _trajectory(a, b, t, observations):
    # Variable projection over time constants after projecting onto panel coordinates.
    c = _fit(a, observations)
    fitted = a@c
    def basis(p):
        return -np.expm1(-t[:,None]*np.exp(p))
    def residual(p):
        f = basis(p)
        return (f@_fit(f, fitted.T)-fitted.T).ravel()
    p = least_squares(residual, np.log([0.42, 1.8]), bounds=(-5., 5.)).x
    f = basis(p)
    return b@(_fit(f, c.T).T@f.T)


def solve_batch(study: dict) -> dict:
    train = study['train']
    transform = _operator(study)
    z = transform(np.asarray(train['x'], float))
    q = transform(np.asarray(study['test']['x'], float))
    y_hat = _scalar(z, q, np.asarray(train['y'], float))
    state_pred = _trajectory(_affine(z), _affine(q), np.asarray(study['eval_times'], float), np.asarray(train['traj'], float))
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
