"""Recover reference coordinates and apply the second panel's nonlinear gain."""
import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _a_design(x, j):
    a, b = x[:, j], x[:, (j + 1) % 4]
    return np.column_stack((np.ones(len(x)), a, b, a*b, a**3))


def _fit_a(u, v):
    designs = [_a_design(u, j) for j in range(4)]
    weights = np.ones(len(u))
    for _ in range(20):
        coef = np.array([np.linalg.lstsq(d*weights[:, None], v[:, j]*weights, rcond=None)[0]
                         for j, d in enumerate(designs)])
        error = np.linalg.norm(np.column_stack([d@c for d, c in zip(designs, coef)]) - v, axis=1)
        weights = 1/np.hypot(1, error/max(np.median(error), 1e-8))
    keep = np.argsort(error)[:-3]
    coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]
                     for j, d in enumerate(designs)]).T
    inverse_seed = np.linalg.lstsq(_affine(v[keep]), u[keep], rcond=None)[0]
    return coef, inverse_seed


def _invert_a(v, coef, inverse_seed):
    bias, lin, neighbor, pair, cubic = coef
    x = _affine(v) @ inverse_seed
    j = np.arange(4)
    for _ in range(20):
        b = np.roll(x, -1, axis=1)
        residual = bias + lin*x + neighbor*b + pair*x*b + cubic*x**3 - v
        if np.max(np.abs(residual), initial=0) < 1e-11:
            break
        jac = np.zeros((len(x), 4, 4))
        jac[:, j, j] = lin + pair*b + 3*cubic*x*x
        jac[:, j, (j+1)%4] = neighbor + pair*x
        x -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -0.5, 0.5)
    return x


def _b_design(x, k):
    return np.column_stack((_affine(x), x*(-np.expm1(-k*x*x))))


def _fit_b(u, v):
    def score(k):
        q = np.linalg.qr(_b_design(u, k))[0]
        err = (v - q @ (q.T @ v))/(1 - (q*q).sum(axis=1, keepdims=True))
        return np.mean(err**2)
    k = minimize_scalar(score, bounds=(0.15, 2.5), method='bounded').x
    return k, np.linalg.lstsq(_b_design(u, k), v, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    coef, seed = _fit_a(a['u'], a['v'])
    k, gain = _fit_b(b['u'], b['v'])
    return {out: _b_design(_invert_a(study[src]['x'], coef, seed), k) @ gain
            for out, src in (('z_tr', 'train'), ('z_te', 'test'))}
