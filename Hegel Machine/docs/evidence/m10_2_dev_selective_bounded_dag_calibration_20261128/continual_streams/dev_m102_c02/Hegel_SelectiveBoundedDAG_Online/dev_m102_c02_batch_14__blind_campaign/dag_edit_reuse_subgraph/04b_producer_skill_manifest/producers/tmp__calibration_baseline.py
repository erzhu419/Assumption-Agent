"""Recover latent coordinates from panel A and apply the panel-B operator."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _latent(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    opposite = (np.arange(u.shape[1]) + 2) % u.shape[1]
    residual = np.linalg.norm(v - _affine(u) @ _ls(_affine(u), v), axis=1)
    keep = residual < max(6 * np.median(residual), 0.1)
    if keep.sum() >= 8:
        u, v = u[keep], v[keep]

    def basis(z, p):
        return np.stack((np.ones_like(z), z, np.tanh(p[0]*z), np.sin(p[1]*z[:, opposite])), axis=-1)

    def project(p, return_coef=False):
        a = basis(u, p)
        coef = np.stack([_ls(a[:, j], v[:, j]) for j in range(u.shape[1])])
        return coef if return_coef else (np.sum(a*coef, axis=-1)-v).ravel()

    p = least_squares(project, [1.8, 2.4], bounds=([0.3, 0.8], [4., 4.])).x
    coef = project(p, True)
    return np.stack([root(lambda z: np.sum(basis(z[None], p)[0]*coef, axis=-1)-q, q).x for q in x])


def _panel_b(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def basis(q, a):
        return np.c_[_affine(q / (1 + a*q*q)), q]
    def loss(a):
        d = basis(u, a)
        return np.mean((d @ _ls(d, v) - v)**2)
    a = minimize_scalar(loss, bounds=(.05, 4.), method='bounded').x
    return basis(z, a) @ _ls(basis(u, a), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    q = np.asarray(study['test']['x'], dtype=float)
    c = _panel_b(study['panel_b'], _latent(study['panel_a'], np.concatenate((x, q))))
    return {'c': c[:len(x)], 'query': c[len(x):]}
