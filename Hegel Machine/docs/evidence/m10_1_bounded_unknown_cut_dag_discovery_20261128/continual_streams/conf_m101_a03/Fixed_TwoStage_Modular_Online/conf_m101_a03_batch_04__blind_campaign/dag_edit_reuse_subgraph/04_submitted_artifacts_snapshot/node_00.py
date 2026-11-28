"""Recover latent coordinates from Panel A, then apply the B and C v2 calibrations."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _panel_a_basis(u):
    neighbor = np.roll(u, -1, axis=1)
    return np.stack((np.ones_like(u), u, neighbor, u * neighbor, u**3), axis=-1)


def _panel_a_fit(panel):
    design = _panel_a_basis(panel['u'])
    v = panel['v']
    w = np.ones(len(v))
    for _ in range(35):
        coef = np.stack([np.linalg.lstsq(design[:, j] * w[:, None], v[:, j] * w,
                                         rcond=None)[0] for j in range(v.shape[1])])
        residual = np.linalg.norm(np.einsum('njk,jk->nj', design, coef) - v, axis=1)
        w = 1 / np.hypot(1, residual / max(np.median(residual), 1e-8))
    return coef


def _invert(x, coef):
    diag = np.arange(coef.shape[0])
    linear = np.diag(coef[:, 1])
    linear[diag, (diag + 1) % len(diag)] = coef[:, 2]
    init = np.linalg.solve(linear, (x - coef[:, 0]).T).T

    def observation(u):
        return np.einsum('jk,jk->j', _panel_a_basis(u[None])[0], coef)

    return np.stack([least_squares(lambda u: observation(u) - row, start).x
                     for row, start in zip(x, init)])


def _transfer(panel, key, z, basis, bounds):
    u, v = panel['u'], panel[key]

    def design(x, rate):
        return np.column_stack((np.ones(len(x)), basis(x, rate)))

    def fit(rate):
        a = design(u, rate)
        q, r = np.linalg.qr(a, mode='reduced')
        proj = q.T @ v
        residual = (q @ proj - v) / (1 - np.sum(q*q, axis=1))[:, None]
        return np.mean(residual**2), np.linalg.solve(r, proj)

    optimum = minimize_scalar(lambda rate: fit(rate)[0], bounds=bounds, method='bounded')
    return design(z, optimum.x) @ fit(optimum.x)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    n = len(train['x'])
    z = _invert(np.concatenate((train['x'], test['x'])), _panel_a_fit(study['panel_a']))
    b = _transfer(study['panel_b'], 'v', z,
                  lambda u, rate: np.column_stack((u, np.tanh(rate*u))), (0.4, 3.0))
    c = _transfer(study['panel_c'], 'v2', z,
                  lambda u, rate: np.column_stack((u, u*u, np.tanh(rate*u*u))), (0.1, 4.0))
    return {'b_train': b[:n], 'b_test': b[n:], 'c_train': c[:n], 'c_test': c[n:],
            'y_train': np.asarray(train['y']), 'traj_train': np.asarray(train['traj'])}
