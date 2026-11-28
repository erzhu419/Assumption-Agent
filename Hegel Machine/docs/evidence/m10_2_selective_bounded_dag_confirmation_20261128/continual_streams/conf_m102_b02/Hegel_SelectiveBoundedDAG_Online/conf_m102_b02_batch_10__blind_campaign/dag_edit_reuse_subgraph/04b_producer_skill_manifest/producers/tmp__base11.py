"""Reconstruct latent input coordinates and compose calibration panels."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _powers(x, *degrees):
    return np.column_stack((np.ones(len(x)), *(x**k for k in degrees)))


def _inverse(panel, measured):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    d = _powers(u, 1, 3)
    keep = np.abs(v) < 3
    coef = np.column_stack([np.linalg.lstsq(d[keep[:, j]], v[keep[:, j], j], rcond=None)[0]
                            for j in range(v.shape[1])])
    p = u.shape[1]
    linear, cubic = coef[1:1+p], coef[1+p:]
    latent = np.linalg.solve(linear.T, (measured - coef[0]).T).T
    for _ in range(18):
        r = coef[0] + latent @ linear + latent**3 @ cubic - measured
        if np.max(np.abs(r)) < 1e-11:
            break
        jac = linear.T[None, :, :] + cubic.T[None, :, :] * (3 * latent[:, None, :]**2)
        latent -= np.clip(np.linalg.solve(jac, r[..., None])[..., 0], -0.35, 0.35)
    return latent


def _panel_b(panel, x):
    u, v = panel['u'], panel['v']
    def design(q, par):
        k, a, b, c = par
        z = erf(np.exp(k)*q + a*q**3) + b*q**3 + c*q
        return _powers(z, 1)
    def fit(par):
        d = design(u, par)
        coef = np.linalg.lstsq(d, v, rcond=None)[0]
        return (d @ coef - v).ravel(), coef
    par = least_squares(lambda par: fit(par)[0], [.63, 4., -.62, -2.13], max_nfev=200).x
    return design(x, par) @ fit(par)[1]


def _panel_c(panel, x):
    return _powers(x, 1, 2) @ np.linalg.lstsq(_powers(panel['u'], 1, 2), panel['v2'], rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train']['x'], study['test']['x']
    x = _inverse(study['panel_a'], np.vstack((train, test)))
    c = _panel_c(study['panel_c'], _panel_b(study['panel_b'], x))
    n = len(train)
    return dict(x_train=x[:n].copy(), x_test=x[n:].copy(), c_train=c[:n].copy(), c_test=c[n:].copy())
