"""Reconstruct latent input coordinates and compose calibration panels."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _powers(x, *degrees):
    return np.column_stack((np.ones(len(x)), *(x**k for k in degrees)))


def _inverse(panel, measured):
    u, v = (np.asarray(panel[k], float) for k in ('u', 'v'))
    keep = np.abs(v) < 3
    def design(q, j, w):
        a, b = q[:, j], q[:, (j+2)%q.shape[1]]
        return np.column_stack((np.ones(len(q)), a, np.sin(w*a), np.sin(w*b)))
    def profile(w, return_coef=False):
        coefs, err = [], []
        for j in range(u.shape[1]):
            d = design(u[keep[:, j]], j, w)
            y = v[keep[:, j], j]
            cf = np.linalg.lstsq(d, y, rcond=None)[0]
            coefs.append(cf)
            err.extend(d @ cf - y)
        return coefs if return_coef else np.asarray(err)
    w = least_squares(lambda a: profile(a[0]), [2.15], bounds=(1.0, 4.0)).x[0]
    coefs = profile(w, True)
    def forward(q):
        return np.column_stack([design(q, j, w) @ cf for j, cf in enumerate(coefs)])
    return np.array([least_squares(lambda q: forward(q[None, :])[0]-r, r,
                                  gtol=1e-12).x for r in measured])


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
    def design(q):
        return np.column_stack((np.ones(len(q)), q, *[q[:, i]*q[:, j]
            for i in range(q.shape[1]) for j in range(i, q.shape[1])]))
    return design(x) @ np.linalg.lstsq(design(panel['u']), panel['v1'], rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train']['x'], study['test']['x']
    x = _inverse(study['panel_a'], np.vstack((train, test)))
    c = _panel_c(study['panel_c'], _panel_b(study['panel_b'], x))
    n = len(train)
    return dict(x_train=x[:n].copy(), x_test=x[n:].copy(), c_train=c[:n].copy(), c_test=c[n:].copy())
