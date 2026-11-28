"""Recover latent coordinates from panel A and apply the panel B fold."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _fit_a(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    neighbor = np.roll(u, -1, axis=1)
    design = np.stack((np.ones_like(u), u, u**3, (u-1)*neighbor), axis=-1)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(12):
        weights = np.array([np.linalg.lstsq(d[keep], w[keep], rcond=None)[0]
                            for d, w in zip(design.transpose(1, 0, 2), v.T)])
        residual = np.linalg.norm(np.sum(design*weights, axis=-1)-v, axis=1)
        good = residual < max(0.001, 3*np.median(residual))
        if np.array_equal(good, keep) or good.sum() < 6:
            break
        keep = good
    return weights.T


def _invert(x, coeff):
    bias, linear, cubic, coupling = coeff
    j = np.arange(x.shape[1])
    nxt = np.roll(j, -1)

    def forward(z):
        return bias + linear*z + cubic*z**3 + coupling*(z-1)*z[nxt]

    def jac(z):
        a = np.diag(linear + 3*cubic*z*z + coupling*z[nxt])
        a[j, nxt] += coupling*(z-1)
        return a

    init = np.linalg.solve(jac(np.zeros_like(bias)), (x-bias).T).T
    return np.array([least_squares(lambda z: forward(z)-row, start, jac=jac,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12).x
                     for row, start in zip(x, init)])


def _fold_b(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def features(q, a):
        return np.column_stack((np.ones(len(a)), erf(q[0]*a) + q[1]*a))

    def fit(q):
        design = features(q, u)
        weights = np.linalg.lstsq(design, v, rcond=1e-10)[0]
        return weights, design @ weights - v

    sols = [least_squares(lambda q: fit(q)[1].ravel(), [a, -.5*a],
                          bounds=([.1, -8], [4, 8]), max_nfev=150)
            for a in (.4, .8, 1.4, 2.2)]
    q = min(sols, key=lambda result: result.cost).x
    return features(q, z) @ fit(q)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = (np.asarray(study[key]['x'], dtype=float) for key in ('train', 'test'))
    n = len(train)
    z = _invert(np.vstack((train, test)), _fit_a(study['panel_a']))
    b = _fold_b(study['panel_b'], z)
    return {'z_train': z[:n].copy(), 'z_test': z[n:].copy(),
            'b_train': b[:n].copy(), 'b_test': b[n:].copy()}
