"""Recover calibrated spatial coordinates and construct panel-C predictors."""
import numpy as np
from scipy.optimize import least_squares


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    distance = np.linalg.norm(u-v, axis=1)
    keep = distance < max(4.0, 4*np.median(distance))
    u, v = u[keep], v[keep]
    partner = np.array([2, 3, 0, 1])

    def design(q, rates):
        z = q[:, partner]
        return np.stack((np.ones_like(q), q, z, np.sin(rates[0]*q),
                         np.sin(rates[1]*z)), axis=2)

    def profile(rates):
        a = design(u, rates)
        w = np.stack([_fit(a[:, j], v[:, j]) for j in range(4)])
        return a, w

    def residual(rates):
        a, w = profile(rates)
        return (np.einsum('njk,jk->nj', a, w) - v).ravel()

    rates = least_squares(residual, [3., 2.4], bounds=([1.5, 1.5], [4.5, 4.5]),
                          max_nfev=80).x
    _, w = profile(rates)
    a = _fit(_affine(u), v)
    q = np.linalg.solve(a[1:].T, (x-a[0]).T).T

    def forward(z):
        return np.einsum('njk,jk->nj', design(z, rates), w)

    def jacobian(z):
        j = np.zeros((len(z), 4, 4))
        i = np.arange(4)
        j[:, i, i] = w[:, 1] + rates[0]*w[:, 3]*np.cos(rates[0]*z)
        j[:, i, partner] = w[:, 2] + rates[1]*w[:, 4]*np.cos(rates[1]*z[:, partner])
        return j

    for _ in range(14):
        error = forward(q)-x
        if not len(q) or np.max(np.abs(error)) < 1e-11:
            break
        delta = np.linalg.solve(jacobian(q), error[..., None])[..., 0]
        q -= np.clip(delta, -0.6, 0.6)
    for i in np.flatnonzero(np.max(np.abs(forward(q)-x), axis=1) > 1e-8):
        q[i] = least_squares(lambda z: forward(z[None, :])[0]-x[i], q[i],
                             jac=lambda z: jacobian(z[None, :])[0], max_nfev=60).x
    return q


def _products(q):
    return np.column_stack((q[:, 0]*q[:, 1], q[:, 0]*q[:, 3],
                            q[:, 1]*q[:, 2], q[:, 2]*q[:, 3]))


def _features(q, panel):
    u = panel['u']
    c = _affine(_products(q)) @ _fit(_affine(_products(u)), panel['v1'])
    cos = lambda z: _affine(np.cos(2.4*z))
    d = cos(q) @ _fit(cos(u), panel['v2'])
    cd = (c[:, :, None]*d[:, None, :]).reshape(len(q), 16)
    return _affine(np.column_stack((q, c, d, cd)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    f = _features(_calibrate(study['panel_a'], x), study['panel_c'])
    n = len(train['x'])
    return {'regression': f[n:] @ _fit(f[:n], train['y']),
            'train_features': f[:n], 'test_features': f[n:]}
