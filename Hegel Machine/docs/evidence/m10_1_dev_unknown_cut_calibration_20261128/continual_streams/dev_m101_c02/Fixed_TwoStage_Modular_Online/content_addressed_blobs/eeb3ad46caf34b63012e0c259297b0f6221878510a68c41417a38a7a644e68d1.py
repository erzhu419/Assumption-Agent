"""Calibrated multi-output regression for the public study."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, raw):
    a = study['panel_a']
    u, v = a['u'], a['v']
    good = np.linalg.norm(v - u, axis=1) < 3
    j = np.arange(raw.shape[1]); k = np.roll(j, -1)

    def terms(x):
        return np.stack((np.ones_like(x), x, x[:, k], x*x[:, k], x**3), axis=-1)

    d = terms(u[good])
    c = np.array([_fit(d[:, i], v[good, i]) for i in j])
    x = raw.copy()
    for _ in range(18):
        err = np.einsum('nij,ij->ni', terms(x), c) - raw
        if np.max(np.abs(err), initial=0.) < 1e-11:
            break
        jac = np.zeros((len(x), len(j), len(j)))
        jac[:, j, j] = c[:, 1] + c[:, 3]*x[:, k] + 3*c[:, 4]*x*x
        jac[:, j, k] = c[:, 2] + c[:, 3]*x
        x -= np.linalg.solve(jac, err[..., None])[..., 0]

    b = study['panel_b']
    def harm(p, w):
        return np.column_stack((_affine(p), np.sin(w*p)))
    def b_loss(w):
        m = harm(b['u'], w)
        return np.sum((m @ _fit(m, b['v']) - b['v'])**2)
    w = minimize_scalar(b_loss, bounds=(2., 3.6), method='bounded').x
    m = harm(b['u'], w)
    return x, harm(x, w) @ _fit(m, b['v'])


def _kernel_options(x, z, n):
    def distance(a):
        a = a / a[:n].std(axis=0)
        return cdist(a, a, 'sqeuclidean') / a.shape[1]

    dx, dz = distance(x), distance(z)
    ds = distance(np.sin(2.8*x))
    for bandwidth in (.24, .34, .48, .68, .96, 1.36, 1.92):
        scale = 2*bandwidth**2
        yield np.exp(-(.5*dx + .5*ds)/scale)
        yield np.exp(-(.5*dx + .25*dz + .25*ds)/scale)


def _predict(x, z, y):
    n = len(y)
    h = _affine(z)
    design = h[:n]
    eye = np.eye(n)
    scores, predictions = [], []
    for k in _kernel_options(x, z, n):
        for ridge in (.001, .005, .02, .08, .32, 1.28):
            factor = cho_factor(k[:n, :n] + ridge*eye, check_finite=False)
            inverse = cho_solve(factor, eye, check_finite=False)
            g = inverse @ design
            trend = np.linalg.solve(design.T @ g, g.T)
            precision = inverse - g @ trend
            weights = precision @ y
            # The projected precision gives deleted-row residuals with the affine trend refitted.
            loo = np.abs(weights / np.diag(precision)[:, None])
            scores.append(loo[:, 0].mean() + 1.4*loo[:, 1:].mean() + .2*loo[:, -1].mean())
            predictions.append(k[n:, :n] @ weights + h[n:] @ (trend @ y))
    scores = np.array(scores)
    blend = np.exp(-(scores - scores.min())/.006)
    return np.einsum('m,mij->ij', blend/blend.sum(), predictions)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, z = _coordinates(study, np.vstack((train['x'], study['test']['x'])))
    p = _predict(x, z, np.column_stack((train['y'], train['traj'])))
    y_hat, state = p[:, 0], p[:, 1:]
    end = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
