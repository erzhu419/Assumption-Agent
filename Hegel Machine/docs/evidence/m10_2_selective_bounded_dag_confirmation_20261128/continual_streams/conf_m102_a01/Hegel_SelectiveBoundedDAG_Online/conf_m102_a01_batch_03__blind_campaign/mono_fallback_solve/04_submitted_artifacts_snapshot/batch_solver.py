"""Calibrated nonlinear prediction with a shared low-rank time response."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    # Pairwise slopes are insensitive to a minority of contaminated readings.
    p = study['panel_a']
    offset, gain = [], []
    for u, v in zip(p['u'].T, p['v'].T):
        i, j = np.triu_indices(len(u), 1)
        d = u[i] - u[j]
        good = np.abs(d) > .08
        slope = np.median((v[i[good]] - v[j[good]]) / d[good])
        bias = np.median(v - slope * u)
        error = np.abs(v - bias - slope * u)
        keep = error < max(.001, 4 * np.median(error))
        if np.count_nonzero(keep) >= 2:
            bias, slope = np.linalg.lstsq(_affine(u[keep]), v[keep], rcond=None)[0]
        offset.append(bias)
        gain.append(slope)
    z = (x - offset) / gain
    p = study['panel_b']
    b = np.linalg.lstsq(_affine(p['u']), p['v'], rcond=None)[0]
    h = np.linalg.solve(b[1:].T, (z - b[0]).T).T
    return z, h


def _polynomial(z, n, target):
    features = [np.ones(len(z))]
    degrees = [0]
    for d in range(1, 6):
        for idx in combinations_with_replacement(range(z.shape[1]), d):
            features.append(np.prod(z[:, idx], axis=1))
            degrees.append(d)
    f = np.column_stack(features)
    f /= np.maximum(np.sqrt(np.mean(f[:n]**2, axis=0)), 1e-10)
    d = np.array(degrees)
    penalty = np.where(d <= 1, 1e-9, .025 * 5.0**(d - 2))
    g = f / np.sqrt(penalty)
    return (g[n:] @ g[:n].T) @ cho_solve(cho_factor(np.eye(n) + g[:n] @ g[:n].T), target)


def _kernel(x, z, n, target, length):
    g = z @ z.T
    diag = g.diagonal()
    d = np.maximum(diag[:, None] + diag[None, :] - 2 * g, 0)
    mats = np.stack((g**2, np.exp(-d / (2 * length**2))))
    mats /= np.maximum(np.sqrt(np.mean(mats[:, :n, :n]**2, axis=(1, 2))), 1e-12)[:, None, None]
    c = np.concatenate((mats[:, :n, :n], np.eye(n)[None]), axis=0)
    base = 2 + 4 * (x @ x[:n].T)
    cross = mats[:, n:, :n]
    pred = []
    for response in target.T:
        def loss(theta):
            w = np.exp(theta)
            k = base[:n] + np.einsum('i,ijk->jk', w, c)
            factor = cho_factor(k, check_finite=False)
            alpha = cho_solve(factor, response, check_finite=False)
            inv = cho_solve(factor, np.eye(n), check_finite=False)
            value = .5 * (response @ alpha) + np.log(np.diag(factor[0])).sum()
            grad = .5 * w * np.einsum('ij,kij->k', inv - np.outer(alpha, alpha), c)
            return value, grad
        fit = minimize(loss, [-3, -3, -9], jac=True, method='L-BFGS-B',
                       bounds=[(-17, 3), (-17, 3), (-13, -3)], options={'maxiter': 80})
        w = np.exp(fit.x)
        k = base[:n] + np.einsum('i,ijk->jk', w, c)
        k_cross = base[n:] + np.einsum('i,ijk->jk', w[:2], cross)
        pred.append(k_cross @ cho_solve(cho_factor(k), response))
    return np.column_stack(pred)


def _components(study):
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z, h = _coordinates(study, x)
    u, s, vt = np.linalg.svd(train['traj'], full_matrices=False)
    r = min(2, len(s))
    target = np.column_stack((train['y'], u[:, :r] * s[:r]))
    poly = .75 * _polynomial(h, n, target) + .25 * _polynomial(x, n, target)
    smooth = (_kernel(x, x, n, target, 1.) + _kernel(x, z, n, target, 2.)
              + _kernel(x, h, n, target, 2.)) / 3
    return poly, smooth, vt[:r]


def _readout(x, z, h, b, y):
    n = len(y)
    coords = (x, z, h, z @ b[1:] + b[0], x @ b[1:] + b[0],
              np.linalg.solve(b[1:].T, (x - b[0]).T).T)
    def terms(g, d):
        return np.column_stack([np.prod(g[:, idx], axis=1)
                                for idx in combinations_with_replacement(range(g.shape[1]), d)])
    def scale(f):
        return f / np.maximum(np.sqrt(np.mean(f[:n]**2, axis=0)), 1e-8)
    base = 10 + 10 * (z @ z[:n].T)
    eye = np.eye(n)
    def evidence(groups):
        fs = [scale(f) for f in groups]
        cov = [f @ f[:n].T / f.shape[1] for f in fs]
        a = np.stack([k[:n] for k in cov] + [eye])
        cross = np.stack([k[n:] for k in cov])
        mu = np.array([-6.] * len(cov) + [-8.])
        def objective(p):
            w = np.exp(p)
            c = cho_factor(base[:n] + np.einsum('i,ijk->jk', w, a))
            alpha = cho_solve(c, y)
            derivative = cho_solve(c, eye) - np.outer(alpha, alpha)
            loss = .5 * y @ alpha + np.log(c[0].diagonal()).sum() + .1 * np.sum((p-mu)**2)
            grad = .5 * w * np.einsum('ijk,jk->i', a, derivative) + .2 * (p-mu)
            return loss, grad
        p = minimize(objective, mu, jac=True, bounds=[(-17, 2)] * len(cov) + [(-13, -4)],
                     method='L-BFGS-B', options={'maxiter': 100}).x
        w = np.exp(p)
        c = cho_factor(base[:n] + np.einsum('i,ijk->jk', w, a))
        return (base[n:] + np.einsum('i,ijk->jk', w[:-1], cross)) @ cho_solve(c, y)
    polys = [[terms(g, d) for d in range(2, 7)] for g in coords]
    groups = []
    for g, p in zip(coords, polys):
        groups.extend(p[:3])
        for v in (np.tanh(.7*g), np.tanh(1.25*g), g/(1+.22*g*g)):
            groups.append(np.column_stack((v, terms(v, 2))))
    g = np.concatenate([scale(f) / np.sqrt(lam) for f, lam in
                        zip([_affine(z)] + polys[4][:3], [1e-7, .035, 1.6, 8.])], axis=1)
    shrink = (g[n:] @ g[:n].T) @ cho_solve(cho_factor(eye + g[:n] @ g[:n].T), y)
    return .55*evidence(groups) + .25*evidence(polys[4]) + .10*shrink + .10*evidence(polys[5])


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.vstack((train['x'], study['test']['x']))
    z, h = _coordinates(study, x)
    panel = study['panel_b']
    b = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    y = _readout(x, z, h, b, np.asarray(train['y']))
    poly, smooth, vt = _components(study)
    state = (.4 * poly[:, 1:] + .6 * smooth[:, 1:]) @ vt
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
