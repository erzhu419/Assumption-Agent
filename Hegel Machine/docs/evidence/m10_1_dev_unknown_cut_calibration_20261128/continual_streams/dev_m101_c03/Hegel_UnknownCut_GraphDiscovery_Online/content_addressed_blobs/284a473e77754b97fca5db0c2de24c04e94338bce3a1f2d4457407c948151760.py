"""Calibrated scalar and three-mode trajectory predictors."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _lstsq(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _next(x):
    return np.roll(x, -1, axis=1)


def _similarity(u, v):
    q = u - u.mean(0)
    l, s, r = np.linalg.svd(q.T @ (v-v.mean(0)))
    m = (l @ r) * (s.sum()/np.sum(q*q))
    return m, v.mean(0) - u.mean(0) @ m


def _features(study, x):
    p = study['panel_a']
    u, v = p['u'], p['v']
    a = []
    for j in range(u.shape[1]):
        q, w = u[:, j], u[:, (j+1) % u.shape[1]]
        basis = _ones(np.column_stack((q, w, q*w, q**3)))
        y = v[:, j]
        weights = np.ones(len(u))
        for _ in range(45):
            coef = _lstsq(basis*weights[:, None], y*weights)
            error = abs(basis @ coef - y)
            weights = 1/np.sqrt(np.maximum(error, 1e-4))
        keep = error < 0.001
        a.append(_lstsq(basis[keep], y[keep]))
    bias, diag, cyc, prod, cubic = np.array(a).T

    def invert(row):
        def residual(z):
            w = np.roll(z, -1)
            return bias + diag*z + cyc*w + prod*z*w + cubic*z**3 - row
        return least_squares(residual, row, gtol=1e-12).x

    z = np.array([invert(row) for row in x])
    p = study['panel_c']
    u = p['u']
    m, k = _similarity(u*_next(u), p['v1'])
    c = (z*_next(z)) @ m + k

    def cosine_fit(rate):
        q = np.cos(rate*u)
        m, k = _similarity(q, p['v2'])
        return np.mean((q @ m + k-p['v2'])**2)

    rate = minimize_scalar(cosine_fit, bounds=(1.8, 3), method='bounded').x
    m, k = _similarity(np.cos(rate*u), p['v2'])
    d = np.cos(rate*z) @ m + k
    return z, c, d


def _powers(x, degree):
    return np.column_stack([np.prod(x[:, j], axis=1)
                            for j in combinations_with_replacement(range(x.shape[1]), degree)])


def _temporal(t, h, groups, pen=0.001, inspect=False):
    n, nt = h.shape
    groups = [(g-g[:n].mean(0))/np.maximum(g[:n].std(0), 1e-8) for g in groups]
    mu = h.mean(0)
    reg = np.diag(np.concatenate([np.full(g.shape[1], np.sqrt(pen)*(0.4 if j==1 else 1))
                                  for j, g in enumerate(groups)]))
    target = np.r_[(h-mu).ravel(), np.zeros(len(reg))]

    def design(rate, train):
        a, b, c = rate
        modes = np.array([-np.expm1(-a*t), t**b, t/(1+c*t)])
        modes /= modes[:, -1:]
        blocks = []
        for mode, g in zip(modes, groups):
            g = g[:n] if train else g[n:]
            blocks.append((g[:, None, :]*mode[None, :, None]).reshape(len(g)*nt, g.shape[1]))
        return np.column_stack(blocks)

    def profile(rate):
        mat = np.vstack((design(rate, True), reg))
        coef = _lstsq(mat, target)
        return mat @ coef-target, coef

    r = least_squares(lambda rate: profile(rate)[0], [1.8, .5, .4],
                      bounds=([.5, .25, .015], [4, .8, 4]), max_nfev=40)
    err, coef = profile(r.x)
    pred = (design(r.x, False) @ coef).reshape(-1, nt) + mu
    return (pred, r.x, coef, np.std(err[:n*nt])) if inspect else pred


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    z, c, d = _features(study, np.vstack((train['x'], study['test']['x'])))
    dot = (c*d).sum(1, keepdims=True)
    p3 = _powers(z, 3)
    scalar = _ones(np.column_stack((c, d, dot, p3, z)))
    y = scalar[n:] @ _lstsq(scalar[:n], train['y'])
    slow = np.c_[d, c, c[:, 0]*d[:, 1], z]
    state = _temporal(study['eval_times'], train['traj'], [np.c_[c, dot], p3, slow])
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
