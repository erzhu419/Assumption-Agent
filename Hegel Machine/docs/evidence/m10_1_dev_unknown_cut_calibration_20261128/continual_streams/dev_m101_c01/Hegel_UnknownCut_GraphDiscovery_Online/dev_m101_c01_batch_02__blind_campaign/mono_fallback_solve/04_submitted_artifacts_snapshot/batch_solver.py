from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    fit = np.column_stack([np.ones(len(u)), u])
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        coef = np.linalg.lstsq(fit[keep], v[keep], rcond=None)[0]
        r = np.linalg.norm(fit @ coef - v, axis=1)
        keep = r < max(0.001, 4 * np.median(r))
    return x @ coef[1:] + coef[0]


def _poly(x, d):
    return np.column_stack([np.prod(x[:, p], axis=1)
                            for p in combinations_with_replacement(range(x.shape[1]), d)]) if d else np.ones((len(x), 1))


def _temporal(t, z):
    def basis(p):
        k, damp, omega = p
        return np.column_stack([-np.expm1(-k * t), np.exp(-damp*t)*np.sin(omega*t)])

    def residual(p):
        f = basis(p)
        return (f @ np.linalg.lstsq(f, z.T, rcond=None)[0] - z.T).ravel()

    p = least_squares(residual, [.5, .2, 1.1], bounds=([.01, 0, .5], [4, 2, 2])).x
    f = basis(p)
    q, _ = np.linalg.qr(f)
    return q


def _quadratic_gram(x):
    f = np.column_stack([_poly(x, j) for j in range(3)])
    return f @ f.T


def _blocks(x):
    d = cdist(x, x, 'sqeuclidean')
    r = np.sqrt(5*d)/2
    base = np.stack([_quadratic_gram(x), np.exp(-d/8), (1+r+r*r/3)*np.exp(-r)])
    harmonic = np.stack([_quadratic_gram(fn(c*x))
                         for fn in (np.tanh, np.sin) for c in (.6, 1.3, 2.)])
    wide = np.stack([np.exp(-d/24), np.exp(-d/3.5)])
    return base, np.concatenate([base, harmonic]), np.concatenate([base, wide])


def _regress(blocks, y):
    n, m = y.shape
    identity = np.eye(n)
    k = blocks[:, :n, :n].copy()
    l = blocks[:, n:, :n].copy()
    norm = np.diagonal(k, axis1=1, axis2=2).mean(axis=1)
    k = np.concatenate([k / norm[:, None, None], identity[None]])
    l = np.concatenate([l / norm[:, None, None], np.zeros_like(l[:1])])
    std = np.maximum(y.std(axis=0), 1e-12)
    y = y / std

    def likelihood(p):
        w = np.exp(p)
        factor = cho_factor(np.einsum('k,kij->ij', w, k), check_finite=False)
        alpha = cho_solve(factor, y, check_finite=False)
        inv = cho_solve(factor, identity, check_finite=False)
        value = .5*np.sum(alpha*y) + m*np.log(np.diag(factor[0])).sum()
        grad = .5*w*np.einsum('kij,ij->k', k, m*inv-alpha@alpha.T)
        return value, grad

    p = minimize(likelihood, np.r_[np.zeros(len(k)-1), -8.], jac=True,
                 method='L-BFGS-B', bounds=[(-19, 8)]*(len(k)-1)+[(-18, -1)]).x
    w = np.exp(p)
    factor = cho_factor(np.einsum('k,kij->ij', w, k), check_finite=False)
    return (np.einsum('k,kij->ij', w, l) @ cho_solve(factor, y, check_finite=False))*std


def solve_batch(study: dict) -> dict:
    train = study['train']
    raw = np.vstack([train['x'], study['test']['x']])
    x = _calibrate(study['panel_a'], raw)
    base, harmonic, wide = _blocks(x)
    target = train['y'][:, None]
    y = 0.5 * (_regress(base, target) + _regress(harmonic, target))[:, 0]

    z = train['traj']
    q = _temporal(study['eval_times'], z)
    amplitude = z @ q
    state = (0.5 * (_regress(base, amplitude) + _regress(wide, amplitude))) @ q.T
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
