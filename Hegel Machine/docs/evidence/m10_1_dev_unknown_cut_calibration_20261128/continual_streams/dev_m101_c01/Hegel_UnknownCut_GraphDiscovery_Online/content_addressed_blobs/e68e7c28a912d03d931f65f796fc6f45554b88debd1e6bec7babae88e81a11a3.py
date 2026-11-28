"""Panel-calibrated spatial regression and low-rank trajectory prediction."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import cdist


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ols(f, y):
    return np.linalg.lstsq(f, y, rcond=None)[0]


def _calibrate(panel, x):
    f = _aff(panel['u'])
    v = panel['v']
    keep = np.ones(len(f), dtype=bool)
    for _ in range(6):
        c = _ols(f[keep], v[keep])
        r = np.linalg.norm(f @ c - v, axis=1)
        keep = r < max(0.001, 4 * np.median(r))
    return _aff(x) @ c


def _degree(x, d):
    return np.column_stack([np.prod(x[:, p], axis=1)
                            for p in combinations_with_replacement(range(x.shape[1]), d)])


def _cross(x, v):
    return (x[:, :, None] * v[:, None, :]).reshape(len(x), -1)


def _gram(f):
    return f @ f.T


def _gp(k, y, floor):
    n = len(y)
    a = k[:, :n, :n]
    b = k[:, n:, :n]
    scale = np.diagonal(a, axis1=1, axis2=2).mean(axis=1)
    eye = np.eye(n)
    a = np.concatenate((a / scale[:, None, None], eye[None]))
    b = b / scale[:, None, None]
    sd = max(float(np.std(y)), 1e-8)
    yy = y / sd
    flat = a.reshape(len(a), -1)

    def objective(p, predict=False):
        w = np.exp(p)
        c = cho_factor((w @ flat).reshape(n, n), check_finite=False)
        alpha = cho_solve(c, yy, check_finite=False)
        if predict:
            return sd * np.einsum('i,ijk,k->j', w[:-1], b, alpha)
        q = cho_solve(c, eye, check_finite=False) - np.outer(alpha, alpha)
        return 0.5 * yy @ alpha + np.log(np.diag(c[0])).sum(), 0.5 * w * (flat @ q.ravel())

    lower = min(0., np.log(floor / sd**2))
    p = minimize(objective, np.r_[np.full(len(k), -2.), -7.], jac=True,
                 method='L-BFGS-B', bounds=[(-15, 6)] * len(k) + [(lower, 1)],
                 options={'maxiter': 180}).x
    return objective(p, True)


def _rational(x, f, y):
    n = len(y)
    squared = x**2

    def design(p):
        return f / (1. + squared @ p)[:, None]

    def residual(p):
        a = design(p)[:n]
        return a @ _ols(a, y) - y

    p = least_squares(residual, np.full(x.shape[1], .2), bounds=(0, 5), max_nfev=40).x
    a = design(p)
    return a[n:] @ _ols(a[:n], y)


def _features(study, raw):
    x = _calibrate(study['panel_a'], raw)
    b = study['panel_b']
    cb = _ols(_aff(b['u']), b['v'])
    c = study['panel_c']
    u = c['u']
    def diagonal(t):
        return np.column_stack((_aff(t), t*t))
    cd = _ols(diagonal(u), c['v2'])

    def c1_features(t, p):
        return _aff(np.tanh(t * p[:4] + p[4:]))

    def c1_residual(p):
        f = c1_features(u, p)
        return (f @ _ols(f, c['v1']) - c['v1']).ravel()

    p = least_squares(c1_residual, np.r_[np.full(4, 1.6), np.zeros(4)],
                      bounds=(np.r_[np.full(4, .2), np.full(4, -2.)],
                              np.r_[np.full(4, 4.), np.full(4, 2.)])).x
    cc = _ols(c1_features(u, p), c['v1'])
    kernels = {'base': _gram(np.column_stack((_aff(x), _degree(x, 2))))}
    rational = []
    for name, t in [('A', x), ('B', _aff(x) @ cb), ('Br', _aff(raw) @ cb)]:
        q2 = np.column_stack((_aff(t), _degree(t, 2)))
        cube = _degree(t, 3)
        q3 = np.column_stack((q2, cube))
        v1 = c1_features(t, p) @ cc
        v2 = diagonal(t) @ cd
        cv2 = _cross(t, v2)
        kernels[name+'3'] = _gram(cube)
        kernels[name+'4'] = _gram(_degree(t, 4))
        kernels[name+'C1'] = _gram(np.column_stack((v1, _cross(t, v1), _degree(v1, 2))))
        kernels[name+'C2'] = _gram(np.column_stack((cv2, _degree(v2, 2))))
        kernels[name+'iso'] = _gram(q3 / (1. + .2 * np.sum(t*t, axis=1))[:, None])
        kernels[name+'aniso'] = _gram(q3 / (1. + (t*t) @ [.4, .5, .25, .1])[:, None])
        if name != 'A':
            rational.append((t, np.column_stack((q2, cv2))))
            h = np.sin(1.6*t)
            kernels[name+'H'] = _gram(np.column_stack((h, _cross(h, t))))
    d = cdist(x, x, 'sqeuclidean')
    for length in (1, 2, 4):
        kernels['R'+str(length)] = np.exp(-d/(2*length**2))
    return kernels, rational


def _stack(kernels, names):
    return np.stack([kernels[k] for k in names.split()])


def solve_batch(study: dict) -> dict:
    train = study['train']
    raw = np.vstack((train['x'], study['test']['x']))
    k, rational = _features(study, raw)
    joint = _stack(k, 'base A3 B3 Br3 A4 B4 Br4 AC1 BC1 BrC1 AC2 BC2 BrC2 Aiso Biso Briso Aaniso Baniso Braniso R1 R2 R4')
    reduced = _stack(k, 'base A3 B3 BC2 BrC2 Biso Briso Baniso Braniso')
    y = np.mean([_gp(joint, train['y'], .00007), _gp(reduced, train['y'], .00007),
                 *(_rational(x, f, train['y']) for x, f in rational)], axis=0)

    z = train['traj']
    g = np.linalg.svd(z, full_matrices=False)[2][0]
    g = g / g[-1]
    amplitude = z @ g / (g @ g)
    branch = _stack(k, 'base BC2 BrC2')
    harmonic = _stack(k, 'base A3 B3 Br3 BC1 BrC1 BH BrH R1 R2 R4')
    a = (0.55 * _gp(joint, amplitude, .000018) + 0.25 * _gp(branch, amplitude, .000018)
         + 0.20 * _gp(harmonic, amplitude, .000018))
    state = a[:, None] * g
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
