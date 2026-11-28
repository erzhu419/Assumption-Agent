import numpy as np
from itertools import combinations
from scipy.optimize import least_squares


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(h, y):
    return np.linalg.lstsq(h, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    w = np.ones(len(u))
    for _ in range(20):
        b = np.stack([_fit(_one(u[:, j:j+1])*w[:, None], v[:, j]*w) for j in range(4)])
        r = np.linalg.norm(v - b[:, 0] - u*b[:, 1], axis=1)
        w = 1./np.hypot(1., r/max(np.median(r), .0002))
    return (x-b[:, 0])/b[:, 1]


def _branch(u, v, z, kind):
    def design(x, p):
        a = x*p[:4]+p[4:]
        nonlin = np.tanh(a) if kind == 1 else np.expm1(-a*a)
        return _one(np.column_stack((x, nonlin)))

    def residual(p):
        h = design(u, p)
        return (h@_fit(h, v)-v).ravel()

    p = least_squares(residual, np.r_[np.full(4, 1.49 if kind == 1 else .542), np.zeros(4)],
                      bounds=(np.r_[np.full(4, .15), np.full(4, -.1)], np.r_[np.full(4, 3.), np.full(4, .1)]),
                      max_nfev=60).x
    return design(z, p) @ _fit(design(u, p), v)


def _predict(study):
    tr = study['train']
    n = len(tr['x'])
    x = np.vstack((tr['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    p = study['panel_c']
    c = _branch(p['u'], p['v1'], z, 1)
    d = _branch(p['u'], p['v2'], z, 2)
    h0 = _one(np.column_stack((z, c, d, np.sum(c*d, axis=1))))
    h1 = _one(np.column_stack((z, c, d, *[a[:, i]*a[:, j] for a in (c,d) for i in range(4) for j in range(i, 4)],
                               (c[:, :, None]*d[:, None, :]).reshape(len(z), 16))))
    t = np.asarray(study['eval_times'])
    q = -np.expm1(-t[:, None]*np.array([.392, 1.857]))
    modes = np.linalg.qr(q, mode='reduced')[0]
    y = np.column_stack((tr['y'], tr['traj']@modes))

    def ridge(h, reg):
        a = h / np.maximum(np.linalg.norm(h[:n], axis=0), 1e-10)
        k = a[:n]@a[:n].T
        return a[n:]@a[:n].T@np.linalg.solve(k+reg*np.eye(n), y)

    a, b = ridge(h0, .00003), ridge(h1, .0003)
    pred = b.copy()
    pred[:, 0] = .65*a[:, 0]+.35*b[:, 0]
    return np.column_stack((pred[:, 0], pred[:, 1:]@modes.T))
