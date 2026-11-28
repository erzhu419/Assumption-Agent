"""Infer the calibration, cyclic readout, and shared trajectory modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.linear_model import lars_path


def _fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _cal_design(x):
    v = np.roll(x, -1, axis=1)
    return np.stack((np.ones_like(x), x, v, x*v, x**3), axis=-1)


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = _cal_design(u)
    good = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = np.stack([_fit(d[good, j], v[good, j]) for j in range(u.shape[1])])
        e = np.linalg.norm(np.einsum('nki,ki->nk', d, w)-v, axis=1)
        good = e <= np.sort(e)[int(.79*len(u))]
    good = e < 12*np.median(e)
    return np.stack([_fit(d[good, j], v[good, j]) for j in range(u.shape[1])])


def _inverse_calibration(measured, w):
    x = (measured-w[:, 0])/w[:, 1]
    j = np.arange(x.shape[1])
    for _ in range(18):
        nxt = np.roll(x, -1, axis=1)
        jac = np.zeros((len(x), len(j), len(j)))
        jac[:, j, j] = w[:, 1]+w[:, 3]*nxt+3*w[:, 4]*x*x
        jac[:, j, (j+1)%len(j)] = w[:, 2]+w[:, 3]*x
        res = np.einsum('nki,ki->nk', _cal_design(x), w)-measured
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        x -= step
        if np.max(np.abs(step), initial=0.) < 1e-11:
            break
    return x


from scipy.linalg import cho_factor, cho_solve


def _branches(study, raw):
    x = _inverse_calibration(raw, _calibration(study['panel_a']))
    axle = study['panel_b']
    def bdesign(x):
        return _affine(x**3 / np.sqrt(1 + .70020846*x**4) - .08899942*x)
    b = bdesign(x) @ _fit(bdesign(axle['u']), axle['v'])
    c = study['panel_c']
    def c1design(v):
        return _affine(v * np.roll(v, -1, axis=1))
    def c2design(v):
        return np.c_[_affine(v), np.cos(1.28247*v)]
    return [x, b, c1design(b)@_fit(c1design(c['u']), c['v1']),
            c2design(b)@_fit(c2design(c['u']), c['v2'])]


def _predict(g, t, n, penalty):
    mu = t.mean(axis=0)
    f = cho_factor(g[:n, :n] + penalty*np.eye(n))
    return mu + g[n:, :n] @ cho_solve(f, t-mu)


def _scalar_design(parts):
    z = np.column_stack(parts[:3])
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((z, z[:, i]*z[:, j]))


def _sparse_readout(design, target, n, terms=25):
    d = (design-design[:n].mean(0))/np.maximum(design[:n].std(0), 1e-8)
    mean = target.mean()
    _, _, path = lars_path(d[:n], target-mean, max_iter=85, method='lasso')
    counts = np.count_nonzero(path, axis=0)
    weights = np.array([np.interp(terms, counts, row) for row in path])
    return mean+d[n:]@weights


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    parts = _branches(study, np.r_[tr['x'], study['test']['x']])
    y = _sparse_readout(_scalar_design(parts), tr['y'], n)
    q = np.column_stack(parts)
    q = (q-q[:n].mean(0))/q[:n].std(0)
    g = q @ q.T / q.shape[1]
    state = _predict((1+g)**2, tr['traj'], n, .002)
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
