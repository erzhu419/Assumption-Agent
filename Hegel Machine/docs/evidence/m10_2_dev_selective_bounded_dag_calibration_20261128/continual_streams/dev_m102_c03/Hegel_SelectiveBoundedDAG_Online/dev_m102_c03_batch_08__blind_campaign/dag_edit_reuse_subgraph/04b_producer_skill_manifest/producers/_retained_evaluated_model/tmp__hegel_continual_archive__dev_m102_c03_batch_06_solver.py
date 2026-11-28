"""Calibrated dual-branch regression with shared exponential dynamics."""
import numpy as np
from scipy.optimize import least_squares


def _fit(h, targets):
    return np.linalg.lstsq(h, targets, rcond=None)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _edge(x):
    return x * np.roll(x, -1, axis=1)


def _cal_design(x, j):
    u, v = x[:, j], x[:, (j + 1) % 4]
    return _one(np.column_stack((u, v, u*v, u**3)))


def _calibration(panel):
    u, v = panel['u'], panel['v']
    h = _one(u)
    r = np.linalg.norm(h @ _fit(h, v) - v, axis=1)
    good = r <= 4 * max(np.median(r), 1e-8)
    for _ in range(4):
        a = np.stack([_fit(_cal_design(u[good], j), v[good, j]) for j in range(4)])
        r = np.linalg.norm(np.column_stack([_cal_design(u, j) @ a[j] for j in range(4)]) - v, axis=1)
        new = r <= 5 * max(np.median(r), 1e-6)
        if new.sum() < 10 or np.array_equal(good, new):
            break
        good = new
    return a


def _inverse(x, a):
    bias, scale, neighbor, mixed, cubic = a.T
    z = (x - bias) / scale
    j = np.arange(4)
    for _ in range(14):
        w = np.roll(z, -1, axis=1)
        r = bias + scale*z + neighbor*w + mixed*z*w + cubic*z**3 - x
        jac = np.zeros((len(x), 4, 4))
        jac[:, j, j] = scale + mixed*w + 3*cubic*z*z
        jac[:, j, (j + 1) % 4] = neighbor + mixed*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= np.clip(step, -1., 1.)
        if np.max(np.abs(step), initial=0.) < 1e-12:
            break
    return z



def _design(study, x):
    """Calibrated cyclic edges, cosine branch, and cubic edge modulation."""
    z = _inverse(x, _calibration(study['panel_a']))
    p = study['panel_c']
    u = np.asarray(p['u'])
    c = _one(_edge(z)) @ _fit(_one(_edge(u)), p['v1'])
    d = _one(np.cos(2.4*z)) @ _fit(_one(np.cos(2.4*u)), p['v2'])
    # Use an independent basis for the products of a coordinate and a cyclic
    # edge, rather than fitting 16 redundant cross products.
    nxt = np.roll(z, -1, axis=1)
    triples = np.column_stack([np.prod(np.delete(z, j, axis=1), axis=1)
                               for j in range(4)])
    return np.column_stack((_one(z), c, d, np.sum(c*d, axis=1),
                            c[:, 0]*d[:, 1], triples, z*z*nxt, z*nxt*nxt))


def _time_basis(t, traj):
    """Fit three shared relaxation rates by variable projection."""
    def basis(r):
        return -np.expm1(-t[:, None]*r)

    def residual(r):
        q = basis(r)
        return (q @ _fit(q, traj.T) - traj.T).ravel()

    rates = least_squares(residual, [.40, 1.8, 4.7],
                          bounds=([.05, .9, 3.0], [1.0, 3.0, 12.0]),
                          max_nfev=120).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x = np.asarray(tr['x'], dtype=float)
    xt = np.asarray(study['test']['x'], dtype=float)
    y = np.asarray(tr['y'], dtype=float)
    traj = np.asarray(tr['traj'], dtype=float)
    t = np.asarray(study['eval_times'], dtype=float)
    h, ht = np.split(_design(study, np.concatenate((x, xt), axis=0)), [len(x)])
    y_hat = ht @ _fit(h, y)
    q = _time_basis(t, traj)
    # Orthogonal projection avoids using ill-conditioned fitted amplitudes
    # as independent response variables.
    qt, _ = np.linalg.qr(q, mode='reduced')
    state_pred = (ht @ _fit(h, traj @ qt)) @ qt.T
    end = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
