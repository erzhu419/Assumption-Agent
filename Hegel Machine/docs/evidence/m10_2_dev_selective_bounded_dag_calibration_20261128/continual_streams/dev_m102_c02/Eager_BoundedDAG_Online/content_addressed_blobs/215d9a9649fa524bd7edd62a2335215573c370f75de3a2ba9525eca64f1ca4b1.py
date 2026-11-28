"""Recover latent coordinates from the two paired reference panels."""
import numpy as np
from scipy.optimize import minimize_scalar


def _a_design(u):
    nxt = np.roll(u, -1, axis=1)
    return np.stack((np.ones_like(u), u, nxt, u * nxt, u**3), axis=-1).transpose(1, 0, 2)


def _fit_a(u, v):
    design = _a_design(u)
    target = v.T[..., None]
    rng = np.random.default_rng(485)
    best = (np.inf, None)
    for _ in range(128):
        idx = rng.choice(len(u), 8, replace=False)
        b = np.linalg.pinv(design[:, idx]) @ target[:, idx]
        residual = (design @ b - target)[..., 0]
        score = np.median(np.linalg.norm(residual, axis=0))
        if score < best[0]:
            best = (score, b)
    b = best[1]
    for _ in range(4):
        r = np.linalg.norm((design @ b - target)[..., 0], axis=0)
        keep = r <= max(6 * np.median(r), 1e-8)
        b = np.linalg.pinv(design[:, keep]) @ target[:, keep]
    return b[:, :, 0].T


def _invert_a(x, b):
    x = np.asarray(x, dtype=float)
    bias, gain, cross, product, cubic = b
    d = x.shape[1]
    k = np.arange(d)
    nxt = (k + 1) % d
    linear = np.zeros((d, d))
    linear[k, k], linear[k, nxt] = gain, cross
    u = np.linalg.solve(linear, (x - bias).T).T
    for _ in range(35):
        v = u[:, nxt]
        r = bias + gain*u + cross*v + product*u*v + cubic*u**3 - x
        loss = np.max(np.abs(r), axis=1)
        if not len(loss) or np.max(loss) <= 1e-12:
            break
        jac = np.zeros((len(u), d, d))
        jac[:, k, k] = gain + product*v + 3*cubic*u*u
        jac[:, k, nxt] = cross + product*u
        delta = np.linalg.solve(jac, r[..., None])[..., 0]
        step = np.ones((len(u), 1))
        for _ in range(18):
            trial = u - step*delta
            v = trial[:, nxt]
            rr = bias + gain*trial + cross*v + product*trial*v + cubic*trial**3 - x
            bad = np.max(np.abs(rr), axis=1) > loss
            if not np.any(bad):
                break
            step[bad] *= 0.5
        u = np.where(bad[:, None], u, trial)
    return u


def _b_design(u, scale):
    return np.column_stack((np.ones(len(u)), u, u*np.exp(-scale*u*u)))


def _fit_b(u, v):
    def fit(scale):
        design = _b_design(u, scale)
        beta = np.linalg.lstsq(design, v, rcond=None)[0]
        return np.mean((design @ beta - v)**2), beta
    scale = minimize_scalar(lambda s: fit(s)[0], bounds=(0.1, 3.0), method='bounded').x
    return scale, fit(scale)[1]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    a_fit = _fit_a(np.asarray(a['u'], float), np.asarray(a['v'], float))
    scale, beta = _fit_b(np.asarray(b['u'], float), np.asarray(b['v'], float))
    def coordinates(x):
        return _b_design(_invert_a(x, a_fit), scale) @ beta
    return {'c': coordinates(study['train']['x']),
            'query': coordinates(study['test']['x'])}
