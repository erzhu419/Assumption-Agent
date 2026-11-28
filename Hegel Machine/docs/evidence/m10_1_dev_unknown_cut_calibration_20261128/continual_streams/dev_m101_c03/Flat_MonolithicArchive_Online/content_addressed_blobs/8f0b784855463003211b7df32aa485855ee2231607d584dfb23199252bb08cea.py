"""Calibrated two-branch prediction with coupled relaxation modes."""
import numpy as np
from scipy.optimize import least_squares, root


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_terms(z):
    w = np.roll(z, -1, axis=-1)
    return np.stack((np.ones_like(z), z, w, z*w, z**3), axis=-1)


def _decode(panel, x):
    u, v = panel['u'], panel['v']
    design = _calibration_terms(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        coef = np.array([_fit(design[keep, j], v[keep, j]) for j in range(4)])
        residual = np.linalg.norm((design * coef).sum(-1) - v, axis=1)
        keep = residual <= 6 * np.median(residual)
    coef = np.array([_fit(design[keep, j], v[keep, j]) for j in range(4)])
    return np.array([root(lambda z: (_calibration_terms(z)*coef).sum(-1) - row,
                          row, tol=1e-10).x for row in x])


def _cross(z):
    a, b, c, d = z.T
    return _affine(np.column_stack((a*b, a*d, b*c, c*d)))


def _second_operator(u, v):
    center = v.mean(axis=0)

    def activation(z, p):
        return z + p[0]*z*z + p[1]*(1 - np.cos(p[2]*z))

    def profile(p):
        x = activation(u, p)
        mean = x.mean(axis=0)
        q = x - mean
        left, values, right = np.linalg.svd(q.T @ (v - center))
        mix = (left @ right) * (values.sum() / np.sum(q*q))
        return q @ mix + center - v, mean, mix

    p = least_squares(lambda p: profile(p)[0].ravel(), [.28, .13, 2.],
                      bounds=([-1., -2., .7], [2., 2., 4.])).x
    _, mean, mix = profile(p)
    return lambda z: (activation(z, p) - mean) @ mix + center


def _features(study, x):
    z = _decode(study['panel_a'], x)
    panel = study['panel_c']
    u = panel['u']
    a = _cross(z) @ _fit(_cross(u), panel['v1'])
    b = _second_operator(u, panel['v2'])(z)
    dot = np.sum(a*b, axis=1)
    mix = a[:, 0]*b[:, 1]
    return _affine(np.column_stack((a, b, dot))), np.column_stack((b, mix)), np.column_stack((a, dot))


def _predict(times, observed, slow, fast, n):
    ns, nf = slow.shape[1], fast.shape[1]

    def temporal(p):
        return -np.expm1(-times[:, None] * p[:2])

    def profile(p):
        f, g = temporal(p).T
        design = np.column_stack(((slow[:n, None, :] * (f + p[2]*g)[None, :, None]).reshape(-1, ns),
                                  (fast[:n, None, :] * g[None, :, None]).reshape(-1, nf)))
        coef = _fit(design, observed.ravel())
        return design @ coef - observed.ravel(), coef

    p = least_squares(lambda p: profile(p)[0], [.4, 1.8, .1],
                      bounds=([.1, .9, -1.], [.89, 4., 1.])).x
    coef = profile(p)[1]
    s = slow[n:] @ coef[:ns]
    f = fast[n:] @ coef[ns:] + p[2]*s
    return np.column_stack((s, f)) @ temporal(p).T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    features, slow, fast = _features(study, x)
    y = features[n:] @ _fit(features[:n], train['y'])
    state = _predict(np.asarray(study['eval_times']), train['traj'], slow, fast, n)
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
