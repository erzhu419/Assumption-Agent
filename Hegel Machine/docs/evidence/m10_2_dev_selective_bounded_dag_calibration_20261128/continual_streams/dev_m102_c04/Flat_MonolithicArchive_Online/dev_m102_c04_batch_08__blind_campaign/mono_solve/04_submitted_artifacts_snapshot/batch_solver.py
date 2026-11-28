import numpy as np
from scipy.optimize import least_squares, minimize_scalar

_PAIRS = ((0, 1), (0, 3), (1, 2), (2, 3))
_TRIPLES = tuple(sorted({tuple(sorted((i, j, k))) for i in range(4) for j, k in _PAIRS}))


def _ls(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _aff(*cols):
    return np.column_stack((np.ones(len(cols[0])), *cols))


def _groups(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        calib = np.column_stack([_ls(_aff(u[keep, j]), v[keep, j]) for j in range(4)])
        e = np.linalg.norm(_aff(u)[:, :1] * calib[0] + u * calib[1] - v, axis=1)
        new = e <= max(4 * np.median(e), 1e-10)
        if np.array_equal(keep, new) or new.sum() < 8:
            break
        keep = new
    z = (x - calib[0]) / calib[1]
    c = study['panel_c']
    u = c['u']

    def quadratic(x):
        return _aff(*(x[:, i] * x[:, j] for i, j in _PAIRS))

    def cosine(x, rate):
        return _aff(np.cos(rate * x))

    def error(rate):
        a = cosine(u, rate)
        return np.sum((a @ _ls(a, c['v2']) - c['v2']) ** 2)

    freq = minimize_scalar(error, bounds=(2.15, 2.65), method='bounded').x
    p = quadratic(z) @ _ls(quadratic(u), c['v1'])
    r = cosine(z, freq) @ _ls(cosine(u, freq), c['v2'])
    fast = np.column_stack((p, np.sum(p * r, axis=1)))
    slow = np.column_stack((r, p[:, 0] * r[:, 1]))
    shared = np.column_stack((z, *(np.prod(z[:, ijk], axis=1) for ijk in _TRIPLES)))
    return fast, slow, shared


def _predict(study, variant='joint'):
    train = study['train']
    n = len(train['x'])
    y, traj = train['y'], train['traj']
    t = np.asarray(study['eval_times'])
    fast, slow, shared = _groups(study, np.concatenate((train['x'], study['test']['x'])))
    scalar = _aff(fast, slow[:, :4], shared)
    if variant == 'linear':
        state = _aff(fast, slow, shared)
        return scalar[n:] @ _ls(scalar[:n], y), state[n:] @ _ls(state[:n], traj)

    def blocks(a, lo, hi):
        f = -np.expm1(-a[0] * t)
        s = -np.expm1(-a[1] * t)
        h = np.sqrt(t) * (a[2] if len(a) == 3 else 1.)
        return np.column_stack(tuple((g[lo:hi, None, :] * v[None, :, None]).reshape((hi - lo)*len(t), g.shape[1]) for g, v in zip((fast, slow, shared), (f, s, h))))

    if variant == 'separate':
        def residual(a):
            b = blocks(a, 0, n)
            return b @ _ls(b, traj.ravel()) - traj.ravel()
        a = least_squares(residual, [1.8, .55], bounds=([.3, .08], [5., 2.])).x
        w = _ls(blocks(a, 0, n), traj.ravel())
        return scalar[n:] @ _ls(scalar[:n], y), (blocks(a, n, len(scalar)) @ w).reshape(-1, len(t))

    # Separate linear readouts for the panel branches, but one spatial cubic
    # latent shared by the scalar and time-dependent measurements.
    k = scalar.shape[1] - shared.shape[1]
    nstate = fast.shape[1] + slow.shape[1]
    weight = 1.6
    top = weight * np.column_stack((scalar[:n, :k], np.zeros((n, nstate)), shared[:n]))
    pad = np.zeros((n * len(t), k))
    target = np.r_[weight * y, traj.ravel()]

    def design(a):
        return np.vstack((top, np.column_stack((pad, blocks(a, 0, n)))))

    def residual(a):
        b = design(a)
        return b @ _ls(b, target) - target

    a = least_squares(residual, [1.8, .55, .55], bounds=([.3, .08, .1], [5., 2., 1.5])).x
    w = _ls(design(a), target)
    yh = scalar[n:, :k] @ w[:k] + shared[n:] @ w[k + nstate:]
    state = (blocks(a, n, len(scalar)) @ w[k:]).reshape(-1, len(t))
    return yh, state


def solve_batch(study: dict) -> dict:
    y, state = _predict(study)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
