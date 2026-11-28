import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_features(x, j):
    p, q = x[:, j], x[:, (j + 1) % 4]
    return _affine(np.column_stack((p, q, p * q, p**3)))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    design = [_calibration_features(u, j) for j in range(4)]
    rng = np.random.default_rng(37)
    best = np.inf
    for _ in range(48):
        rows = rng.choice(len(u), 8, replace=False)
        fit = np.column_stack([a @ _ls(a[rows], v[rows, j]) for j, a in enumerate(design)])
        errors = np.linalg.norm(v - fit, axis=1)
        score = np.median(errors)
        if score < best:
            best = score
            keep = errors <= 5 * score
    return np.array([_ls(a[keep], v[keep, j]) for j, a in enumerate(design)])


def _unwarp(x, coeff):
    offset, gain, neighbor, cross, cubic = coeff.T
    j = np.arange(4)
    k = (j + 1) % 4
    linear = np.diag(gain)
    linear[j, k] = neighbor
    u = np.linalg.solve(linear, (x - offset).T).T
    jac = np.zeros((len(x), 4, 4))
    for _ in range(20):
        q = u[:, k]
        residual = offset + gain*u + neighbor*q + cross*u*q + cubic*u**3 - x
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, j, j] = gain + cross*q + 3*cubic*u**2
        jac[:, j, k] = neighbor + cross*u
        delta = np.linalg.solve(jac, residual[..., None])[..., 0]
        u -= delta / np.maximum(1, np.linalg.norm(delta, axis=1, keepdims=True))
    return u


def _operator(panel, key, features, start, bounds):
    u, v = panel['u'], panel[key]

    def residual(p):
        f = _affine(features(u, p))
        return (f @ _ls(f, v) - v).ravel()

    p = least_squares(residual, start, bounds=bounds).x
    c = _ls(_affine(features(u, p)), v)
    return lambda x: _affine(features(x, p)) @ c


def _panel_b(panel, x):
    transform = _operator(panel, 'v',
                          lambda u, p: np.sign(u)*np.maximum(abs(u)-p[0], 0)**p[1],
                          [.35, 1.5], ([.1, .8], [.7, 2.3]))
    return transform(x)


def _trajectory(z, zz, traj, times):
    q, r = np.linalg.qr(z, mode='reduced')
    target = (q.T @ traj).T

    def basis(p):
        return -np.expm1(-times[:, None]*p)

    def residual(p):
        h = basis(p)
        return (h @ _ls(h, target) - target).ravel()

    h = basis(least_squares(residual, [.4, 1.8], bounds=([.05, .85], [.85, 8])).x)
    return zz @ _ls(r, _ls(h, target).T) @ h.T


def _scalar(z):
    # The scalar readout couples two pairs of panel B coordinates.
    return np.column_stack((z, z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = _unwarp(np.concatenate((train['x'], study['test']['x'])), _calibrate(study['panel_a']))
    b = _panel_b(study['panel_b'], x)
    f = _scalar(b)
    y = f[n:] @ _ls(f[:n], train['y'])
    z = _affine(b)
    trajectory = _trajectory(z[:n], z[n:], train['traj'], study['eval_times'])
    end = trajectory[:, -1]
    return {'y_hat': y, 'state_pred': trajectory,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
