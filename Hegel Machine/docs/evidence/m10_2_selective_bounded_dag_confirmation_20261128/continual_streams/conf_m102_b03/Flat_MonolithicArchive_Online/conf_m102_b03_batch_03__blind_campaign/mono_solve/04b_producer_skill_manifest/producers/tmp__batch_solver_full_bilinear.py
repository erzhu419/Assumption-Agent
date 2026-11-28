"""Calibrate the four channels and fit two coupled response branches."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    linear = _design(u)
    rng = np.random.default_rng(45)
    fits = (_ls(linear[i], v[i]) for i in (rng.choice(len(u), 8, replace=False) for _ in range(120)))
    pilot = min(fits, key=lambda q: np.median(np.linalg.norm(linear @ q - v, axis=1)))
    residual = np.linalg.norm(linear @ pilot - v, axis=1)
    good = residual <= max(0.005, 5 * np.median(residual))
    pilot = _ls(linear[good], v[good])
    coefs = []
    for j in range(u.shape[1]):
        a, b = u[:, j], u[:, (j + 1) % u.shape[1]]
        X = _design(np.column_stack((a, b, a*b, a**3)))
        coefs.append(_ls(X[good], v[good, j]))
    bias, diag, adjacent, interaction, cubic = np.array(coefs).T
    z = np.linalg.solve(pilot[1:].T, (x-pilot[0]).T).T
    k = np.arange(z.shape[1])
    for _ in range(18):
        w = np.roll(z, -1, axis=1)
        r = bias + diag*z + adjacent*w + interaction*z*w + cubic*z**3 - x
        if not r.size or np.max(np.abs(r)) < 1e-12:
            break
        J = np.zeros((len(z), len(k), len(k)))
        J[:, k, k] = diag + interaction*w + 3*cubic*z*z
        J[:, k, (k+1) % len(k)] = adjacent + interaction*z
        z -= np.linalg.solve(J, r[..., None])[..., 0]
    return z


def _branches(panel, x):
    u = panel['u']
    out = []
    for key, f, initial in [('v1', lambda w: w/(1+np.abs(w)), 0.84),
                            ('v2', lambda w: w*expit(w), 1.45)]:
        v = panel[key]
        def design(z, h):
            return _design(np.column_stack((z, f(h[0]*z))))
        def residual(h):
            A = design(u, h)
            return (A @ _ls(A, v) - v).ravel()
        h = least_squares(residual, [initial], bounds=(0.15, 5)).x
        A = design(u, h)
        out.append(design(x, h) @ _ls(A, v))
    return out


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    c, d = _branches(study['panel_c'], z)
    pair = c[:, :, None] * d[:, None, :]
    main = np.column_stack((c, d, np.trace(pair, axis1=1, axis2=2)))
    off = ~np.eye(c.shape[1], dtype=bool)
    cross = np.column_stack((c, d, pair.reshape(len(z), c.shape[1]*d.shape[1])))
    return _design(main), _design(cross)


def _time_basis(t, traj):
    def basis(rate):
        return -np.expm1(-t[:, None]*rate)
    def residual(rate):
        H = basis(rate)
        return (H @ _ls(H, traj.T) - traj.T).ravel()
    rate = least_squares(residual, [1.8, 0.4], bounds=([0.7, 0.02], [6., 1.2])).x
    return basis(rate)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    y_design, state_design = _features(study, np.concatenate((train['x'], study['test']['x'])))
    y_hat = y_design[n:] @ _ls(y_design[:n], train['y'])
    H = _time_basis(study['eval_times'], train['traj'])
    amps = _ls(H, train['traj'].T).T
    state_pred = state_design[n:] @ _ls(state_design[:n], amps) @ H.T
    last = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
