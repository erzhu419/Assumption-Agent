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



def _panel_features(study, x):
    z = _calibrate(study['panel_a'], x)
    panel = study['panel_c']
    u = np.asarray(panel['u'])

    def products(a):
        return _design(a * np.roll(a, -1, axis=1))

    def cosines(a):
        return _design(np.cos(2.4 * a))

    p = products(z) @ _ls(products(u), panel['v1'])
    q = cosines(z) @ _ls(cosines(u), panel['v2'])
    dot = np.einsum('ij,ij->i', p, q)
    cross01 = p[:, 0] * q[:, 1]
    scalar = _design(np.column_stack((p, q, dot)))
    fast = _design(np.column_stack((p, q, dot, cross01)))
    slow = _design(np.column_stack((q, cross01)))
    return scalar, fast, slow


def _state(times, observations, fast, slow):
    n = len(observations)

    def matrix(rates, f, g):
        h = -np.expm1(-times[:, None] * rates)
        terms = (f[:, None, :] * h[None, :, 0, None],
                 g[:, None, :] * h[None, :, 1, None])
        return np.concatenate(terms, axis=2).reshape(len(f) * len(times), f.shape[1] + g.shape[1])

    target = observations.ravel()

    def residual(rates):
        d = matrix(rates, fast[:n], slow[:n])
        return d @ _ls(d, target) - target

    fit = least_squares(residual, [1.8, 0.45], bounds=([0.8, 0.05], [4., 0.78]), ftol=1e-10)
    coef = _ls(matrix(fit.x, fast[:n], slow[:n]), target)
    return (matrix(fit.x, fast[n:], slow[n:]) @ coef).reshape(len(fast) - n, len(times))


def solve_batch(study: dict) -> dict:
    """Predict endpoints and two-mode relaxation traces from calibrated panel C features."""
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x']))
    scalar, fast, slow = _panel_features(study, x)
    y = scalar[n:] @ _ls(scalar[:n], np.asarray(train['y']))
    state = _state(np.asarray(study['eval_times']), np.asarray(train['traj']), fast, slow)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
