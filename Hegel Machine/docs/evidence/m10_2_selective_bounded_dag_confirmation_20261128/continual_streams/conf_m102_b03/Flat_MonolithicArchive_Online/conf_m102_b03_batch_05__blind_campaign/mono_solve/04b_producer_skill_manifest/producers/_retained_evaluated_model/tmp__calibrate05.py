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


