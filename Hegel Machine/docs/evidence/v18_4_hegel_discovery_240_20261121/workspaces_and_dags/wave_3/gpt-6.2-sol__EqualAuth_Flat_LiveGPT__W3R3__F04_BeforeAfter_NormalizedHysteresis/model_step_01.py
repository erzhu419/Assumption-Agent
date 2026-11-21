import numpy as np
from scipy.optimize import least_squares

_scale = 0.02
_params = None


def fit(train_rows):
    global _params
    u_now, u_past, h, y = np.array([[r['u_now'], r['u_past'], r['H'], r['y']] for r in train_rows], dtype=float).T
    u = u_now / _scale
    delta = (u_now - u_past) / _scale

    def residual(p):
        gain, tau, k_eq = p
        return gain * delta * np.exp(-h / tau) + k_eq * u - y

    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x


def predict(row) -> float:
    gain, tau, k_eq = _params
    u = row['u_now'] / _scale
    delta = (row['u_now'] - row['u_past']) / _scale
    return float(gain * delta * np.exp(-row['H'] / tau) + k_eq * u)
