import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')


def _forecast(data, params):
    a_slow, b_couple, vmax, rho, tau_fast = params
    u, x, z, s, horizon = data.T
    u_n = u / 0.005
    inflow = vmax * (1.0 - rho * s / (1.0 + s)) * u_n / (1.2 + u_n)
    n = max(1, math.ceil(float(np.max(horizon)) / 0.005))
    generator = np.zeros((len(data), 3, 3))
    generator[:, 0, 0] = -a_slow
    generator[:, 0, 1] = b_couple
    generator[:, 0, 2] = inflow
    generator[:, 1, 0] = 0.52 / tau_fast
    generator[:, 1, 1] = -1.0 / tau_fast
    m = generator * (horizon / n)[:, None, None]
    eye = np.eye(3)
    # One classical RK4 step for the augmented state (x, z, 1).
    step = eye + m @ (eye + m @ (eye / 2.0 + m @ (eye / 6.0 + m / 24.0)))
    initial = np.stack((x, z, np.ones_like(x)), axis=-1)
    return (np.linalg.matrix_power(step, n) @ initial[..., None])[:, 0, 0]


def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    data = np.array([[row[key] for key in _KEYS] for row in rows], dtype=float)
    target = np.array([row['y'] for row in rows], dtype=float)
    result = least_squares(
        lambda params: _forecast(data, params) - target,
        x0=[1.5, 1.0, 6.0, 0.7, 0.2],
        bounds=([0.0, 0.0, 0.0, 0.0, 0.005], [20.0, 20.0, 25.0, 1.0, 5.0]),
        x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    _PARAMS = result.x


def predict(row) -> float:
    data = np.array([[row[key] for key in _KEYS]], dtype=float)
    return float(_forecast(data, _PARAMS)[0])