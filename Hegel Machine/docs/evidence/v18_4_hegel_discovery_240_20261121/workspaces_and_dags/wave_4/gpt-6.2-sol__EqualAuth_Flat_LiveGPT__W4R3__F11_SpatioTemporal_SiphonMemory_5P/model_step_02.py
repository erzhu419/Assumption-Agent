import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 'x_now', 'mem_conv', 's_grad', 'H')
_params = None


def _forecast(p, data):
    a, b, log_tau, vmax, rho = p
    tau = math.exp(log_tau)
    u, x, z, s, H = data.T
    u = u / 0.05
    J = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    n = max(128, math.ceil(float(np.max(H)) / 0.005))
    dt = H / n

    def rhs(x, z):
        return -a * x + b * z + J, (0.52 * x - z) / tau

    for _ in range(n):
        k1, l1 = rhs(x, z)
        k2, l2 = rhs(x + 0.5 * dt * k1, z + 0.5 * dt * l1)
        k3, l3 = rhs(x + 0.5 * dt * k2, z + 0.5 * dt * l2)
        k4, l4 = rhs(x + dt * k3, z + dt * l3)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        z = z + dt * (l1 + 2 * l2 + 2 * l3 + l4) / 6.0
    return x


def fit(train_rows):
    global _params
    rows = list(train_rows)
    data = np.array([[r[k] for k in _KEYS] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    _params = least_squares(
        lambda p: _forecast(p, data) - y,
        [1.0, 1.0, math.log(0.4), 4.0, 0.4],
        bounds=([0.0, 0.0, math.log(0.005), 0.0, 0.0],
                [np.inf, np.inf, np.inf, np.inf, 1.0]),
    ).x
    return _params


def predict(row):
    data = np.array([[row[k] for k in _KEYS]], dtype=float)
    return float(_forecast(_params, data)[0])
